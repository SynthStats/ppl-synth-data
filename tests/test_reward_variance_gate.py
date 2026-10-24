"""Behaviour tests for the package reward-variance gate."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppl_synth_data.gates import reward_variance as gate

FLOOR = gate.TRAINING_LOG_REWARD_FLOOR


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(row_id: str, *, runtime: str = "pymc") -> dict[str, object]:
    return {
        "row_id": row_id,
        "runtime": runtime,
        "split": "train",
        "data_source": "fixture",
        "structural_class": None,
        "prompt": 'Observed data JSON: {"y": [1]}',
        "visible_data": {"y": [1]},
        "heldout_data": {"y": [2]},
        "dataset": gate.DATASET,
        "revision": "a" * 40,
    }


def _verifier_settings(*, data_shape: str = "lists") -> gate.VerifierSettings:
    return gate.VerifierSettings(
        data_shape=data_shape,
        pymc_draws=50,
        pymc_tune=50,
        pymc_timeout_sec=60,
        lazyppl_timeout_sec=600,
        lazyppl_harbor_image="lazyppl-harbor:test",
    )


def test_build_parser_wires_system_prompt_and_thinking(tmp_path: Path) -> None:
    system_prompt = tmp_path / "system.txt"
    args = gate.build_parser().parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "sample",
            "--model",
            "policy",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--system-prompt-file",
            str(system_prompt),
            "--thinking",
        ]
    )
    assert args.system_prompt_file == system_prompt
    assert args.thinking is True


def test_build_parser_accepts_score_measurement_flags(tmp_path: Path) -> None:
    args = gate.build_parser().parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "score",
            "--data-shape",
            "arrays",
            "--pymc-draws",
            "125",
            "--pymc-tune",
            "75",
            "--pymc-timeout-sec",
            "180",
        ]
    )
    assert args.data_shape == "arrays"
    assert args.pymc_draws == 125
    assert args.pymc_tune == 75
    assert args.pymc_timeout_sec == 180


def test_build_parser_uses_the_pymc_scorer_defaults(tmp_path: Path) -> None:
    import inspect

    from ppl_synth_data.validation.pymc_semantics import score_pymc_semantics

    scorer = inspect.signature(score_pymc_semantics).parameters
    args = gate.build_parser().parse_args(["--out-dir", str(tmp_path), "score"])
    assert args.data_shape == "lists"
    assert args.pymc_draws == scorer["draws"].default
    assert args.pymc_tune == scorer["tune"].default
    assert args.pymc_timeout_sec == scorer["timeout_sec"].default


@pytest.mark.parametrize("command", ["report", "--help"])
def test_build_parser_does_not_need_the_pipeline_config(tmp_path: Path, command: str) -> None:
    # Only scoring resolves verifier settings, and Settings() needs configs/default.yaml. Import a
    # copy of the package in isolated mode from a directory without configs/, so a checkout's
    # configs cannot hide the dependency; RUN_ROOT is dropped for the same reason.
    shutil.copytree(
        Path(gate.__file__).resolve().parents[1],
        tmp_path / "src" / "ppl_synth_data",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        root = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(root / "src"))

        from ppl_synth_data.gates import reward_variance

        assert Path(reward_variance.__file__).resolve().is_relative_to(root)
        assert not (root / "configs").exists()
        reward_variance.build_parser().parse_args(["--out-dir", str(root / "run"), sys.argv[2]])
        """
    )
    env = {key: value for key, value in os.environ.items() if key.lower() != "run_root"}
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(tmp_path), command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    if command == "--help":
        assert "usage:" in completed.stdout


def test_main_passes_score_measurement_flags_to_stage(tmp_path: Path) -> None:
    row = _row("row")
    _write_jsonl(tmp_path / "rows.jsonl", [row])
    _write_jsonl(
        tmp_path / "completions.jsonl",
        [{"row_id": "row", "sample_idx": 0, "text": "program"}],
    )
    _write_jsonl(
        tmp_path / "scores.jsonl",
        [{"row_id": "row", "sample_idx": 0, "data_shape": "arrays"}],
    )
    manifest = gate.sample_manifest(
        rows=[row],
        model="policy",
        base_url="http://127.0.0.1:8000/v1",
        k=4,
        temperature=0.7,
        max_tokens=128,
        thinking=False,
        system_prompt=None,
    )
    expected = replace(
        gate.verifier_settings(),
        data_shape="arrays",
        pymc_draws=125,
        pymc_tune=75,
        pymc_timeout_sec=180,
    )
    manifest["score"] = asdict(expected)
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    assert (
        gate.main(
            [
                "--out-dir",
                str(tmp_path),
                "score",
                "--data-shape",
                "arrays",
                "--pymc-draws",
                "125",
                "--pymc-tune",
                "75",
                "--pymc-timeout-sec",
                "180",
            ]
        )
        == 0
    )
    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written["score"] == asdict(expected)


@pytest.mark.parametrize(
    "argv",
    [
        ["sample", "--model", "m", "--base-url", "http://host", "--k", "1"],
        ["sample", "--model", "m", "--base-url", "http://host", "--concurrency", "0"],
        ["sample", "--model", "m", "--base-url", "http://host", "--max-tokens", "0"],
        ["score", "--workers", "0"],
        ["score", "--pymc-draws", "0"],
        ["score", "--pymc-tune", "-1"],
        ["score", "--pymc-timeout-sec", "0"],
        ["report", "--k", "1"],
    ],
)
def test_build_parser_rejects_invalid_positive_counts(tmp_path: Path, argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        gate.build_parser().parse_args(["--out-dir", str(tmp_path), *argv])


def test_visible_data_parses_pymc_prompt() -> None:
    prompt = (
        "Write a PyMC probabilistic model. Scenario: crashes by road length. "
        'Assume `pm` is available. Observed data JSON: {"road_km":[2.1,3.5],"crashes":[0,3]}. '
        "Return only one fenced block."
    )
    assert gate.visible_data_from_prompt(prompt) == {"road_km": [2.1, 3.5], "crashes": [0, 3]}


def test_visible_data_parses_lazyppl_prompt() -> None:
    prompt = (
        'Scenario: claims. Observed row fields JSON: {"count":[7,0,0]} '
        "Return only one fenced block."
    )
    assert gate.visible_data_from_prompt(prompt) == {"count": [7, 0, 0]}


def test_visible_data_raises_without_marker() -> None:
    with pytest.raises(ValueError):
        gate.visible_data_from_prompt("Scenario: nothing observed here.")


def test_strip_null_columns_drops_loader_padding() -> None:
    assert gate.strip_null_columns({"y": [1.0], "x": None, "groupId": None}) == {"y": [1.0]}


def test_pull_rejects_missing_declared_holdout(tmp_path: Path, monkeypatch) -> None:
    record = {
        "metadata": {
            "row_id": "declared-heldout",
            "data_source": "fixture",
            "structural_class": None,
            "hidden_data_sha256": "b" * 64,
        },
        "prompt": 'Observed data JSON: {"y": [1]}',
    }

    def fake_load_dataset(_dataset: str, config: str, *, revision: str):
        del revision
        if config.endswith("_heldout"):
            return {"heldout": []}
        return {"train": [record]}

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )
    with pytest.raises(ValueError, match="declared-heldout"):
        gate.stage_pull(out_dir=tmp_path, revision="a" * 40)


def test_pull_records_null_when_row_does_not_declare_holdout(tmp_path: Path, monkeypatch) -> None:
    record = {
        "metadata": {
            "row_id": "no-heldout",
            "data_source": "fixture",
            "structural_class": None,
            "hidden_data_sha256": None,
        },
        "prompt": 'Observed data JSON: {"y": [1]}',
    }

    def fake_load_dataset(_dataset: str, config: str, *, revision: str):
        del revision
        if config.endswith("_heldout"):
            return {"heldout": []}
        return {"train": [record]}

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )
    gate.stage_pull(out_dir=tmp_path, revision="a" * 40)
    assert all(row["heldout_data"] is None for row in gate.read_jsonl(tmp_path / "rows.jsonl"))


def test_missing_sample_requests_preserve_partial_rows() -> None:
    rows = [_row("partial"), _row("complete")]
    completions = [
        {"row_id": "partial", "sample_idx": 1},
        *({"row_id": "complete", "sample_idx": idx} for idx in range(4)),
    ]
    requests = gate.missing_sample_requests(rows, completions, k=4)
    assert [(row["row_id"], indices) for row, indices in requests] == [("partial", (0, 2, 3))]


def test_sample_row_assigns_missing_indices_and_preserves_empty_completion() -> None:
    captured: dict[str, object] = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    index=1,
                    message=SimpleNamespace(content="second"),
                    finish_reason="stop",
                ),
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(content=None),
                    finish_reason="length",
                ),
            ],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    completions = asyncio.run(
        gate._sample_row(
            client,
            model="policy",
            row=_row("partial"),
            sample_indices=(0, 3),
            temperature=0.7,
            max_tokens=32,
            thinking=False,
            system_prompt=None,
            semaphore=asyncio.Semaphore(1),
        )
    )
    assert captured["n"] == 2
    assert [(item.sample_idx, item.text, item.finish_reason) for item in completions] == [
        (0, "", "length"),
        (3, "second", "stop"),
    ]


def test_manifest_drift_names_the_field(tmp_path: Path) -> None:
    manifest = gate.sample_manifest(
        rows=[_row("row")],
        model="policy-a",
        base_url="http://127.0.0.1:8000/v1",
        k=4,
        temperature=0.7,
        max_tokens=128,
        thinking=False,
        system_prompt="instructions",
    )
    gate.ensure_sample_manifest(tmp_path, manifest)
    changed = {**manifest, "model": "policy-b"}
    with pytest.raises(ValueError, match="model"):
        gate.ensure_sample_manifest(tmp_path, changed)


def test_manifest_records_prompt_hash_and_host() -> None:
    manifest = gate.sample_manifest(
        rows=[_row("row")],
        model="policy",
        base_url="http://127.0.0.1:8000/v1",
        k=4,
        temperature=0.7,
        max_tokens=128,
        thinking=True,
        system_prompt="instructions",
    )
    assert manifest["base_url_host"] == "127.0.0.1"
    assert manifest["system_prompt_sha256"] == (
        "238fa28a94976c7da14563bc873c2729bd5cd325389085bb4c6dd0de28923590"
    )


def test_score_settings_are_recorded_in_the_sample_manifest(tmp_path: Path) -> None:
    manifest = gate.sample_manifest(
        rows=[_row("row")],
        model="policy",
        base_url="http://127.0.0.1:8000/v1",
        k=4,
        temperature=0.7,
        max_tokens=128,
        thinking=False,
        system_prompt=None,
    )
    gate.ensure_sample_manifest(tmp_path, manifest)
    gate._record_score_manifest(tmp_path, _verifier_settings())
    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written["score"] == {
        "data_shape": "lists",
        "pymc_draws": 50,
        "pymc_tune": 50,
        "pymc_timeout_sec": 60,
        "lazyppl_timeout_sec": 600,
        "lazyppl_harbor_image": "lazyppl-harbor:test",
    }


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("data_shape", "arrays"),
        ("pymc_draws", 100),
        ("pymc_tune", 100),
        ("pymc_timeout_sec", 120),
        ("lazyppl_harbor_image", "lazyppl-harbor:changed"),
    ],
)
def test_score_manifest_drift_with_existing_scores_names_field(
    tmp_path: Path,
    field: str,
    changed_value: object,
) -> None:
    manifest = gate.sample_manifest(
        rows=[_row("row")],
        model="policy",
        base_url="http://127.0.0.1:8000/v1",
        k=4,
        temperature=0.7,
        max_tokens=128,
        thinking=False,
        system_prompt=None,
    )
    gate.ensure_sample_manifest(tmp_path, manifest)
    settings = _verifier_settings()
    gate._record_score_manifest(tmp_path, settings)
    _write_jsonl(
        tmp_path / "scores.jsonl",
        [{"row_id": "row", "sample_idx": 0, "data_shape": "lists"}],
    )

    with pytest.raises(ValueError, match=field):
        gate._record_score_manifest(
            tmp_path,
            replace(settings, **{field: changed_value}),
        )


def test_score_manifest_drift_with_existing_scores_names_lazyppl_timeout_field(
    tmp_path: Path,
) -> None:
    manifest = gate.sample_manifest(
        rows=[_row("row")],
        model="policy",
        base_url="http://127.0.0.1:8000/v1",
        k=4,
        temperature=0.7,
        max_tokens=128,
        thinking=False,
        system_prompt=None,
    )
    gate.ensure_sample_manifest(tmp_path, manifest)
    settings = _verifier_settings()
    gate._record_score_manifest(tmp_path, settings)
    _write_jsonl(
        tmp_path / "scores.jsonl",
        [{"row_id": "row", "sample_idx": 0, "data_shape": "lists"}],
    )

    with pytest.raises(ValueError, match="lazyppl_timeout_sec"):
        gate._record_score_manifest(
            tmp_path,
            replace(settings, lazyppl_timeout_sec=601),
        )


def test_reward_law_floors_non_holdout_metrics() -> None:
    assert gate.to_log_reward("avg_holdout_logp", -3.2) == -3.2
    assert gate.to_log_reward("avg_holdout_logp", -40.0) == FLOOR
    assert gate.to_log_reward(None, None) == FLOOR
    assert gate.to_log_reward("other_metric", 1.0) == FLOOR


def test_pymc_real_reward_records_reward_source(monkeypatch) -> None:
    from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime
    from ppl_synth_data.validation import pymc_semantics

    monkeypatch.setattr(PyMCRuntime, "extract_code", lambda self, text: "model")
    monkeypatch.setattr(
        pymc_semantics,
        "score_pymc_semantics",
        lambda *args, **kwargs: SimpleNamespace(
            ok=True,
            metric_name="avg_holdout_logp",
            metric_value=-3.0,
            status="pass",
            message="scored",
        ),
    )
    score = gate.score_one(
        _row("pymc"),
        {"row_id": "pymc", "sample_idx": 0, "text": "program"},
        _verifier_settings(),
    )
    assert score.training_log_reward == -3.0
    assert score.reward_source == "validation.pymc_elpd"


def test_score_one_shapes_pymc_array_data_and_records_provenance(monkeypatch) -> None:
    from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime
    from ppl_synth_data.validation import pymc_semantics

    captured: dict[str, object] = {}

    def fake_score(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ok=True,
            metric_name="avg_holdout_logp",
            metric_value=-3.0,
            status="pass",
            message="scored",
        )

    monkeypatch.setattr(PyMCRuntime, "extract_code", lambda self, text: "model")
    monkeypatch.setattr(pymc_semantics, "score_pymc_semantics", fake_score)
    row = _row("pymc")
    row["visible_data"] = {"y": [1, 2], "scale": 3, "label": "observed"}
    row["heldout_data"] = {"y": [4, 5], "scale": 6, "label": "heldout"}

    score = gate.score_one(
        row,
        {"row_id": "pymc", "sample_idx": 0, "text": "program"},
        _verifier_settings(data_shape="arrays"),
    )

    visible_data = captured["visible_data"]
    hidden_data = captured["hidden_data"]
    assert isinstance(visible_data, dict)
    assert isinstance(hidden_data, dict)
    assert isinstance(visible_data["y"], np.ndarray)
    assert isinstance(hidden_data["y"], np.ndarray)
    assert visible_data["y"].tolist() == [1, 2]
    assert hidden_data["y"].tolist() == [4, 5]
    assert visible_data["scale"] == 3
    assert visible_data["label"] == "observed"
    assert hidden_data["scale"] == 6
    assert hidden_data["label"] == "heldout"
    assert captured["draws"] == 50
    assert captured["tune"] == 50
    assert captured["timeout_sec"] == 60
    assert score.data_shape == "arrays"


def test_lazyppl_valid_artifact_uses_unmodified_training_reward(monkeypatch) -> None:
    from ppl_synth_data.runtimes.lazyppl_runtime import LazyPPLRuntime
    from ppl_synth_data.validation import lazyppl_semantics

    monkeypatch.setattr(LazyPPLRuntime, "extract_code", lambda self, text: "model")
    monkeypatch.setattr(
        lazyppl_semantics,
        "score_lazyppl_semantics",
        lambda *args, **kwargs: SimpleNamespace(
            ok=True,
            details={
                "reward_artifact": {
                    "reward_source": "validation.lazyppl_elpd",
                    "training_log_reward": -42.0,
                }
            },
            metric_name="avg_holdout_logp",
            metric_value=-7.0,
            status="pass",
            message="scored",
        ),
    )
    score = gate.score_one(
        _row("lazy", runtime="lazyppl"),
        {"row_id": "lazy", "sample_idx": 0, "text": "program"},
        _verifier_settings(),
    )
    assert score.training_log_reward == -42.0
    assert score.metric_value == -7.0
    assert score.reward_source == "validation.lazyppl_elpd"


@pytest.mark.parametrize(
    ("ok", "source", "reward"),
    [
        (False, "validation.lazyppl_elpd", -7.0),
        (True, "heuristic.lazyppl_elpd_error", -7.0),
        (True, "validation.lazyppl_elpd", math.nan),
        (True, "validation.lazyppl_elpd", "not-a-number"),
    ],
)
def test_lazyppl_failed_artifact_contract_floors(
    monkeypatch, ok: bool, source: str, reward: object
) -> None:
    from ppl_synth_data.runtimes.lazyppl_runtime import LazyPPLRuntime
    from ppl_synth_data.validation import lazyppl_semantics

    monkeypatch.setattr(LazyPPLRuntime, "extract_code", lambda self, text: "model")
    monkeypatch.setattr(
        lazyppl_semantics,
        "score_lazyppl_semantics",
        lambda *args, **kwargs: SimpleNamespace(
            ok=ok,
            details={
                "reward_artifact": {
                    "reward_source": source,
                    "training_log_reward": reward,
                }
            },
            metric_name="avg_holdout_logp",
            metric_value=-8.0,
            status="critical",
            message="failed contract",
        ),
    )
    score = gate.score_one(
        _row("lazy", runtime="lazyppl"),
        {"row_id": "lazy", "sample_idx": 0, "text": "program"},
        _verifier_settings(),
    )
    assert score.training_log_reward == FLOOR
    assert score.metric_value == -8.0
    assert score.reason == "failed contract"
    assert score.reward_source == "floor"


def test_empty_completion_floors_with_specific_reason() -> None:
    score = gate.score_one(
        _row("empty"),
        {"row_id": "empty", "sample_idx": 2, "text": ""},
        _verifier_settings(),
    )
    assert score.training_log_reward == FLOOR
    assert score.reason == "empty completion"
    assert score.reward_source == "floor"
    assert score.data_shape == "lists"


def test_classify_dead_flat_live() -> None:
    assert gate.classify([FLOOR] * 4, floor=FLOOR, eps=1e-6)[0] == "DEAD"
    assert gate.classify([-2.5, -2.5, -2.5, -2.5], floor=FLOOR, eps=1e-6)[0] == "FLAT"
    verdict, variance, spread, n_floor = gate.classify(
        [-2.5, -2.5, FLOOR, -1.0], floor=FLOOR, eps=1e-6
    )
    assert verdict == "LIVE"
    assert n_floor == 1
    assert spread == pytest.approx(FLOOR * -1 - 1.0)
    assert variance > 0


def test_classify_treats_one_floor_among_real_scores_as_live() -> None:
    verdict, _, _, n_floor = gate.classify([FLOOR, -1.0, -1.0, -1.0], floor=FLOOR, eps=1e-6)
    assert verdict == "LIVE"
    assert n_floor == 1


@pytest.mark.parametrize(
    ("rewards", "eps", "message"),
    [
        ([FLOOR], 1e-6, "at least 2"),
        ([FLOOR, math.inf], 1e-6, "finite"),
        ([FLOOR, -1.0], -1e-6, "eps"),
    ],
)
def test_classify_rejects_invalid_inputs(rewards: list[float], eps: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gate.classify(rewards, floor=FLOOR, eps=eps)


def test_classify_rejects_nan_eps() -> None:
    with pytest.raises(ValueError, match="eps"):
        gate.classify([-1.0, -2.0], floor=FLOOR, eps=math.nan)


def test_report_marks_partial_scorable_row_incomplete(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "rows.jsonl", [_row("complete"), _row("partial")])
    _write_jsonl(
        tmp_path / "completions.jsonl",
        [{"row_id": "complete", "sample_idx": idx, "text": "program"} for idx in range(4)],
    )
    _write_jsonl(
        tmp_path / "scores.jsonl",
        [
            *(
                {
                    "row_id": "complete",
                    "sample_idx": idx,
                    "training_log_reward": FLOOR,
                }
                for idx in range(4)
            ),
            *(
                {
                    "row_id": "partial",
                    "sample_idx": idx,
                    "training_log_reward": -1.0,
                }
                for idx in range(3)
            ),
        ],
    )
    (tmp_path / "keep_row_ids.txt").write_text("stale\n", encoding="utf-8")

    assert gate.stage_report(out_dir=tmp_path, k=4, eps=1e-6) == 1
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "incomplete: 1" in summary
    assert "partial" in summary
    assert not (tmp_path / "keep_row_ids.txt").exists()
    assert [row["row_id"] for row in gate.read_jsonl(tmp_path / "gate.jsonl")] == ["complete"]


@pytest.mark.parametrize("filename", ["rows.jsonl", "completions.jsonl", "scores.jsonl"])
def test_report_requires_every_input_file(tmp_path: Path, filename: str) -> None:
    for required in ("rows.jsonl", "completions.jsonl", "scores.jsonl"):
        if required != filename:
            (tmp_path / required).write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=filename):
        gate.stage_report(out_dir=tmp_path, k=4, eps=1e-6)

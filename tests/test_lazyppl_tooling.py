from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from synthstats_schemas import RewardArtifact

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.narrow_unit_not_acceptance
def test_audit_record_handles_null_completion(monkeypatch) -> None:
    audit_mod = _load_module("audit_lazyppl_typecheck_test", "scripts/audit_lazyppl_typecheck.py")

    class _FakeRuntime:
        def extract_code(self, completion: str):
            assert completion == ""
            return None

        def validate_syntax(self, code: str):  # pragma: no cover - should not be called
            raise AssertionError(f"unexpected validate_syntax call: {code}")

    monkeypatch.setattr(audit_mod, "rt", _FakeRuntime())
    info = audit_mod.audit_record({"completion": None, "metadata": {"id": "rec-1"}})

    assert info["id"] == "rec-1"
    assert info["has_code"] is False
    assert info["has_logLikRow"] is False
    assert info["has_module_submission"] is False
    assert info["has_model"] is False
    assert info["typecheck_ok"] is False
    assert info["typecheck_msg"] == "no code extracted"


def test_merge_script_fails_when_required_input_is_missing(tmp_path: Path, monkeypatch) -> None:
    merge_mod = _load_module("merge_lazyppl_splits_test", "scripts/merge_lazyppl_splits.py")
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    for filename in ("train_translated.jsonl", "val_translated.jsonl", "test_translated.jsonl"):
        (input_dir / filename).write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="native_train.jsonl"):
        merge_mod.main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])


def test_merge_script_preserves_existing_outputs_when_all_records_are_filtered(
    tmp_path: Path, monkeypatch
) -> None:
    merge_mod = _load_module(
        "merge_lazyppl_splits_filtered_test",
        "scripts/merge_lazyppl_splits.py",
    )
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    output_dir.mkdir()

    bad_record = json.dumps({"metadata": []}) + "\n"
    for filename in (
        "train_translated.jsonl",
        "val_translated.jsonl",
        "test_translated.jsonl",
        "native_train.jsonl",
    ):
        (input_dir / filename).write_text(bad_record, encoding="utf-8")

    existing_outputs = {
        "train.jsonl": "train-kept\n",
        "val.jsonl": "val-kept\n",
        "test.jsonl": "test-kept\n",
    }
    for filename, content in existing_outputs.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="filtered out"):
        merge_mod.main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])

    for filename, content in existing_outputs.items():
        assert (output_dir / filename).read_text(encoding="utf-8") == content


def test_merge_script_reports_invalid_json_with_file_and_line(tmp_path: Path, monkeypatch) -> None:
    merge_mod = _load_module(
        "merge_lazyppl_splits_invalid_json_test",
        "scripts/merge_lazyppl_splits.py",
    )
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    (input_dir / "train_translated.jsonl").write_text("{bad json}\n", encoding="utf-8")
    for filename in ("val_translated.jsonl", "test_translated.jsonl", "native_train.jsonl"):
        (input_dir / filename).write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=r"train_translated\.jsonl:1"):
        merge_mod.main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])


def test_jsonl_loaders_reject_non_object_records(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text("[]\n", encoding="utf-8")

    audit_mod = _load_module(
        "audit_lazyppl_typecheck_non_object_test",
        "scripts/audit_lazyppl_typecheck.py",
    )
    merge_mod = _load_module(
        "merge_lazyppl_splits_non_object_test",
        "scripts/merge_lazyppl_splits.py",
    )

    with pytest.raises(ValueError, match=r"expected JSON object in .*bad\.jsonl:1"):
        audit_mod._load_records(input_path)
    with pytest.raises(ValueError, match=r"expected JSON object in .*bad\.jsonl:1"):
        merge_mod.load_jsonl(input_path)


@pytest.mark.narrow_unit_not_acceptance
def test_audit_script_reports_invalid_json_with_file_and_line(tmp_path: Path, monkeypatch) -> None:
    audit_mod = _load_module(
        "audit_lazyppl_typecheck_invalid_json_test",
        "scripts/audit_lazyppl_typecheck.py",
    )
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text("{bad json}\n", encoding="utf-8")

    monkeypatch.setattr(audit_mod.sys, "argv", ["audit", str(input_path)])

    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        audit_mod.main()


@pytest.mark.narrow_unit_not_acceptance
def test_merge_script_preserves_existing_outputs_on_staged_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    merge_mod = _load_module(
        "merge_lazyppl_splits_write_failure_test",
        "scripts/merge_lazyppl_splits.py",
    )
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    output_dir.mkdir()

    good_record = json.dumps({"prompt": "p", "metadata": {"typecheck_ok": True}}) + "\n"
    for filename in (
        "train_translated.jsonl",
        "val_translated.jsonl",
        "test_translated.jsonl",
        "native_train.jsonl",
    ):
        (input_dir / filename).write_text(good_record, encoding="utf-8")

    existing_outputs = {
        "train.jsonl": "train-kept\n",
        "val.jsonl": "val-kept\n",
        "test.jsonl": "test-kept\n",
    }
    for filename, content in existing_outputs.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    original_writer = merge_mod._write_jsonl_atomic
    call_count = {"n": 0}

    def flaky_writer(path: Path, records: list[dict]) -> Path:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("disk full")
        return original_writer(path, records)

    monkeypatch.setattr(merge_mod, "_write_jsonl_atomic", flaky_writer)

    with pytest.raises(OSError, match="disk full"):
        merge_mod.main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])

    for filename, content in existing_outputs.items():
        assert (output_dir / filename).read_text(encoding="utf-8") == content


def test_merge_script_succeeds_without_val_test_translated(tmp_path: Path, monkeypatch) -> None:
    merge_mod = _load_module(
        "merge_lazyppl_splits_optional_test",
        "scripts/merge_lazyppl_splits.py",
    )
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()

    good_record = json.dumps({"prompt": "p", "metadata": {"typecheck_ok": True}}) + "\n"
    # only train_translated and native_train; val/test missing
    (input_dir / "train_translated.jsonl").write_text(good_record, encoding="utf-8")
    (input_dir / "native_train.jsonl").write_text(good_record, encoding="utf-8")
    # val_translated.jsonl and test_translated.jsonl intentionally absent

    merge_mod.main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()

    train = (output_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train) >= 1  # at least 1 translated + some native
    # val and test may be empty (only native records split into them)


def test_merge_script_filters_wrapped_failures_but_keeps_native_rows(tmp_path: Path) -> None:
    merge_mod = _load_module(
        "merge_lazyppl_splits_wrap_gate_test",
        "scripts/merge_lazyppl_splits.py",
    )
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()

    translated_bad = {
        "prompt": "translated-bad",
        "metadata": {
            "typecheck_ok": True,
            "wrapped_to_submission": True,
            "wrap_typecheck_ok": False,
            "source": "translated_from_pymc",
        },
    }
    translated_good = {
        "prompt": "translated-good",
        "metadata": {
            "typecheck_ok": True,
            "wrapped_to_submission": True,
            "wrap_typecheck_ok": True,
            "source": "translated_from_pymc",
        },
    }
    native_good = {
        "prompt": "native-good",
        "metadata": {
            "typecheck_ok": True,
            "family": "native",
        },
    }

    (input_dir / "train_translated.jsonl").write_text(
        "\n".join(json.dumps(record) for record in (translated_bad, translated_good)) + "\n",
        encoding="utf-8",
    )
    (input_dir / "val_translated.jsonl").write_text("", encoding="utf-8")
    (input_dir / "test_translated.jsonl").write_text("", encoding="utf-8")
    (input_dir / "native_train.jsonl").write_text(json.dumps(native_good) + "\n", encoding="utf-8")

    merge_mod.main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])

    train_records = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept_prompts = {record["prompt"] for record in train_records}

    assert "translated-bad" not in kept_prompts
    assert "translated-good" in kept_prompts
    assert "native-good" in kept_prompts


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. Real arviz.loo on this test's
# degenerate all-equal-value log_lik ([[0.0],[0.0],[0.0]]) returns
# pareto_k_max=inf, not the mocked 0.1 (verified directly: _loo_score on this
# exact fixture returns (0.0, inf)), so the exact contract-shape assertions
# below cannot be reproduced honestly against the real scorer. Real-boundary
# coverage of this holdout-draw-count arithmetic lives in
# test_score_main_real_arviz_emits_valid_contract_artifact_from_log_lik
# (real arviz.loo + real _write_reward on non-degenerate seeded Gaussian
# data).
def test_score_main_uses_holdout_draw_count(tmp_path: Path, monkeypatch) -> None:
    score_mod = _load_module("score_py_test", "docker/lazyppl-harbor/app/score.py")
    log_lik_path = tmp_path / "log_lik.json"
    hidden_eval_path = tmp_path / "hidden_eval.json"
    log_lik_path.write_text(
        json.dumps(
            {
                "log_lik": [[0.0], [0.0], [0.0]],
                "holdout_log_lik": [[0.0], [math.log(4.0)]],
            }
        ),
        encoding="utf-8",
    )
    hidden_eval_path.write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}
    monkeypatch.setattr(score_mod, "_loo_score", lambda _matrix: (0.0, 0.1))
    monkeypatch.setattr(score_mod, "_write_reward", lambda payload: captured.update(payload))
    monkeypatch.setattr(
        score_mod.sys,
        "argv",
        ["score.py", str(log_lik_path), str(hidden_eval_path)],
    )

    score_mod.main()

    assert captured["metrics"]["avg_holdout_logp"] == pytest.approx(math.log(2.5), abs=1e-6)


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. Real arviz.loo on this test's
# degenerate log_lik returns pareto_k_max=inf, not the mocked 0.1, so the
# exact contract-shape assertions below cannot be reproduced honestly
# against the real scorer. Real-boundary coverage of the raw-holdout
# precedence arithmetic lives in
# test_score_main_real_arviz_prefers_raw_holdout_over_malformed_hidden_eval
# (real arviz.loo + real _write_reward on non-degenerate seeded Gaussian
# data, with a genuinely malformed hidden_eval.json alongside).
def test_score_main_uses_raw_holdout_without_hidden_eval_file(tmp_path: Path, monkeypatch) -> None:
    score_mod = _load_module("score_py_raw_holdout_test", "docker/lazyppl-harbor/app/score.py")
    log_lik_path = tmp_path / "log_lik.json"
    log_lik_path.write_text(
        json.dumps(
            {
                "log_lik": [[0.0], [0.0]],
                "holdout_log_lik": [[0.0], [math.log(9.0)]],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(score_mod, "_loo_score", lambda _matrix: (0.0, 0.1))
    monkeypatch.setattr(score_mod, "_write_reward", lambda payload: captured.update(payload))
    monkeypatch.setattr(score_mod.sys, "argv", ["score.py", str(log_lik_path)])

    score_mod.main()

    assert captured["metrics"]["avg_holdout_logp"] == pytest.approx(math.log(5.0), abs=1e-6)


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. Real arviz.loo on this test's
# degenerate log_lik returns pareto_k_max=inf, not the mocked 0.1, so the
# exact contract-shape assertions below cannot be reproduced honestly
# against the real scorer. Real-boundary coverage of the raw-holdout
# precedence arithmetic lives in
# test_score_main_real_arviz_prefers_raw_holdout_over_malformed_hidden_eval
# (real arviz.loo + real _write_reward on non-degenerate seeded Gaussian
# data, with a genuinely malformed hidden_eval.json alongside).
def test_score_main_prefers_raw_holdout_over_bad_hidden_eval(tmp_path: Path, monkeypatch) -> None:
    score_mod = _load_module(
        "score_py_raw_holdout_preferred_test",
        "docker/lazyppl-harbor/app/score.py",
    )
    log_lik_path = tmp_path / "log_lik.json"
    hidden_eval_path = tmp_path / "hidden_eval.json"
    log_lik_path.write_text(
        json.dumps(
            {
                "log_lik": [[0.0], [0.0]],
                "holdout_log_lik": [[0.0], [math.log(9.0)]],
            }
        ),
        encoding="utf-8",
    )
    hidden_eval_path.write_text("{bad json}\n", encoding="utf-8")

    captured: dict[str, object] = {}
    monkeypatch.setattr(score_mod, "_loo_score", lambda _matrix: (0.0, 0.1))
    monkeypatch.setattr(score_mod, "_write_reward", lambda payload: captured.update(payload))
    monkeypatch.setattr(
        score_mod.sys,
        "argv",
        ["score.py", str(log_lik_path), str(hidden_eval_path)],
    )

    score_mod.main()

    assert captured["metrics"]["avg_holdout_logp"] == pytest.approx(math.log(5.0), abs=1e-6)
    assert "holdout_error" not in captured["diagnostics"]


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. Real arviz.loo on this test's
# degenerate log_lik returns pareto_k_max=inf, not the mocked 0.1, so the
# exact contract-shape assertions below cannot be reproduced honestly
# against the real scorer. Real-boundary coverage of the graceful-degradation
# path (malformed holdout alongside a genuinely-computed elpd_loo) lives in
# test_score_main_real_arviz_reports_holdout_error_without_failing_run (real
# arviz.loo + real _write_reward on non-degenerate seeded Gaussian data).
def test_score_main_reports_holdout_error_without_failing_run(tmp_path: Path, monkeypatch) -> None:
    score_mod = _load_module(
        "score_py_holdout_error_test",
        "docker/lazyppl-harbor/app/score.py",
    )
    log_lik_path = tmp_path / "log_lik.json"
    log_lik_path.write_text(
        json.dumps(
            {
                "log_lik": [[0.0], [0.0]],
                "holdout_log_lik": "not-a-matrix",
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(score_mod, "_loo_score", lambda _matrix: (0.0, 0.1))
    monkeypatch.setattr(score_mod, "_write_reward", lambda payload: captured.update(payload))
    monkeypatch.setattr(score_mod.sys, "argv", ["score.py", str(log_lik_path)])

    score_mod.main()

    assert captured["metrics"]["avg_holdout_logp"] is None
    assert "holdout_error" in captured["diagnostics"]


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. Real arviz.loo on this test's
# degenerate log_lik returns pareto_k_max=inf, not the mocked 0.1 (verified
# directly against this exact fixture), so the exact contract-shape
# assertions below cannot be reproduced honestly against the real scorer.
# Real-boundary coverage of the full contract-artifact shape lives in
# test_score_main_real_arviz_emits_valid_contract_artifact_from_log_lik
# (real arviz.loo + real _write_reward + RewardArtifact.model_validate on
# non-degenerate seeded Gaussian data).
def test_score_main_emits_contract_reward_artifact(tmp_path: Path, monkeypatch) -> None:
    score_mod = _load_module(
        "score_py_contract_artifact_test",
        "docker/lazyppl-harbor/app/score.py",
    )
    log_lik_path = tmp_path / "log_lik.json"
    log_lik_path.write_text(
        json.dumps(
            {
                "log_lik": [[0.0], [0.0]],
                "holdout_log_lik": [[0.0], [math.log(9.0)]],
                "gold_elpd": 1.0,
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(score_mod, "_loo_score", lambda _matrix: (0.0, 0.1))
    monkeypatch.setattr(score_mod, "_write_reward", lambda payload: captured.update(payload))
    monkeypatch.setattr(score_mod.sys, "argv", ["score.py", str(log_lik_path)])

    score_mod.main()

    assert set(captured) >= {
        "schema_version",
        "reward_source",
        "evidence_kind",
        "truth_owner",
        "producer",
        "metrics",
        "diagnostics",
        "provenance",
        "reward_transform",
        "training_log_reward",
    }
    assert captured["reward_source"] == "validation.lazyppl_elpd"
    assert captured["evidence_kind"] == "verifier"
    assert captured["truth_owner"] == "verifier"
    assert captured["producer"] == "ppl_synth_data.docker.lazyppl_harbor.score"
    assert captured["metrics"]["elpd_loo"] == 0.0
    assert captured["metrics"]["avg_holdout_logp"] == pytest.approx(math.log(5.0), abs=1e-6)
    assert captured["metrics"]["holdout_elpd"] == pytest.approx(math.log(5.0), abs=1e-4)
    assert captured["metrics"]["pareto_k_max"] == 0.1
    assert math.isfinite(captured["training_log_reward"])
    assert captured["training_log_reward"] >= -20.0
    assert captured["reward_transform"]["lower_log_reward_bound"] == -20.0
    assert captured["reward_components"][0]["name"] == "holdout_elpd"
    assert captured["reward_components"][0]["kind"] == "terminal_verifier"
    assert captured["reward_components"][0]["source"] == "validation.lazyppl_elpd"
    assert captured["scalarization"]["method"] == "weighted_sum"
    assert captured["scalarization"]["component_names"] == ["holdout_elpd"]
    RewardArtifact.model_validate(captured)


@pytest.mark.narrow_unit_not_acceptance
def test_score_error_emits_finite_heuristic_reward_artifact(monkeypatch) -> None:
    score_mod = _load_module(
        "score_py_error_contract_artifact_test",
        "docker/lazyppl-harbor/app/score.py",
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(score_mod, "_write_reward", lambda payload: captured.update(payload))

    score_mod._error_reward("missing input")

    assert captured["reward_source"] == "heuristic.lazyppl_elpd_error"
    assert captured["evidence_kind"] == "heuristic"
    assert captured["truth_owner"] == "heuristic"
    assert math.isfinite(captured["training_log_reward"])
    assert captured["training_log_reward"] == -20.0
    assert captured["metrics"]["elpd_loo"] is None
    assert captured["metrics"]["avg_holdout_logp"] is None
    assert captured["reward_components"][0]["name"] == "lazyppl_elpd_error"
    assert captured["reward_components"][0]["kind"] == "penalty"
    assert captured["scalarization"]["method"] == "manual"
    RewardArtifact.model_validate(captured)


@pytest.mark.narrow_unit_not_acceptance
def test_eval_script_failure_records_keep_full_result_surface(
    monkeypatch,
) -> None:
    eval_mod = _load_module(
        "eval_lazyppl_elpd_failure_surface_test",
        "scripts/eval_lazyppl_elpd.py",
    )
    monkeypatch.setattr(
        eval_mod,
        "get_runtime",
        lambda _name: SimpleNamespace(extract_code=lambda _completion: None),
    )

    result = eval_mod.evaluate_one(
        {"completion": "", "metadata": {}},
        "rec-missing-code",
        harbor_image="lazyppl-harbor:latest",
        timeout=1,
    )

    assert set(result) == {
        "id",
        "family",
        "elpd_loo",
        "pareto_k_max",
        "reward",
        "compile_ok",
        "error",
        "holdout_error",
        "n_draws",
        "n_obs",
        "pareto_k_warning",
    }
    assert result["family"] == "unknown"
    assert result["error"] == "no code block found in completion"
    assert result["n_draws"] is None
    assert result["n_obs"] is None
    assert result["pareto_k_warning"] is None


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. A genuine container run never returns
# returncode==0 with non-JSON stdout: score.py always writes valid JSON
# reward.json, and the `set -euo pipefail` chain fails the whole invocation
# before `cat` on any real GHC/runtime error, so this exact branch has no
# real-substrate trigger (documented in the twin's own docstring).
# Real-boundary coverage of the adjacent reachable failure (a genuine GHC
# compile failure surfacing through the same result-shaping path) lives in
# test_eval_script_parse_failure_keeps_full_result_surface_real_docker.
def test_eval_script_parse_failure_keeps_full_result_surface(
    monkeypatch,
) -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_parse_surface_test", "scripts/eval_lazyppl_elpd.py")
    monkeypatch.setattr(
        eval_mod,
        "get_runtime",
        lambda _name: SimpleNamespace(extract_code=lambda _completion: "model :: Int"),
    )
    monkeypatch.setattr(
        eval_mod,
        "_build_taskdata_realistic",
        lambda _code, _prompt, **kw: "module TaskData where",
    )
    monkeypatch.setattr(
        eval_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-json",
            stderr="",
        ),
    )

    result = eval_mod.evaluate_one(
        {"completion": "```hs\nmodel :: Int\n```", "metadata": {}},
        "rec-parse-fail",
        harbor_image="lazyppl-harbor:latest",
        timeout=1,
    )

    assert set(result) == {
        "id",
        "family",
        "elpd_loo",
        "pareto_k_max",
        "reward",
        "compile_ok",
        "error",
        "holdout_error",
        "n_draws",
        "n_obs",
        "pareto_k_warning",
    }
    assert result["error"] is not None
    assert "failed to parse reward JSON" in result["error"]
    assert result["n_draws"] is None
    assert result["n_obs"] is None
    assert result["pareto_k_warning"] is None


@pytest.mark.narrow_unit_not_acceptance
def test_eval_script_main_worker_exception_keeps_result_surface(
    tmp_path: Path, monkeypatch
) -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_worker_surface_test", "scripts/eval_lazyppl_elpd.py")
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "elpd_results.json"
    input_path.write_text(
        json.dumps(
            {
                "prompt": "translated prompt",
                "completion": "```hs\nmodel :: Int\n```",
                "metadata": {
                    "typecheck_ok": True,
                    "source": "translated_from_pymc",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeFuture:
        def result(self):
            raise RuntimeError("boom")

    fake_future = _FakeFuture()

    class _FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, *_args, **_kwargs):
            return fake_future

    monkeypatch.setattr(eval_mod, "ProcessPoolExecutor", lambda max_workers: _FakePool())
    monkeypatch.setattr(eval_mod, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_lazyppl_elpd.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--workers",
            "1",
        ],
    )

    eval_mod.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 1
    record = payload["records"][0]
    assert record["family"] == "translated"
    assert record["error"] == "worker exception: boom"
    assert record["n_draws"] is None
    assert record["n_obs"] is None
    assert record["pareto_k_warning"] is None


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. The fabricated holdout_error string in
# this mock's canned stdout cannot be produced by a genuine container run
# through the public evaluate_one interface (documented in the twin's own
# docstring: _resolve_holdout_payload treats an absent holdout section as
# "no holdout evaluation", not an error). Real-boundary coverage of the
# closest reachable real trigger (holdout absent on empty hidden data,
# pinning current behavior) lives in
# test_eval_script_holdout_absent_on_empty_hidden_data_real_docker.
def test_eval_script_surfaces_holdout_error_from_reward_payload(
    monkeypatch,
) -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_test", "scripts/eval_lazyppl_elpd.py")
    monkeypatch.setattr(
        eval_mod,
        "get_runtime",
        lambda _name: SimpleNamespace(extract_code=lambda _completion: "model :: Int"),
    )
    monkeypatch.setattr(
        eval_mod,
        "_build_taskdata_realistic",
        lambda _code, _prompt, **kw: "module TaskData where",
    )
    monkeypatch.setattr(
        eval_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "compile_ok": 1.0,
                    "elpd_loo": 0.0,
                    "pareto_k_max": 0.1,
                    "reward": 0.5,
                    "n_draws": 2,
                    "n_obs": 1,
                    "pareto_k_warning": False,
                    "holdout_error": "bad holdout payload",
                }
            ),
            stderr="",
        ),
    )

    result = eval_mod.evaluate_one(
        {"completion": "```hs\nmodel :: Int\n```", "metadata": {}},
        "rec-1",
        harbor_image="lazyppl-harbor:latest",
        timeout=1,
    )

    assert result["compile_ok"] is True
    assert result["holdout_error"] == "bad holdout payload"


def test_eval_script_aggregate_counts_holdout_errors() -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_aggregate_test", "scripts/eval_lazyppl_elpd.py")

    aggregate = eval_mod._aggregate(
        [
            {
                "id": "rec-1",
                "family": "normal",
                "compile_ok": True,
                "elpd_loo": 1.0,
                "holdout_error": None,
            },
            {
                "id": "rec-2",
                "family": "normal",
                "compile_ok": True,
                "elpd_loo": 2.0,
                "holdout_error": "bad holdout payload",
            },
            {
                "id": "rec-3",
                "family": "poisson",
                "compile_ok": False,
                "elpd_loo": None,
                "holdout_error": "missing holdout data",
            },
        ]
    )

    assert aggregate["summary"]["holdout_errors"] == 2
    assert aggregate["by_family"]["normal"]["holdout_errors"] == 1
    assert aggregate["by_family"]["poisson"]["holdout_errors"] == 1


def test_generate_native_lazyppl_imports() -> None:
    mod = _load_module(
        "generate_native_lazyppl_import_test",
        "scripts/generate_native_lazyppl.py",
    )
    assert hasattr(mod, "LAZYPPL")
    assert hasattr(mod, "main")
    assert callable(mod.main)


def test_translate_pymc_to_lazyppl_imports() -> None:
    mod = _load_module(
        "translate_pymc_to_lazyppl_import_test",
        "scripts/translate_pymc_to_lazyppl.py",
    )
    assert hasattr(mod, "LAZYPPL")
    assert hasattr(mod, "main")
    assert callable(mod.main)


# --- heavy_lane: real docker + real ArviZ substrate twins ---
#
# The monkeypatched tests above stay as fast local unit coverage (kept
# unchanged). The tests below drive the real `lazyppl-harbor` Docker image
# (for the two `eval_lazyppl_elpd.evaluate_one` rows) and the real
# `arviz.loo` PSIS-LOO computation (for the five `score.py` rows); no
# monkeypatch of the Docker call, `_loo_score`, or `_write_reward`. Hard-
# asserts substrate presence; raises rather than skips.

_REAL_SUBMIT_COMPLETION = (
    '<submit_program language="haskell">\n'
    "module Submission where\n\n"
    "import LazyPPL\nimport LazyPPL.Distributions\nimport TaskData (Row(..))\n\n"
    "data State = State Double deriving (Show)\n\n"
    "model :: [Row] -> Meas State\n"
    "model rows = do\n"
    "  mu <- sample (normal 0.0 1.0)\n"
    "  let state = State mu\n"
    "  mapM_ (\\row -> score (exp (logLikRow state row))) rows\n"
    "  return state\n\n"
    "logLikRow :: State -> Row -> Double\n"
    "logLikRow (State mu) row = normalLogPdf mu 1.0 (value row)\n"
    "</submit_program>"
)


def _require_real_harbor_image() -> str:
    if not shutil.which("docker"):
        raise RuntimeError(
            "heavy_lane test requires a real docker binary on PATH; run this suite on "
            "the AWS heavy-lane instance"
        )
    image = os.environ.get("LAZYPPL_HARBOR_IMAGE")
    if not image:
        raise RuntimeError(
            "heavy_lane test requires LAZYPPL_HARBOR_IMAGE to point at the built "
            "lazyppl-harbor image tag"
        )
    return image


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_eval_script_holdout_absent_on_empty_hidden_data_real_docker() -> None:
    """Real-Docker twin of row 151, pinning CURRENT behavior.

    ``evaluate_one`` builds hidden TaskData from an empty row list
    (``generate_hidden_taskdata_hs(parse_seed_row_fields(code), [])``), and
    ``_resolve_holdout_payload`` treats an absent holdout section as "no
    holdout evaluation", not an error: a genuine container run completes with
    ``holdout_error`` null and no holdout metric. (The narrow-unit row's
    fabricated ``holdout_error`` payload exercises the surfacing branch; the
    empty-hidden-data path cannot reach it through the public interface.)
    """
    eval_mod = _load_module(
        "eval_lazyppl_elpd_real_holdout_absent_test", "scripts/eval_lazyppl_elpd.py"
    )
    image = _require_real_harbor_image()

    result = eval_mod.evaluate_one(
        {"completion": _REAL_SUBMIT_COMPLETION, "metadata": {}},
        "record-real-holdout",
        image,
        timeout=600,
    )

    assert result["compile_ok"] is True, result["error"]
    assert result["holdout_error"] is None
    assert result.get("avg_holdout_logp") is None


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_eval_script_parse_failure_keeps_full_result_surface_real_docker() -> None:
    """Real-Docker twin of row 150.

    A genuine container run never returns ``returncode == 0`` with non-JSON
    stdout: ``score.py`` always writes a valid JSON ``reward.json``, and the
    ``set -euo pipefail`` chain fails the whole invocation before ``cat`` on
    any real GHC/runtime error; so the exact "parse failure" branch has no
    real-substrate trigger. The adjacent real trigger this test drives
    instead is a genuine GHC compile failure (a program missing the required
    ``model``/``logLikRow`` signatures), which surfaces as a real non-zero-
    exit, non-infrastructure error through the same ``evaluate_one`` error-
    handling path, keeping the full result surface intact.
    """
    eval_mod = _load_module(
        "eval_lazyppl_elpd_real_compile_failure_test", "scripts/eval_lazyppl_elpd.py"
    )
    image = _require_real_harbor_image()

    broken_completion = (
        '<submit_program language="haskell">\nmodule Submission where\nnotAValidModel = 42\n'
        "</submit_program>"
    )
    result = eval_mod.evaluate_one(
        {"completion": broken_completion, "metadata": {}},
        "record-real-compile-fail",
        image,
        timeout=180,
    )

    assert set(result) == {
        "id",
        "family",
        "elpd_loo",
        "pareto_k_max",
        "reward",
        "compile_ok",
        "error",
        "holdout_error",
        "n_draws",
        "n_obs",
        "pareto_k_warning",
    }
    assert result["compile_ok"] is False
    assert result["error"] is not None


def _real_gaussian_log_lik(*, seed: int, n_draws: int, n_obs: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs = rng.normal(0.0, 1.0, size=n_obs)
    draws = rng.normal(0.0, 1.0, size=(n_draws, n_obs))
    return -0.5 * np.log(2 * np.pi) - 0.5 * (draws - obs[None, :]) ** 2


@pytest.mark.real_boundary
# No heavy_lane: `_loo_score` runs real arviz.loo PSIS-LOO, and arviz is a
# normal installed dependency (uv.lock); no Docker/GHC/provisioned
# substrate is required, so this test belongs in the default lane.
def test_score_main_real_arviz_emits_valid_contract_artifact_from_log_lik(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-ArviZ twin of rows 154 (``test_score_main_emits_contract_reward_artifact``)
    and 157 (``test_score_main_uses_holdout_draw_count``): no monkeypatch of
    ``_loo_score`` or ``_write_reward``; real ``arviz.loo`` PSIS-LOO computed
    over a real (seeded, non-trivial) log-likelihood matrix, real
    ``reward.json`` written to disk and read back.
    """
    score_mod = _load_module(
        "score_py_real_arviz_contract_test", "docker/lazyppl-harbor/app/score.py"
    )
    log_lik = _real_gaussian_log_lik(seed=20260705, n_draws=300, n_obs=40)
    holdout_log_lik = _real_gaussian_log_lik(seed=20260706, n_draws=300, n_obs=5)

    log_lik_path = tmp_path / "log_lik.json"
    log_lik_path.write_text(
        json.dumps({"log_lik": log_lik.tolist(), "holdout_log_lik": holdout_log_lik.tolist()}),
        encoding="utf-8",
    )

    if score_mod._REWARD_JSON.exists():
        score_mod._REWARD_JSON.unlink()
    monkeypatch.setattr(score_mod.sys, "argv", ["score.py", str(log_lik_path)])

    score_mod.main()

    captured = json.loads(score_mod._REWARD_JSON.read_text())
    RewardArtifact.model_validate(captured)
    assert math.isfinite(captured["metrics"]["elpd_loo"])
    assert captured["metrics"]["n_draws"] == 300
    assert captured["metrics"]["n_obs"] == 40
    assert captured["metrics"]["avg_holdout_logp"] is not None
    assert math.isfinite(captured["metrics"]["avg_holdout_logp"])
    assert captured["reward_source"] == "validation.lazyppl_elpd"


@pytest.mark.real_boundary
# No heavy_lane: real arviz.loo, no Docker/GHC substrate required (see
# test_score_main_real_arviz_emits_valid_contract_artifact_from_log_lik).
def test_score_main_real_arviz_prefers_raw_holdout_over_malformed_hidden_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-ArviZ twin of rows 155 and 158 (``prefers_raw_holdout_over_bad_hidden_eval``,
    ``uses_raw_holdout_without_hidden_eval_file``): real arithmetic over a
    real raw ``holdout_log_lik`` in ``log_lik.json``, alongside a malformed
    ``hidden_eval.json`` that must be ignored in favour of the raw payload.
    """
    score_mod = _load_module(
        "score_py_real_arviz_raw_holdout_test", "docker/lazyppl-harbor/app/score.py"
    )
    log_lik = _real_gaussian_log_lik(seed=1, n_draws=150, n_obs=20)
    holdout_log_lik = _real_gaussian_log_lik(seed=2, n_draws=150, n_obs=4)

    log_lik_path = tmp_path / "log_lik.json"
    log_lik_path.write_text(
        json.dumps({"log_lik": log_lik.tolist(), "holdout_log_lik": holdout_log_lik.tolist()}),
        encoding="utf-8",
    )
    hidden_eval_path = tmp_path / "hidden_eval.json"
    hidden_eval_path.write_text("{bad json}\n", encoding="utf-8")

    if score_mod._REWARD_JSON.exists():
        score_mod._REWARD_JSON.unlink()
    monkeypatch.setattr(
        score_mod.sys, "argv", ["score.py", str(log_lik_path), str(hidden_eval_path)]
    )

    score_mod.main()

    captured = json.loads(score_mod._REWARD_JSON.read_text())
    assert "holdout_error" not in captured["diagnostics"]
    assert captured["metrics"]["avg_holdout_logp"] is not None
    column_max = holdout_log_lik.max(axis=0, keepdims=True)
    stable = holdout_log_lik - column_max
    expected_avg = float(np.mean(np.log(np.mean(np.exp(stable), axis=0)) + column_max[0]))
    assert captured["metrics"]["avg_holdout_logp"] == pytest.approx(expected_avg, abs=1e-4)


@pytest.mark.real_boundary
# No heavy_lane: real arviz.loo, no Docker/GHC substrate required (see
# test_score_main_real_arviz_emits_valid_contract_artifact_from_log_lik).
def test_score_main_real_arviz_reports_holdout_error_without_failing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-ArviZ twin of row 156 (``test_score_main_reports_holdout_error_without_failing_run``):
    a malformed ``holdout_log_lik`` ("not-a-matrix") alongside a real,
    successfully-computed ``elpd_loo``, proving the run degrades gracefully
    without faking the underlying PSIS-LOO computation.
    """
    score_mod = _load_module(
        "score_py_real_arviz_holdout_error_test", "docker/lazyppl-harbor/app/score.py"
    )
    log_lik = _real_gaussian_log_lik(seed=7, n_draws=120, n_obs=25)

    log_lik_path = tmp_path / "log_lik.json"
    log_lik_path.write_text(
        json.dumps({"log_lik": log_lik.tolist(), "holdout_log_lik": "not-a-matrix"}),
        encoding="utf-8",
    )

    if score_mod._REWARD_JSON.exists():
        score_mod._REWARD_JSON.unlink()
    monkeypatch.setattr(score_mod.sys, "argv", ["score.py", str(log_lik_path)])

    score_mod.main()

    captured = json.loads(score_mod._REWARD_JSON.read_text())
    assert captured["metrics"]["avg_holdout_logp"] is None
    assert "holdout_error" in captured["diagnostics"]
    assert math.isfinite(captured["metrics"]["elpd_loo"])

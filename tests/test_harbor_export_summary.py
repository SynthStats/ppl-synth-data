"""Summary publication laws against the installed Harbor exporter."""

from __future__ import annotations

import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppl_synth_data.harbor import export_pymc
from ppl_synth_data.pipeline.io import write_jsonl
from ppl_synth_data.runtime_registry import build_harbor_task_id

RUN_ID = "export-summary-laws"
VALID_CODE = """\
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        y = data["y"]
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
"""
POTENTIAL_CODE = VALID_CODE.replace(
    "    return m", '        pm.Potential("hack", mu * 0)\n    return m'
)


@pytest.fixture(scope="session")
def exporter():
    return export_pymc


def _completion(sequence: int, *, code: str = VALID_CODE) -> dict:
    return {
        "id": f"comp-{sequence:06d}",
        "prompt_id": "prompt-000000",
        "prompt_text": "Model the relationship.",
        "raw_completion": code,
        "parsed_program": code,
        "metadata": {
            "env_id": "env001",
            "generative_family": "regression_continuous",
            "validation": {"ok": True},
        },
    }


def _skipped_completion(sequence: int) -> dict:
    record = _completion(sequence)
    del record["metadata"]["env_id"]
    return record


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """The existing test_harbor_export.py fixture: 10 visible + 3 hidden rows."""
    run = tmp_path / RUN_ID
    for stage in ("stage_A_envspec", "stage_C_data", "stage_D_prompts", "stage_F_validation"):
        (run / stage).mkdir(parents=True)
    write_jsonl(
        run / "stage_A_envspec" / "envspecs.jsonl",
        [
            {
                "env_id": "env001",
                "env_name": "TestEnv",
                "domain": "testing",
                "complexity": "simple",
                "model_type": "regression",
                "key_concepts": ["linear"],
                "io_schema": {
                    "input_type": "float",
                    "input_description": "x values",
                    "output_type": "float",
                    "target_name": "y",
                    "feature_names": ["x"],
                    "feature_domains": {"x": [0.0, 10.0]},
                },
                "data_plan": {"n_observations": 10},
                "generative_family": "regression_continuous",
                "generative_params": {},
                "prompt_guidance": {},
            }
        ],
    )
    data = run / "stage_C_data" / "data"
    (data / "hidden").mkdir(parents=True)
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 13)
    y = 0.4 + 1.1 * x + rng.normal(scale=0.1, size=13)
    pd.DataFrame({"x": x[:10], "y": y[:10]}).to_parquet(data / "env001.parquet", index=False)
    pd.DataFrame({"x": x[10:], "y": y[10:]}).to_parquet(
        data / "hidden" / "env001.parquet", index=False
    )
    write_jsonl(
        run / "stage_C_data" / "data_refs.jsonl",
        [
            {
                "env_id": "env001",
                "env_name": "TestEnv",
                "data_ref": "stage_C_data/data/env001.parquet",
                "hidden_data_ref": "stage_C_data/data/hidden/env001.parquet",
                "split_policy": "ordered_tail",
            }
        ],
    )
    write_jsonl(
        run / "stage_D_prompts" / "prompts.jsonl",
        [
            {
                "id": "prompt-000000",
                "split": "train",
                "env_id": "env001",
                "prompt_text": "Model the relationship between x and y using linear regression.",
                "metadata": {"env_name": "TestEnv", "domain": "testing"},
            }
        ],
    )
    write_jsonl(run / "stage_F_validation" / "validated.jsonl", [_completion(0)])
    return run


@contextmanager
def _observe_publication(summary_path: Path):
    """Observe real filesystem publication without replacing any exporter code."""
    replacements = []
    active = True

    def observe(event, args):
        if not active or event != "os.rename":
            return
        source, destination, source_fd, destination_fd = args
        if Path(os.fsdecode(destination)).absolute() != summary_path.absolute():
            return
        source_path = Path(os.fsdecode(source)).absolute()
        replacements.append(
            {
                "source": source_path,
                "source_fd": source_fd,
                "destination_fd": destination_fd,
                "payload": json.loads(source_path.read_text(encoding="utf-8")),
            }
        )

    # CPython audits os.replace as os.rename. Hooks cannot be removed, so
    # this observer becomes inert as soon as its context ends.
    sys.addaudithook(observe)
    try:
        yield replacements
    finally:
        active = False


def _read_summary(out: Path) -> dict:
    path = out / "export_summary.json"
    assert path.is_file(), f"Exporter must persist its summary at {path}"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(summary, dict)
    assert {
        "run_id",
        "validated",
        "prepared",
        "tasks_created",
        "skipped",
        "gold_failures",
        "reference_failures",
        "task_dirs",
    } <= summary.keys()
    return summary


def _assert_counts(summary: dict, *, validated: int, prepared: int, created: int) -> None:
    assert summary["run_id"] == RUN_ID
    for key, expected in (
        ("validated", validated),
        ("prepared", prepared),
        ("tasks_created", created),
    ):
        assert type(summary[key]) is int, (key, summary[key])
        assert summary[key] == expected


def _assert_publication(out: Path, summary: dict, replacements: list) -> None:
    assert len(replacements) == 1, "Summary must be published with a same-directory atomic rename"
    replacement = replacements[0]
    assert replacement["source"].parent == out.absolute()
    assert replacement["source"] != out / "export_summary.json"
    assert replacement["source_fd"] == replacement["destination_fd"] == -1
    assert replacement["payload"] == summary, "Complete JSON must exist before publication"
    assert not replacement["source"].exists(), "The temporary file must be consumed by the rename"


def _assert_tasks(out: Path, summary: dict, expected: list[str]) -> None:
    names = summary["task_dirs"]
    assert isinstance(names, list)
    assert sorted(names) == sorted(expected)
    assert sorted(path.name for path in out.iterdir() if path.is_dir()) == sorted(expected)
    for name in expected:
        task = out / name
        for relative in (
            "task.toml",
            "instruction.md",
            "environment/Dockerfile",
            "environment/workspace/data.parquet",
            "environment/workspace/submission.py",
            "tests/holdout.parquet",
            "tests/elpd_reference.json",
            "tests/validator.py",
            "tests/test.sh",
            "solution/model.py",
            "solution/solve.sh",
        ):
            assert (task / relative).is_file(), relative
        assert (task / "solution/model.py").read_text(encoding="utf-8") == VALID_CODE
        reference = json.loads((task / "tests/elpd_reference.json").read_text(encoding="utf-8"))
        assert math.isfinite(reference["gold_elpd"])
        assert math.isfinite(reference["baseline_elpd"])


def test_success_persists_summary_and_preserves_return(exporter, run_dir: Path, tmp_path: Path):
    out = tmp_path / "success"
    with _observe_publication(out / "export_summary.json") as replacements:
        result = exporter.export_harbor_tasks(
            run_dir,
            out,
            run_id=RUN_ID,
            attempt_id="summary-test",
            l6_draws=15,
            l6_tune=15,
            allow_unlocked_reference_runtime=True,
        )

    assert result == {
        "run_id": RUN_ID,
        "tasks_created": 1,
        "tasks_skipped": 0,
        "skipped_details": [],
    }
    summary = _read_summary(out)
    _assert_counts(summary, validated=1, prepared=1, created=1)
    assert summary["skipped"] == summary["gold_failures"] == summary["reference_failures"] == []
    _assert_tasks(out, summary, [build_harbor_task_id("pymc", "env001", sequence="000000")])
    _assert_publication(out, summary, replacements)


@pytest.mark.parametrize("reference_workers", [1, 2])
def test_both_failures_are_reported_after_summary_is_written(
    exporter,
    run_dir: Path,
    tmp_path: Path,
    reference_workers: int,
):
    # A second real visible split passes structural validation. Its malformed
    # hidden artifact fails during phase 2, inside the reference-failure collector.
    stage_a = run_dir / "stage_A_envspec" / "envspecs.jsonl"
    env = json.loads(stage_a.read_text(encoding="utf-8").strip())
    reference_env = {**env, "env_id": "env002"}
    write_jsonl(stage_a, [env, reference_env])
    stage_d = run_dir / "stage_D_prompts" / "prompts.jsonl"
    prompt = json.loads(stage_d.read_text(encoding="utf-8").strip())
    write_jsonl(stage_d, [prompt, {**prompt, "id": "prompt-000002", "env_id": "env002"}])
    data = run_dir / "stage_C_data" / "data"
    (data / "env002.parquet").write_bytes((data / "env001.parquet").read_bytes())
    (data / "hidden" / "env002.parquet").write_bytes(b"invalid held-out parquet\n")
    refs_path = run_dir / "stage_C_data" / "data_refs.jsonl"
    refs = json.loads(refs_path.read_text(encoding="utf-8").strip())
    write_jsonl(
        refs_path,
        [
            refs,
            {
                **refs,
                "env_id": "env002",
                "data_ref": "stage_C_data/data/env002.parquet",
                "hidden_data_ref": "stage_C_data/data/hidden/env002.parquet",
            },
        ],
    )
    reference_record = _completion(2)
    reference_record["metadata"]["env_id"] = "env002"
    reference_record["prompt_id"] = "prompt-000002"
    write_jsonl(
        run_dir / "stage_F_validation" / "validated.jsonl",
        [
            _completion(1, code=POTENTIAL_CODE),
            reference_record,
            _skipped_completion(3),
            _completion(0),
        ],
    )

    out = tmp_path / "mixed"
    summary_path = out / "export_summary.json"
    with _observe_publication(summary_path) as replacements:
        with pytest.raises(RuntimeError) as caught:
            exporter.export_harbor_tasks(
                run_dir,
                out,
                run_id=RUN_ID,
                attempt_id="summary-test",
                validate_gold=True,
                l6_draws=15,
                l6_tune=15,
                reference_workers=reference_workers,
                allow_unlocked_reference_runtime=True,
            )
        # Read immediately after the exception: no retry or repair may create it.
        summary = _read_summary(out)

    _assert_counts(summary, validated=4, prepared=2, created=1)
    assert summary["skipped"] == [{"id": "comp-000003", "reason": "missing env_id"}]
    assert len(summary["gold_failures"]) == len(summary["reference_failures"]) == 1
    gold = summary["gold_failures"][0]
    reference = summary["reference_failures"][0]
    assert set(gold) == {"id", "env_id", "reason"}
    assert set(reference) == {"id", "env_id", "error"}
    assert (gold["id"], gold["env_id"]) == ("comp-000001", "env001")
    assert "Potential detected" in gold["reason"]
    assert (reference["id"], reference["env_id"]) == ("comp-000002", "env002")
    assert isinstance(reference["error"], str) and "parquet" in reference["error"].lower()
    message = str(caught.value)
    for detail in (
        gold["id"],
        gold["env_id"],
        gold["reason"],
        reference["id"],
        reference["env_id"],
        reference["error"],
        str(summary_path),
    ):
        assert detail in message, f"Aggregate error omitted {detail!r}: {message}"
    assert message.index(gold["id"]) < message.index(reference["id"])
    _assert_tasks(out, summary, [build_harbor_task_id("pymc", "env001", sequence="000000")])
    _assert_publication(out, summary, replacements)


def test_all_skipped_records_are_persisted_without_task_directories(
    exporter, run_dir: Path, tmp_path: Path
):
    missing_prompt = _completion(2)
    missing_prompt["prompt_id"] = "prompt-absent"
    write_jsonl(
        run_dir / "stage_F_validation" / "validated.jsonl",
        [
            _skipped_completion(1),
            missing_prompt,
        ],
    )
    out = tmp_path / "all-skipped"
    skipped = [
        {"id": "comp-000001", "reason": "missing env_id"},
        {"id": "comp-000002", "reason": "no prompt for prompt-absent"},
    ]
    with _observe_publication(out / "export_summary.json") as replacements:
        with pytest.raises(RuntimeError, match="created 0 tasks") as caught:
            exporter.export_harbor_tasks(
                run_dir,
                out,
                run_id=RUN_ID,
                attempt_id="summary-test",
                allow_unlocked_reference_runtime=True,
            )

    assert str(out / "export_summary.json") in str(caught.value)
    for record in skipped:
        assert record["id"] in str(caught.value)
        assert record["reason"] in str(caught.value)
    summary = _read_summary(out)
    _assert_counts(summary, validated=2, prepared=0, created=0)
    assert summary["skipped"] == skipped
    assert summary["gold_failures"] == summary["reference_failures"] == []
    _assert_tasks(out, summary, [])
    _assert_publication(out, summary, replacements)

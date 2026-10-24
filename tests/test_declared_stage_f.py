"""Declared Stage F CLI builds real models from canonical Stage B/C artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ppl_synth_data.cli import app
from ppl_synth_data.specs.models import CompletionRecord, PromptRecord

SOURCE_RUN = Path(__file__).resolve().parents[1] / "artifacts" / "reference"
FAMILIES = (
    "regression_continuous",
    "regression_count",
    "zero_inflated_count",
    "binary_choice",
)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def prepared_run(tmp_path):
    canonical = _rows(SOURCE_RUN / "stage_B_envs/envspecs.jsonl")
    specs = [
        next(row for row in canonical if row["generative_family"] == family) for family in FAMILIES
    ]
    ids = {row["env_id"] for row in specs}
    run = tmp_path / "run"
    for stage in ("stage_A_envspec", "stage_B_envs"):
        _write(run / stage / "envspecs.jsonl", specs)
    real_prompts = {
        row["env_id"]: row for row in _rows(SOURCE_RUN / "stage_D_prompts/prompts.jsonl")
    }
    prompts = []
    for index, spec in enumerate(specs):
        prompt = real_prompts.get(spec["env_id"])
        if prompt is None:
            # An explicit test input, from the real scenario, needs no LLM call.
            prompt = PromptRecord(
                id=f"declared-test-{index}",
                env_id=spec["env_id"],
                split="train",
                prompt_text=spec["scenario"]["intent"],
            ).model_dump(mode="json")
        prompts.append(prompt)
        for hidden in (False, True):
            relative = Path("stage_C_data/data")
            if hidden:
                relative /= "hidden"
            relative /= f"{spec['env_id']}.parquet"
            target = run / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE_RUN / relative, target)
    _write(run / "stage_D_prompts/prompts.jsonl", prompts)
    _write(
        run / "stage_C_data/data_refs.jsonl",
        [row for row in _rows(SOURCE_RUN / "stage_C_data/data_refs.jsonl") if row["env_id"] in ids],
    )
    return tmp_path, run, specs, prompts


def _invoke(root):
    return CliRunner().invoke(
        app,
        [
            "stage",
            "f-declared",
            "--run-id",
            "run",
            "--run-root",
            str(root),
        ],
    )


def _diagnostic(result):
    return result.output + "\n" + str(result.exception)


def test_declared_stage_f_command_is_available():
    result = CliRunner().invoke(app, ["stage", "f-declared", "--help"])
    assert result.exit_code == 0, _diagnostic(result)


@pytest.mark.heavy_lane(reason="Requires Stage B, C, and D artifacts in artifacts/reference")
def test_declared_stage_f_writes_one_executed_gold_per_family(prepared_run):
    root, run, specs, prompts = prepared_run
    result = _invoke(root)
    assert result.exit_code == 0, _diagnostic(result)
    records = [
        CompletionRecord.model_validate(row)
        for row in _rows(run / "stage_F_validation/validated.jsonl")
    ]
    assert len(records) == 4
    assert {record.env_id for record in records} == {spec["env_id"] for spec in specs}
    by_env = {prompt["env_id"]: prompt for prompt in prompts}
    import numpy as np
    import pandas as pd
    import pymc as pm

    for record in records:
        assert record.prompt_id == by_env[record.env_id]["id"]
        assert record.prompt_text == by_env[record.env_id]["prompt_text"]
        assert record.metadata["gold_source"] == "declared"
        program = record.parsed_program
        assert program
        assert (
            record.metadata["gold_program_sha256"]
            == hashlib.sha256(program.encode("utf-8")).hexdigest()
        )
        namespace = {"pm": pm, "np": np}
        exec(program, namespace)
        frame = pd.read_parquet(run / "stage_C_data/data" / f"{record.env_id}.parquet")
        model = namespace["model"]({column: frame[column].to_numpy() for column in frame})
        assert isinstance(model, pm.Model)
        assert model.observed_RVs
        assert np.isfinite(model.compile_logp()(model.initial_point()))
        assert record.validation is not None
        assert record.validation.semantic_score is None


@pytest.mark.heavy_lane(reason="Requires Stage B, C, and D artifacts in artifacts/reference")
def test_declared_stage_f_refuses_missing_visible_data(prepared_run):
    root, run, specs, _ = prepared_run
    missing = run / "stage_C_data/data" / f"{specs[0]['env_id']}.parquet"
    missing.rename(missing.with_suffix(".unavailable"))
    result = _invoke(root)
    assert result.exit_code != 0
    assert specs[0]["env_id"] in _diagnostic(result)
    assert "parquet" in _diagnostic(result)


@pytest.mark.heavy_lane(reason="Requires Stage B, C, and D artifacts in artifacts/reference")
def test_declared_stage_f_refuses_incomplete_prompts(prepared_run):
    root, run, _, prompts = prepared_run
    _write(run / "stage_D_prompts/prompts.jsonl", prompts[:-1])
    result = _invoke(root)
    assert result.exit_code != 0
    assert prompts[-1]["env_id"] in _diagnostic(result)


@pytest.mark.parametrize(
    "artifact", ["stage_B_envs/envspecs.jsonl", "stage_D_prompts/prompts.jsonl"]
)
@pytest.mark.heavy_lane(reason="Requires Stage B, C, and D artifacts in artifacts/reference")
def test_declared_stage_f_refuses_duplicate_coverage(prepared_run, artifact):
    root, run, _, _ = prepared_run
    rows = _rows(run / artifact)
    _write(run / artifact, rows + [rows[0]])
    result = _invoke(root)
    assert result.exit_code != 0
    assert "duplicate" in _diagnostic(result).lower()
    assert rows[0]["env_id"] in _diagnostic(result)


@pytest.mark.heavy_lane(reason="Requires Stage B, C, and D artifacts in artifacts/reference")
def test_declared_stage_f_refuses_invalid_binary_observations(prepared_run):
    import pandas as pd

    root, run, specs, _ = prepared_run
    spec = next(row for row in specs if row["generative_family"] == "binary_choice")
    path = run / "stage_C_data/data" / f"{spec['env_id']}.parquet"
    frame = pd.read_parquet(path)
    frame["y"] = 2  # Real data mutation: outside Bernoulli support.
    frame.to_parquet(path, index=False)
    result = _invoke(root)
    assert result.exit_code != 0
    assert spec["env_id"] in _diagnostic(result)


@pytest.mark.heavy_lane(reason="Requires Stage B, C, and D artifacts in artifacts/reference")
def test_declared_stage_f_deferred_export_consumes_records(prepared_run):
    root, run, specs, _ = prepared_run
    result = _invoke(root)
    assert result.exit_code == 0, _diagnostic(result)
    out = root / "export"
    result = CliRunner().invoke(
        app,
        [
            "export-harbor",
            "--allow-unlocked-reference-runtime",
            "--run-id",
            "run",
            "--run-root",
            str(root),
            "--out",
            str(out),
            "--defer-references",
            "--no-validate-gold",
        ],
    )
    assert result.exit_code == 0, _diagnostic(result)
    tasks = list(out.rglob("task.toml"))
    assert len(tasks) == len(specs)
    for task in tasks:
        assert (task.parent / "solution/model.py").is_file()
        assert not (task.parent / "tests/elpd_reference.json").exists()

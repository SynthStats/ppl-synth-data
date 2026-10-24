"""Installed export CLI with canonical input fixtures and rejection controls.

The public CLI disables gold validation and defers reference fits, so no model code executes.
"""

import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "export_zero_task_canonical"
VALIDATED = Path("stage_F_validation/validated.jsonl")


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def canonical_run(tmp_path):
    run = tmp_path / "runs" / "canonical-cli-control"
    shutil.copytree(FIXTURE, run)
    return run


def _controls(run):
    canonical = _read_rows(run / VALIDATED)[0]
    empty = copy.deepcopy(canonical)
    empty.update(id="canonical-empty", parsed_program="", raw_completion="")
    no_spec = copy.deepcopy(canonical)
    no_spec.update(id="canonical-no-spec", env_id="missing_canonical_spec")
    no_prompt = copy.deepcopy(canonical)
    no_prompt.update(id="canonical-no-prompt", prompt_id="missing-canonical-prompt")
    expected = [
        {"id": "canonical-empty", "reason": "empty completion"},
        {"id": "canonical-no-spec", "reason": "no EnvSpec for missing_canonical_spec"},
        {"id": "canonical-no-prompt", "reason": "no prompt for missing-canonical-prompt"},
    ]
    return canonical, [empty, no_spec, no_prompt], expected


def _export(run, out):
    cli = Path(sys.executable).parent / "ppldf"
    assert cli.is_file(), f"Required installed real CLI missing: {cli}"
    return subprocess.run(
        [
            str(cli),
            "export-harbor",
            "--run-id",
            run.name,
            "--attempt-id",
            "canonical-cli-test",
            "--run-root",
            str(run.parent),
            "--out",
            str(out),
            "--defer-references",
            "--no-validate-gold",
            "--allow-unlocked-reference-runtime",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_all_canonical_controls_fail_with_count_and_every_reason(canonical_run, tmp_path):
    _, controls, expected = _controls(canonical_run)
    _write_rows(canonical_run / VALIDATED, controls)
    out = tmp_path / "all-rejected"
    result = _export(canonical_run, out)
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
    assert result.returncode != 0, output
    diagnostic = output.rsplit("PyMC Harbor export created", 1)[-1]
    diagnostic = " ".join(diagnostic.split())
    assert "0 tasks" in diagnostic
    assert "3 of 3" in diagnostic
    for row in expected:
        assert f"- {row['id']}: {row['reason']}" in diagnostic
    assert not list(out.glob("*/task.toml"))


def test_partial_canonical_export_succeeds_with_files_and_every_skip(canonical_run, tmp_path):
    canonical, controls, expected = _controls(canonical_run)
    _write_rows(canonical_run / VALIDATED, [canonical, *controls])
    out = tmp_path / "partial"
    result = _export(canonical_run, out)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["tasks_created"] == 1
    assert summary["tasks_skipped"] == 3
    assert summary["skipped_details"] == expected
    tasks = list(out.glob("*/task.toml"))
    assert len(tasks) == 1
    task = tasks[0].parent
    for relative in [
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "environment/workspace/data.parquet",
        "environment/workspace/submission.py",
        "solution/model.py",
        "solution/solve.sh",
        "tests/validator.py",
        "tests/holdout.parquet",
        "tests/test.sh",
    ]:
        assert (task / relative).is_file(), relative
    visible = (task / "environment/workspace/data.parquet").read_bytes()
    source_visible = next((canonical_run / "stage_C_data/data").glob("*.parquet"))
    assert visible == source_visible.read_bytes()
    assert visible != (task / "tests/holdout.parquet").read_bytes()
    assert (task / "solution/model.py").read_text().strip() == canonical["parsed_program"].strip()
    assert not (task / "tests/elpd_reference.json").exists()
    assert summary["references"] == {task.name: "PENDING"}


def test_empty_canonical_validated_file_fails(canonical_run, tmp_path):
    _write_rows(canonical_run / VALIDATED, [])
    result = _export(canonical_run, tmp_path / "empty")
    assert result.returncode != 0, result.stdout
    assert "no validated records" in result.stdout + result.stderr

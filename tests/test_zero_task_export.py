"""An empty export must not look successful to its host-job caller.

These exercise real file inputs and the installed CLI; no fit is required
because both records fail the exporter's input checks.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from ppl_synth_data.harbor.export_pymc import export_harbor_tasks
from ppl_synth_data.pipeline.io import write_jsonl
from tests.test_harbor_export import _make_completion, run_dir  # noqa: F401


def rejected_records(pipeline_dir: Path) -> None:
    absent_spec = _make_completion("absent-env")
    absent_spec["id"] = "row-no-spec"
    absent_prompt = _make_completion()
    absent_prompt["id"] = "row-no-prompt"
    absent_prompt["prompt_id"] = "absent-prompt"
    write_jsonl(pipeline_dir / "stage_F_validation/validated.jsonl", [absent_spec, absent_prompt])


@pytest.mark.parametrize("defer", [False, True])
def test_zero_tasks_reports_each_rejected_row(run_dir, tmp_path, defer):  # noqa: F811
    rejected_records(run_dir)
    with pytest.raises(RuntimeError) as caught:
        export_harbor_tasks(
            run_dir,
            tmp_path / "out",
            "test-run",
            defer_references=defer,
            allow_unlocked_reference_runtime=True,
            attempt_id="test-attempt",
        )
    message = str(caught.value)
    assert "0 tasks" in message
    assert "2 of 2" in message
    assert "row-no-spec: no EnvSpec for absent-env" in message
    assert "row-no-prompt: no prompt for absent-prompt" in message


def test_empty_validated_input_fails(run_dir, tmp_path):  # noqa: F811
    write_jsonl(run_dir / "stage_F_validation/validated.jsonl", [])
    with pytest.raises(RuntimeError, match="no validated records"):
        export_harbor_tasks(
            run_dir,
            tmp_path / "out",
            "test-run",
            defer_references=True,
            allow_unlocked_reference_runtime=True,
            attempt_id="test-attempt",
        )


def test_zero_tasks_cli_exits_nonzero(run_dir, tmp_path):  # noqa: F811
    rejected_records(run_dir)
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "ppldf"),
            "export-harbor",
            "--run-id",
            "test-run",
            "--run-root",
            str(run_dir.parent),
            "--out",
            str(tmp_path / "cli-out"),
            "--attempt-id",
            "attempt-1",
            "--defer-references",
            "--allow-unlocked-reference-runtime",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    output = result.stdout + result.stderr
    assert "0 tasks" in output
    assert "row-no-spec" in output
    assert "row-no-prompt" in output


def test_partial_export_retains_success_and_skip_details(run_dir, tmp_path):  # noqa: F811
    valid = _make_completion()
    absent_spec = _make_completion("absent-env")
    absent_spec["id"] = "row-no-spec"
    write_jsonl(run_dir / "stage_F_validation/validated.jsonl", [valid, absent_spec])
    result = export_harbor_tasks(
        run_dir,
        tmp_path / "partial",
        "test-run",
        defer_references=True,
        allow_unlocked_reference_runtime=True,
        attempt_id="test-attempt",
    )
    assert result["tasks_created"] == 1
    assert result["skipped_details"] == [
        {"id": "row-no-spec", "reason": "no EnvSpec for absent-env"}
    ]

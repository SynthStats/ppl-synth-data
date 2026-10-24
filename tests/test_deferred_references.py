import tomllib

import pytest

from ppl_synth_data.harbor.export_pymc import export_harbor_tasks
from tests.test_harbor_export import _single_exported_task_dir, run_dir  # noqa: F401


def test_deferred_export_keeps_data_and_leaves_reference_absent(run_dir, tmp_path):  # noqa: F811
    out = tmp_path / "deferred"
    result = export_harbor_tasks(
        run_dir,
        out,
        "test-run",
        defer_references=True,
        allow_unlocked_reference_runtime=True,
        attempt_id="test-attempt",
    )
    assert result["tasks_created"] == 1
    task = _single_exported_task_dir(out)
    assert not (task / "tests/elpd_reference.json").exists()
    assert (task / "tests/holdout.parquet").is_file()
    assert (task / "environment/workspace/data.parquet").is_file()
    assert not (task / "environment/workspace/holdout.parquet").exists()
    assert tomllib.loads((task / "task.toml").read_text())["metadata"]["references"] == "PENDING"
    assert result["references"] == {task.name: "PENDING"}


def test_reference_workers_reject_zero_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="reference_workers"):
        export_harbor_tasks(
            tmp_path / "absent",
            tmp_path / "out",
            "test-run",
            reference_workers=0,
            allow_unlocked_reference_runtime=True,
            attempt_id="test-attempt",
        )

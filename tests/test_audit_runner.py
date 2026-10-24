from __future__ import annotations

import json
from pathlib import Path

from ppl_synth_data.audit.runner import AuditJob, load_outputs, outputs_complete


def test_audit_job_save_and_load(tmp_path: Path) -> None:
    job = AuditJob(job_id="batch_abc", request_ids=["batch-0", "batch-1"], model="gpt-4.1")
    job.save(tmp_path)

    loaded = AuditJob.load(tmp_path)
    assert loaded.job_id == "batch_abc"
    assert loaded.request_ids == ["batch-0", "batch-1"]
    assert loaded.model == "gpt-4.1"


def test_audit_job_save_writes_valid_json(tmp_path: Path) -> None:
    job = AuditJob(job_id="x", request_ids=[], model="m")
    job.save(tmp_path)

    raw = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    assert raw["job_id"] == "x"
    assert raw["request_ids"] == []


def test_load_outputs_reads_existing_files(tmp_path: Path) -> None:
    (tmp_path / "batch_0_output.txt").write_text("result-0", encoding="utf-8")
    (tmp_path / "batch_2_output.txt").write_text("result-2", encoding="utf-8")

    results = load_outputs(tmp_path, [0, 1, 2])
    assert results == {0: "result-0", 2: "result-2"}
    assert 1 not in results


def test_load_outputs_empty_when_no_files(tmp_path: Path) -> None:
    assert load_outputs(tmp_path, [0, 1]) == {}


def test_outputs_complete_true_when_all_present(tmp_path: Path) -> None:
    for n in [0, 1, 2]:
        (tmp_path / f"batch_{n}_output.txt").write_text("x", encoding="utf-8")

    assert outputs_complete(tmp_path, [0, 1, 2]) is True


def test_outputs_complete_false_when_missing(tmp_path: Path) -> None:
    (tmp_path / "batch_0_output.txt").write_text("x", encoding="utf-8")

    assert outputs_complete(tmp_path, [0, 1]) is False


def test_outputs_complete_empty_batch_list(tmp_path: Path) -> None:
    assert outputs_complete(tmp_path, []) is True

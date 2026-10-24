from __future__ import annotations

import json
from pathlib import Path

from ppl_synth_data.pipeline.artifacts import resolve_run_paths
from ppl_synth_data.pipeline.runner import _mark_downstream_skipped, run_all
from ppl_synth_data.pipeline.types import StageResult


def _fake_runner(stage: str, status: str = "success"):
    def runner(paths):
        stage_dir = paths.stage_dirs[stage]
        stage_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"status": status}
        (stage_dir / "manifest.json").write_text(json.dumps(manifest))
        return StageResult(stage=stage, status=status, outputs=manifest)

    return runner


def test_run_all_success(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "test-run")
    runners = {s: _fake_runner(s) for s in "ABCDEFG"}
    results = run_all(paths, runners=runners, resume=False)
    assert len(results) == 7
    assert all(r.status == "success" for r in results.values())
    run_manifest = json.loads((paths.run_dir / "run_manifest.json").read_text())
    assert run_manifest["run_id"] == "test-run"


def test_run_all_failure_skips_downstream(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "fail-run")
    runners = {s: _fake_runner(s) for s in "ABCDEFG"}
    runners["C"] = _fake_runner("C", status="failed")
    results = run_all(paths, runners=runners, resume=False)
    assert results["A"].status == "success"
    assert results["B"].status == "success"
    assert results["C"].status == "failed"
    assert results["D"].status == "skipped"
    assert results["G"].status == "skipped"


def test_run_all_resume_skips_completed(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "resume-run")
    stage_a_dir = paths.stage_dirs["A"]
    stage_a_dir.mkdir(parents=True)
    (stage_a_dir / "manifest.json").write_text(json.dumps({"status": "success"}))
    (stage_a_dir / "envspecs.jsonl").write_text("")

    call_count = {"A": 0}

    def counting_runner(paths):
        call_count["A"] += 1
        return _fake_runner("A")(paths)

    runners = {s: _fake_runner(s) for s in "ABCDEFG"}
    runners["A"] = counting_runner
    results = run_all(paths, runners=runners, resume=True)
    assert results["A"].status == "skipped"
    assert call_count["A"] == 0


def test_mark_downstream_skipped() -> None:
    results: dict[str, StageResult] = {
        "A": StageResult(stage="A", status="success"),
        "B": StageResult(stage="B", status="failed"),
    }
    _mark_downstream_skipped(results, failed_stage="B")
    assert "C" in results
    assert results["C"].status == "skipped"
    assert results["G"].status == "skipped"
    assert results["A"].status == "success"


def test_run_all_exception_in_stage(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "err-run")

    def boom(paths):
        raise RuntimeError("stage exploded")

    runners = {s: _fake_runner(s) for s in "ABCDEFG"}
    runners["B"] = boom
    results = run_all(paths, runners=runners, resume=False)
    assert results["B"].status == "failed"
    assert "exploded" in (results["B"].message or "")
    assert results["C"].status == "skipped"

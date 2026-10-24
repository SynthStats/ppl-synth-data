from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppl_synth_data.pipeline.artifacts import (
    manifest_is_success,
    resolve_run_paths,
    run_manifest_payload,
    stage_is_resumable,
)
from ppl_synth_data.pipeline.types import StageResult


def test_resolve_run_paths() -> None:
    paths = resolve_run_paths("/tmp/runs", "my-run-01")
    assert paths.run_id == "my-run-01"
    assert paths.run_dir == Path("/tmp/runs/my-run-01")
    assert "A" in paths.stage_dirs


def test_resolve_run_paths_bad_id() -> None:
    with pytest.raises(ValueError, match="run_id must match"):
        resolve_run_paths("/tmp", "bad/id")

    with pytest.raises(ValueError, match="run_id must match"):
        resolve_run_paths("/tmp", " spaces")


def test_manifest_is_success_true(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "success"}))
    assert manifest_is_success(manifest) is True


def test_manifest_is_success_false(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "failed"}))
    assert manifest_is_success(manifest) is False


def test_manifest_is_success_missing(tmp_path: Path) -> None:
    assert manifest_is_success(tmp_path / "nope.json") is False


def test_manifest_is_success_invalid_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("not json")
    assert manifest_is_success(manifest) is False


def test_stage_is_resumable_success(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_A_envspec"
    stage_dir.mkdir()
    (stage_dir / "manifest.json").write_text(json.dumps({"status": "success"}))
    (stage_dir / "envspecs.jsonl").write_text("")
    assert stage_is_resumable("A", stage_dir) is True


def test_stage_is_resumable_missing_artifact(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_A_envspec"
    stage_dir.mkdir()
    (stage_dir / "manifest.json").write_text(json.dumps({"status": "success"}))
    assert stage_is_resumable("A", stage_dir) is False


def test_stage_is_resumable_failed(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_A_envspec"
    stage_dir.mkdir()
    (stage_dir / "manifest.json").write_text(json.dumps({"status": "failed"}))
    assert stage_is_resumable("A", stage_dir) is False


def test_stage_b_resumable_checks_compiled_modules(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_B_envs"
    stage_dir.mkdir()
    envs_dir = stage_dir / "envs"
    envs_dir.mkdir()
    mod_file = stage_dir / "mod.py"
    mod_file.write_text("x=1")
    manifest = {"status": "success", "compiled_modules": [str(mod_file)]}
    (stage_dir / "manifest.json").write_text(json.dumps(manifest))
    assert stage_is_resumable("B", stage_dir) is True


def test_stage_b_resumable_missing_module(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_B_envs"
    stage_dir.mkdir()
    envs_dir = stage_dir / "envs"
    envs_dir.mkdir()
    manifest = {"status": "success", "compiled_modules": ["/no/such/file.py"]}
    (stage_dir / "manifest.json").write_text(json.dumps(manifest))
    assert stage_is_resumable("B", stage_dir) is False


def test_stage_is_resumable_runtime_sensitive_stage_checks_runtime(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_D_prompts"
    stage_dir.mkdir()
    (stage_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "runtime": "pymc"}),
        encoding="utf-8",
    )
    (stage_dir / "prompts.jsonl").write_text("", encoding="utf-8")

    assert stage_is_resumable("D", stage_dir, requested_runtime="pymc") is True
    assert stage_is_resumable("D", stage_dir, requested_runtime="lazyppl") is False
    assert stage_is_resumable("D", stage_dir, requested_runtime=None) is False


def test_stage_f_resumable_requires_traces_artifact(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_F_validation"
    stage_dir.mkdir()
    (stage_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "runtime": "pymc"}),
        encoding="utf-8",
    )
    (stage_dir / "validated.jsonl").write_text("", encoding="utf-8")
    (stage_dir / "rejected.jsonl").write_text("", encoding="utf-8")
    assert stage_is_resumable("F", stage_dir, requested_runtime="pymc") is False

    (stage_dir / "traces.jsonl").write_text("", encoding="utf-8")
    assert stage_is_resumable("F", stage_dir, requested_runtime="pymc") is True


def test_run_manifest_payload() -> None:
    paths = resolve_run_paths("/tmp/runs", "run1")
    results = {
        "A": StageResult(stage="A", status="success"),
        "B": StageResult(stage="B", status="failed"),
    }
    payload = run_manifest_payload(paths, results, runtime_name="lazyppl")
    assert payload["run_id"] == "run1"
    assert payload["runtime"] == "lazyppl"
    assert payload["status_by_stage"]["A"] == "success"
    assert payload["status_by_stage"]["B"] == "failed"

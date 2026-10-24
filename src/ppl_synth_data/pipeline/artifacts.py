from __future__ import annotations

import json
from pathlib import Path

from ppl_synth_data.pipeline.contracts import REQUIRED_STAGE_ARTIFACTS, RUN_ID_RE, STAGE_DIR_NAMES
from ppl_synth_data.pipeline.types import RunPaths, StageResult

_RUNTIME_SENSITIVE_STAGES = frozenset({"D", "E", "F", "G"})


def resolve_run_paths(run_root: Path | str, run_id: str) -> RunPaths:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must match ^[A-Za-z0-9][A-Za-z0-9_-]*$ (no whitespace or slashes)")
    root = Path(run_root)
    run_dir = root / run_id
    stage_dirs = {stage: run_dir / dirname for stage, dirname in STAGE_DIR_NAMES.items()}
    return RunPaths(run_root=root, run_id=run_id, run_dir=run_dir, stage_dirs=stage_dirs)


def manifest_is_success(path: Path) -> bool:
    payload = _read_manifest_payload(path)
    return bool(payload and payload.get("status") == "success")


def stage_manifest_runtime(stage_dir: Path) -> str | None:
    payload = _read_manifest_payload(stage_dir / "manifest.json")
    if payload is None or payload.get("status") != "success":
        return None
    runtime = payload.get("runtime")
    if isinstance(runtime, str) and runtime:
        return runtime
    return None


def _positive_int_payload_value(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return int(value) > 0


def _stage_b_modules_intact(payload: dict[str, object], stage_dir: Path) -> bool:
    compiled_modules = payload.get("compiled_modules", [])
    if not isinstance(compiled_modules, list) or not compiled_modules:
        return False
    for item in compiled_modules:
        if not isinstance(item, str):
            return False
        module_path = Path(item)
        if not module_path.is_absolute():
            module_path = stage_dir / module_path
        if not module_path.is_file():
            return False
    return True


def stage_is_resumable(
    stage: str,
    stage_dir: Path,
    *,
    requested_runtime: str | None = None,
) -> bool:
    manifest_path = stage_dir / "manifest.json"
    payload = _read_manifest_payload(manifest_path)
    if payload is None or payload.get("status") != "success":
        return False

    for artifact in REQUIRED_STAGE_ARTIFACTS.get(stage, ()):  # pragma: no branch
        artifact_path = stage_dir / artifact
        if artifact_path.suffix:
            if not artifact_path.is_file():
                return False
        elif not artifact_path.is_dir():
            return False

    if stage == "B" and not _stage_b_modules_intact(payload, stage_dir):
        return False

    if stage in {"B", "C"} and _positive_int_payload_value(payload, "skipped_env_count"):
        return False

    if stage == "G":
        if _positive_int_payload_value(payload, "count_dropped"):
            return False
        count_written = payload.get("count_written")
        if isinstance(count_written, int | float) and not isinstance(count_written, bool):
            if int(count_written) <= 0:
                return False

    if stage in _RUNTIME_SENSITIVE_STAGES:
        if requested_runtime is None:
            return False
        if stage_manifest_runtime(stage_dir) != requested_runtime:
            return False

    return True


def run_manifest_payload(
    paths: RunPaths,
    results: dict[str, StageResult],
    *,
    runtime_name: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": paths.run_id,
        "status_by_stage": {stage: result.status for stage, result in results.items()},
    }
    if runtime_name is not None:
        payload["runtime"] = runtime_name
    return payload


def _read_manifest_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload

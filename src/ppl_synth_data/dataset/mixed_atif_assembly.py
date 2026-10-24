"""Assemble strict mixed-source ATIF bundles.

This module copies already-exported ATIF trial directories into one mixed
bundle and writes the explicit source map required by the verifier. It does not
generate data, run validators, or infer source runs from trial directory names.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthstats_schemas import ATIFRecord, MixedATIFAssemblyManifest, RewardArtifact

from ppl_synth_data.pipeline.io import write_json

_MIXED_ATIF_SCHEMA_VERSION = "ppl_synth_data.mixed_atif_assembly.v1"


@dataclass(frozen=True)
class MixedATIFSource:
    source_run_id: str
    atif_dir: Path
    run_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def _safe_id(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    path = Path(value)
    if path.name != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a single safe path segment: {value!r}")
    return value


def _assert_fresh_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")


def _runtime_from_atif(atif: ATIFRecord, reward: RewardArtifact) -> str:
    dataset_row = atif.metadata.get("dataset_row")
    if isinstance(dataset_row, dict):
        runtime = dataset_row.get("runtime")
        if isinstance(runtime, str) and runtime:
            return runtime
    runtime = reward.provenance.get("runtime")
    if isinstance(runtime, str) and runtime:
        return runtime
    raise ValueError(f"ATIF row {atif.session_id!r} has no explicit runtime provenance")


def _load_strict_trial(trial_dir: Path) -> tuple[ATIFRecord, RewardArtifact]:
    trajectory_path = trial_dir / "trajectory.json"
    reward_path = trial_dir / "reward.json"
    atif = ATIFRecord.model_validate_json(trajectory_path.read_text(encoding="utf-8"))
    reward = RewardArtifact.model_validate_json(reward_path.read_text(encoding="utf-8"))
    if atif.reward_artifact is None:
        raise ValueError(f"{trial_dir} trajectory.json has no embedded reward_artifact")
    embedded = atif.reward_artifact.model_dump(mode="json")
    sidecar = reward.model_dump(mode="json")
    if embedded != sidecar:
        raise ValueError(f"{trial_dir} reward.json differs from embedded reward_artifact")
    return atif, reward


def assemble_mixed_atif_sources(
    sources: list[MixedATIFSource],
    out_dir: Path,
    *,
    min_exported: int | None = None,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one ATIF source is required")
    if min_exported is not None and min_exported <= 0:
        raise ValueError("min_exported must be a positive integer")

    source_ids: set[str] = set()
    source_run_dirs: dict[str, str] = {}
    source_atif_dirs: dict[str, str] = {}
    source_manifest_sha256: dict[str, str] = {}
    source_trial_artifacts: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_run_id = _safe_id(source.source_run_id, field_name="source_run_id")
        if source_run_id in source_ids:
            raise ValueError(f"duplicate source_run_id: {source_run_id!r}")
        source_ids.add(source_run_id)
        run_dir = source.run_dir.resolve()
        atif_dir = source.atif_dir.resolve()
        if not run_dir.is_dir():
            raise ValueError(f"source run directory does not exist: {source.run_dir}")
        if not atif_dir.is_dir():
            raise ValueError(f"source ATIF directory does not exist: {source.atif_dir}")
        manifest_path = atif_dir / "manifest.json"
        source_manifest = _read_json_object(manifest_path)
        raw_trial_artifacts = source_manifest.get("trial_artifacts")
        if not isinstance(raw_trial_artifacts, dict) or not raw_trial_artifacts:
            raise ValueError(
                f"source ATIF manifest requires non-empty trial_artifacts: {manifest_path}"
            )
        source_run_dirs[source_run_id] = str(run_dir)
        source_atif_dirs[source_run_id] = str(atif_dir)
        source_manifest_sha256[source_run_id] = _sha256(manifest_path)
        source_trial_artifacts[source_run_id] = raw_trial_artifacts

    out_dir = out_dir.resolve()
    _assert_fresh_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_source_run_ids: dict[str, str] = {}
    atif_ids: list[str] = []
    trial_dirs: list[str] = []
    runtime_counts: Counter[str] = Counter()
    reward_source_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    trial_artifacts: dict[str, dict[str, Any]] = {}

    for source in sources:
        source_run_id = _safe_id(source.source_run_id, field_name="source_run_id")
        source_run_dir = source.run_dir.resolve()
        source_atif_dir = source.atif_dir.resolve()
        for trial_dir in sorted(path for path in source_atif_dir.iterdir() if path.is_dir()):
            trial_name = _safe_id(trial_dir.name, field_name="trial directory name")
            if trial_name in trial_source_run_ids:
                raise ValueError(f"duplicate mixed ATIF trial directory: {trial_name!r}")
            source_artifact = source_trial_artifacts[source_run_id].get(trial_name)
            if not isinstance(source_artifact, dict):
                raise ValueError(
                    f"source manifest for {source_run_id!r} has no trial_artifacts "
                    f"entry for {trial_name!r}"
                )
            atif, reward = _load_strict_trial(trial_dir)
            source_trajectory_sha256 = _sha256(trial_dir / "trajectory.json")
            source_reward_sha256 = _sha256(trial_dir / "reward.json")
            if source_artifact.get("trajectory_sha256") != source_trajectory_sha256:
                raise ValueError(f"source manifest trajectory_sha256 mismatch for {trial_name!r}")
            if source_artifact.get("reward_sha256") != source_reward_sha256:
                raise ValueError(f"source manifest reward_sha256 mismatch for {trial_name!r}")
            target = out_dir / trial_name
            shutil.copytree(trial_dir, target)
            trial_source_run_ids[trial_name] = source_run_id
            atif_ids.append(atif.session_id)
            trial_dirs.append(str(target))
            runtime = _runtime_from_atif(atif, reward)
            runtime_counts.update([runtime])
            reward_source_counts.update([reward.reward_source])
            source_counts.update([source_run_id])
            split = source_artifact.get("split")
            if not isinstance(split, str) or not split:
                raise ValueError(
                    f"source manifest for {source_run_id!r} has no split for {trial_name!r}"
                )
            split_counts.update([split])
            trial_artifacts[trial_name] = {
                "atif_id": atif.session_id,
                "trial_dir": str(target),
                "source_run_id": source_run_id,
                "source_run_dir": str(source_run_dir),
                "source_atif_dir": str(source_atif_dir),
                "runtime": runtime,
                "split": split,
                "local_record_id": source_artifact.get("local_record_id"),
                "prompt_id": source_artifact.get("prompt_id"),
                "env_id": source_artifact.get("env_id"),
                "reward_source": reward.reward_source,
                "training_log_reward": reward.training_log_reward,
                "data_ref": source_artifact.get("data_ref"),
                "data_ref_sha256": source_artifact.get("data_ref_sha256"),
                "prompt_sha256": source_artifact.get("prompt_sha256"),
                "completion_sha256": source_artifact.get("completion_sha256"),
                "parsed_program_sha256": source_artifact.get("parsed_program_sha256"),
                "validation_sha256": source_artifact.get("validation_sha256"),
                "env_spec_sha256": source_artifact.get("env_spec_sha256"),
                "stage_g_row_sha256": source_artifact.get("stage_g_row_sha256"),
                "source_trajectory_sha256": source_trajectory_sha256,
                "source_reward_sha256": source_reward_sha256,
                "trajectory_sha256": _sha256(target / "trajectory.json"),
                "reward_sha256": _sha256(target / "reward.json"),
            }

    if min_exported is not None and len(trial_source_run_ids) < min_exported:
        raise ValueError(
            f"mixed ATIF assembly produced {len(trial_source_run_ids)} rows, "
            f"below required minimum {min_exported}"
        )
    if len(set(atif_ids)) != len(atif_ids):
        raise ValueError("duplicate ATIF session_id values in mixed assembly")

    manifest = {
        "status": "success",
        "schema_version": _MIXED_ATIF_SCHEMA_VERSION,
        "digest_algorithm": "sha256",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "out_dir": str(out_dir),
        "source_run_dirs": source_run_dirs,
        "source_atif_dirs": source_atif_dirs,
        "source_manifest_sha256": source_manifest_sha256,
        "trial_source_run_ids": dict(sorted(trial_source_run_ids.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "runtime_counts": dict(sorted(runtime_counts.items())),
        "reward_source_counts": dict(sorted(reward_source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "count_exported": len(trial_source_run_ids),
        "row_count": len(trial_source_run_ids),
        "count_unique_atif_ids": len(set(atif_ids)),
        "atif_ids": atif_ids,
        "trial_dirs": trial_dirs,
        "trial_artifacts": dict(sorted(trial_artifacts.items())),
    }
    manifest = MixedATIFAssemblyManifest.model_validate(manifest).model_dump(mode="json")
    write_json(out_dir / "manifest.json", manifest)
    return manifest


__all__ = ["MixedATIFSource", "assemble_mixed_atif_sources"]

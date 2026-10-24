"""Validate exported ATIF directories against SynthStats contracts.

This is intentionally an artifact verifier, not a unit-test helper. It reads an
export directory produced by ``ppldf export-atif`` and checks the serialized
payloads that downstream ``synthstats`` training will consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from synthstats_schemas import (
    ATIFExportManifest,
    ATIFRecord,
    GenerationRecord,
    MixedATIFAssemblyManifest,
    ReleaseManifestAdapter,
    RewardArtifact,
    TraceRecord,
    TrialRecord,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_value_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def _json_payload_sha256(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return _json_value_sha256(json.loads(value))
    return _json_value_sha256(value)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _safe_manifest_id(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    path = Path(value)
    if path.name != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a single safe path segment: {value!r}")
    return value


def _resolve_run_dir_value(raw: Any, producer_root: Path, *, field_name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field_name} must be a non-empty string")
    path = Path(raw)
    if not path.is_absolute():
        if any(part == ".." for part in path.parts):
            raise ValueError(f"{field_name} must not escape producer_root: {raw!r}")
        path = producer_root / path
    return path.resolve()


@dataclass(frozen=True)
class _ManifestSources:
    mode: str
    run_dir: Path | None
    source_run_dirs: dict[str, Path]
    trial_source_run_ids: dict[str, str]

    def run_dir_for_trial(self, trial_dir: Path) -> Path:
        if self.run_dir is not None:
            return self.run_dir
        source_run_id = self.trial_source_run_ids[trial_dir.name]
        return self.source_run_dirs[source_run_id]

    def source_run_id_for_trial(self, trial_dir: Path) -> str | None:
        if self.run_dir is not None:
            return None
        return self.trial_source_run_ids[trial_dir.name]


def _resolve_mixed_sources(
    manifest: MixedATIFAssemblyManifest,
    producer_root: Path,
    trial_dirs: list[Path],
) -> _ManifestSources:
    source_run_dirs: dict[str, Path] = {}
    for raw_source_id, raw_run_dir in manifest.source_run_dirs.items():
        source_id = _safe_manifest_id(raw_source_id, field_name="source_run_id")
        if source_id in source_run_dirs:
            raise ValueError(f"duplicate source_run_id: {source_id!r}")
        source_run_dirs[source_id] = _resolve_run_dir_value(
            raw_run_dir,
            producer_root,
            field_name=f"source_run_dirs[{source_id!r}]",
        )

    trial_source_run_ids: dict[str, str] = {}
    for raw_trial_name, raw_source_id in manifest.trial_source_run_ids.items():
        trial_name = _safe_manifest_id(raw_trial_name, field_name="trial directory name")
        source_id = _safe_manifest_id(raw_source_id, field_name=f"source id for {trial_name!r}")
        if source_id not in source_run_dirs:
            raise ValueError(
                f"trial_source_run_ids[{trial_name!r}] references unknown "
                f"source_run_id {source_id!r}"
            )
        trial_source_run_ids[trial_name] = source_id

    trial_names = {trial_dir.name for trial_dir in trial_dirs}
    mapped_trial_names = set(trial_source_run_ids)
    missing = sorted(trial_names - mapped_trial_names)
    if missing:
        raise ValueError(f"mixed ATIF manifest has unmapped trial directories: {missing}")
    extra = sorted(mapped_trial_names - trial_names)
    if extra:
        raise ValueError(f"mixed ATIF manifest maps trial directories that are absent: {extra}")

    return _ManifestSources(
        mode="mixed",
        run_dir=None,
        source_run_dirs=source_run_dirs,
        trial_source_run_ids=trial_source_run_ids,
    )


def _resolve_manifest_sources(
    manifest: ATIFExportManifest | MixedATIFAssemblyManifest,
    producer_root: Path,
    trial_dirs: list[Path],
) -> _ManifestSources:
    if isinstance(manifest, ATIFExportManifest):
        run_dir = _resolve_run_dir_value(
            manifest.run_dir,
            producer_root,
            field_name="ATIF manifest run_dir",
        )
        return _ManifestSources(
            mode="single",
            run_dir=run_dir,
            source_run_dirs={run_dir.name: run_dir},
            trial_source_run_ids={},
        )
    if isinstance(manifest, MixedATIFAssemblyManifest):
        return _resolve_mixed_sources(manifest, producer_root, trial_dirs)
    raise TypeError(f"unsupported ATIF manifest type: {type(manifest).__name__}")


def _manifest_trial_artifacts(
    manifest: ATIFExportManifest | MixedATIFAssemblyManifest,
) -> dict[str, dict[str, Any]]:
    return {
        trial_name: artifact.model_dump(mode="json")
        for trial_name, artifact in manifest.trial_artifacts.items()
    }


def _resolve_data_ref(run_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("dataset row is missing data_ref")
    ref = Path(value)
    if ref.is_absolute() or any(part == ".." for part in ref.parts):
        raise ValueError(f"data_ref is not safely run-relative: {value!r}")
    resolved = (run_dir / ref).resolve()
    resolved.relative_to(run_dir)
    return resolved


def _iter_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(value)
    return rows


def _load_stage_g_row(
    *,
    run_dir: Path,
    runtime: str,
    split: str,
    local_record_id: str,
) -> dict[str, Any]:
    if not runtime or not split or not local_record_id:
        raise ValueError(
            "runtime, split, and local_record_id are required to re-check Stage G lineage"
        )
    path = run_dir / "stage_G_dataset" / f"{runtime}_with_data.{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Stage G with-data split is missing: {path}")
    matches = [row for row in _iter_jsonl_objects(path) if row.get("id") == local_record_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one Stage G row id={local_record_id!r} in {path}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _canonical_reward(value: dict[str, Any] | RewardArtifact) -> dict[str, Any]:
    artifact = value if isinstance(value, RewardArtifact) else RewardArtifact.model_validate(value)
    return artifact.model_dump(mode="json")


def _float_matches(left: object, right: object) -> bool:
    if not isinstance(left, int | float) or isinstance(left, bool):
        return False
    if not isinstance(right, int | float) or isinstance(right, bool):
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _first_step_message(atif: ATIFRecord, source: str) -> str:
    for step in atif.steps:
        if step.source == source:
            return _text(step.message)
    return ""


def _check_manifest_artifact(
    *,
    manifest_artifact: dict[str, Any] | None,
    require_release_digests: bool,
    trial_dir: Path,
    run_dir: Path,
    source_run_id: str | None,
    atif: ATIFRecord,
    reward: RewardArtifact,
    trajectory_sha256: str,
    reward_sha256: str,
    dataset_row: dict[str, Any] | None,
    data_path: Path | None,
) -> list[str]:
    failures: list[str] = []
    if manifest_artifact is None:
        if require_release_digests:
            failures.append(f"manifest trial_artifacts is missing entry for {trial_dir.name!r}")
        return failures

    expected_pairs: dict[str, object] = {
        "atif_id": atif.session_id,
        "reward_source": reward.reward_source,
        "trajectory_sha256": trajectory_sha256,
        "reward_sha256": reward_sha256,
    }
    if source_run_id is not None:
        expected_pairs["source_run_id"] = source_run_id
    if dataset_row is not None:
        expected_pairs["runtime"] = dataset_row.get("runtime")
        expected_pairs["split"] = dataset_row.get("split")
        expected_pairs["local_record_id"] = dataset_row.get("local_record_id")
        expected_pairs["prompt_id"] = dataset_row.get("prompt_id")
        expected_pairs["env_id"] = dataset_row.get("env_id")
        expected_pairs["data_ref"] = dataset_row.get("data_ref")
    for field_name, expected in expected_pairs.items():
        if expected is None:
            continue
        if manifest_artifact.get(field_name) != expected:
            failures.append(
                f"manifest trial_artifacts[{trial_dir.name!r}].{field_name} "
                f"does not match artifact value"
            )

    if not _float_matches(
        manifest_artifact.get("training_log_reward"),
        reward.training_log_reward,
    ):
        failures.append(
            f"manifest trial_artifacts[{trial_dir.name!r}].training_log_reward "
            "does not match reward.json"
        )

    if data_path is not None and data_path.is_file():
        actual_data_ref_sha256 = _sha256(data_path)
        if manifest_artifact.get("data_ref_sha256") != actual_data_ref_sha256:
            failures.append(
                f"manifest trial_artifacts[{trial_dir.name!r}].data_ref_sha256 "
                "does not match data_ref file"
            )

    prompt_message = _first_step_message(atif, "user")
    completion_message = _first_step_message(atif, "agent")
    digest_pairs = {
        "prompt_sha256": _sha256_text(prompt_message),
        "completion_sha256": _sha256_text(completion_message),
    }
    for field_name, actual in digest_pairs.items():
        if manifest_artifact.get(field_name) != actual:
            failures.append(
                f"manifest trial_artifacts[{trial_dir.name!r}].{field_name} "
                "does not match trajectory message"
            )

    if not require_release_digests or dataset_row is None:
        return failures

    try:
        stage_g_row = _load_stage_g_row(
            run_dir=run_dir,
            runtime=_text(dataset_row.get("runtime")),
            split=_text(dataset_row.get("split")),
            local_record_id=_text(dataset_row.get("local_record_id")),
        )
    except Exception as exc:  # noqa: BLE001 - report lineage check failures.
        failures.append(f"Stage G lineage lookup failed: {type(exc).__name__}: {exc}")
        return failures

    validation_payload = json.loads(_text(stage_g_row.get("validation_json")) or "{}")
    release_digests = {
        "prompt_sha256": _sha256_text(_text(stage_g_row.get("prompt"))),
        "completion_sha256": _sha256_text(_text(stage_g_row.get("completion"))),
        "env_spec_sha256": _json_payload_sha256(stage_g_row.get("env_spec_json")),
        "validation_sha256": _json_value_sha256(validation_payload),
        "stage_g_row_sha256": _json_value_sha256(stage_g_row),
    }
    for field_name, actual in release_digests.items():
        if actual is None:
            failures.append(f"Stage G row lacks required release digest source for {field_name}")
            continue
        if manifest_artifact.get(field_name) != actual:
            failures.append(
                f"manifest trial_artifacts[{trial_dir.name!r}].{field_name} "
                "does not match Stage G source row"
            )

    return failures


def _validate_trial(
    trial_dir: Path,
    *,
    run_dir: Path,
    source_run_id: str | None,
    manifest_artifact: dict[str, Any] | None,
    require_readable_data: bool,
    require_shared_records: bool,
    require_explicit_reward: bool,
    require_release_digests: bool,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    trajectory_path = trial_dir / "trajectory.json"
    reward_path = trial_dir / "reward.json"
    trial_report: dict[str, Any] = {
        "trial_dir": str(trial_dir),
        "trajectory_path": str(trajectory_path),
        "reward_path": str(reward_path),
        "source_run_id": source_run_id,
        "source_run_dir": str(run_dir),
    }

    try:
        trajectory_raw = _read_json(trajectory_path)
        reward_raw = _read_json(reward_path)
        atif = ATIFRecord.model_validate(trajectory_raw)
        reward = RewardArtifact.model_validate(reward_raw)
    except Exception as exc:  # noqa: BLE001 - report all artifact validation failures.
        failures.append(f"contract validation failed: {type(exc).__name__}: {exc}")
        trial_report["status"] = "failed"
        return trial_report, failures

    trial_report.update(
        {
            "status": "validated",
            "atif_id": atif.session_id,
            "reward_source": reward.reward_source,
            "training_log_reward": reward.training_log_reward,
            "trajectory_sha256": _sha256(trajectory_path),
            "reward_sha256": _sha256(reward_path),
            "manifest_artifact_present": manifest_artifact is not None,
        }
    )
    trajectory_sha256 = trial_report["trajectory_sha256"]
    reward_sha256 = trial_report["reward_sha256"]

    if require_explicit_reward and reward.reward is None:
        failures.append("reward.json has no explicit reward field")
    if (
        require_explicit_reward
        and atif.reward_artifact is not None
        and atif.reward_artifact.reward is None
    ):
        failures.append("embedded ATIF reward_artifact has no explicit reward field")
    if atif.reward_artifact is None:
        failures.append("ATIF record has no embedded reward_artifact")
    else:
        embedded = _canonical_reward(atif.reward_artifact)
        sidecar = _canonical_reward(reward)
        if embedded != sidecar:
            failures.append("sidecar reward.json differs from embedded ATIF reward_artifact")

    metadata = atif.metadata
    dataset_row = metadata.get("dataset_row")
    data_path: Path | None = None
    if not isinstance(dataset_row, dict):
        failures.append("ATIF metadata.dataset_row is missing or not an object")
    else:
        try:
            data_path = _resolve_data_ref(run_dir, dataset_row.get("data_ref"))
            trial_report["data_ref"] = str(data_path)
            if not data_path.is_file():
                failures.append(f"data_ref does not exist: {data_path}")
            elif require_readable_data:
                frame = pd.read_parquet(data_path)
                trial_report["data_rows"] = int(len(frame))
                trial_report["data_columns"] = list(map(str, frame.columns))
                if frame.empty:
                    failures.append(f"data_ref is readable but empty: {data_path}")
        except Exception as exc:  # noqa: BLE001 - report all artifact validation failures.
            failures.append(f"data_ref validation failed: {type(exc).__name__}: {exc}")

    failures.extend(
        _check_manifest_artifact(
            manifest_artifact=manifest_artifact,
            require_release_digests=require_release_digests,
            trial_dir=trial_dir,
            run_dir=run_dir,
            source_run_id=source_run_id,
            atif=atif,
            reward=reward,
            trajectory_sha256=trajectory_sha256,
            reward_sha256=reward_sha256,
            dataset_row=dataset_row if isinstance(dataset_row, dict) else None,
            data_path=data_path,
        )
    )

    if require_shared_records:
        shared_generation = metadata.get("shared_generation_record")
        shared_trial = metadata.get("shared_trial_record")
        shared_trace = metadata.get("shared_trace_record")
        try:
            generation = GenerationRecord.model_validate(shared_generation)
            trial = TrialRecord.model_validate(shared_trial)
            trace = TraceRecord.model_validate(shared_trace)
            if generation.record_id != atif.session_id:
                failures.append("shared GenerationRecord.record_id does not match ATIF session_id")
            if trial.trial_id != atif.session_id:
                failures.append("shared TrialRecord.trial_id does not match ATIF session_id")
            if trace.trial_id != atif.session_id:
                failures.append("shared TraceRecord.trial_id does not match ATIF session_id")
            if trial.trace_id != trace.trace_id:
                failures.append("shared TrialRecord.trace_id does not match TraceRecord.trace_id")
            if (
                not isinstance(generation.parsed_program, str)
                or not generation.parsed_program.strip()
            ):
                failures.append("shared GenerationRecord.parsed_program is empty")
            if not isinstance(trial.program, str) or not trial.program.strip():
                failures.append("shared TrialRecord.program is empty")
            if generation.parsed_program != trial.program:
                failures.append(
                    "shared GenerationRecord.parsed_program does not match TrialRecord.program"
                )
        except Exception as exc:  # noqa: BLE001 - report all artifact validation failures.
            failures.append(f"shared record validation failed: {type(exc).__name__}: {exc}")

    return trial_report, failures


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def validate_atif_dir(args: argparse.Namespace) -> dict[str, Any]:
    atif_dir = Path(args.atif_dir).resolve()
    producer_root = Path(args.producer_root).resolve()
    manifest_path = atif_dir / "manifest.json"
    manifest_model = ReleaseManifestAdapter.validate_json(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_model, ATIFExportManifest | MixedATIFAssemblyManifest):
        raise ValueError(
            "ATIF directory manifest must be an ATIF export or mixed ATIF assembly manifest"
        )
    manifest = manifest_model.model_dump(mode="json")
    trial_dirs = [path for path in sorted(atif_dir.iterdir()) if path.is_dir()]
    sources = _resolve_manifest_sources(manifest_model, producer_root, trial_dirs)
    trial_artifacts = _manifest_trial_artifacts(manifest_model)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    rewards: list[float] = []

    if args.require_release_digests and not trial_artifacts:
        failures.append(
            {
                "trial_dir": str(atif_dir),
                "failure": "manifest requires non-empty trial_artifacts for release validation",
            }
        )

    for trial_dir in trial_dirs:
        raw_manifest_artifact = trial_artifacts.get(trial_dir.name)
        manifest_artifact = (
            raw_manifest_artifact if isinstance(raw_manifest_artifact, dict) else None
        )
        report, trial_failures = _validate_trial(
            trial_dir,
            run_dir=sources.run_dir_for_trial(trial_dir),
            source_run_id=sources.source_run_id_for_trial(trial_dir),
            manifest_artifact=manifest_artifact,
            require_readable_data=args.require_readable_data,
            require_shared_records=args.require_shared_records,
            require_explicit_reward=args.require_explicit_reward,
            require_release_digests=args.require_release_digests,
        )
        reports.append(report)
        value = report.get("training_log_reward")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            rewards.append(float(value))
        for failure in trial_failures:
            failures.append({"trial_dir": str(trial_dir), "failure": failure})

    if args.min_trials is not None and len(trial_dirs) < args.min_trials:
        failures.append(
            {
                "trial_dir": str(atif_dir),
                "failure": (
                    f"expected at least {args.min_trials} trial directories, "
                    f"found {len(trial_dirs)}"
                ),
            }
        )
    if args.require_reward_variance and _variance(rewards) <= 0.0:
        failures.append(
            {
                "trial_dir": str(atif_dir),
                "failure": "training_log_reward variance is zero or unavailable",
            }
        )
    manifest_count = manifest.get("count_exported")
    if isinstance(manifest_count, int) and manifest_count != len(trial_dirs):
        failures.append(
            {
                "trial_dir": str(atif_dir),
                "failure": (
                    f"manifest count_exported={manifest_count} but found "
                    f"{len(trial_dirs)} trial directories"
                ),
            }
        )

    return {
        "schema": "ppl_synth_data.verify_atif_contracts.v1",
        "status": "failed" if failures else "passed",
        "atif_dir": str(atif_dir),
        "producer_root": str(producer_root),
        "source_mode": sources.mode,
        "run_dir": str(sources.run_dir) if sources.run_dir is not None else None,
        "source_run_dirs": {
            source_id: str(path) for source_id, path in sorted(sources.source_run_dirs.items())
        },
        "trial_source_run_ids": dict(sorted(sources.trial_source_run_ids.items())),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "count_trials": len(trial_dirs),
        "reward_variance": _variance(rewards),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "failures": failures,
        "trials": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ATIF + RewardArtifact export directories against the shared contracts."
        ),
    )
    parser.add_argument("--atif-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--producer-root", type=Path, default=Path.cwd())
    parser.add_argument("--min-trials", type=int, default=None)
    parser.add_argument("--require-readable-data", action="store_true")
    parser.add_argument("--require-reward-variance", action="store_true")
    parser.add_argument("--require-shared-records", action="store_true")
    parser.add_argument("--require-explicit-reward", action="store_true")
    parser.add_argument("--require-release-digests", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = validate_atif_dir(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("status", "count_trials", "reward_variance")},
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

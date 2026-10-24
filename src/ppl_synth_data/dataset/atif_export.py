"""Export validated pipeline rows into ATIF directories.

This module is owned by ``ppl-synth-data`` because it translates local pipeline
artifacts into the shared ATIF/reward ABI. It does not import ``synthstats``.
Training-side policy tokenisation remains in SynthStats.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

from synthstats_schemas import (
    TRAINING_LOG_REWARD_MIN,
    ATIFExportManifest,
    ATIFRecord,
    RewardArtifact,
    RewardComponent,
    RewardComponentKind,
    RewardScalarization,
    RewardScalarizationMethod,
)

from ppl_synth_data.pipeline.io import write_json

_ATIF_SCHEMA_VERSION = "ATIF-v1.6"
_PYMC_PRODUCER = "ppl_synth_data.validation.pymc_semantics"
_LAZYPPL_REWARD_SOURCE = "validation.lazyppl_elpd"
_REWARD_SCHEMA_VERSION = "synthstats.harbor.reward.v1"
_SEMANTIC_METRIC_REWARD_SOURCES = {
    "avg_holdout_logp": "validation.pymc_semantic_holdout_logp",
}
_EXPORTABLE_PYMC_SEMANTIC_STATUSES = ("pass", "flag")

BaselineKind = Literal["mean", "median", "zero"]
SplitKind = Literal["train", "validation", "test", "all"]
_SOURCE_PROVENANCE_KEYS = (
    "source_category",
    "source_url_or_repo",
    "license_signal",
    "adaptation_mode",
    "source_retrieval_date",
)


def _sha256_file(path: Path) -> str:
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


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(value)
    return rows


def _json_object_from_string(value: object, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return parsed


def _source_provenance_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    source: dict[str, str] = {}
    for key in _SOURCE_PROVENANCE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            source[key] = value
    if source.get("source_category") == "synthetic_invented":
        source.setdefault("source_url_or_repo", "not_applicable_invented")
        source.setdefault("license_signal", "not_applicable_invented")
        source.setdefault("adaptation_mode", "invented")
        source.setdefault("source_retrieval_date", "not_applicable_invented")
    return source


def _source_provenance_from_stage_g_row(row: dict[str, Any]) -> dict[str, str]:
    payload = _json_object_from_string(
        row.get("provenance_json"),
        field_name="provenance_json",
    )
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        source = _source_provenance_from_payload(provenance)
        if source:
            return source
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return _source_provenance_from_payload(metadata)
    return {}


def _load_stage_g_records(
    run_dir: Path,
    *,
    runtime_name: str,
    split: SplitKind,
) -> list[dict[str, Any]]:
    stage_dir = run_dir / "stage_G_dataset"
    splits = ("train", "validation", "test") if split == "all" else (split,)
    records: list[dict[str, Any]] = []
    for split_name in splits:
        path = stage_dir / f"{runtime_name}_with_data.{split_name}.jsonl"
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            validation = _json_object_from_string(
                row.get("validation_json"),
                field_name="validation_json",
            )
            source_provenance = _source_provenance_from_stage_g_row(row)
            records.append(
                {
                    "id": row.get("id"),
                    "prompt_id": row.get("prompt_id"),
                    "env_id": row.get("env_id"),
                    "split": row.get("split", split_name),
                    "prompt_text": row.get("prompt"),
                    "raw_completion": row.get("completion"),
                    "parsed_program": row.get("parsed_program", ""),
                    "validation": validation,
                    "data_ref": row.get("data_ref"),
                    "trace_id": row.get("trace_id"),
                    "source_provenance": source_provenance,
                    "stage_g_row": row,
                }
            )
    return records


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    candidate = float(value)
    return candidate if math.isfinite(candidate) else None


def _semantic_score(record: dict[str, Any], metric_name: str) -> float | None:
    validation = record.get("validation")
    if not isinstance(validation, dict):
        return None
    if validation.get("semantic_metric") == metric_name:
        score = _finite_float(validation.get("semantic_score"))
        if score is not None:
            return score
    details = validation.get("semantic_details")
    if isinstance(details, dict):
        return _finite_float(details.get(metric_name))
    return None


def _pymc_semantic_export_rejection(record: dict[str, Any]) -> str | None:
    validation = record.get("validation")
    if not isinstance(validation, dict):
        return "missing validation payload"
    if validation.get("ok") is not True:
        return "validation.ok is not true"
    semantic_status = validation.get("semantic_status")
    if semantic_status not in _EXPORTABLE_PYMC_SEMANTIC_STATUSES:
        return (
            "semantic_status must be one of "
            f"{list(_EXPORTABLE_PYMC_SEMANTIC_STATUSES)!r}; got {semantic_status!r}"
        )
    return None


def _pymc_semantic_status(record: dict[str, Any]) -> str:
    validation = record.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("missing validation payload")
    semantic_status = validation.get("semantic_status")
    if semantic_status not in _EXPORTABLE_PYMC_SEMANTIC_STATUSES:
        raise ValueError(
            "PyMC reward export requires semantic_status to be one of "
            f"{list(_EXPORTABLE_PYMC_SEMANTIC_STATUSES)!r}; got {semantic_status!r}"
        )
    return str(semantic_status)


def _embedded_lazyppl_reward_artifact(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    validation = record.get("validation")
    if not isinstance(validation, dict):
        return None, "missing validation payload"
    details = validation.get("semantic_details")
    if not isinstance(details, dict):
        return None, "missing semantic_details"
    candidate = details.get("reward_artifact")
    if not isinstance(candidate, dict):
        return None, "missing embedded LazyPPL reward_artifact"
    try:
        artifact = RewardArtifact.model_validate(candidate).model_dump(mode="json")
    except Exception as exc:
        return None, f"invalid embedded LazyPPL reward_artifact: {type(exc).__name__}: {exc}"

    if artifact.get("reward_source") != _LAZYPPL_REWARD_SOURCE:
        return None, (
            "embedded LazyPPL reward_artifact has unsupported reward_source: "
            f"{artifact.get('reward_source')!r}"
        )
    diagnostics = artifact.get("diagnostics")
    if isinstance(diagnostics, dict):
        if diagnostics.get("holdout_error") is not None:
            return None, "embedded LazyPPL reward_artifact has holdout error"
        metrics = artifact.get("metrics")
        has_holdout_metric = (
            isinstance(metrics, dict) and _finite_float(metrics.get("avg_holdout_logp")) is not None
        )
        if diagnostics.get("pareto_k_error") is not None and not has_holdout_metric:
            return None, "embedded LazyPPL reward_artifact has Pareto-k error"
        if diagnostics.get("pareto_k_warning") is True and not has_holdout_metric:
            return None, "embedded LazyPPL reward_artifact has Pareto-k warning"

    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    source_provenance = record.get("source_provenance")
    if not isinstance(source_provenance, dict):
        source_provenance = {}
    artifact["provenance"] = {
        **provenance,
        "runtime": "lazyppl",
        "record_id": _text(record.get("id")),
        "atif_id": _text(record.get("atif_id")),
        "prompt_id": _text(record.get("prompt_id")),
        "env_id": _text(record.get("env_id")),
        "source_provenance": source_provenance,
    }
    return RewardArtifact.model_validate(artifact).model_dump(mode="json"), None


def _baseline(scores: list[float], kind: BaselineKind) -> float:
    if kind == "zero":
        return 0.0
    if kind == "median":
        return float(median(scores))
    return float(mean(scores))


def _population_std(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    mu = mean(scores)
    return math.sqrt(sum((score - mu) ** 2 for score in scores) / len(scores))


def _temperature(scores: list[float], requested: float | None) -> float:
    if requested is not None:
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("reward_temperature must be finite and positive")
        return float(requested)
    return max(_population_std(scores), 1.0)


def _reward_source_for_metric(metric_name: str) -> str:
    try:
        return _SEMANTIC_METRIC_REWARD_SOURCES[metric_name]
    except KeyError:
        supported = ", ".join(sorted(_SEMANTIC_METRIC_REWARD_SOURCES))
        raise ValueError(
            f"unsupported ATIF reward metric {metric_name!r}; supported metrics: {supported}"
        ) from None


def _assert_fresh_output_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    entries = [entry for entry in out_dir.iterdir() if entry.name != ".DS_Store"]
    if entries:
        raise FileExistsError(
            f"ATIF output directory already contains files: {out_dir}. "
            "Choose a fresh output directory to avoid mixing stale trajectories."
        )


def _safe_trial_dir_name(value: str) -> str:
    if not value:
        raise ValueError("ATIF trial id must be non-empty")
    if value in {".", ".."} or "/" in value or "\\" in value or Path(value).name != value:
        raise ValueError(f"ATIF trial id is not path-safe: {value!r}")
    return value


def _stable_atif_id(
    *,
    run_dir: Path,
    runtime_name: str,
    local_record_id: str,
) -> str:
    run_id = _safe_trial_dir_name(run_dir.name)
    runtime_id = _safe_trial_dir_name(runtime_name)
    record_id = _safe_trial_dir_name(local_record_id)
    return _safe_trial_dir_name(f"{runtime_id}__{run_id}__{record_id}")


def _safe_existing_data_ref(run_dir: Path, value: str) -> str:
    if not value:
        raise ValueError("Stage G row is missing data_ref")
    ref_path = Path(value)
    if ref_path.is_absolute() or value in {".", ".."}:
        raise ValueError(f"Stage G data_ref is not run-relative: {value!r}")
    if any(part == ".." for part in ref_path.parts):
        raise ValueError(f"Stage G data_ref escapes the run directory: {value!r}")
    run_root = run_dir.resolve()
    resolved = (run_root / ref_path).resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"Stage G data_ref escapes the run directory: {value!r}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Stage G data_ref does not exist: {value!r}")
    return value


def _source_lineage_digests(
    record: dict[str, Any],
    *,
    run_dir: Path,
    data_ref: str,
) -> dict[str, str | None]:
    stage_g_row = record.get("stage_g_row")
    if not isinstance(stage_g_row, dict):
        stage_g_row = {}
    data_path = (run_dir.resolve() / data_ref).resolve()
    data_path.relative_to(run_dir.resolve())
    return {
        "data_ref": data_ref,
        "data_ref_sha256": _sha256_file(data_path),
        "prompt_sha256": _sha256_text(_text(record.get("prompt_text"))),
        "completion_sha256": _sha256_text(_text(record.get("raw_completion"))),
        "parsed_program_sha256": _sha256_text(_text(record.get("parsed_program"))),
        "validation_sha256": _json_value_sha256(record.get("validation", {})),
        "env_spec_sha256": _json_payload_sha256(stage_g_row.get("env_spec_json")),
        "stage_g_row_sha256": _json_value_sha256(stage_g_row),
    }


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _dataset_row(record: dict[str, Any], *, runtime_name: str) -> dict[str, Any]:
    atif_id = _text(record.get("atif_id")) or _text(record.get("id"))
    local_record_id = _text(record.get("local_record_id")) or _text(record.get("id"))
    local_trace_id = _text(record.get("trace_id"))
    source_provenance = record.get("source_provenance")
    if not isinstance(source_provenance, dict):
        source_provenance = {}
    row = {
        "id": atif_id,
        "atif_id": atif_id,
        "local_record_id": local_record_id,
        "env_id": _text(record.get("env_id")),
        "prompt_id": _text(record.get("prompt_id")),
        "split": _text(record.get("split")),
        "trace_id": f"{atif_id}:validation",
        "local_trace_id": local_trace_id,
        "runtime": runtime_name,
        "data_ref": _text(record.get("data_ref")),
        "source_provenance": source_provenance,
    }
    lineage_digests = record.get("lineage_digests")
    if isinstance(lineage_digests, dict):
        row["lineage_digests"] = lineage_digests
    row.update(source_provenance)
    return row


def _reward_artifact(
    *,
    record: dict[str, Any],
    metric_name: str,
    score: float,
    baseline_value: float,
    temperature_value: float,
) -> dict[str, Any]:
    reward_source = _reward_source_for_metric(metric_name)
    normalized = (score - baseline_value) / temperature_value
    training_log_reward = max(float(normalized), TRAINING_LOG_REWARD_MIN)
    reward = math.exp(min(training_log_reward, 20.0))
    atif_id = _text(record.get("atif_id"))
    local_record_id = _text(record.get("local_record_id")) or _text(record.get("id"))
    validation = record.get("validation")
    semantic_details = {}
    if isinstance(validation, dict) and isinstance(validation.get("semantic_details"), dict):
        semantic_details = dict(validation["semantic_details"])
    semantic_status = _pymc_semantic_status(record)
    source_provenance = record.get("source_provenance")
    if not isinstance(source_provenance, dict):
        source_provenance = {}
    reward_component = RewardComponent.model_validate(dict(
        name="semantic_score",
        kind=RewardComponentKind.terminal_verifier,
        value=score,
        source=reward_source,
        evidence_kind="verifier",
        truth_owner="verifier",
        producer=_PYMC_PRODUCER,
        weight=1.0,
        log_contribution=training_log_reward,
        confidence=1.0,
        metrics={metric_name: score},
        provenance={
            "metric": metric_name,
            "record_id": local_record_id,
            "atif_id": atif_id,
            "env_id": _text(record.get("env_id")),
            "source_provenance": source_provenance,
        },
    ))
    validation_success = (
        1.0 if isinstance(validation, dict) and validation.get("ok") is True else 0.0
    )
    validation_component = RewardComponent.model_validate(dict(
        name="validation_success",
        kind=RewardComponentKind.diagnostic,
        value=validation_success,
        source=reward_source,
        evidence_kind="verifier",
        truth_owner="verifier",
        producer=_PYMC_PRODUCER,
        weight=0.0,
        log_contribution=0.0 if validation_success else TRAINING_LOG_REWARD_MIN,
        confidence=1.0,
        metrics={"validation_ok": validation_success},
        provenance={
            "metric": "validation_ok",
            "record_id": local_record_id,
            "atif_id": atif_id,
            "env_id": _text(record.get("env_id")),
            "semantic_status": semantic_status,
            "component_role": "diagnostic_only",
            "semantic_validation_gate": "pre_export_hard_gate",
            "source_provenance": source_provenance,
        },
    ))
    artifact = RewardArtifact.model_validate(dict(
        schema_version=_REWARD_SCHEMA_VERSION,
        reward=float(reward),
        training_log_reward=training_log_reward,
        reward_transform={
            "kind": "run_standardized_semantic_score",
            "metric": metric_name,
            "baseline": baseline_value,
            "temperature": temperature_value,
            "lower_log_reward_bound": TRAINING_LOG_REWARD_MIN,
        },
        reward_source=reward_source,
        evidence_kind="verifier",
        truth_owner="verifier",
        producer=_PYMC_PRODUCER,
        metrics={
            metric_name: score,
            "baseline": baseline_value,
            "temperature": temperature_value,
        },
        components={
            "semantic_score": score,
            "validation_success": validation_success,
        },
        reward_components=(reward_component, validation_component),
        scalarization=RewardScalarization(
            method=RewardScalarizationMethod.weighted_sum,
            component_names=("semantic_score", "validation_success"),
            weights={"semantic_score": 1.0, "validation_success": 0.0},
            lower_log_reward_bound=TRAINING_LOG_REWARD_MIN,
            temperature=temperature_value,
            diagnostics={
                "aggregate": "training_log_reward",
                "semantic_validation_gate": "pre_export_hard_gate",
                "accepted_semantic_statuses": list(_EXPORTABLE_PYMC_SEMANTIC_STATUSES),
                "validation_success_component_role": "diagnostic_only",
            },
        ),
        diagnostics={
            "semantic_status": semantic_status,
            "semantic_validation_gate": "pre_export_hard_gate",
            "accepted_semantic_statuses": list(_EXPORTABLE_PYMC_SEMANTIC_STATUSES),
            "validation_success_component_role": "diagnostic_only",
        },
        provenance={
            "runtime": _text(record.get("runtime")) or "pymc",
            "record_id": local_record_id,
            "atif_id": atif_id,
            "prompt_id": _text(record.get("prompt_id")),
            "env_id": _text(record.get("env_id")),
            "semantic_status": semantic_status,
            "semantic_details": semantic_details,
            "source_provenance": source_provenance,
        },
    ))
    return artifact.model_dump(mode="json")


def _atif_record(
    record: dict[str, Any],
    reward_artifact: dict[str, Any],
    *,
    runtime_name: str,
) -> dict[str, Any]:
    local_record_id = _text(record.get("local_record_id")) or _text(record.get("id"))
    atif_id = _text(record.get("atif_id")) or local_record_id
    prompt_id = _text(record.get("prompt_id"))
    env_id = _text(record.get("env_id"))
    prompt = _text(record.get("prompt_text"))
    completion = _text(record.get("raw_completion"))
    parsed_program = _text(record.get("parsed_program"))
    dataset_row = _dataset_row(record, runtime_name=runtime_name)
    trace_id = _text(dataset_row.get("trace_id"))
    source_provenance = record.get("source_provenance")
    if not isinstance(source_provenance, dict):
        source_provenance = {}
    atif = ATIFRecord.model_validate(dict(
        schema_version=_ATIF_SCHEMA_VERSION,
        session_id=atif_id,
        metadata={
            "atif_id": atif_id,
            "local_record_id": local_record_id,
            "source_provenance": source_provenance,
            "dataset_row": dataset_row,
            "shared_generation_record": {
                "schema_version": "ppl.platform.record.v0",
                "record_id": atif_id,
                "task_id": env_id,
                "prompt_id": prompt_id,
                "split": _text(record.get("split")),
                "runtime": {"name": runtime_name},
                "raw_output": completion,
                "parsed_program": parsed_program,
                "validation": record.get("validation", {}),
                "metadata": {
                    "atif_id": atif_id,
                    "local_record_id": local_record_id,
                    "data_ref": dataset_row["data_ref"],
                    "source_provenance": source_provenance,
                },
            },
            "shared_trial_record": {
                "schema_version": "ppl.platform.record.v0",
                "trial_id": atif_id,
                "trace_id": trace_id,
                "task_id": env_id,
                "program": parsed_program,
                "runtime": {"name": runtime_name},
                "metadata": {
                    "atif_id": atif_id,
                    "stage": "G",
                    "split": _text(record.get("split")),
                    "prompt_id": prompt_id,
                    "data_ref": dataset_row["data_ref"],
                    "local_trial_id": local_record_id,
                    "local_record_id": local_record_id,
                    "source_provenance": source_provenance,
                },
            },
            "shared_trace_record": {
                "schema_version": "ppl.platform.record.v0",
                "trace_id": trace_id,
                "task_id": env_id,
                "trial_id": atif_id,
                "metadata": {
                    "atif_id": atif_id,
                    "stage": "G",
                    "runtime": runtime_name,
                    "local_record_id": local_record_id,
                    "local_trace_id": _text(record.get("trace_id")),
                    "source_provenance": source_provenance,
                },
            },
        },
        steps=[
            {
                "step_id": "step-system",
                "source": "system",
                "message": "You are a probabilistic programming assistant.",
            },
            {
                "step_id": "step-user",
                "source": "user",
                "message": prompt,
            },
            {
                "step_id": "step-agent",
                "source": "agent",
                "message": completion,
                "metrics": {},
                "metadata": {
                    "parsed_program": parsed_program,
                    "policy_score_status": "not_recorded",
                    "training_input_mode": "message_only_requires_policy_rescore",
                    "source": "ppl-synth-data.stage_F_validation",
                },
            },
        ],
        reward_artifact=RewardArtifact.model_validate(reward_artifact),
    ))
    return atif.model_dump(mode="json")


def export_atif_from_run(
    run_dir: Path,
    out_dir: Path,
    *,
    metric_name: str = "avg_holdout_logp",
    baseline: BaselineKind = "mean",
    reward_temperature: float | None = None,
    runtime_name: str = "pymc",
    split: SplitKind = "train",
    min_exported: int | None = None,
) -> dict[str, Any]:
    """Export Stage G dataset rows into ATIF trial directories.

    Stage G is the dataset boundary that applies split and drop decisions. ATIF
    export intentionally consumes that boundary instead of Stage F directly, so
    offline training cannot bypass train/validation/test partitioning or rows
    dropped because their materialized data is unavailable.
    """
    records = _load_stage_g_records(run_dir, runtime_name=runtime_name, split=split)
    use_embedded_reward = runtime_name == "lazyppl"
    scored: list[tuple[dict[str, Any], float, dict[str, Any] | None]] = []
    dropped: list[dict[str, Any]] = []
    if not use_embedded_reward:
        _reward_source_for_metric(metric_name)
    for index, record in enumerate(records):
        local_record_id = _safe_trial_dir_name(_text(record.get("id")) or f"record-{index:06d}")
        record["id"] = local_record_id
        record["local_record_id"] = local_record_id
        record["atif_id"] = _stable_atif_id(
            run_dir=run_dir,
            runtime_name=runtime_name,
            local_record_id=local_record_id,
        )
        raw_data_ref = _text(record.get("data_ref"))
        try:
            record["data_ref"] = _safe_existing_data_ref(
                run_dir,
                raw_data_ref,
            )
        except ValueError as exc:
            if not raw_data_ref:
                raise ValueError(
                    f"Stage G row {record.get('id')!r} is missing data_ref; "
                    "refusing to export an unscoreable ATIF row"
                ) from exc
            raise
        if not _text(record.get("parsed_program")).strip():
            dropped.append(
                {
                    "id": record.get("id"),
                    "reason": "missing parsed_program",
                }
            )
            continue
        if use_embedded_reward:
            artifact, reason = _embedded_lazyppl_reward_artifact(record)
            if artifact is None:
                dropped.append(
                    {
                        "id": record.get("id"),
                        "reason": reason or "missing LazyPPL reward artifact",
                    }
                )
                continue
            score = _finite_float(artifact.get("training_log_reward"))
            if score is None:
                dropped.append(
                    {
                        "id": record.get("id"),
                        "reason": (
                            "embedded LazyPPL reward_artifact has no finite training_log_reward"
                        ),
                    }
                )
                continue
            scored.append((record, score, artifact))
            continue
        rejection = _pymc_semantic_export_rejection(record)
        if rejection is not None:
            dropped.append(
                {
                    "id": record.get("id"),
                    "reason": rejection,
                }
            )
            continue
        score = _semantic_score(record, metric_name)
        if score is None:
            dropped.append(
                {
                    "id": record.get("id"),
                    "reason": f"missing finite semantic metric: {metric_name}",
                }
            )
            continue
        scored.append((record, score, None))
    if not scored:
        if use_embedded_reward:
            raise ValueError(
                f"no Stage G {runtime_name!r} rows for split={split!r} expose a "
                f"valid {_LAZYPPL_REWARD_SOURCE!r} RewardArtifact"
            )
        raise ValueError(
            f"no Stage G {runtime_name!r} rows for split={split!r} expose metric {metric_name!r}"
        )
    if min_exported is not None:
        try:
            min_exported_int = int(min_exported)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_exported must be a positive integer") from exc
        if isinstance(min_exported, bool) or min_exported_int <= 0:
            raise ValueError("min_exported must be a positive integer")
        if len(scored) < min_exported_int:
            raise ValueError(
                f"ATIF export produced {len(scored)} rows, "
                f"below required minimum {min_exported_int}"
            )

    scores = [score for _, score, _ in scored]
    if use_embedded_reward:
        baseline_value = 0.0
        temperature_value = 1.0
    else:
        baseline_value = _baseline(scores, baseline)
        temperature_value = _temperature(scores, reward_temperature)

    _assert_fresh_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_dirs: list[str] = []
    atif_ids: list[str] = []
    trial_artifacts: dict[str, dict[str, Any]] = {}
    seen_atif_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for index, (record, score, embedded_reward_artifact) in enumerate(scored):
        atif_id = _safe_trial_dir_name(
            _text(record.get("atif_id")) or f"{runtime_name}__{run_dir.name}__record-{index:06d}"
        )
        if atif_id in seen_atif_ids:
            raise ValueError(f"duplicate ATIF id: {atif_id!r}")
        seen_atif_ids.add(atif_id)
        trial_dir = out_dir / atif_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        record["runtime"] = runtime_name
        lineage_digests = _source_lineage_digests(
            record,
            run_dir=run_dir,
            data_ref=_text(record.get("data_ref")),
        )
        record_split = _text(record.get("split"))
        if not record_split:
            raise ValueError(
                f"Stage G row {record.get('id')!r} is missing split; "
                "refusing to export release ATIF without split provenance"
            )
        split_counts.update([record_split])
        record["lineage_digests"] = lineage_digests
        reward_artifact = (
            embedded_reward_artifact
            if embedded_reward_artifact is not None
            else _reward_artifact(
                record=record,
                metric_name=metric_name,
                score=score,
                baseline_value=baseline_value,
                temperature_value=temperature_value,
            )
        )
        atif = _atif_record(record, reward_artifact, runtime_name=runtime_name)
        trajectory_path = trial_dir / "trajectory.json"
        reward_path = trial_dir / "reward.json"
        write_json(trajectory_path, atif)
        write_json(reward_path, reward_artifact)
        trial_dirs.append(str(trial_dir))
        atif_ids.append(atif_id)
        trial_artifacts[atif_id] = {
            "atif_id": atif_id,
            "trial_dir": str(trial_dir),
            "runtime": runtime_name,
            "split": record_split,
            "local_record_id": _text(record.get("local_record_id")),
            "prompt_id": _text(record.get("prompt_id")),
            "env_id": _text(record.get("env_id")),
            "reward_source": reward_artifact["reward_source"],
            "training_log_reward": reward_artifact["training_log_reward"],
            **lineage_digests,
            "trajectory_sha256": _sha256_file(trajectory_path),
            "reward_sha256": _sha256_file(reward_path),
        }

    manifest = {
        "status": "success",
        "schema_version": "ppl_synth_data.atif_export.v1",
        "digest_algorithm": "sha256",
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "metric_name": metric_name,
        "baseline": baseline,
        "baseline_value": baseline_value,
        "reward_temperature": temperature_value,
        "runtime": runtime_name,
        "split": split,
        "reward_mode": ("embedded_reward_artifact" if use_embedded_reward else "semantic_metric"),
        "reward_source": (
            _LAZYPPL_REWARD_SOURCE
            if use_embedded_reward
            else _reward_source_for_metric(metric_name)
        ),
        "policy_logprobs_present": False,
        "training_input_mode": "message_only_requires_policy_rescore",
        "direct_offline_collect_training_supported": False,
        "identity_mode": "runtime_run_local_record_id",
        "count_input": len(records),
        "count_exported": len(scored),
        "count_unique_atif_ids": len(set(atif_ids)),
        "split_counts": dict(sorted(split_counts.items())),
        "atif_ids": atif_ids,
        "count_dropped": len(dropped),
        "trial_dirs": trial_dirs,
        "trial_artifacts": dict(sorted(trial_artifacts.items())),
        "dropped": dropped,
    }
    manifest = ATIFExportManifest.model_validate(manifest).model_dump(mode="json")
    write_json(out_dir / "manifest.json", manifest)
    tmp_path = out_dir / ".dropped.jsonl.tmp"
    tmp_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dropped),
        encoding="utf-8",
    )
    os.replace(tmp_path, out_dir / "dropped.jsonl")
    return manifest


__all__ = ["export_atif_from_run"]

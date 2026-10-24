from __future__ import annotations

import json
import logging
from typing import Any

from ppl_synth_data import __version__
from ppl_synth_data.dataset.features import sft_features, with_data_features
from ppl_synth_data.dataset.hf_upload import (
    build_datasetdict_from_jsonl_splits,
    push_datasetdict_to_hub,
)
from ppl_synth_data.dataset.split import deterministic_split
from ppl_synth_data.pipeline.io import (
    read_jsonl,
    write_json,
    write_jsonl,
    write_split_files,
)
from ppl_synth_data.pipeline.runtime_contracts import (
    ensure_upstream_stage_runtime,
)
from ppl_synth_data.pipeline.stages._envspecs import read_canonical_envspecs
from ppl_synth_data.pipeline.types import RunPaths, StageResult
from ppl_synth_data.pipeline.validation.semantic_evidence import (
    lazyppl_semantic_evidence_failure,
)
from ppl_synth_data.runtimes.base import TargetRuntime
from ppl_synth_data.specs.models import CompletionRecord
from ppl_synth_data.specs.record_contracts import normalize_completion_record

log = logging.getLogger(__name__)

# Return-type aliases for _materialize_record / _validate_record_fields:
# _DropRecord signals the record should be skipped (with reason metadata).
# _MaterializedPair is the (sft_row, with_data_row) success case.
_DropRecord = dict[str, Any]
_MaterializedPair = tuple[dict[str, Any], dict[str, Any]]
_TraceMap = dict[str, dict[str, Any]]


def _drop(rec: CompletionRecord, env_id: str, reason: str, **extra: Any) -> _DropRecord:
    base = {"id": rec.id, "prompt_id": rec.prompt_id, "reason": reason}
    if env_id:
        base["env_id"] = env_id
    base.update(extra)
    return base


def _record_key_concepts(rec: CompletionRecord, env_spec: Any) -> list[str]:
    values: Any
    if rec.provenance is not None and rec.provenance.key_concepts:
        values = rec.provenance.key_concepts
    else:
        values = getattr(env_spec, "key_concepts", [])
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]


def _record_has_observed_data(rec: CompletionRecord) -> bool:
    if rec.provenance is not None and rec.provenance.has_observed_data is not None:
        return bool(rec.provenance.has_observed_data)
    return True


def _record_contract_str(
    rec: CompletionRecord,
    provenance_attr: str,
    fallback: str,
) -> str:
    if rec.provenance is not None:
        value = getattr(rec.provenance, provenance_attr)
        if isinstance(value, str) and value:
            return value
    return fallback


def _validate_record_fields(
    rec: CompletionRecord,
    env_id: str,
    envspec_by_id: dict[str, Any],
    train_ratio: float,
    val_ratio: float,
) -> _DropRecord | tuple[str, Any, list[str], str, int]:
    """Validate and extract fields. Returns a drop dict on failure, or validated fields tuple."""
    split = str(
        rec.split or deterministic_split(env_id, train_ratio=train_ratio, val_ratio=val_ratio)
    )
    if split not in ("train", "validation", "test"):
        return _drop(rec, env_id, "invalid_split", split=split)

    env_spec = envspec_by_id.get(env_id)
    if env_spec is None:
        return _drop(rec, env_id, "unknown_env_id")

    key_concepts = _record_key_concepts(rec, env_spec)

    completion_text = rec.raw_completion or (rec.parsed_program or "")
    if not completion_text.strip():
        return _drop(rec, env_id, "empty_completion")

    seed_value = rec.metadata.get("seed", 0)
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        return _drop(rec, env_id, "invalid_seed", seed=seed_value)

    return split, env_spec, key_concepts, completion_text, seed


def _build_rows(
    rec: CompletionRecord,
    env_id: str,
    split: str,
    env_spec: Any,
    key_concepts: list[str],
    completion_text: str,
    seed: int,
    data_refs: dict[str, str],
    trace_records: _TraceMap,
    runtime: TargetRuntime,
) -> _MaterializedPair:
    structured_provenance = (
        rec.provenance.model_dump(mode="json") if rec.provenance is not None else None
    )
    validation_payload = (
        rec.validation.model_dump(mode="json") if rec.validation is not None else {}
    )
    provenance = {
        "id": rec.id,
        "prompt_id": rec.prompt_id,
        "provenance": structured_provenance,
        "metadata": rec.metadata,
    }
    version_key = runtime.metadata_version_key()
    version_val = str(
        (
            rec.provenance.runtime_version
            if rec.provenance is not None and rec.provenance.runtime_version
            else None
        )
        or runtime.runtime_version_tag()
    )
    trace_record = trace_records.get(rec.id, {})
    trace_id = str(trace_record.get("trace_id", "")) if trace_record else ""
    base_row = {
        "id": rec.id,
        "prompt_id": rec.prompt_id,
        "env_id": env_id,
        "split": split,
        "prompt": rec.prompt_text,
        "completion": completion_text,
        "model_type": _record_contract_str(
            rec,
            "model_type",
            str(env_spec.model_type),
        ),
        "domain": _record_contract_str(rec, "domain", str(env_spec.domain)),
        "complexity": _record_contract_str(
            rec,
            "complexity",
            str(env_spec.complexity.value),
        ),
        "key_concepts": key_concepts,
        "has_observed_data": _record_has_observed_data(rec),
        version_key: version_val,
        "trace_id": trace_id,
        "provenance_json": json.dumps(provenance, ensure_ascii=False),
        "candidate_outputs_json": json.dumps(rec.candidate_outputs, ensure_ascii=False),
    }
    with_data_row = {
        **base_row,
        "env_name": env_spec.env_name,
        "env_spec_json": env_spec.model_dump_json(),
        "data_ref": data_refs.get(env_id, ""),
        "parsed_program": rec.parsed_program or "",
        "seed": seed,
        "generator_version": __version__,
        "trace_json": json.dumps(trace_record, ensure_ascii=False),
        "validation_json": json.dumps(validation_payload, ensure_ascii=False),
    }
    return base_row, with_data_row


def _materialize_record(
    rec: CompletionRecord,
    envspec_by_id: dict[str, Any],
    data_refs: dict[str, str],
    trace_records: _TraceMap,
    runtime: TargetRuntime,
    train_ratio: float = 0.85,
    val_ratio: float = 0.075,
) -> _MaterializedPair | _DropRecord:
    env_id = rec.env_id
    if not env_id:
        return _drop(rec, env_id, "missing_env_id")
    if not data_refs.get(env_id):
        return _drop(rec, env_id, "missing_data_ref")

    record_runtime = (
        rec.provenance.runtime_name
        if rec.provenance is not None and rec.provenance.runtime_name
        else None
    )
    if record_runtime is None:
        return _drop(
            rec,
            env_id,
            "runtime_unknown",
            requested_runtime=runtime.name,
        )
    if record_runtime != runtime.name:
        return _drop(
            rec,
            env_id,
            "runtime_mismatch",
            record_runtime=record_runtime,
            requested_runtime=runtime.name,
        )
    if runtime.name == "lazyppl":
        stale_reason = lazyppl_semantic_evidence_failure(rec)
        if stale_reason is not None:
            return _drop(
                rec,
                env_id,
                "missing_lazyppl_semantic_evidence",
                detail=stale_reason,
            )

    result = _validate_record_fields(
        rec,
        env_id,
        envspec_by_id,
        train_ratio,
        val_ratio,
    )
    if isinstance(result, dict):
        return result

    split, env_spec, key_concepts, completion_text, seed = result
    return _build_rows(
        rec,
        env_id,
        split,
        env_spec,
        key_concepts,
        completion_text,
        seed,
        data_refs,
        trace_records,
        runtime,
    )


def _upload_datasets(
    sft_ds: Any,
    with_data_ds: Any,
    stage_dir: Any,
    *,
    repo_id: str,
    hf_token: str,
    private: bool,
    revision: str | None,
    commit_message: str | None,
    runtime: TargetRuntime,
) -> dict[str, Any]:
    sft_config = f"{runtime.name}_sft"
    with_data_config = f"{runtime.name}_with_data"
    upload_manifest: dict[str, Any] = {}
    upload_manifest[sft_config] = push_datasetdict_to_hub(
        sft_ds,
        repo_id=repo_id,
        token=hf_token,
        config_name=sft_config,
        private=private,
        revision=revision,
        commit_message=commit_message,
    )
    upload_manifest[with_data_config] = push_datasetdict_to_hub(
        with_data_ds,
        repo_id=repo_id,
        token=hf_token,
        config_name=with_data_config,
        private=private,
        revision=revision,
        commit_message=commit_message,
    )
    write_json(stage_dir / "hf_upload_manifest.json", upload_manifest)
    return upload_manifest


def _load_run_artifacts(
    paths: RunPaths,
) -> tuple[list[CompletionRecord], dict[str, Any], dict[str, str], _TraceMap]:
    validated = [
        CompletionRecord.model_validate(row)
        for row in read_jsonl(paths.stage_dirs["F"] / "validated.jsonl")
    ]
    envspecs = read_canonical_envspecs(paths)
    envspec_by_id: dict[str, Any] = {}
    for spec in envspecs:
        envspec_by_id[spec.env_id] = spec
    data_refs: dict[str, str] = {}
    data_refs_path = paths.stage_dirs["C"] / "data_refs.jsonl"
    if data_refs_path.exists():
        for row in read_jsonl(data_refs_path):
            eid = str(row["env_id"])
            data_refs[eid] = str(row["data_ref"])
    trace_records: _TraceMap = {}
    traces_path = paths.stage_dirs["F"] / "traces.jsonl"
    if traces_path.exists():
        for row in read_jsonl(traces_path):
            trial_id = row.get("trial_id")
            if isinstance(trial_id, str) and trial_id:
                trace_records[trial_id] = row
    return validated, envspec_by_id, data_refs, trace_records


def run_stage_g(
    paths: RunPaths,
    *,
    repo_id: str | None,
    hf_token: str | None,
    private: bool = True,
    revision: str | None = None,
    commit_message: str | None = None,
    skip_upload: bool = False,
    extra_run_dirs: list[RunPaths] | None = None,
    train_ratio: float = 0.85,
    val_ratio: float = 0.075,
    runtime: TargetRuntime | None = None,
) -> StageResult:
    if runtime is None:
        raise ValueError(
            "runtime is required for stage G. Pass --runtime pymc or --runtime lazyppl."
        )
    ensure_upstream_stage_runtime(paths, "F", runtime.name)

    stage = "G"
    stage_dir = paths.stage_dirs[stage]
    stage_dir.mkdir(parents=True, exist_ok=True)

    raw_validated, envspec_by_id, data_refs, trace_records = _load_run_artifacts(paths)
    validated = [normalize_completion_record(rec) for rec in raw_validated]
    if not validated:
        stage_f_manifest_path = paths.stage_dirs["F"] / "manifest.json"
        count_total = 0
        if stage_f_manifest_path.exists():
            manifest = json.loads(stage_f_manifest_path.read_text(encoding="utf-8"))
            raw_count_total = manifest.get("count_total")
            if isinstance(raw_count_total, int):
                count_total = raw_count_total
        if count_total > 0:
            raise RuntimeError(
                "stage G found no validated completions after Stage F processed "
                f"{count_total} completions"
            )

    seen_env_ids: set[str] = {rec.env_id for rec in validated if rec.env_id}
    for extra in extra_run_dirs or []:
        (
            extra_validated_raw,
            extra_specs,
            extra_data_refs,
            extra_trace_records,
        ) = _load_run_artifacts(extra)
        for spec_id, spec in extra_specs.items():
            if spec_id not in envspec_by_id:
                envspec_by_id[spec_id] = spec
        data_refs.update(extra_data_refs)
        trace_records.update(extra_trace_records)
        for rec in extra_validated_raw:
            rec = normalize_completion_record(rec)
            eid = rec.env_id
            if eid and eid not in seen_env_ids:
                validated.append(rec)
                seen_env_ids.add(eid)

    _splits = ("train", "validation", "test")
    sft_rows_by_split: dict[str, list[dict[str, Any]]] = {k: [] for k in _splits}
    with_data_rows_by_split: dict[str, list[dict[str, Any]]] = {k: [] for k in _splits}
    dropped: list[dict[str, Any]] = []

    for rec in validated:
        result = _materialize_record(
            rec,
            envspec_by_id,
            data_refs,
            trace_records,
            runtime,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        if isinstance(result, dict):
            dropped.append(result)
        else:
            sft_row, with_data_row = result
            split = sft_row["split"]
            sft_rows_by_split[split].append(sft_row)
            with_data_rows_by_split[split].append(with_data_row)

    write_jsonl(stage_dir / "dropped.jsonl", dropped)

    count_written = sum(len(rows) for rows in sft_rows_by_split.values())
    if validated and count_written == 0:
        raise RuntimeError(
            "stage G dropped every validated completion; "
            f"first dropped record: {dropped[0] if dropped else 'unknown'}"
        )

    sft_prefix = f"{runtime.name}_sft"
    with_data_prefix = f"{runtime.name}_with_data"

    sft_paths = write_split_files(stage_dir, sft_prefix, sft_rows_by_split)
    with_data_paths = write_split_files(stage_dir, with_data_prefix, with_data_rows_by_split)

    sft_ds = build_datasetdict_from_jsonl_splits(
        sft_paths["train"],
        sft_paths["validation"],
        sft_paths["test"],
        features=sft_features(runtime.name),
    )
    with_data_ds = build_datasetdict_from_jsonl_splits(
        with_data_paths["train"],
        with_data_paths["validation"],
        with_data_paths["test"],
        features=with_data_features(runtime.name),
    )

    if not skip_upload:
        if not repo_id:
            raise ValueError("repo_id is required when skip_upload is False")
        if not hf_token:
            raise ValueError("hf_token is required when skip_upload is False")
        _upload_datasets(
            sft_ds,
            with_data_ds,
            stage_dir,
            repo_id=repo_id,
            hf_token=hf_token,
            private=private,
            revision=revision,
            commit_message=commit_message,
            runtime=runtime,
        )

    manifest = {
        "status": "success",
        "status_detail": "success_with_drops" if dropped else "success",
        "resumable": not dropped and count_written > 0,
        "count_input_validated": len(validated),
        "count_written": count_written,
        "count_dropped": len(dropped),
        "count_rows_with_trace": sum(
            1 for rows in with_data_rows_by_split.values() for row in rows if row["trace_id"]
        ),
        "sft_counts": {k: len(v) for k, v in sft_rows_by_split.items()},
        "with_data_counts": {k: len(v) for k, v in with_data_rows_by_split.items()},
        "runtime": runtime.name,
        "uploaded": not skip_upload,
        "repo_id": repo_id,
    }
    write_json(stage_dir / "manifest.json", manifest)
    return StageResult(stage=stage, status="success", outputs=manifest)

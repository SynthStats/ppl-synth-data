from __future__ import annotations

from ppl_synth_data.pipeline.artifacts import stage_manifest_runtime
from ppl_synth_data.pipeline.types import RunPaths
from ppl_synth_data.specs.models import CompletionRecord, PromptRecord
from ppl_synth_data.specs.record_contracts import (
    normalize_completion_record,
    normalize_prompt_record,
)


def ensure_upstream_stage_runtime(
    paths: RunPaths,
    upstream_stage: str,
    expected_runtime: str,
) -> None:
    actual_runtime = stage_manifest_runtime(paths.stage_dirs[upstream_stage])
    if actual_runtime is None or actual_runtime == expected_runtime:
        return
    raise ValueError(
        f"stage {upstream_stage} artifacts were produced for runtime "
        f"{actual_runtime!r}, not {expected_runtime!r}"
    )


def _ensure_record_runtime(
    *,
    stage_name: str,
    record_kind: str,
    record_id: str,
    actual_runtime: str | None,
    expected_runtime: str,
) -> None:
    if actual_runtime is None:
        raise ValueError(
            f"stage {stage_name} {record_kind} {record_id!r} is missing runtime provenance"
        )
    if actual_runtime != expected_runtime:
        raise ValueError(
            f"stage {stage_name} {record_kind} {record_id!r} was produced for runtime "
            f"{actual_runtime!r}, not {expected_runtime!r}"
        )


def normalize_and_require_prompt_runtime(
    prompt: PromptRecord,
    *,
    expected_runtime: str,
    stage_name: str,
) -> PromptRecord:
    normalized = normalize_prompt_record(prompt)
    runtime_name = normalized.provenance.runtime_name if normalized.provenance is not None else None
    _ensure_record_runtime(
        stage_name=stage_name,
        record_kind="prompt",
        record_id=normalized.id,
        actual_runtime=runtime_name,
        expected_runtime=expected_runtime,
    )
    return normalized


def normalize_and_require_completion_runtime(
    record: CompletionRecord,
    *,
    expected_runtime: str,
    stage_name: str,
) -> CompletionRecord:
    normalized = normalize_completion_record(record)
    runtime_name = normalized.provenance.runtime_name if normalized.provenance is not None else None
    _ensure_record_runtime(
        stage_name=stage_name,
        record_kind="completion",
        record_id=normalized.id,
        actual_runtime=runtime_name,
        expected_runtime=expected_runtime,
    )
    return normalized

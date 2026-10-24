from __future__ import annotations

from typing import Any

from synthstats_schemas.trace import TraceRecord, TraceStep

from ppl_synth_data.evidence import EvidenceKind, SemanticStatus, TruthOwner
from ppl_synth_data.runtimes.base import TargetRuntime
from ppl_synth_data.specs.models import CompletionRecord
from ppl_synth_data.validation.pipeline import ValidationTraceEntry


def trace_step(
    *,
    attempt: int,
    phase: str,
    ok: bool,
    message: str,
    evidence_kind: EvidenceKind = EvidenceKind.verifier,
    truth_owner: TruthOwner = TruthOwner.verifier,
    semantic_status: SemanticStatus | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_semantic_status = (
        semantic_status
        if semantic_status is not None
        else (SemanticStatus.not_scored if ok else SemanticStatus.critical)
    )
    return {
        "attempt": attempt,
        "phase": phase,
        "ok": ok,
        "message": message,
        "evidence_kind": str(evidence_kind),
        "truth_owner": str(truth_owner),
        "semantic_status": str(resolved_semantic_status),
        "metadata": metadata or {},
    }


def trace_payload(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"steps": steps}


def trace_entry_payload(entry: ValidationTraceEntry) -> dict[str, Any]:
    return trace_step(
        attempt=entry.attempt,
        phase=entry.phase,
        ok=entry.ok,
        message=entry.message,
        evidence_kind=entry.evidence_kind,
        truth_owner=entry.truth_owner,
        semantic_status=entry.semantic_status,
        metadata=dict(entry.metadata),
    )


def validation_trace_record(
    rec: CompletionRecord,
    *,
    trace_steps: list[dict[str, Any]],
    runtime: TargetRuntime,
    outcome: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build the stage-F trace artifact as a real contracts TraceRecord.

    synthstats_schemas.trace.TraceStep has no first-class evidence_kind /
    truth_owner / semantic_status fields (unlike the now-removed
    producer-owned shadow TraceStep it replaced), so the taxonomy is folded
    into `metadata` under the same key names. The return value is the model
    dumped to a plain JSON-safe dict (this is what gets embedded as
    `trace_json` on the dataset row and later re-parsed by
    schemas.adapters.trace_record_from_dataset_row via TraceRecord.model_validate).
    """
    steps: list[TraceStep] = []
    for index, step in enumerate(trace_steps):
        raw_metadata = step.get("metadata")
        step_metadata: dict[str, Any] = {
            "attempt": step.get("attempt"),
            "phase": step.get("phase"),
            "ok": step.get("ok"),
            **(dict(raw_metadata) if isinstance(raw_metadata, dict) else {}),
        }
        for field_name in ("evidence_kind", "truth_owner", "semantic_status"):
            field_value = step.get(field_name)
            if field_value is not None:
                step_metadata[field_name] = str(field_value)
        steps.append(
            TraceStep(
                step_id=f"step-{index:04d}",
                role="validator",
                content=str(step.get("message") or ""),
                metadata=step_metadata,
            )
        )

    artifacts: dict[str, Any] = {
        "outcome": outcome,
        "prompt_id": rec.prompt_id,
        "split": rec.split or None,
    }
    if reason is not None:
        artifacts["reason"] = reason

    record = TraceRecord(
        trace_id=f"{rec.id}:validation",
        task_id=rec.env_id or None,
        trial_id=rec.id,
        steps=steps,
        artifacts=artifacts,
        metadata={
            "stage": "F",
            "runtime": runtime.name,
            "env_id": rec.env_id or None,
            "split": rec.split or None,
        },
    )
    return record.model_dump(mode="json")

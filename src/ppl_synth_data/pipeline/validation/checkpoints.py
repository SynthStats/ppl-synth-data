from __future__ import annotations

from pathlib import Path
from typing import Any

from ppl_synth_data.pipeline.io import read_jsonl, write_json, write_jsonl


def stage_f_artifact_paths(stage_dir: Path) -> dict[str, Path]:
    return {
        "validated": stage_dir / "validated.jsonl",
        "rejected": stage_dir / "rejected.jsonl",
        "traces": stage_dir / "traces.jsonl",
        "progress": stage_dir / "progress.json",
    }


def load_stage_f_checkpoint(
    stage_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paths = stage_f_artifact_paths(stage_dir)
    validated = read_jsonl(paths["validated"]) if paths["validated"].exists() else []
    rejected = read_jsonl(paths["rejected"]) if paths["rejected"].exists() else []
    traces = read_jsonl(paths["traces"]) if paths["traces"].exists() else []
    if len(traces) != len(validated) + len(rejected):
        raise RuntimeError(
            "Stage F checkpoint is inconsistent: traces.jsonl must contain "
            "one trace per validated/rejected completion"
        )
    return validated, rejected, traces


def filter_stage_f_checkpoint_to_current_inputs(
    *,
    validated: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    current_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    current_validated = [row for row in validated if str(row.get("id")) in current_ids]
    current_rejected = [row for row in rejected if str(row.get("id")) in current_ids]
    current_traces = [row for row in traces if str(row.get("trial_id")) in current_ids]
    if len(current_traces) != len(current_validated) + len(current_rejected):
        raise RuntimeError(
            "Stage F checkpoint is inconsistent with current Stage E inputs: "
            "traces.jsonl must contain one current trace per current "
            "validated/rejected completion"
        )
    return current_validated, current_rejected, current_traces


def write_stage_f_checkpoint(
    stage_dir: Path,
    *,
    validated: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    total: int,
    runtime_name: str,
    status: str,
    semantic_validation: dict[str, Any],
    current_record_id: str | None = None,
) -> None:
    paths = stage_f_artifact_paths(stage_dir)
    write_jsonl(paths["validated"], validated)
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["traces"], traces)
    processed = len(validated) + len(rejected)
    write_json(
        paths["progress"],
        {
            "status": status,
            "count_total": total,
            "count_processed": processed,
            "count_validated": len(validated),
            "count_rejected": len(rejected),
            "count_traces": len(traces),
            "current_record_id": current_record_id,
            "runtime": runtime_name,
            "semantic_validation": semantic_validation,
        },
    )

from __future__ import annotations

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import LLMBackend
from ppl_synth_data.pipeline.backends import (
    build_backend,
    resolve_backend_name,
    resolve_judge_backend_name,
    resolve_stage_model,
)
from ppl_synth_data.pipeline.config_defaults import load_default_pipeline_defaults
from ppl_synth_data.pipeline.io import read_jsonl, write_json
from ppl_synth_data.pipeline.runtime_contracts import (
    ensure_upstream_stage_runtime,
    normalize_and_require_completion_runtime,
)
from ppl_synth_data.pipeline.types import RunPaths, StageResult
from ppl_synth_data.pipeline.validation.checkpoints import (
    filter_stage_f_checkpoint_to_current_inputs,
    load_stage_f_checkpoint,
    write_stage_f_checkpoint,
)
from ppl_synth_data.pipeline.validation.leakage_gate import (
    DETERMINISTIC_JUDGE_BACKENDS,
)
from ppl_synth_data.pipeline.validation.record_validation import (
    DEFAULT_STAGE_F_SEMANTIC_SETTINGS,
    StageFSemanticSettings,
    validate_record,
)
from ppl_synth_data.pipeline.validation.traces import validation_trace_record
from ppl_synth_data.runtimes.base import TargetRuntime
from ppl_synth_data.specs.models import CompletionRecord


def run_stage_f(
    paths: RunPaths,
    *,
    backend_name: str | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    judge_backend_name: str | None = None,
    max_retries: int = 3,
    backend: LLMBackend | None = None,
    runtime: TargetRuntime | None = None,
    settings: Settings | None = None,
    pymc_semantic_draws: int = DEFAULT_STAGE_F_SEMANTIC_SETTINGS.pymc_draws,
    pymc_semantic_tune: int = DEFAULT_STAGE_F_SEMANTIC_SETTINGS.pymc_tune,
    pymc_semantic_timeout_sec: float = (DEFAULT_STAGE_F_SEMANTIC_SETTINGS.pymc_timeout_sec),
    acceptance_min_pymc_semantic_draws: int | None = None,
    acceptance_min_pymc_semantic_tune: int | None = None,
) -> StageResult:
    if runtime is None:
        raise ValueError(
            "runtime is required for stage F. Pass --runtime pymc or --runtime lazyppl."
        )
    semantic_settings = StageFSemanticSettings(
        pymc_draws=pymc_semantic_draws,
        pymc_tune=pymc_semantic_tune,
        pymc_timeout_sec=pymc_semantic_timeout_sec,
    )
    acceptance_requirements = {
        "min_pymc_draws": acceptance_min_pymc_semantic_draws,
        "min_pymc_tune": acceptance_min_pymc_semantic_tune,
    }
    _require_acceptance_semantic_budget(
        runtime=runtime,
        semantic_settings=semantic_settings,
        min_draws=acceptance_min_pymc_semantic_draws,
        min_tune=acceptance_min_pymc_semantic_tune,
    )
    semantic_validation = {
        "settings": semantic_settings.as_dict(),
        "acceptance_requirements": acceptance_requirements,
    }
    ensure_upstream_stage_runtime(paths, "E", runtime.name)

    stage = "F"
    stage_dir = paths.stage_dirs[stage]
    stage_dir.mkdir(parents=True, exist_ok=True)

    defaults = load_default_pipeline_defaults()
    resolved_backend_name = resolve_backend_name(backend_name, defaults=defaults)
    resolved_model = resolve_stage_model("repair", model, defaults=defaults)
    resolved_judge_model = resolve_stage_model("judge", judge_model, defaults=defaults)
    resolved_judge_backend = resolve_judge_backend_name(
        judge_backend_name,
        defaults=defaults,
    )
    use_deterministic_judge = resolved_judge_backend.strip().lower() in DETERMINISTIC_JUDGE_BACKENDS
    input_records = [
        normalize_and_require_completion_runtime(
            CompletionRecord.model_validate(row),
            expected_runtime=runtime.name,
            stage_name="F",
        )
        for row in read_jsonl(paths.stage_dirs["E"] / "completions.jsonl")
    ]
    data_dir = paths.stage_dirs["C"] / "data"

    validation_llm = backend
    judge_llm: LLMBackend | None = None
    if input_records and not use_deterministic_judge:
        judge_llm = build_backend(resolved_judge_backend, settings=settings)

    def get_validation_llm() -> LLMBackend:
        nonlocal validation_llm
        if validation_llm is None:
            validation_llm = build_backend(resolved_backend_name, settings=settings)
        if validation_llm is None:
            raise RuntimeError("validation backend was not initialized")
        return validation_llm

    validated, rejected, traces = load_stage_f_checkpoint(stage_dir)
    current_ids = {rec.id for rec in input_records}
    validated, rejected, traces = filter_stage_f_checkpoint_to_current_inputs(
        validated=validated,
        rejected=rejected,
        traces=traces,
        current_ids=current_ids,
    )
    processed_ids = {str(row["id"]) for row in (*validated, *rejected) if row.get("id") is not None}
    total = len(input_records)
    write_stage_f_checkpoint(
        stage_dir,
        validated=validated,
        rejected=rejected,
        traces=traces,
        total=total,
        runtime_name=runtime.name,
        status="running",
        semantic_validation=semantic_validation,
    )
    for rec in input_records:
        if rec.id in processed_ids:
            continue
        rejection = validate_record(
            rec,
            judge_llm=judge_llm,
            judge_model=resolved_judge_model,
            use_deterministic_judge=use_deterministic_judge,
            get_llm=get_validation_llm,
            repair_model=resolved_model,
            max_retries=max_retries,
            data_dir=data_dir,
            runtime=runtime,
            semantic_settings=semantic_settings,
        )
        if rejection is not None:
            rejected.append(rejection)
            trace_steps = list(rejection.get("validation_trace", {}).get("steps", []))
            traces.append(
                validation_trace_record(
                    rec,
                    trace_steps=trace_steps,
                    runtime=runtime,
                    outcome="rejected",
                    reason=str(rejection["reason"]),
                )
            )
        else:
            validated_row = rec.model_dump(mode="json")
            validated.append(validated_row)
            trace_steps = list(
                validated_row.get("validation", {}).get("trace", {}).get("steps", [])
            )
            traces.append(
                validation_trace_record(
                    rec,
                    trace_steps=trace_steps,
                    runtime=runtime,
                    outcome="validated",
                )
            )
        processed_ids.add(rec.id)
        write_stage_f_checkpoint(
            stage_dir,
            validated=validated,
            rejected=rejected,
            traces=traces,
            total=total,
            runtime_name=runtime.name,
            status="running",
            semantic_validation=semantic_validation,
            current_record_id=rec.id,
        )

    pass_rate = (len(validated) / total) if total else 0.0
    manifest = {
        "status": "success",
        "count_total": total,
        "count_validated": len(validated),
        "count_rejected": len(rejected),
        "count_traces": len(traces),
        "validation_pass_rate": pass_rate,
        "judge_backend": resolved_judge_backend,
        "judge_model": resolved_judge_model,
        "leakage_judge": ("deterministic_banned_terms" if use_deterministic_judge else "llm_judge"),
        "runtime": runtime.name,
        "semantic_validation": semantic_validation,
    }
    write_stage_f_checkpoint(
        stage_dir,
        validated=validated,
        rejected=rejected,
        traces=traces,
        total=total,
        runtime_name=runtime.name,
        status="success",
        semantic_validation=semantic_validation,
    )
    write_json(stage_dir / "manifest.json", manifest)
    return StageResult(stage=stage, status="success", outputs=manifest)


def _require_acceptance_semantic_budget(
    *,
    runtime: TargetRuntime,
    semantic_settings: StageFSemanticSettings,
    min_draws: int | None,
    min_tune: int | None,
) -> None:
    if runtime.name != "pymc":
        return
    if min_draws is not None and semantic_settings.pymc_draws < min_draws:
        raise ValueError(
            "Stage F PyMC acceptance mode requires at least "
            f"{min_draws} semantic draws; got {semantic_settings.pymc_draws}"
        )
    if min_tune is not None and semantic_settings.pymc_tune < min_tune:
        raise ValueError(
            "Stage F PyMC acceptance mode requires at least "
            f"{min_tune} semantic tune steps; got {semantic_settings.pymc_tune}"
        )

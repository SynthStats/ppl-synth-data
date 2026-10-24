from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ppl_synth_data.evidence import EvidenceKind, SemanticStatus, TruthOwner
from ppl_synth_data.llm.backend import LLMBackend
from ppl_synth_data.pipeline.validation.data_loading import try_load_sample_data
from ppl_synth_data.pipeline.validation.leakage_gate import (
    completion_generative_family,
    judge_leakage,
    judge_leakage_deterministic,
    prompt_text_for_leakage_scan,
)
from ppl_synth_data.pipeline.validation.semantic_evidence import (
    LAZYPPL_SEMANTIC_EVIDENCE_VERSION,
)
from ppl_synth_data.pipeline.validation.traces import (
    trace_entry_payload,
    trace_payload,
    trace_step,
)
from ppl_synth_data.runtimes.base import TargetRuntime
from ppl_synth_data.specs.models import CompletionRecord, ValidationInfo
from ppl_synth_data.validation.lazyppl_semantics import score_lazyppl_semantics
from ppl_synth_data.validation.pipeline import (
    coerce_validation_run_result,
    validate_and_repair,
)
from ppl_synth_data.validation.pymc_semantics import score_pymc_semantics

PYMC_STAGE_F_SEMANTIC_DRAWS = 100
PYMC_STAGE_F_SEMANTIC_TUNE = 100
PYMC_STAGE_F_SEMANTIC_TIMEOUT_SEC = 180


@dataclass(frozen=True)
class StageFSemanticSettings:
    pymc_draws: int = PYMC_STAGE_F_SEMANTIC_DRAWS
    pymc_tune: int = PYMC_STAGE_F_SEMANTIC_TUNE
    pymc_timeout_sec: float = PYMC_STAGE_F_SEMANTIC_TIMEOUT_SEC

    def __post_init__(self) -> None:
        if self.pymc_draws <= 0:
            raise ValueError("pymc semantic draws must be positive")
        if self.pymc_tune <= 0:
            raise ValueError("pymc semantic tune must be positive")
        if self.pymc_timeout_sec <= 0:
            raise ValueError("pymc semantic timeout must be positive")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "pymc_draws": self.pymc_draws,
            "pymc_tune": self.pymc_tune,
            "pymc_timeout_sec": self.pymc_timeout_sec,
        }


DEFAULT_STAGE_F_SEMANTIC_SETTINGS = StageFSemanticSettings()


def reject_completion(rec: CompletionRecord, reason: str, **extra: Any) -> dict[str, Any]:
    evidence_kind = str(extra.pop("evidence_kind", EvidenceKind.verifier))
    truth_owner = str(extra.pop("truth_owner", TruthOwner.verifier))
    semantic_status = str(extra.pop("semantic_status", SemanticStatus.critical))
    return {
        "id": rec.id,
        "prompt_id": rec.prompt_id,
        "reason": reason,
        "evidence_kind": evidence_kind,
        "truth_owner": truth_owner,
        "semantic_status": semantic_status,
        **extra,
    }


def validate_record(
    rec: CompletionRecord,
    *,
    judge_llm: LLMBackend | None,
    judge_model: str,
    use_deterministic_judge: bool,
    get_llm: Callable[[], LLMBackend],
    repair_model: str,
    max_retries: int,
    data_dir: Path,
    runtime: TargetRuntime,
    semantic_settings: StageFSemanticSettings,
) -> dict[str, Any] | None:
    """Return a rejection payload, or None on success. Mutates the completion."""
    actual_family = completion_generative_family(rec)
    leakage_prompt_text, masked_leakage_markers = prompt_text_for_leakage_scan(rec.prompt_text)
    if use_deterministic_judge:
        leakage = judge_leakage_deterministic(leakage_prompt_text, actual_family)
    else:
        if judge_llm is None:
            raise RuntimeError("LLM leakage judge backend was not initialized")
        leakage = judge_leakage(
            leakage_prompt_text,
            judge_llm,
            judge_model,
            actual_family,
        )
    trace_steps = [
        trace_step(
            attempt=0,
            phase="leakage_judge",
            ok=not leakage["leaked"],
            message=leakage["rating"],
            evidence_kind=EvidenceKind.heuristic,
            truth_owner=TruthOwner.heuristic,
            semantic_status=(
                SemanticStatus.pass_ if not leakage["leaked"] else SemanticStatus.critical
            ),
            metadata={
                "evidence": leakage["evidence"],
                "recovered_family": leakage["recovered_family"],
                "method": leakage.get("method", "llm_judge"),
                "scan_text_char_count": len(leakage_prompt_text),
                "masked_runtime_markers": list(masked_leakage_markers),
            },
        )
    ]
    if leakage["leaked"]:
        return reject_completion(
            rec,
            "prompt_leakage",
            evidence_kind=EvidenceKind.heuristic,
            truth_owner=TruthOwner.heuristic,
            semantic_status=SemanticStatus.critical,
            leakage_detail=leakage,
            validation_trace=trace_payload(trace_steps),
        )

    fmt = runtime.validate_completion_format(rec.raw_completion)
    trace_steps.append(
        trace_step(
            attempt=0,
            phase="completion_format",
            ok=fmt.ok,
            message="ok" if fmt.ok else "completion format validation failed",
            evidence_kind=EvidenceKind.verifier,
            truth_owner=TruthOwner.verifier,
            metadata={"issues": list(fmt.issues)},
        )
    )
    if not fmt.ok:
        return reject_completion(
            rec,
            "completion_format",
            issues=fmt.issues,
            validation_trace=trace_payload(trace_steps),
        )

    original_code = runtime.extract_code(rec.raw_completion)
    trace_steps.append(
        trace_step(
            attempt=0,
            phase="code_extraction",
            ok=bool(original_code),
            message="code block extracted" if original_code else "missing code block",
            evidence_kind=EvidenceKind.verifier,
            truth_owner=TruthOwner.verifier,
        )
    )
    if not original_code:
        return reject_completion(
            rec,
            "missing_code_block",
            validation_trace=trace_payload(trace_steps),
        )

    env_id = rec.env_id or None
    sample_data, sample_load_error = try_load_sample_data(env_id, data_dir)
    hidden_data, hidden_load_error = try_load_sample_data(
        env_id,
        data_dir,
        hidden=True,
    )
    if sample_data is None:
        message = sample_load_error or "missing Stage C visible data"
        trace_steps.append(
            trace_step(
                attempt=0,
                phase="stage_c_data",
                ok=False,
                message=message,
                evidence_kind=EvidenceKind.verifier,
                truth_owner=TruthOwner.verifier,
                semantic_status=SemanticStatus.critical,
                metadata={"env_id": env_id},
            )
        )
        return reject_completion(
            rec,
            "missing_stage_c_data",
            message=message,
            sample_data_error=message,
            validation_trace=trace_payload(trace_steps),
        )

    validation_result = coerce_validation_run_result(
        validate_and_repair(
            original_code,
            rec.prompt_text,
            get_llm(),
            model=repair_model,
            max_retries=max_retries,
            sample_data=sample_data,
            runtime=runtime,
        )
    )
    trace_steps.extend(trace_entry_payload(entry) for entry in validation_result.trace)
    if not validation_result.ok:
        return reject_completion(
            rec,
            "runtime_or_static_validation",
            message=validation_result.message,
            validation_trace=trace_payload(trace_steps),
        )

    repaired_code = validation_result.final_code
    msg = validation_result.message
    rec.parsed_program = repaired_code
    rec.raw_completion = runtime.replace_code(rec.raw_completion, repaired_code)
    semantic_status = SemanticStatus.not_scored
    semantic_metric: str | None = None
    semantic_score: float | None = None
    semantic_details: dict[str, Any] = {}
    if runtime.name == "pymc" and sample_data is not None:
        semantic_result = score_pymc_semantics(
            repaired_code,
            visible_data=sample_data,
            hidden_data=hidden_data,
            draws=semantic_settings.pymc_draws,
            tune=semantic_settings.pymc_tune,
            timeout_sec=int(semantic_settings.pymc_timeout_sec),
        )
        semantic_status = semantic_result.status
        semantic_metric = semantic_result.metric_name
        semantic_score = semantic_result.metric_value
        semantic_details = dict(semantic_result.details)
        trace_steps.append(
            trace_step(
                attempt=max_retries + 1,
                phase="semantic_validation",
                ok=semantic_result.ok,
                message=semantic_result.message,
                evidence_kind=EvidenceKind.verifier,
                truth_owner=TruthOwner.verifier,
                semantic_status=semantic_result.status,
                metadata={
                    "status": semantic_result.status,
                    "metric_name": semantic_metric,
                    "metric_value": semantic_score,
                    "semantic_settings": semantic_settings.as_dict(),
                    "details": semantic_details,
                },
            )
        )
        if semantic_result.status == SemanticStatus.critical:
            return reject_completion(
                rec,
                "semantic_validation",
                message=semantic_result.message,
                evidence_kind=EvidenceKind.verifier,
                truth_owner=TruthOwner.verifier,
                semantic_status=semantic_result.status,
                semantic_metric=semantic_metric,
                semantic_score=semantic_score,
                semantic_details=semantic_details,
                parsed_program=repaired_code,
                validation_trace=trace_payload(trace_steps),
            )
    elif runtime.name == "lazyppl":
        semantic_result = score_lazyppl_semantics(
            repaired_code,
            prompt=rec.prompt_text,
            sample_data=sample_data,
            hidden_data=hidden_data,
        )
        semantic_status = semantic_result.status
        semantic_metric = semantic_result.metric_name
        semantic_score = semantic_result.metric_value
        semantic_details = {
            **dict(semantic_result.details),
            "semantic_evidence_version": LAZYPPL_SEMANTIC_EVIDENCE_VERSION,
        }
        trace_steps.append(
            trace_step(
                attempt=max_retries + 1,
                phase="semantic_validation",
                ok=semantic_result.ok,
                message=semantic_result.message,
                evidence_kind=EvidenceKind.verifier,
                truth_owner=TruthOwner.verifier,
                semantic_status=semantic_result.status,
                metadata={
                    "status": semantic_result.status,
                    "metric_name": semantic_metric,
                    "metric_value": semantic_score,
                    "details": semantic_details,
                },
            )
        )
        if semantic_result.status == SemanticStatus.critical:
            return reject_completion(
                rec,
                "semantic_validation",
                message=semantic_result.message,
                evidence_kind=EvidenceKind.verifier,
                truth_owner=TruthOwner.verifier,
                semantic_status=semantic_result.status,
                semantic_metric=semantic_metric,
                semantic_score=semantic_score,
                semantic_details=semantic_details,
                parsed_program=repaired_code,
                validation_trace=trace_payload(trace_steps),
            )

    validation_info: dict[str, Any] = {
        "ok": True,
        "message": msg,
        "repaired": repaired_code != original_code,
        "leakage_rating": leakage["rating"],
        "evidence_kind": EvidenceKind.verifier,
        "truth_owner": TruthOwner.verifier,
        "semantic_status": semantic_status,
        "semantic_metric": semantic_metric,
        "semantic_score": semantic_score,
        "semantic_settings": semantic_settings.as_dict(),
        "semantic_details": semantic_details,
        "trace": trace_payload(trace_steps),
    }
    if sample_load_error is not None:
        validation_info["sample_data_error"] = sample_load_error
    if hidden_load_error is not None:
        validation_info["semantic_details"] = {
            **semantic_details,
            "hidden_data_error": hidden_load_error,
        }
    rec.validation = ValidationInfo.model_validate(validation_info)
    return None

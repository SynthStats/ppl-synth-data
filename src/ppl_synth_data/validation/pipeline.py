from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ppl_synth_data.evidence import EvidenceKind, SemanticStatus, TruthOwner
from ppl_synth_data.llm.backend import LLMBackend, LLMRequest
from ppl_synth_data.runtimes.base import TargetRuntime

log = logging.getLogger(__name__)
_FENCED_CODE_RE = re.compile(
    r"^\s*```(?:python)?\s*\n(?P<code>.*?)\n?```(?:\s*)$",
    re.IGNORECASE | re.DOTALL,
)
_SUBMIT_PROGRAM_RE = re.compile(
    r"^\s*<submit_program(?:\s+[^>]*)?>\s*(?P<code>.*?)\s*</submit_program>\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ValidationTraceEntry:
    attempt: int
    phase: str
    ok: bool
    message: str
    evidence_kind: EvidenceKind = EvidenceKind.verifier
    truth_owner: TruthOwner = TruthOwner.verifier
    semantic_status: SemanticStatus = SemanticStatus.not_scored
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationRunResult:
    ok: bool
    final_code: str
    message: str
    trace: tuple[ValidationTraceEntry, ...] = ()

    def __iter__(self) -> Iterator[bool | str]:
        yield self.ok
        yield self.final_code
        yield self.message


def coerce_validation_run_result(
    result: ValidationRunResult | tuple[bool, str, str],
) -> ValidationRunResult:
    if isinstance(result, ValidationRunResult):
        return result
    ok, final_code, message = result
    return ValidationRunResult(
        ok=ok,
        final_code=final_code,
        message=message,
    )


def _is_infra_failure(message: str) -> bool:
    return message.startswith("INFRA:")


def validate_and_repair(
    code: str,
    prompt: str,
    backend: LLMBackend,
    model: str,
    max_retries: int = 3,
    sample_data: dict[str, Any] | None = None,
    *,
    runtime: TargetRuntime,
) -> ValidationRunResult:
    """Returns success/final code/message plus attempt-level trace data."""
    current = code
    trace: list[ValidationTraceEntry] = []
    for attempt in range(1 + max_retries):
        syntax_result = runtime.validate_static(current, sample_data=sample_data)
        trace.append(
            ValidationTraceEntry(
                attempt=attempt,
                phase="syntax",
                ok=syntax_result.ok,
                message=syntax_result.message,
                semantic_status=(
                    SemanticStatus.not_scored if syntax_result.ok else SemanticStatus.critical
                ),
            )
        )
        if not syntax_result.ok:
            log.debug("attempt %d: syntax check failed: %s", attempt, syntax_result.message)
            if _is_infra_failure(syntax_result.message):
                return ValidationRunResult(
                    ok=False,
                    final_code=current,
                    message=f"syntax check failed: {syntax_result.message}",
                    trace=tuple(trace),
                )
            if attempt < max_retries:
                repaired = _call_repair_with_runtime(
                    backend,
                    model,
                    prompt,
                    current,
                    syntax_result.message,
                    runtime,
                )
                trace.append(
                    ValidationTraceEntry(
                        attempt=attempt,
                        phase="repair",
                        ok=True,
                        message="repair requested after syntax failure",
                        semantic_status=SemanticStatus.not_scored,
                        metadata={
                            "trigger": "syntax",
                            "code_changed": repaired != current,
                        },
                    )
                )
                current = repaired
                continue
            return ValidationRunResult(
                ok=False,
                final_code=current,
                message=f"syntax check failed: {syntax_result.message}",
                trace=tuple(trace),
            )

        runtime_result = runtime.validate_runtime(current, sample_data=sample_data)
        trace.append(
            ValidationTraceEntry(
                attempt=attempt,
                phase="runtime",
                ok=runtime_result.ok,
                message=runtime_result.message,
                semantic_status=(
                    SemanticStatus.not_scored if runtime_result.ok else SemanticStatus.critical
                ),
            )
        )
        if runtime_result.ok:
            return ValidationRunResult(
                ok=True,
                final_code=current,
                message="ok",
                trace=tuple(trace),
            )

        log.debug("attempt %d: runtime check failed: %s", attempt, runtime_result.message)
        if _is_infra_failure(runtime_result.message):
            return ValidationRunResult(
                ok=False,
                final_code=current,
                message=f"runtime check failed: {runtime_result.message}",
                trace=tuple(trace),
            )
        if attempt < max_retries:
            repaired = _call_repair_with_runtime(
                backend,
                model,
                prompt,
                current,
                runtime_result.message,
                runtime,
            )
            trace.append(
                ValidationTraceEntry(
                    attempt=attempt,
                    phase="repair",
                    ok=True,
                    message="repair requested after runtime failure",
                    semantic_status=SemanticStatus.not_scored,
                    metadata={
                        "trigger": "runtime",
                        "code_changed": repaired != current,
                    },
                )
            )
            current = repaired
            continue
        return ValidationRunResult(
            ok=False,
            final_code=current,
            message=f"runtime check failed: {runtime_result.message}",
            trace=tuple(trace),
        )
    return ValidationRunResult(
        ok=False,
        final_code=current,
        message="exhausted retries",
        trace=tuple(trace),
    )


def _call_repair_with_runtime(
    backend: LLMBackend,
    model: str,
    prompt: str,
    broken_code: str,
    error_trace: str,
    runtime: TargetRuntime,
) -> str:
    log.info("requesting LLM repair for %s code failure", runtime.name)
    system, user = runtime.repair_system_prompt(prompt, broken_code, error_trace)
    request = LLMRequest(model=model, system=system, user=user)
    result = backend.complete(request)
    return _normalize_repair_code(result.content)


def _normalize_repair_code(repaired: str) -> str:
    stripped = repaired.strip()
    submit_match = _SUBMIT_PROGRAM_RE.match(stripped)
    if submit_match:
        stripped = submit_match.group("code").strip()

    match = _FENCED_CODE_RE.match(stripped)
    if not match:
        return stripped
    return match.group("code").strip()

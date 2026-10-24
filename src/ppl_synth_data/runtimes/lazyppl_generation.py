from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ppl_synth_data.llm.backend import LLMBackend, LLMRequest
from ppl_synth_data.llm.completion_markup import extract_think_text, strip_think_block
from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.pipeline.io import atomic_write_text
from ppl_synth_data.runtimes import get_runtime
from ppl_synth_data.runtimes.base import TargetRuntime
from ppl_synth_data.validation.pipeline import (
    coerce_validation_run_result,
    validate_and_repair,
)


@dataclass(frozen=True)
class LazyPPLGenerationResult:
    code: str | None
    completion: str
    think: str | None
    repaired: bool
    typecheck_ok: bool
    error: str | None


def generate_lazyppl_completion(
    prompt_text: str,
    backend: LLMBackend,
    model: str,
    *,
    max_repairs: int = 3,
    temperature: float = 0.4,
    strip_thinking: bool = True,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    runtime: TargetRuntime | None = None,
) -> LazyPPLGenerationResult:
    rt = runtime or get_runtime("lazyppl")
    system_prompt = load_prompt_text(rt.completion_system_prompt_name())
    req = LLMRequest(
        model=model,
        system=system_prompt,
        user=prompt_text,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    result = backend.complete(req)
    raw_completion = result.content

    code = rt.extract_code(raw_completion)
    if code is None:
        return LazyPPLGenerationResult(
            code=None,
            completion=raw_completion,
            think=None,
            repaired=False,
            typecheck_ok=False,
            error="no code block extracted",
        )

    think_text = extract_think_text(raw_completion)
    validation_result = coerce_validation_run_result(
        validate_and_repair(
            code,
            prompt_text,
            backend,
            model,
            max_retries=max_repairs,
            runtime=rt,
        )
    )
    ok = validation_result.ok
    final_code = validation_result.final_code
    msg = validation_result.message

    repaired = final_code != code
    if repaired:
        raw_completion = rt.replace_code(raw_completion, final_code)

    completion_out = raw_completion
    if strip_thinking:
        completion_out = strip_think_block(completion_out)

    return LazyPPLGenerationResult(
        code=final_code,
        completion=completion_out,
        think=think_text,
        repaired=repaired,
        typecheck_ok=ok,
        error=None if ok else msg,
    )


def build_lazyppl_record(
    *,
    prompt: str,
    complexity: str,
    metadata: Mapping[str, Any] | None,
    result: LazyPPLGenerationResult,
) -> dict[str, Any]:
    merged_metadata = dict(metadata or {})
    merged_metadata["typecheck_ok"] = result.typecheck_ok
    merged_metadata["repaired"] = result.repaired
    if result.think:
        merged_metadata["think"] = result.think
    if result.error:
        merged_metadata["error"] = result.error

    return {
        "prompt": prompt,
        "completion": result.completion,
        "metadata": merged_metadata,
        "complexity": complexity,
    }


def build_lazyppl_api_error_record(
    *,
    prompt: str,
    complexity: str,
    metadata: Mapping[str, Any] | None,
    error_message: str,
) -> dict[str, Any]:
    return build_lazyppl_record(
        prompt=prompt,
        complexity=complexity,
        metadata=metadata,
        result=LazyPPLGenerationResult(
            code=None,
            completion="",
            think=None,
            repaired=False,
            typecheck_ok=False,
            error=error_message,
        ),
    )


def write_jsonl_records(path: Path, records: list[Mapping[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))

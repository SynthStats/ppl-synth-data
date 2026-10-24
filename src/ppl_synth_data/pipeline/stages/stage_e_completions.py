from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import LLMBackend, LLMRequest, LLMResult
from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.llm.response_utils import extract_candidate_texts
from ppl_synth_data.pipeline.backends import (
    build_backend,
    resolve_backend_name,
    resolve_stage_model,
)
from ppl_synth_data.pipeline.concurrency import execute_requests_raw
from ppl_synth_data.pipeline.config_defaults import (
    effective_reasoning_effort,
    load_default_pipeline_defaults,
)
from ppl_synth_data.pipeline.io import read_jsonl, write_json, write_jsonl
from ppl_synth_data.pipeline.runtime_contracts import (
    ensure_upstream_stage_runtime,
    normalize_and_require_prompt_runtime,
)
from ppl_synth_data.pipeline.types import RunPaths, StageResult
from ppl_synth_data.runtimes.base import TargetRuntime
from ppl_synth_data.specs.io import read_envspecs
from ppl_synth_data.specs.models import CompletionRecord, EnvSpec, FourFamilySpec, PromptRecord

log = logging.getLogger(__name__)


_PROGRESS_FILE = "progress.json"


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provider_response_id(raw: dict[str, Any]) -> str | None:
    response_id = raw.get("id")
    return response_id if isinstance(response_id, str) and response_id else None


def _provider_usage(raw: dict[str, Any]) -> dict[str, Any] | None:
    usage = raw.get("usage")
    return usage if isinstance(usage, dict) else None


def _load_envspecs_by_id(paths: RunPaths) -> dict[str, EnvSpec | FourFamilySpec]:
    envspec_path = paths.stage_dirs["A"] / "envspecs.jsonl"
    if not envspec_path.exists():
        return {}
    return {spec.env_id: spec for spec in read_envspecs(envspec_path)}


def _final_answer_contract(*, runtime_name: str) -> str:
    if runtime_name == "lazyppl":
        return (
            "Final answer contract for this request:\n"
            "- Return a non-empty answer with exactly one `<think>...</think>` block "
            "followed by exactly one "
            '`<submit_program language="haskell">...</submit_program>` block.\n'
            "- The submitted Haskell must define `data State = ... deriving (Show)`, "
            "`model :: [Row] -> Meas State`, and `logLikRow :: State -> Row -> Double`.\n"
            "- If the ideal model is uncertain, submit the simplest schema-faithful "
            "baseline model that compiles and scores every row. Never return a blank, "
            "analysis-only, markdown-only, or refusal-style answer."
        )
    return (
        "Final answer contract for this request:\n"
        "- Return a non-empty answer with exactly one `<think>...</think>` block "
        "followed by exactly one `<submit_program>...</submit_program>` block.\n"
        "- The submitted Python must define `def model(data):`, use the exact observed "
        "data keys listed above, and return a `pm.Model`.\n"
        "- If the ideal model is uncertain, submit the simplest validation-fast "
        "baseline model that samples cleanly. Never return a blank, analysis-only, "
        "markdown-only, or refusal-style answer."
    )


def _append_final_answer_contract(text: str, *, runtime_name: str) -> str:
    return f"{text}\n\n{_final_answer_contract(runtime_name=runtime_name)}"


def _completion_user_prompt(
    prompt: PromptRecord,
    spec: EnvSpec | FourFamilySpec | None,
    *,
    runtime_name: str,
) -> str:
    if (
        runtime_name == "lazyppl"
        and "## LazyPPL TaskData schema" in prompt.prompt_text
        and "data Row = Row" in prompt.prompt_text
    ):
        return _append_final_answer_contract(
            f"{prompt.prompt_text}\n\n"
            "LazyPPL submission constraint: the `## LazyPPL TaskData schema` "
            "section above is the only Row accessor contract. Use only the "
            "accessors and Haskell types listed in that materialized Row type. "
            "Do not use EnvSpec feature names, prose-only covariates, or data "
            "dictionary names that are not literal Row accessors in that schema.",
            runtime_name=runtime_name,
        )

    if runtime_name == "pymc" and "## PyMC data dictionary" in prompt.prompt_text:
        return _append_final_answer_contract(
            f"{prompt.prompt_text}\n\n"
            "PyMC submission constraint: the `## PyMC data dictionary` section "
            "above is the only `data` key contract. Use only those literal keys "
            "in `def model(data)`. Do not use EnvSpec feature names, prose-only "
            "covariates, or dictionary keys that are not listed there.",
            runtime_name=runtime_name,
        )

    if spec is None:
        return _append_final_answer_contract(
            prompt.prompt_text,
            runtime_name=runtime_name,
        )

    # io_schema is an EnvSpec field; a FourFamilySpec has none, and reading it
    # raises exactly this error rather than degrading the data dictionary.
    if isinstance(spec, FourFamilySpec):
        raise AttributeError(
            f"'{type(spec).__name__}' object has no attribute 'io_schema'"
        )

    feature_names = list(spec.io_schema.feature_names)
    target_name = spec.io_schema.target_name
    feature_lines = "\n".join(f'- "{name}"' for name in feature_names) or "- none"
    base = (
        f"{prompt.prompt_text}\n\n"
        "Data dictionary contract for the submitted program:\n"
        f'- observed target key: "{target_name}"\n'
        "observed predictor keys:\n"
        f"{feature_lines}\n\n"
        "Use exactly these data dictionary keys in `def model(data)`. "
        "Do not guess alternate column names."
    )
    if runtime_name != "lazyppl":
        return _append_final_answer_contract(base, runtime_name=runtime_name)

    predictor_accessor_lines = "\n".join(f"- `{name} row`" for name in feature_names) or "- none"
    allowed_accessor_lines = "\n".join(f"- `{name} row`" for name in [target_name, *feature_names])
    return _append_final_answer_contract(
        (
            f"{prompt.prompt_text}\n\n"
            "Data dictionary contract for the submitted LazyPPL program:\n"
            f"- observed target Row accessor: `{target_name} row`\n"
            "observed predictor Row accessors:\n"
            f"{predictor_accessor_lines}\n\n"
            "Allowed `Row` accessors:\n"
            f"{allowed_accessor_lines}\n\n"
            "Use only these `TaskData.Row` accessor identifiers in `model` and "
            "`logLikRow`. Do not guess alternate field names, helper names, or "
            "dictionary keys."
        ),
        runtime_name=runtime_name,
    )


def _load_existing_completion_records(stage_dir: Path) -> dict[str, CompletionRecord]:
    completions_path = stage_dir / "completions.jsonl"
    if not completions_path.exists():
        return {}
    records: dict[str, CompletionRecord] = {}
    for row in read_jsonl(completions_path):
        record = CompletionRecord.model_validate(row)
        records[record.id] = record
    return records


def _load_existing_skipped_records(stage_dir: Path) -> dict[str, dict[str, str]]:
    skipped_path = stage_dir / "skipped_completions.jsonl"
    if not skipped_path.exists():
        return {}
    skipped: dict[str, dict[str, str]] = {}
    for row in read_jsonl(skipped_path):
        request_id = row.get("id")
        if isinstance(request_id, str):
            skipped[request_id] = {str(key): str(value) for key, value in row.items()}
    return skipped


def _write_stage_e_progress(
    stage_dir: Path,
    *,
    status: str,
    count_completed: int,
    count_skipped: int,
    count_requested: int,
    current_request_id: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "status": status,
        "count_completed": count_completed,
        "count_skipped": count_skipped,
        "count_recorded": count_completed + count_skipped,
        "count_requested": count_requested,
    }
    if current_request_id is not None:
        payload["current_request_id"] = current_request_id
    write_json(stage_dir / _PROGRESS_FILE, payload)


def _missing_output_skip(
    request_id: str,
    prompt: PromptRecord,
    *,
    provider_failure: str | None = None,
) -> dict[str, str]:
    skipped = {
        "id": request_id,
        "prompt_id": prompt.id,
        "env_id": prompt.env_id,
        "reason": "missing_or_empty_batch_output",
    }
    if provider_failure:
        skipped["provider_failure"] = provider_failure
    return skipped


def _empty_output_skip(request_id: str, prompt: PromptRecord) -> dict[str, str]:
    return {
        "id": request_id,
        "prompt_id": prompt.id,
        "env_id": prompt.env_id,
        "reason": "empty_output",
    }


def _completion_record_from_output(
    *,
    request_id: str,
    prompt: PromptRecord,
    user_prompt: str,
    output: LLMResult,
) -> CompletionRecord | dict[str, str]:
    if not output.content.strip():
        return _empty_output_skip(request_id, prompt)
    candidate_outputs = [
        candidate
        for candidate in extract_candidate_texts(output.raw)
        if candidate != output.content
    ]
    evidence_metadata: dict[str, Any] = {
        "provider_request_id": request_id,
        "provider_response_sha256": _json_sha256(output.raw),
        "raw_completion_sha256": _text_sha256(output.content),
    }
    response_id = _provider_response_id(output.raw)
    if response_id is not None:
        evidence_metadata["provider_response_id"] = response_id
    usage = _provider_usage(output.raw)
    if usage is not None:
        evidence_metadata["provider_usage"] = usage
    return CompletionRecord(
        id=request_id,
        prompt_id=prompt.id,
        env_id=prompt.env_id,
        split=prompt.split,
        prompt_text=user_prompt,
        raw_completion=output.content,
        candidate_outputs=candidate_outputs,
        provider_response=output.raw,
        provenance=prompt.provenance,
        metadata={
            **prompt.metadata,
            "stage_d_prompt_text": prompt.prompt_text,
            "stage_e_prompt_augmented": user_prompt != prompt.prompt_text,
            **evidence_metadata,
        },
    )


def _provider_failure_for_request(
    batch_failures: dict[str, str],
    request_id: str,
) -> str | None:
    direct = batch_failures.get(request_id)
    if direct:
        return direct
    missing = batch_failures.get("__missing__")
    if missing and request_id in missing:
        return f"missing from batch output: {missing}"
    return None


def _write_partial_outputs(
    stage_dir: Path,
    records: list[CompletionRecord],
    skipped: list[dict[str, str]],
) -> None:
    write_jsonl(
        stage_dir / "completions.jsonl",
        [rec.model_dump(mode="json") for rec in records],
    )
    write_jsonl(stage_dir / "skipped_completions.jsonl", skipped)


def _execute_sync_with_progress(
    llm: LLMBackend,
    *,
    requests: list[tuple[str, LLMRequest]],
    stage_dir: Path,
    prompt_by_request: dict[str, PromptRecord],
    user_prompt_by_request: dict[str, str],
) -> tuple[list[CompletionRecord], list[dict[str, str]]]:
    records_by_request = _load_existing_completion_records(stage_dir)
    skipped_by_request = _load_existing_skipped_records(stage_dir)
    completed_or_skipped = set(records_by_request) | set(skipped_by_request)

    records = [
        records_by_request[request_id]
        for request_id, _ in requests
        if request_id in records_by_request
    ]
    skipped = [
        skipped_by_request[request_id]
        for request_id, _ in requests
        if request_id in skipped_by_request
    ]
    _write_partial_outputs(stage_dir, records, skipped)

    for request_id, req in requests:
        if request_id in completed_or_skipped:
            continue
        _write_stage_e_progress(
            stage_dir,
            status="running",
            count_completed=len(records),
            count_skipped=len(skipped),
            count_requested=len(requests),
            current_request_id=request_id,
        )
        output = llm.complete(req)
        result = _completion_record_from_output(
            request_id=request_id,
            prompt=prompt_by_request[request_id],
            user_prompt=user_prompt_by_request[request_id],
            output=output,
        )
        if isinstance(result, CompletionRecord):
            records.append(result)
        else:
            skipped.append(result)
        completed_or_skipped.add(request_id)
        _write_partial_outputs(stage_dir, records, skipped)
        _write_stage_e_progress(
            stage_dir,
            status="running",
            count_completed=len(records),
            count_skipped=len(skipped),
            count_requested=len(requests),
        )

    return records, skipped


def run_stage_e(
    paths: RunPaths,
    *,
    limit: int | None = None,
    backend_name: str | None = None,
    use_batch: bool = True,
    model: str | None = None,
    backend: LLMBackend | None = None,
    runtime: TargetRuntime | None = None,
    settings: Settings | None = None,
) -> StageResult:
    stage = "E"
    stage_dir = paths.stage_dirs[stage]
    stage_dir.mkdir(parents=True, exist_ok=True)

    defaults = load_default_pipeline_defaults()
    stage_defaults = defaults.stages.completions
    resolved_backend_name = resolve_backend_name(backend_name, defaults=defaults)
    resolved_model = resolve_stage_model("completions", model, defaults=defaults)
    reasoning_effort = effective_reasoning_effort(stage_defaults.reasoning_effort)
    max_output_tokens = stage_defaults.max_output_tokens

    prompt_rows = read_jsonl(paths.stage_dirs["D"] / "prompts.jsonl")
    prompts = [PromptRecord.model_validate(row) for row in prompt_rows]
    if limit is not None:
        prompts = prompts[:limit]

    if runtime is None:
        raise ValueError(
            "runtime is required for stage E. Pass --runtime pymc or --runtime lazyppl."
        )
    ensure_upstream_stage_runtime(paths, "D", runtime.name)
    system = load_prompt_text(runtime.completion_system_prompt_name())
    llm = backend or build_backend(resolved_backend_name, settings=settings)
    envspecs_by_id = _load_envspecs_by_id(paths)

    requests: list[tuple[str, LLMRequest]] = []
    prompt_by_request: dict[str, PromptRecord] = {}
    user_prompt_by_request: dict[str, str] = {}
    for idx, prompt in enumerate(prompts):
        prompt = normalize_and_require_prompt_runtime(
            prompt,
            expected_runtime=runtime.name,
            stage_name="E",
        )
        request_id = f"completion-{idx:06d}"
        user_prompt = _completion_user_prompt(
            prompt,
            envspecs_by_id.get(prompt.env_id),
            runtime_name=runtime.name,
        )
        prompt_by_request[request_id] = prompt
        user_prompt_by_request[request_id] = user_prompt
        requests.append(
            (
                request_id,
                LLMRequest(
                    model=resolved_model,
                    system=system,
                    user=user_prompt,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                ),
            )
        )

    if use_batch and llm.capabilities.supports_batch:
        outputs = execute_requests_raw(
            llm,
            requests=requests,
            use_batch=use_batch,
            state_dir=stage_dir,
            allow_partial_batch_failures=True,
        )
        batch_failures = getattr(llm, "last_batch_failures", {})
        if not isinstance(batch_failures, dict):
            batch_failures = {}
        records: list[CompletionRecord] = []
        skipped: list[dict[str, str]] = []
        for request_id, _ in requests:
            if request_id not in outputs:
                skipped.append(
                    _missing_output_skip(
                        request_id,
                        prompt_by_request[request_id],
                        provider_failure=_provider_failure_for_request(
                            {str(key): str(value) for key, value in batch_failures.items()},
                            request_id,
                        ),
                    )
                )
                continue
            result = _completion_record_from_output(
                request_id=request_id,
                prompt=prompt_by_request[request_id],
                user_prompt=user_prompt_by_request[request_id],
                output=outputs[request_id],
            )
            if isinstance(result, CompletionRecord):
                records.append(result)
            else:
                skipped.append(result)
        _write_partial_outputs(stage_dir, records, skipped)
    else:
        records, skipped = _execute_sync_with_progress(
            llm,
            requests=requests,
            stage_dir=stage_dir,
            prompt_by_request=prompt_by_request,
            user_prompt_by_request=user_prompt_by_request,
        )

    out_path = stage_dir / "completions.jsonl"
    skipped_path = stage_dir / "skipped_completions.jsonl"
    batch_receipt_path = stage_dir / "batch_receipt.json"
    if not records:
        raise RuntimeError("Stage E produced no non-empty completions")
    manifest = {
        "status": "success",
        "count": len(records),
        "count_requested": len(requests),
        "skipped_count": len(skipped),
        "request_ids": [request_id for request_id, _ in requests],
        "success_request_ids": [record.id for record in records],
        "skipped_request_ids": [item["id"] for item in skipped if isinstance(item.get("id"), str)],
        "count_with_candidates": sum(1 for record in records if record.candidate_outputs),
        "output": str(out_path),
        "skipped_output": str(skipped_path),
        "batch_receipt": str(batch_receipt_path) if batch_receipt_path.exists() else None,
        "backend": resolved_backend_name,
        "model": resolved_model,
        "configured_reasoning_effort": stage_defaults.reasoning_effort,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "use_batch": use_batch,
        "runtime": runtime.name,
    }
    write_json(stage_dir / "manifest.json", manifest)
    _write_stage_e_progress(
        stage_dir,
        status="success",
        count_completed=len(records),
        count_skipped=len(skipped),
        count_requested=len(requests),
    )
    return StageResult(stage=stage, status="success", outputs=manifest)

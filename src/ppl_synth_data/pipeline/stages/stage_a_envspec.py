from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import LLMBackend, LLMRequest
from ppl_synth_data.llm.prompt_loader import load_prompt_text
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
from ppl_synth_data.pipeline.io import write_json, write_jsonl
from ppl_synth_data.pipeline.types import RunPaths, StageResult
from ppl_synth_data.runtime_registry import normalize_runtime_name
from ppl_synth_data.specs.models import EnvSpec
from ppl_synth_data.specs.sampling import build_generation_plan
from ppl_synth_data.validation.leakage import public_source_or_sentinel

log = logging.getLogger(__name__)

_ENVSPEC_TEXT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "EnvSpec",
    "schema": EnvSpec.model_json_schema(),
    # EnvSpec intentionally contains open maps (`generative_params` and
    # `prompt_guidance`). OpenAI strict schemas require every object to set
    # `additionalProperties: false`, which would make those maps unusable.
    # Keep schema guidance on, then validate the provider output with Pydantic.
    "strict": False,
}


def run_stage_a(
    paths: RunPaths,
    *,
    limit: int = 25,
    seed: int = 0,
    backend_name: str | None = None,
    use_batch: bool = True,
    model: str | None = None,
    source_plan_path: Path | None = None,
    runtime_target: str | None = None,
    backend: LLMBackend | None = None,
    settings: Settings | None = None,
) -> StageResult:
    stage = "A"
    stage_dir = paths.stage_dirs[stage]
    stage_dir.mkdir(parents=True, exist_ok=True)

    defaults = load_default_pipeline_defaults()
    stage_defaults = defaults.stages.envspec
    resolved_backend_name = resolve_backend_name(backend_name, defaults=defaults)
    resolved_model = resolve_stage_model("envspec", model, defaults=defaults)
    configured_reasoning_effort = stage_defaults.reasoning_effort
    reasoning_effort = effective_reasoning_effort(configured_reasoning_effort)
    max_output_tokens = stage_defaults.max_output_tokens

    system = load_prompt_text("envspec_gen.md")
    llm = backend or build_backend(resolved_backend_name, settings=settings)
    text_format = (
        dict(_ENVSPEC_TEXT_FORMAT)
        if llm.capabilities.supports_json_schema
        else {"type": "json_object"}
    )
    log.info(
        "stage A using backend=%s model=%s envspec_output=%s",
        resolved_backend_name,
        resolved_model,
        text_format["type"],
    )

    plan, source_plan_manifest = _generation_plan(
        limit=limit,
        seed=seed,
        source_plan_path=source_plan_path,
        runtime_target=runtime_target,
    )

    requests: list[tuple[str, LLMRequest]] = []
    for idx, slot in enumerate(plan):
        source_provenance = _source_provenance_from_slot(slot)
        user = (
            "Create one EnvSpec.\n"
            f"Family: {slot['family']}\n"
            f"Domain: {slot['domain']}\n"
            f"Complexity: {slot['complexity']}\n"
            f"Diversity seed: {slot['diversity_seed']}\n"
            + _source_plan_hint_suffix(slot)
            + _source_plan_prompt_suffix(source_provenance)
            + "Output valid JSON only."
        )
        req = LLMRequest(
            model=resolved_model,
            system=system,
            user=user,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            text_format=text_format,
        )
        requests.append((f"spec-{idx:06d}", req))

    raw_outputs = execute_requests_raw(
        llm,
        requests=requests,
        use_batch=use_batch,
        state_dir=stage_dir,
        allow_partial_batch_failures=True,
    )

    specs: list[EnvSpec] = []
    skipped: list[dict[str, str]] = []
    for idx, (request_id, _) in enumerate(requests):
        output = raw_outputs.get(request_id)
        if output is None:
            skipped.append(
                {
                    "id": request_id,
                    "reason": "missing_or_failed_batch_output",
                }
            )
            continue
        text = output.content
        try:
            spec = EnvSpec.model_validate_json(text)
        except ValidationError as exc:
            skipped.append(
                {
                    "id": request_id,
                    "reason": "invalid_envspec",
                    "error": json.dumps(exc.errors(), default=str),
                }
            )
            continue

        expected_family = plan[idx]["family"]
        if spec.generative_family != expected_family:
            log.warning(
                "family mismatch for %s: requested %r, got %r",
                request_id,
                expected_family,
                spec.generative_family,
            )
        source_provenance = _source_provenance_from_slot(plan[idx])
        if source_provenance:
            spec.prompt_guidance = dict(spec.prompt_guidance)
            spec.prompt_guidance["source_provenance"] = source_provenance
        specs.append(spec)

    out_path = stage_dir / "envspecs.jsonl"
    skipped_path = stage_dir / "skipped_envspecs.jsonl"
    write_jsonl(out_path, [spec.model_dump(mode="json") for spec in specs])
    write_jsonl(skipped_path, skipped)
    if not specs:
        raise RuntimeError("Stage A produced no valid EnvSpecs")
    manifest = {
        "status": "success",
        "count": len(specs),
        "count_requested": len(requests),
        "skipped_count": len(skipped),
        "output": str(out_path),
        "skipped_output": str(skipped_path),
        "backend": resolved_backend_name,
        "model": resolved_model,
        "configured_reasoning_effort": configured_reasoning_effort,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "seed": seed,
        **source_plan_manifest,
    }
    write_json(stage_dir / "manifest.json", manifest)
    return StageResult(stage=stage, status="success", outputs=manifest)


def _generation_plan(
    *,
    limit: int,
    seed: int,
    source_plan_path: Path | None,
    runtime_target: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source_plan_path is None:
        return build_generation_plan(limit, seed=seed), {}

    raw_slots = _load_source_plan(source_plan_path)
    if not raw_slots:
        raise ValueError(f"source plan is empty: {source_plan_path}")

    eligible_slots = raw_slots
    runtime_manifest: dict[str, Any] = {}
    if runtime_target is not None:
        normalized_runtime_target = normalize_runtime_name(runtime_target)
        if normalized_runtime_target is None:
            raise ValueError(f"unknown source-plan runtime target: {runtime_target!r}")
        eligible_slots = _filter_source_plan_by_runtime(
            raw_slots,
            runtime_target=normalized_runtime_target,
        )
        if not eligible_slots:
            raise ValueError(
                "source plan has no rows for runtime_target="
                f"{normalized_runtime_target!r}: {source_plan_path}"
            )
        runtime_manifest = {
            "source_plan_runtime_target": normalized_runtime_target,
            "source_plan_runtime_eligible_count": len(eligible_slots),
            "source_plan_runtime_skipped_count": len(raw_slots) - len(eligible_slots),
        }

    plan = eligible_slots[:limit]
    manifest = {
        "source_plan": str(source_plan_path),
        "source_plan_count": len(raw_slots),
        "source_plan_used_count": len(plan),
        "source_plan_sha256": _sha256_file(source_plan_path),
        **runtime_manifest,
    }
    return plan, manifest


def _filter_source_plan_by_runtime(
    slots: list[dict[str, Any]],
    *,
    runtime_target: str,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for idx, slot in enumerate(slots):
        raw_slot_runtime = slot.get("runtime_target")
        if raw_slot_runtime is None:
            eligible.append(slot)
            continue
        if not isinstance(raw_slot_runtime, str) or not raw_slot_runtime.strip():
            raise ValueError(f"source plan row {idx} has invalid runtime_target")
        normalized_slot_runtime = normalize_runtime_name(raw_slot_runtime)
        if normalized_slot_runtime is None:
            raise ValueError(
                f"source plan row {idx} has unknown runtime_target {raw_slot_runtime!r}"
            )
        if normalized_slot_runtime == runtime_target:
            eligible.append(slot)
    return eligible


def _load_source_plan(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("items") or payload.get("slots") or payload.get("rows")
        if not isinstance(payload, list):
            raise ValueError(
                "source plan must be a JSON list, JSONL, or object with items/slots/rows"
            )
        rows = payload

    plan: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"source plan row {idx} must be an object")
        slot = _normalize_source_plan_slot(row, idx=idx)
        plan.append(slot)
    return plan


def _normalize_source_plan_slot(row: dict[str, Any], *, idx: int) -> dict[str, Any]:
    family = row.get("family") or row.get("generative_family")
    domain = row.get("domain")
    complexity = row.get("complexity", "medium")
    if not isinstance(family, str) or not family:
        raise ValueError(f"source plan row {idx} missing family/generative_family")
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"source plan row {idx} missing domain")
    if complexity not in {"simple", "medium", "complex"}:
        raise ValueError(f"source plan row {idx} has invalid complexity {complexity!r}")

    diversity_seed = row.get("diversity_seed")
    if not isinstance(diversity_seed, int):
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).digest()
        diversity_seed = int.from_bytes(digest[:8], "big")

    slot = dict(row)
    slot.update(
        {
            "family": family,
            "domain": domain,
            "complexity": complexity,
            "diversity_seed": diversity_seed,
        }
    )
    return slot


def _source_provenance_from_slot(slot: dict[str, Any]) -> dict[str, str | None]:
    nested = slot.get("source_provenance")
    if isinstance(nested, dict):
        slot = {**slot, **nested}

    keys = (
        "source_category",
        "source_url_or_repo",
        "license_signal",
        "adaptation_mode",
        "source_retrieval_date",
    )
    provenance = {
        key: slot[key] if isinstance(slot.get(key), str) else None
        for key in keys
        if key in slot or isinstance(nested, dict)
    }
    if provenance.get("source_category") == "synthetic_invented":
        provenance.setdefault("source_url_or_repo", None)
        provenance.setdefault("license_signal", None)
        provenance.setdefault("adaptation_mode", None)
        provenance.setdefault("source_retrieval_date", None)
        provenance["source_url_or_repo"] = (
            provenance.get("source_url_or_repo") or "not_applicable_invented"
        )
        provenance["license_signal"] = provenance.get("license_signal") or "not_applicable_invented"
        provenance["adaptation_mode"] = provenance.get("adaptation_mode") or "invented"
        provenance["source_retrieval_date"] = (
            provenance.get("source_retrieval_date") or "not_applicable_invented"
        )
    raw_source = provenance.get("source_url_or_repo")
    if isinstance(raw_source, str):
        safe_source = public_source_or_sentinel(raw_source)
        if safe_source != raw_source:
            log.warning(
                "stage A: replaced non-public source_url_or_repo from the source "
                "plan with a sentinel to prevent prompt leakage"
            )
        provenance["source_url_or_repo"] = safe_source
    return provenance


def _source_plan_prompt_suffix(source_provenance: dict[str, str | None]) -> str:
    if not source_provenance:
        return ""
    encoded = json.dumps(source_provenance, sort_keys=True)
    return (
        "Source provenance: "
        f"{encoded}\n"
        "Set prompt_guidance.source_provenance to this exact object.\n"
    )


def _source_plan_hint_suffix(slot: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, label in (
        ("runtime_target", "Runtime target"),
        ("problem_hint", "Problem inspiration"),
        ("target_model_shape", "Target model shape"),
    ):
        value = slot.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"{label}: {value.strip()}")
    return "\n".join(lines) + "\n" if lines else ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

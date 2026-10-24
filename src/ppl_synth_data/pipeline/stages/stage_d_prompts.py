from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from ppl_synth_data.config import Settings
from ppl_synth_data.dataset.split import deterministic_split
from ppl_synth_data.llm.backend import LLMBackend, LLMRequest, LLMResult
from ppl_synth_data.llm.bedrock_converse_backend import (
    BedrockConverseBackend,
    retry_converse_call,
    validate_converse_receipt,
)
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
from ppl_synth_data.pipeline.contracts import PYMC_ROW_COUNT_KEY_LINE
from ppl_synth_data.pipeline.io import read_jsonl, write_json, write_jsonl
from ppl_synth_data.pipeline.stages._envspecs import read_canonical_envspecs
from ppl_synth_data.pipeline.types import RunPaths, StageResult
from ppl_synth_data.runtimes.base import TargetRuntime
from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec, PromptProvenance, PromptRecord
from ppl_synth_data.validation.leakage import public_source_or_sentinel

log = logging.getLogger(__name__)


def persist_prompt_receipt(
    path: Path,
    env_id: str,
    prompt_text: str,
    result: LLMResult,
) -> None:
    """Durably append provider evidence before starting another prompt request."""
    receipt = validate_converse_receipt(result.raw.get("receipt"))
    row = {
        **receipt,
        "env_id": env_id,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def persist_throttle_event(path: Path, env_id: str, event: dict[str, Any]) -> None:
    """Append the failed attempt before waiting or propagating the final error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"env_id": env_id, **event}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_resumable_prompts(
    prompts_path: Path,
    receipts_path: Path,
) -> dict[str, PromptRecord]:
    """Reuse only nonempty prompts with complete provider evidence for these bytes."""
    if not prompts_path.exists() or not receipts_path.exists():
        return {}
    evidence: set[tuple[str, str]] = set()
    for row in read_jsonl(receipts_path):
        try:
            validate_converse_receipt(row)
        except RuntimeError:
            # Incomplete evidence never licenses reuse of a paid result.
            continue
        env_id, digest = row.get("env_id"), row.get("prompt_sha256")
        if isinstance(env_id, str) and isinstance(digest, str):
            evidence.add((env_id, digest))
    records: dict[str, PromptRecord] = {}
    for row in read_jsonl(prompts_path):
        record = PromptRecord.model_validate(row)
        digest = hashlib.sha256(record.prompt_text.encode("utf-8")).hexdigest()
        if record.prompt_text.strip() and (record.env_id, digest) in evidence:
            if record.env_id in records:
                raise ValueError(f"duplicate resumable prompt env_id: {record.env_id}")
            records[record.env_id] = record
    return records


def _is_count_name(column_name: str) -> bool:
    return column_name == "count" or column_name.endswith("_count")


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _all_integer_like(values: pd.Series) -> bool:
    return all(math.isfinite(float(value)) and float(value).is_integer() for value in values)


def _haskell_type_for_series(column_name: str, series: pd.Series) -> str:
    dtype_name = str(series.dtype)
    if dtype_name == "bool":
        return "Bool"
    if dtype_name.startswith(("int", "uint")):
        if _is_count_name(column_name):
            return "Integer"
        return "Int"
    if dtype_name.startswith("float"):
        return "Double"

    numeric = _numeric_series(series)
    if len(numeric) == len(series.dropna()) and not numeric.empty:
        if _is_count_name(column_name):
            if not _all_integer_like(numeric):
                return "Double"
            return "Integer"
        if (
            column_name.endswith("_id")
            or column_name.endswith("Id")
            or column_name in {"t", "idx", "index"}
        ) and _all_integer_like(numeric):
            return "Int"
        return "Double"
    return "String"


def _materialized_accessors(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
) -> tuple[str, list[str]]:
    columns = [str(column) for column in dataframe.columns]
    if target_name in columns:
        target = target_name
    else:
        target = columns[-1]
    features = [column for column in columns if column != target]
    return target, features


def _format_accessor_list(names: list[str]) -> str:
    if not names:
        return "(none; model only the target accessor and intercept/noise terms)."
    return ", ".join(f"`{name}`" for name in names) + "."


def _haskell_type_for_column(column_name: str, dtype: Any) -> str:
    if _is_count_name(column_name):
        return "Integer"
    if column_name == "t" or column_name.endswith("_id") or column_name.endswith("Id"):
        return "Int"
    dtype_name = str(dtype)
    if dtype_name == "bool":
        return "Bool"
    if dtype_name.startswith(("int", "uint")):
        return "Int"
    return "Double"


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _taskdata_summary_lines(dataframe: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "Observed TaskData summaries from the exact rows used by the verifier:",
    ]
    for column_name in dataframe.columns:
        series = dataframe[column_name]
        dtype = str(series.dtype)
        if dtype == "bool":
            true_count = int(series.astype(bool).sum())
            lines.append(f"- `{column_name}`: Bool, true_count={true_count}, n={len(series)}.")
            continue
        if not pd.api.types.is_numeric_dtype(series):
            observed_values = sorted({str(value) for value in series.dropna().unique()})
            preview = ", ".join(observed_values[:8])
            lines.append(f"- `{column_name}`: {dtype}, observed_values={preview}, n={len(series)}.")
            continue

        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            lines.append(f"- `{column_name}`: {dtype}, no finite numeric rows.")
            continue
        lines.append(
            "- "
            f"`{column_name}`: {dtype}, "
            f"min={_format_float(float(numeric.min()))}, "
            f"q25={_format_float(float(numeric.quantile(0.25)))}, "
            f"median={_format_float(float(numeric.median()))}, "
            f"mean={_format_float(float(numeric.mean()))}, "
            f"q75={_format_float(float(numeric.quantile(0.75)))}, "
            f"max={_format_float(float(numeric.max()))}, "
            f"n={len(numeric)}."
        )
    return lines


def _lazy_ppl_schema_appendix(
    *,
    spec: EnvSpec | FourFamilySpec,
    data_ref: dict[str, Any] | None,
    run_dir: Path,
) -> str:
    if data_ref is None:
        return ""
    data_ref_path = data_ref.get("data_ref")
    if not isinstance(data_ref_path, str) or not data_ref_path:
        return ""
    parquet_path = run_dir / data_ref_path
    if not parquet_path.exists():
        return ""

    dataframe = pd.read_parquet(parquet_path)
    if dataframe.empty:
        return ""

    target_accessor, feature_accessors = _materialized_accessors(
        dataframe=dataframe,
        target_name="y" if isinstance(spec, FourFamilySpec) else spec.io_schema.target_name,
    )
    lines = [
        "",
        "## LazyPPL TaskData schema",
        "",
        "The generated Haskell `TaskData` module exports this exact row type.",
        "Use these accessor names and Haskell types literally; do not infer",
        "different types from the prose description.",
        "",
        "```haskell",
        "data Row = Row",
    ]
    for index, column_name in enumerate(dataframe.columns):
        prefix = "  { " if index == 0 else "  , "
        hs_type = _haskell_type_for_series(column_name, dataframe[column_name])
        lines.append(f"{prefix}{column_name} :: {hs_type}")
    lines.append("  } deriving (Show)")
    lines.append("```")
    lines.extend(
        [
            "",
            f"Target accessor: `{target_accessor}`.",
            "Feature accessors: " + _format_accessor_list(feature_accessors),
            "Only the accessors in the Row type above exist in TaskData. Do not",
            "use EnvSpec feature names, prose variable names, or domain concepts",
            "unless they appear literally in the Row type above.",
            *_taskdata_summary_lines(dataframe),
            "",
            "If a concept is described as binary but the schema says `Double`,",
            "treat it as a numeric covariate or threshold it explicitly; do not",
            "write `if field row then ...` unless the schema type is `Bool`.",
            "Never apply `fromIntegral` to a `Double` accessor. Use the value",
            "directly, or convert to `Int` with an explicit `round (...) :: Int`",
            "only at the point where an integer-valued PMF requires it.",
            "Apply `fromIntegral` to `Int` or `Integer` accessors when they enter",
            "continuous arithmetic, but pass `Integer` counts directly to",
            "`poissonLogPmf` and related count PMFs.",
            "",
            "The verifier scores against the generated TaskData rows, not against",
            "domain-scale examples in the prose. For feature centering, scaling,",
            "offsets, and weakly informative priors, use row-derived quantities",
            "inside `model` when possible; otherwise use the observed TaskData",
            "summaries above rather than prose-only domain constants.",
        ]
    )
    return "\n".join(lines)


def _pymc_data_schema_appendix(
    *,
    spec: EnvSpec | FourFamilySpec,
    data_ref: dict[str, Any] | None,
    run_dir: Path,
) -> str:
    if data_ref is None:
        return ""
    data_ref_path = data_ref.get("data_ref")
    if not isinstance(data_ref_path, str) or not data_ref_path:
        return ""
    parquet_path = run_dir / data_ref_path
    if not parquet_path.exists():
        return ""

    dataframe = pd.read_parquet(parquet_path)
    if dataframe.empty:
        return ""

    target_key, feature_keys = _materialized_accessors(
        dataframe=dataframe,
        target_name="y" if isinstance(spec, FourFamilySpec) else spec.io_schema.target_name,
    )
    allowed_key_lines = "\n".join(f"- `{name}`" for name in dataframe.columns)
    feature_lines = "\n".join(f"- `{name}`" for name in feature_keys) or "- none"
    lines = [
        "",
        "## PyMC data dictionary",
        "",
        "The verifier calls `def model(data)` with a Python dict built from the",
        "exact materialized visible Stage C dataframe. Use these keys literally;",
        "do not infer alternate names from EnvSpec, prose variable names, or",
        "domain concepts unless they appear in the allowed key list below.",
        "",
        f"Observed target key: `{target_key}`.",
        "Observed predictor keys:",
        feature_lines,
        "",
        "Allowed `data` keys:",
        PYMC_ROW_COUNT_KEY_LINE,
        allowed_key_lines,
        *_taskdata_summary_lines(dataframe),
        "",
        "If the prose describes a variable name that is absent from the allowed",
        "key list, ignore that prose-only name and use the materialized key",
        "above. If several measurement columns are present, a compact model may",
        "include multiple observed variables, but every observed variable and",
        "covariate must use only the allowed `data` keys.",
    ]
    return "\n".join(lines)


_SOURCE_PROVENANCE_KEYS = (
    "source_category",
    "source_url_or_repo",
    "license_signal",
    "adaptation_mode",
    "source_retrieval_date",
)


def _source_provenance_from_spec(spec: EnvSpec | FourFamilySpec) -> dict[str, str | None]:
    if isinstance(spec, FourFamilySpec):
        return {
            "source_category": "synthetic_regenerated",
            "source_url_or_repo": None,
            "license_signal": None,
            "adaptation_mode": "regenerated",
            "source_retrieval_date": None,
        }
    values: dict[str, str | None] = {
        "source_category": "synthetic_invented",
        "source_url_or_repo": "not_applicable_invented",
        "license_signal": "not_applicable_invented",
        "adaptation_mode": "invented",
        "source_retrieval_date": "not_applicable_invented",
    }
    candidates: list[dict[str, Any]] = []
    for container in (spec.prompt_guidance, spec.generative_params):
        if isinstance(container, dict):
            candidates.append(container)
            nested = container.get("source_provenance")
            if isinstance(nested, dict):
                candidates.append(nested)
    for candidate in candidates:
        for key in _SOURCE_PROVENANCE_KEYS:
            raw_value = candidate.get(key)
            if isinstance(raw_value, str) and raw_value:
                values[key] = raw_value
    if values.get("source_category") == "synthetic_invented":
        values["source_url_or_repo"] = values.get("source_url_or_repo") or "not_applicable_invented"
        values["license_signal"] = values.get("license_signal") or "not_applicable_invented"
        values["adaptation_mode"] = values.get("adaptation_mode") or "invented"
        values["source_retrieval_date"] = (
            values.get("source_retrieval_date") or "not_applicable_invented"
        )
    raw_source = values.get("source_url_or_repo")
    if isinstance(raw_source, str):
        safe_source = public_source_or_sentinel(raw_source)
        if safe_source != raw_source:
            log.warning(
                "stage D: replaced non-public source_url_or_repo from the EnvSpec "
                "with a sentinel to prevent prompt leakage"
            )
        values["source_url_or_repo"] = safe_source or "not_applicable_invented"
    return values


def _missing_prompt_skip(request_id: str, spec: EnvSpec | FourFamilySpec) -> dict[str, str]:
    return {
        "id": request_id,
        "env_id": spec.env_id,
        "reason": "missing_or_failed_batch_output",
    }


def _empty_prompt_skip(request_id: str, spec: EnvSpec | FourFamilySpec) -> dict[str, str]:
    return {
        "id": request_id,
        "env_id": spec.env_id,
        "reason": "empty_output",
    }


def _prompt_text_from_output(
    *,
    output: LLMResult,
    runtime: TargetRuntime,
    spec: EnvSpec | FourFamilySpec,
    data_ref: dict[str, Any] | None,
    run_dir: Path,
) -> str:
    prompt_text = output.content
    if runtime.name == "pymc":
        prompt_text = (
            prompt_text.rstrip()
            + _pymc_data_schema_appendix(
                spec=spec,
                data_ref=data_ref,
                run_dir=run_dir,
            )
        ).rstrip()
    if runtime.name == "lazyppl":
        prompt_text = (
            prompt_text.rstrip()
            + _lazy_ppl_schema_appendix(
                spec=spec,
                data_ref=data_ref,
                run_dir=run_dir,
            )
        ).rstrip()
    return prompt_text


def build_prompt_record(
    *,
    request_id: str,
    spec: EnvSpec | FourFamilySpec,
    prompt_text: str,
    runtime: TargetRuntime,
) -> PromptRecord:
    """Assemble a prompt with source lineage kept outside learner-facing text."""
    return PromptRecord(
        id=request_id,
        split=deterministic_split(spec.env_id),
        env_id=spec.env_id,
        prompt_text=prompt_text,
        provenance=PromptProvenance(
            env_name=spec.env_name,
            domain=spec.domain,
            complexity=None if isinstance(spec, FourFamilySpec) else spec.complexity.value,
            model_type=None if isinstance(spec, FourFamilySpec) else spec.model_type,
            generative_family=spec.generative_family,
            key_concepts=[] if isinstance(spec, FourFamilySpec) else list(spec.key_concepts),
            has_observed_data=True,
            runtime_name=runtime.name,
            runtime_version=runtime.runtime_version_tag(),
            **_source_provenance_from_spec(spec),
        ),
        metadata=(
            {"scenario": spec.scenario.model_dump(mode="json"), "data_origin": "regenerated"}
            if isinstance(spec, FourFamilySpec)
            else {}
        ),
    )


def run_stage_d(
    paths: RunPaths,
    *,
    limit: int | None = None,
    backend_name: str | None = None,
    use_batch: bool = True,
    model: str | None = None,
    backend: LLMBackend | None = None,
    runtime: TargetRuntime | None = None,
    settings: Settings | None = None,
    resume: bool = False,
    prompt_rule: str = "default",
) -> StageResult:
    if prompt_rule not in ("default", "no-numeric-summaries-v1"):
        raise ValueError(f"unknown prompt_rule: {prompt_rule!r}")
    stage = "D"
    stage_dir = paths.stage_dirs[stage]
    stage_dir.mkdir(parents=True, exist_ok=True)

    if runtime is None:
        raise ValueError(
            "runtime is required for stage D. Pass --runtime pymc or --runtime lazyppl."
        )
    _rt = runtime

    defaults = load_default_pipeline_defaults()
    stage_defaults = defaults.stages.prompts
    resolved_backend_name = resolve_backend_name(backend_name, defaults=defaults)
    resolved_model = resolve_stage_model("prompts", model, defaults=defaults)
    configured_reasoning_effort = stage_defaults.reasoning_effort
    reasoning_effort = effective_reasoning_effort(configured_reasoning_effort)
    max_output_tokens = stage_defaults.max_output_tokens

    summaries = read_jsonl(paths.stage_dirs["C"] / "summaries.jsonl")
    data_refs_path = paths.stage_dirs["C"] / "data_refs.jsonl"
    data_refs = read_jsonl(data_refs_path) if data_refs_path.exists() else []
    data_ref_by_env = {
        str(row["env_id"]): row for row in data_refs if row.get("env_id") is not None
    }
    summary_by_env: dict[str, dict] = {}
    for row in summaries:
        eid = row["env_id"]
        if eid in summary_by_env:
            raise ValueError(f"duplicate env_id in summaries: {eid}")
        summary_by_env[eid] = row
    all_specs = read_canonical_envspecs(paths)
    specs = [spec for spec in all_specs if spec.env_id in summary_by_env]
    if limit is not None:
        specs = specs[:limit]
    if all_specs and not specs and (limit is None or limit > 0):
        raise RuntimeError(
            "stage D found no materialized environment summaries for canonical EnvSpecs; "
            "rerun or inspect Stage C instead of creating an empty prompt set"
        )

    if prompt_rule == "no-numeric-summaries-v1":
        for spec in specs:
            if not isinstance(spec, FourFamilySpec):
                raise ValueError(
                    f"prompt_rule {prompt_rule!r} requires FourFamilySpec: {spec.env_id}"
                )

    system = load_prompt_text(
        "prompt_gen_no_numbers.md" if prompt_rule == "no-numeric-summaries-v1" else "prompt_gen.md"
    )
    llm = backend or build_backend(resolved_backend_name, settings=settings)
    receipts_path = stage_dir / "receipts.jsonl"
    cached = load_resumable_prompts(stage_dir / "prompts.jsonl", receipts_path) if resume else {}
    if prompt_rule == "no-numeric-summaries-v1":
        cached = {
            env_id: record
            for env_id, record in cached.items()
            if record.metadata.get("prompt_rule") == prompt_rule
        }
    requests: list[tuple[str, LLMRequest]] = []
    spec_by_request: dict[str, EnvSpec | FourFamilySpec] = {}
    for idx, spec in enumerate(specs):
        if spec.env_id in cached:
            continue
        if prompt_rule == "no-numeric-summaries-v1":
            declared = spec.model_dump(mode="json")
            redacted = {
                key: declared[key] for key in ("env_id", "env_name", "domain", "generative_family")
            }
            redacted["scenario"] = {"intent": declared["scenario"]["intent"]}
            redacted["features"] = [
                {
                    "name": feature["name"],
                    **({"units": feature["units"]} if "units" in feature else {}),
                    "law": {"dist": feature["law"]["dist"]},
                }
                for feature in declared["features"]
            ]
            redacted["response"] = {
                key: declared["response"][key]
                for key in ("family", "count_distribution", "link")
                if key in declared["response"]
            }
            spec_json = json.dumps(redacted, ensure_ascii=False)
            user = f"Environment specification JSON:\n{spec_json}\n"
        else:
            summary = summary_by_env[spec.env_id]
            spec_json = json.dumps(spec.model_dump(mode="json"), ensure_ascii=False)
            user = (
                f"Environment specification JSON:\n{spec_json}\n\n"
                f"Data summary JSON:\n{json.dumps(summary, ensure_ascii=False)}\n"
            )
        request_id = f"prompt-{idx:06d}"
        spec_by_request[request_id] = spec
        requests.append(
            (
                request_id,
                LLMRequest(
                    model=resolved_model,
                    system=system,
                    user=user,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                ),
            )
        )

    receipt_required = resolved_backend_name.strip().lower() == "bedrock_converse" or isinstance(
        llm, BedrockConverseBackend
    )
    outputs = (
        {}
        if receipt_required
        else execute_requests_raw(
            llm,
            requests=requests,
            use_batch=use_batch,
            state_dir=stage_dir,
            allow_partial_batch_failures=True,
        )
    )

    # Include every verified checkpoint before a new call: an interruption must
    # not drop completed environments that appear later in the canonical order.
    records: list[PromptRecord] = list(cached.values())
    skipped: list[dict[str, str]] = []
    for request_id, request in requests:
        spec = spec_by_request[request_id]
        if receipt_required:
            # Finish all data-file reads before the paid call, so an appendix
            # failure cannot discard an already completed provider response.
            appendix = ""
            if _rt.name == "pymc":
                appendix = _pymc_data_schema_appendix(
                    spec=spec,
                    data_ref=data_ref_by_env.get(spec.env_id),
                    run_dir=paths.run_dir,
                )
            elif _rt.name == "lazyppl":
                appendix = _lazy_ppl_schema_appendix(
                    spec=spec,
                    data_ref=data_ref_by_env.get(spec.env_id),
                    run_dir=paths.run_dir,
                )
            outputs[request_id] = retry_converse_call(
                partial(llm.complete, request),
                on_throttle=partial(
                    persist_throttle_event,
                    stage_dir / "throttle-log.jsonl",
                    spec.env_id,
                ),
            )
        if request_id not in outputs:
            skipped.append(_missing_prompt_skip(request_id, spec))
            continue
        if receipt_required:
            prompt_text = outputs[request_id].content
            if _rt.name in {"pymc", "lazyppl"}:
                prompt_text = (prompt_text.rstrip() + appendix).rstrip()
            persist_prompt_receipt(receipts_path, spec.env_id, prompt_text, outputs[request_id])
        else:
            prompt_text = _prompt_text_from_output(
                output=outputs[request_id],
                runtime=_rt,
                spec=spec,
                data_ref=data_ref_by_env.get(spec.env_id),
                run_dir=paths.run_dir,
            )
        if not prompt_text.strip():
            skipped.append(_empty_prompt_skip(request_id, spec))
            continue
        record = build_prompt_record(
            request_id=request_id,
            spec=spec,
            prompt_text=prompt_text,
            runtime=_rt,
        )
        if prompt_rule == "no-numeric-summaries-v1":
            record.metadata["prompt_rule"] = prompt_rule
        records.append(record)
        if receipt_required:
            write_jsonl(
                stage_dir / "prompts.jsonl",
                [rec.model_dump(mode="json") for rec in records],
            )

    out_path = stage_dir / "prompts.jsonl"
    skipped_path = stage_dir / "skipped_prompts.jsonl"
    if not resume or requests or not out_path.exists():
        write_jsonl(out_path, [rec.model_dump(mode="json") for rec in records])
    write_jsonl(skipped_path, skipped)
    if requests and not records:
        raise RuntimeError("Stage D produced no non-empty prompts")
    manifest = {
        "status": "success",
        "count": len(records),
        "count_requested": len(specs),
        "skipped_count": len(skipped),
        "output": str(out_path),
        "skipped_output": str(skipped_path),
        "backend": resolved_backend_name,
        "model": resolved_model,
        "configured_reasoning_effort": configured_reasoning_effort,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "use_batch": use_batch,
        "runtime": _rt.name,
    }
    if receipt_required:
        manifest["receipts_output"] = str(receipts_path)
        manifest["throttle_log_output"] = str(stage_dir / "throttle-log.jsonl")
    if resume:
        manifest["resume"] = True
        manifest["count_reused"] = sum(spec.env_id in cached for spec in specs)
        manifest["count_called"] = len(requests)
    write_json(stage_dir / "manifest.json", manifest)
    return StageResult(stage=stage, status="success", outputs=manifest)

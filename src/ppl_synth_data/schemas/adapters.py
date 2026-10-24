from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from synthstats_schemas.trace import (
    GenerationRecord,
    PromptView,
    RuntimeDescriptor,
    TraceRecord,
    TraceStep,
    TrialRecord,
)

from ppl_synth_data.eval.task_models import EvaluationSpec, ObservationSchema, TaskSpec
from ppl_synth_data.runtime_registry import (
    get_runtime_spec,
    normalize_runtime_name,
    runtime_name_from_metadata,
    runtime_version_from_metadata,
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        numeric = _number(raw)
        if numeric is not None:
            out[str(key)] = numeric
    return out


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _json_string_list(value: Any) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if value is not None}


def _set_if_present(metadata: dict[str, Any], key: str, value: Any) -> None:
    """Fold a taxonomy value (tool_call_id/evidence_kind/truth_owner/semantic_status)
    into a TraceStep metadata dict.

    synthstats_schemas.trace.TraceStep has no first-class fields for these
    (unlike the now-removed producer-owned shadow TraceStep it replaced), so
    every producer maps them into `metadata` under the same key names instead.
    """
    if value is not None:
        metadata[key] = value


def runtime_descriptor_from_provenance(
    provenance: Mapping[str, Any] | None,
) -> RuntimeDescriptor | None:
    if provenance is None:
        return None
    runtime_name = normalize_runtime_name(_string(provenance.get("runtime_name")))
    runtime_version = _string(provenance.get("runtime_version"))
    if runtime_name is None and runtime_version is None:
        return None
    execution_backend = None
    if runtime_name is not None:
        execution_backend = get_runtime_spec(runtime_name).execution_backend
    return RuntimeDescriptor(
        name=runtime_name or "unknown",
        version=runtime_version,
        execution_backend=execution_backend,
    )


def runtime_descriptor_from_dataset_row(
    dataset_row: Mapping[str, Any],
) -> RuntimeDescriptor | None:
    runtime_name = runtime_name_from_metadata(dataset_row)
    if runtime_name is None:
        return None
    runtime_spec = get_runtime_spec(runtime_name)
    version = runtime_version_from_metadata(dataset_row, runtime_name)
    return RuntimeDescriptor(
        name=runtime_spec.name,
        version=version,
        execution_backend=runtime_spec.execution_backend,
    )


def _runtime_descriptor_from_language(language: str | None) -> RuntimeDescriptor | None:
    if language is None:
        return None
    normalized = normalize_runtime_name(language)
    if normalized is None:
        return RuntimeDescriptor(name=language)
    runtime_spec = get_runtime_spec(normalized)
    return RuntimeDescriptor(
        name=runtime_spec.name,
        execution_backend=runtime_spec.execution_backend,
    )


def task_spec_from_env_spec(
    env_spec: Mapping[str, Any],
    *,
    runtime_name: str | None = None,
    runtime_version: str | None = None,
) -> TaskSpec:
    io_schema = _mapping(env_spec.get("io_schema"))
    data_plan = _mapping(env_spec.get("data_plan"))

    runtime = None
    normalized_runtime_name = normalize_runtime_name(runtime_name)
    if normalized_runtime_name is not None or runtime_version is not None:
        execution_backend = None
        if normalized_runtime_name is not None:
            execution_backend = get_runtime_spec(normalized_runtime_name).execution_backend
        runtime = RuntimeDescriptor(
            name=normalized_runtime_name or "unknown",
            version=runtime_version,
            execution_backend=execution_backend,
        )

    return TaskSpec(
        task_id=str(env_spec["env_id"]),
        title=str(env_spec.get("env_name") or env_spec["env_id"]),
        domain=_string(env_spec.get("domain")),
        complexity=_string(env_spec.get("complexity")),
        model_type=_string(env_spec.get("model_type")),
        generative_family=_string(env_spec.get("generative_family")),
        key_concepts=_string_list(env_spec.get("key_concepts")),
        prompt_guidance=_dict(env_spec.get("prompt_guidance")),
        observation_schema=ObservationSchema(
            input_type=_string(io_schema.get("input_type")),
            input_description=_string(io_schema.get("input_description")),
            output_type=_string(io_schema.get("output_type")),
            target_name=_string(io_schema.get("target_name")),
            feature_names=_string_list(io_schema.get("feature_names")),
            grouping=_string_list(data_plan.get("grouping")),
            covariates=_string_list(data_plan.get("covariates")),
            n_observations=_int_value(data_plan.get("n_observations")),
        ),
        evaluation_spec=EvaluationSpec(
            primary_metric="semantic_validation",
            metrics=["runtime_validity", "semantic_fidelity"],
            split_policy="deterministic_env_split",
            verifier="validator_cascade",
            metadata={"source": "ppl_synth_data.env_spec"},
        ),
        runtime=runtime,
        metadata={"generative_params": _dict(env_spec.get("generative_params"))},
    )


def prompt_view_from_prompt_record(prompt_record: Mapping[str, Any]) -> PromptView:
    provenance = _dict(prompt_record.get("provenance"))
    return PromptView(
        view_id=str(prompt_record["id"]),
        task_id=_string(prompt_record.get("env_id")),
        split=_string(prompt_record.get("split")),
        content=str(prompt_record.get("prompt_text") or ""),
        provenance=provenance,
        metadata=_dict(prompt_record.get("metadata")),
    )


def generation_record_from_completion_record(
    completion_record: Mapping[str, Any],
) -> GenerationRecord:
    provenance = _dict(completion_record.get("provenance"))
    runtime = runtime_descriptor_from_provenance(provenance)
    validation = _dict(completion_record.get("validation"))

    prompt = PromptView(
        view_id=str(completion_record.get("prompt_id") or completion_record["id"]),
        task_id=_string(completion_record.get("env_id")),
        split=_string(completion_record.get("split")),
        content=str(completion_record.get("prompt_text") or ""),
        provenance=provenance,
    )

    return GenerationRecord(
        record_id=str(completion_record["id"]),
        task_id=_string(completion_record.get("env_id")),
        prompt_id=_string(completion_record.get("prompt_id")),
        split=_string(completion_record.get("split")),
        runtime=runtime,
        prompt=prompt,
        raw_output=str(completion_record.get("raw_completion") or ""),
        candidate_outputs=_string_list(completion_record.get("candidate_outputs")),
        provider_response=_dict(completion_record.get("provider_response")),
        parsed_program=_string(completion_record.get("parsed_program")),
        validation=validation,
        metadata=_dict(completion_record.get("metadata")),
    )


def _message_value(message: Any, key: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(key)
    return getattr(message, key, None)


def _message_metadata(message: Any) -> dict[str, Any]:
    raw = _message_value(message, "metadata")
    return _dict(raw)


def _message_taxonomy_value(
    message: Any,
    metadata: Mapping[str, Any],
    key: str,
) -> str | None:
    return _string(_message_value(message, key)) or _string(metadata.get(key))


def trace_record_from_messages(
    messages: Sequence[Any],
    *,
    trace_id: str,
    task_id: str | None = None,
    trial_id: str | None = None,
    artifacts: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TraceRecord:
    steps: list[TraceStep] = []
    for index, message in enumerate(messages):
        raw_step_metadata = _message_metadata(message)
        step_metadata = {
            key: value
            for key, value in raw_step_metadata.items()
            if key not in {"evidence_kind", "truth_owner", "semantic_status"}
        }
        _set_if_present(
            step_metadata,
            "tool_call_id",
            _string(_message_value(message, "tool_call_id")),
        )
        _set_if_present(
            step_metadata,
            "evidence_kind",
            _message_taxonomy_value(message, raw_step_metadata, "evidence_kind"),
        )
        _set_if_present(
            step_metadata,
            "truth_owner",
            _message_taxonomy_value(message, raw_step_metadata, "truth_owner"),
        )
        _set_if_present(
            step_metadata,
            "semantic_status",
            _message_taxonomy_value(message, raw_step_metadata, "semantic_status"),
        )
        steps.append(
            TraceStep(
                step_id=f"step-{index:04d}",
                role=str(_message_value(message, "role") or "unknown"),
                content=str(_message_value(message, "content") or ""),
                reasoning_content=_string(_message_value(message, "reasoning_content")),
                metadata=step_metadata,
            )
        )
    return TraceRecord(
        trace_id=trace_id,
        task_id=task_id,
        trial_id=trial_id,
        steps=steps,
        artifacts=_dict(artifacts),
        metadata=_dict(metadata),
    )


def _trace_step_from_validation_step(
    step_payload: Mapping[str, Any],
    *,
    index: int,
) -> TraceStep:
    raw_metadata = _dict(step_payload.get("metadata"))
    metadata = {
        **(
            {"attempt": attempt}
            if (attempt := _int_value(step_payload.get("attempt"))) is not None
            else {}
        ),
        **({"phase": phase} if (phase := _string(step_payload.get("phase"))) is not None else {}),
        **({"ok": ok} if isinstance((ok := step_payload.get("ok")), bool) else {}),
        **{
            key: value
            for key, value in raw_metadata.items()
            if key not in {"evidence_kind", "truth_owner", "semantic_status"}
        },
    }
    _set_if_present(
        metadata,
        "evidence_kind",
        _message_taxonomy_value(step_payload, raw_metadata, "evidence_kind"),
    )
    _set_if_present(
        metadata,
        "truth_owner",
        _message_taxonomy_value(step_payload, raw_metadata, "truth_owner"),
    )
    _set_if_present(
        metadata,
        "semantic_status",
        _message_taxonomy_value(step_payload, raw_metadata, "semantic_status"),
    )

    return TraceStep(
        step_id=f"step-{index:04d}",
        role="validator",
        content=_string(step_payload.get("message")) or "",
        metadata=metadata,
    )


def _trace_record_from_validation_trace(
    dataset_row: Mapping[str, Any],
    *,
    trace_id: str,
) -> TraceRecord | None:
    validation = _json_dict(dataset_row.get("validation_json"))
    trace_payload = _mapping(validation.get("trace"))
    raw_steps = trace_payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    return TraceRecord(
        trace_id=trace_id,
        task_id=_string(dataset_row.get("env_id")),
        trial_id=_string(dataset_row.get("id")),
        steps=[
            _trace_step_from_validation_step(_mapping(step_payload), index=index)
            for index, step_payload in enumerate(raw_steps)
        ],
        metadata={
            "source": "ppl_synth_data.dataset_row",
            "trace_origin": "validation_json.trace",
        },
    )


def _enrich_empty_trace_record(
    trace: TraceRecord,
    dataset_row: Mapping[str, Any],
) -> TraceRecord:
    validation_trace = _trace_record_from_validation_trace(
        dataset_row,
        trace_id=trace.trace_id,
    )
    if validation_trace is None or validation_trace.steps == []:
        return trace

    merged_metadata = dict(trace.metadata)
    merged_metadata["trace_origin"] = "trace_json.empty+validation_json.trace"

    return TraceRecord(
        trace_id=trace.trace_id,
        task_id=trace.task_id or validation_trace.task_id,
        trial_id=trace.trial_id or validation_trace.trial_id,
        steps=validation_trace.steps,
        artifacts=trace.artifacts or validation_trace.artifacts,
        metadata=merged_metadata,
    )


def _validate_trace_dataset_join_keys(
    trace: TraceRecord,
    dataset_row: Mapping[str, Any],
) -> None:
    expected_pairs = (
        ("trace_id", _string(dataset_row.get("trace_id")), trace.trace_id),
        ("task_id", _string(dataset_row.get("env_id")), trace.task_id),
        ("trial_id", _string(dataset_row.get("id")), trace.trial_id),
    )
    for key, expected, actual in expected_pairs:
        if expected is not None and actual is not None and actual != expected:
            raise ValueError(
                f"dataset row {key} does not match trace_json.{key}: {expected!r} != {actual!r}"
            )


def trial_record_from_agent_result(
    agent_result: Mapping[str, Any],
    *,
    trial_id: str,
    trace_id: str | None = None,
) -> TrialRecord:
    task_config = _mapping(agent_result.get("task_config"))
    env_cfg = _mapping(task_config.get("env"))
    raw_language = _string(agent_result.get("language"))
    runtime = _runtime_descriptor_from_language(raw_language)
    language = runtime.name if runtime is not None else raw_language
    return TrialRecord(
        trial_id=trial_id,
        task_id=_string(env_cfg.get("name")),
        language=language,
        program=_string(agent_result.get("program")),
        reward=_number(agent_result.get("reward")),
        runtime=runtime,
        trace_id=trace_id,
        artifacts={"task_config": _dict(task_config)},
    )


def trial_record_from_dataset_row(
    dataset_row: Mapping[str, Any],
) -> TrialRecord:
    runtime = runtime_descriptor_from_dataset_row(dataset_row)
    return TrialRecord(
        trial_id=str(dataset_row["id"]),
        task_id=_string(dataset_row.get("env_id")),
        language=runtime.name if runtime is not None else None,
        program=_string(dataset_row.get("completion")),
        runtime=runtime,
        trace_id=_string(dataset_row.get("trace_id")),
        artifacts={
            "prompt_id": _string(dataset_row.get("prompt_id")),
            "split": _string(dataset_row.get("split")),
            "prompt": _string(dataset_row.get("prompt")),
            "env_name": _string(dataset_row.get("env_name")),
            "data_ref": _string(dataset_row.get("data_ref")),
            "provenance": _json_dict(dataset_row.get("provenance_json")),
            "validation": _json_dict(dataset_row.get("validation_json")),
        },
        metadata={"source": "ppl_synth_data.dataset_row"},
    )


def generation_record_from_dataset_row(
    dataset_row: Mapping[str, Any],
) -> GenerationRecord:
    runtime = runtime_descriptor_from_dataset_row(dataset_row)
    provenance = _json_dict(dataset_row.get("provenance_json"))
    prompt_provenance = _dict(provenance.get("provenance")) if provenance else {}
    prompt_metadata = _dict(provenance.get("metadata")) if provenance else {}

    return GenerationRecord(
        record_id=str(dataset_row["id"]),
        task_id=_string(dataset_row.get("env_id")),
        prompt_id=_string(dataset_row.get("prompt_id")),
        split=_string(dataset_row.get("split")),
        runtime=runtime,
        prompt=PromptView(
            view_id=_string(dataset_row.get("prompt_id")) or str(dataset_row["id"]),
            task_id=_string(dataset_row.get("env_id")),
            split=_string(dataset_row.get("split")),
            content=_string(dataset_row.get("prompt")) or "",
            provenance=prompt_provenance,
            metadata=prompt_metadata,
        ),
        raw_output=_string(dataset_row.get("completion")) or "",
        candidate_outputs=_json_string_list(dataset_row.get("candidate_outputs_json")),
        parsed_program=None,
        validation=_json_dict(dataset_row.get("validation_json")),
        metadata={
            "source": "ppl_synth_data.dataset_row",
            "env_name": _string(dataset_row.get("env_name")),
            "data_ref": _string(dataset_row.get("data_ref")),
            "trace_id": _string(dataset_row.get("trace_id")),
        },
    )


def trace_record_from_dataset_row(
    dataset_row: Mapping[str, Any],
) -> TraceRecord | None:
    trace_payload = _json_dict(dataset_row.get("trace_json"))
    if trace_payload:
        trace = TraceRecord.model_validate(trace_payload)
        trace = _enrich_empty_trace_record(trace, dataset_row)
        _validate_trace_dataset_join_keys(trace, dataset_row)
        return trace

    trace_id = _string(dataset_row.get("trace_id"))
    if trace_id is None:
        return None

    validation_trace = _trace_record_from_validation_trace(
        dataset_row,
        trace_id=trace_id,
    )
    if validation_trace is not None:
        return validation_trace

    return TraceRecord(
        trace_id=trace_id,
        task_id=_string(dataset_row.get("env_id")),
        trial_id=_string(dataset_row.get("id")),
        metadata={"source": "ppl_synth_data.dataset_row"},
    )


__all__ = [
    "generation_record_from_completion_record",
    "generation_record_from_dataset_row",
    "prompt_view_from_prompt_record",
    "runtime_descriptor_from_dataset_row",
    "runtime_descriptor_from_provenance",
    "task_spec_from_env_spec",
    "trace_record_from_dataset_row",
    "trace_record_from_messages",
    "trial_record_from_agent_result",
    "trial_record_from_dataset_row",
]

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ppl_synth_data.runtime_registry import (
    get_runtime_spec,
    normalize_runtime_name,
    runtime_name_from_metadata,
    runtime_version_from_metadata,
    supported_runtime_names,
)
from ppl_synth_data.specs.models import (
    CompletionRecord,
    PromptProvenance,
    PromptRecord,
)

_RUNTIME_VERSION_KEYS = {
    runtime_name: get_runtime_spec(runtime_name).metadata_version_key
    for runtime_name in supported_runtime_names()
}
_CONTRACT_METADATA_KEYS = {
    "env_name",
    "domain",
    "complexity",
    "model_type",
    "generative_family",
    "key_concepts",
    "has_observed_data",
    "source_category",
    "source_url_or_repo",
    "license_signal",
    "adaptation_mode",
    "source_retrieval_date",
    "validation",
    "env_id",
    "split",
    *_RUNTIME_VERSION_KEYS.values(),
}


def infer_runtime_name_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    return runtime_name_from_metadata(metadata)


def _metadata_str(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _metadata_bool(metadata: Mapping[str, Any], key: str) -> bool | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    return None


def _metadata_str_list(metadata: Mapping[str, Any], key: str) -> list[str] | None:
    value = metadata.get(key)
    if not isinstance(value, list) or not value:
        return None
    return [str(item) for item in value]


def _normalize_provenance(
    provenance: PromptProvenance | None,
    metadata: Mapping[str, Any],
) -> PromptProvenance | None:
    values = provenance.model_dump(mode="python") if provenance is not None else {}

    for key in (
        "env_name",
        "domain",
        "complexity",
        "model_type",
        "generative_family",
        "source_category",
        "source_url_or_repo",
        "license_signal",
        "adaptation_mode",
        "source_retrieval_date",
    ):
        if not values.get(key):
            meta_value = _metadata_str(metadata, key)
            if meta_value is not None:
                values[key] = meta_value

    if not values.get("key_concepts"):
        key_concepts = _metadata_str_list(metadata, "key_concepts")
        if key_concepts is not None:
            values["key_concepts"] = key_concepts

    if values.get("has_observed_data") is None:
        has_observed_data = _metadata_bool(metadata, "has_observed_data")
        if has_observed_data is not None:
            values["has_observed_data"] = has_observed_data

    runtime_name = normalize_runtime_name(values.get("runtime_name"))
    if runtime_name is not None:
        values["runtime_name"] = runtime_name
    if not runtime_name:
        runtime_name = infer_runtime_name_from_metadata(metadata)
        if runtime_name is not None:
            values["runtime_name"] = runtime_name

    if not values.get("runtime_version") and runtime_name in _RUNTIME_VERSION_KEYS:
        runtime_version = runtime_version_from_metadata(metadata, runtime_name)
        if runtime_version is not None:
            values["runtime_version"] = runtime_version

    if not values:
        return None
    return PromptProvenance.model_validate(values)


def _strip_contract_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key not in _CONTRACT_METADATA_KEYS}


def normalize_prompt_record(record: PromptRecord) -> PromptRecord:
    metadata = dict(record.metadata)
    return record.model_copy(
        update={
            "provenance": _normalize_provenance(record.provenance, metadata),
            "metadata": _strip_contract_metadata(metadata),
        }
    )


def normalize_completion_record(record: CompletionRecord) -> CompletionRecord:
    metadata = dict(record.metadata)

    env_id = record.env_id or _metadata_str(metadata, "env_id") or ""
    split = record.split or _metadata_str(metadata, "split") or ""

    return record.model_copy(
        update={
            "env_id": env_id,
            "split": split,
            "provenance": _normalize_provenance(record.provenance, metadata),
            "validation": record.validation,
            "metadata": _strip_contract_metadata(metadata),
        }
    )

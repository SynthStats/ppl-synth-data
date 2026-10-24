from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ppl_synth_data.harbor_timeouts import pymc_harbor_verifier_timeout_sec


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    execution_backend: str
    default_version: str
    metadata_version_key: str
    score_artifact_key: str
    harbor_task_id_prefix: str
    harbor_docker_image: str | None
    harbor_verifier_timeout_sec: float
    harbor_agent_timeout_sec: float
    aliases: tuple[str, ...] = ()


DEFAULT_RUNTIME_NAME = "pymc"

_RUNTIME_SPECS = (
    RuntimeSpec(
        name="pymc",
        execution_backend="python",
        default_version="v5",
        metadata_version_key="pymc_version",
        score_artifact_key="elpd",
        harbor_task_id_prefix="pymc",
        harbor_docker_image=None,
        harbor_verifier_timeout_sec=pymc_harbor_verifier_timeout_sec(),
        harbor_agent_timeout_sec=600.0,
    ),
    RuntimeSpec(
        name="lazyppl",
        execution_backend="haskell",
        default_version="lazyppl-1.0",
        metadata_version_key="lazyppl_version",
        score_artifact_key="log_lik_matrix",
        harbor_task_id_prefix="lazyppl",
        harbor_docker_image=None,
        harbor_verifier_timeout_sec=180.0,
        harbor_agent_timeout_sec=600.0,
        aliases=("haskell", "lazy-ppl", "lazy_ppl"),
    ),
)

_ALIASES: dict[str, str] = {}
_BY_NAME: dict[str, RuntimeSpec] = {}
for _spec in _RUNTIME_SPECS:
    _BY_NAME[_spec.name] = _spec
    _ALIASES[_spec.name] = _spec.name
    for _alias in _spec.aliases:
        _ALIASES[_alias] = _spec.name


def default_runtime_name() -> str:
    return DEFAULT_RUNTIME_NAME


def supported_runtime_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in _RUNTIME_SPECS)


def iter_runtime_specs() -> tuple[RuntimeSpec, ...]:
    return _RUNTIME_SPECS


def normalize_runtime_name(
    name: str | None,
    *,
    default: str | None = None,
) -> str | None:
    candidate = name if name is not None else default
    if candidate is None:
        return None
    normalized = _ALIASES.get(candidate.strip().lower())
    if normalized is not None:
        return normalized
    return None


def get_runtime_spec(name: str) -> RuntimeSpec:
    normalized = normalize_runtime_name(name)
    if normalized is None:
        available = ", ".join(sorted(supported_runtime_names()))
        raise ValueError(f"unknown runtime: {name!r}. Available runtimes: {available}")
    return _BY_NAME[normalized]


def runtime_name_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    inferred = [
        spec.name
        for spec in _RUNTIME_SPECS
        if isinstance(metadata.get(spec.metadata_version_key), str)
        and metadata.get(spec.metadata_version_key)
    ]
    if len(inferred) == 1:
        return inferred[0]
    return None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def runtime_name_from_payload(
    payload: Mapping[str, Any],
    *,
    default: str | None = None,
) -> str | None:
    shared_trial = payload.get("shared_trial_record")
    if isinstance(shared_trial, Mapping):
        language = normalize_runtime_name(_string(shared_trial.get("language")))
        if language is not None:
            return language
        runtime = shared_trial.get("runtime")
        if isinstance(runtime, Mapping):
            runtime_name = normalize_runtime_name(_string(runtime.get("name")))
            if runtime_name is not None:
                return runtime_name

    language = normalize_runtime_name(_string(payload.get("language")))
    if language is not None:
        return language

    metadata_runtime = runtime_name_from_metadata(payload)
    if metadata_runtime is not None:
        return metadata_runtime

    return normalize_runtime_name(default)


def runtime_version_from_metadata(
    metadata: Mapping[str, Any],
    runtime_name: str,
) -> str | None:
    spec = get_runtime_spec(runtime_name)
    version = metadata.get(spec.metadata_version_key)
    if isinstance(version, str) and version:
        return version
    return None


def runtime_spec_from_payload(
    payload: Mapping[str, Any],
    *,
    default: str | None = None,
) -> RuntimeSpec | None:
    runtime_name = runtime_name_from_payload(payload, default=default)
    if runtime_name is None:
        return None
    return get_runtime_spec(runtime_name)


def score_artifact_key_from_payload(
    payload: Mapping[str, Any],
    *,
    default: str | None = None,
) -> str | None:
    runtime_spec = runtime_spec_from_payload(payload, default=default)
    if runtime_spec is None:
        return None
    return runtime_spec.score_artifact_key


def harbor_task_id_prefix(runtime_name: str) -> str:
    return get_runtime_spec(runtime_name).harbor_task_id_prefix


def build_harbor_task_id(
    runtime_name: str,
    label: str,
    *,
    sequence: int | str,
    max_label_len: int = 40,
) -> str:
    prefix = harbor_task_id_prefix(runtime_name)
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    slug = slug[:max_label_len] or "task"
    sequence_value = int(sequence)
    return f"{prefix}-{slug}-{sequence_value:05d}"


__all__ = [
    "DEFAULT_RUNTIME_NAME",
    "RuntimeSpec",
    "build_harbor_task_id",
    "default_runtime_name",
    "get_runtime_spec",
    "harbor_task_id_prefix",
    "iter_runtime_specs",
    "normalize_runtime_name",
    "runtime_name_from_payload",
    "runtime_name_from_metadata",
    "runtime_spec_from_payload",
    "runtime_version_from_metadata",
    "score_artifact_key_from_payload",
    "supported_runtime_names",
]

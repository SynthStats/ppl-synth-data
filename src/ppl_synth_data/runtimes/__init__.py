from __future__ import annotations

from ppl_synth_data.runtime_registry import normalize_runtime_name, supported_runtime_names
from ppl_synth_data.runtimes.base import CompletionFormatResult, TargetRuntime, ValidationResult
from ppl_synth_data.runtimes.lazyppl_runtime import LazyPPLRuntime
from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime

__all__ = [
    "CompletionFormatResult",
    "TargetRuntime",
    "ValidationResult",
    "get_runtime",
]

_RUNTIME_FACTORIES: dict[str, type[TargetRuntime]] = {
    "pymc": PyMCRuntime,
    "lazyppl": LazyPPLRuntime,
}


def get_runtime(name: str) -> TargetRuntime:
    normalized_name = normalize_runtime_name(name)
    factory = _RUNTIME_FACTORIES.get(normalized_name or "")
    if factory is None:
        raise ValueError(
            f"unknown target runtime: {name!r}. Available: {sorted(supported_runtime_names())}"
        )
    return factory()

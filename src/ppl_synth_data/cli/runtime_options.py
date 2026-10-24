from __future__ import annotations

from collections.abc import Iterable

import typer

from ppl_synth_data.pipeline.artifacts import stage_manifest_runtime
from ppl_synth_data.pipeline.types import RunPaths
from ppl_synth_data.runtime_registry import default_runtime_name, normalize_runtime_name
from ppl_synth_data.runtimes import TargetRuntime, get_runtime

DEFAULT_RUNTIME_NAME = default_runtime_name()
_RUNTIME_INFERENCE_STAGES = ("G", "F", "E", "D")


def _existing_stage_runtimes(paths: RunPaths) -> set[str]:
    runtimes: set[str] = set()
    for stage in _RUNTIME_INFERENCE_STAGES:
        runtime = stage_manifest_runtime(paths.stage_dirs[stage])
        if runtime is not None:
            runtimes.add(normalize_runtime_name(runtime, default=runtime) or runtime)
    return runtimes


def _format_runtime_sources(runtimes: Iterable[str]) -> str:
    return ", ".join(sorted(runtimes))


def resolve_runtime(
    runtime_name: str | None,
    *,
    default_name: str = DEFAULT_RUNTIME_NAME,
    paths: RunPaths | None = None,
) -> tuple[str, TargetRuntime]:
    candidate = runtime_name
    if not candidate:
        discovered = _existing_stage_runtimes(paths) if paths is not None else set()
        if len(discovered) > 1:
            raise typer.BadParameter(
                "Conflicting runtime metadata exists in this run "
                f"({_format_runtime_sources(discovered)}). Pass --runtime explicitly."
            )
        if discovered:
            candidate = next(iter(discovered))
        else:
            candidate = default_name

    try:
        runtime = get_runtime(candidate)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--runtime") from exc
    return runtime.name, runtime

from __future__ import annotations

from pathlib import Path

from ppl_synth_data.pipeline.types import RunPaths
from ppl_synth_data.specs.io import read_envspecs
from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec


def canonical_envspecs_path(paths: RunPaths) -> Path:
    return paths.stage_dirs["B"] / "envspecs.jsonl"


def read_canonical_envspecs(paths: RunPaths) -> list[EnvSpec | FourFamilySpec]:
    path = canonical_envspecs_path(paths)
    if path.exists():
        return read_envspecs(path)
    return read_envspecs(paths.stage_dirs["A"] / "envspecs.jsonl")

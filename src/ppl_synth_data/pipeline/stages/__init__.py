from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ppl_synth_data.pipeline.types import StageRunner

_STAGE_MAP = {
    "run_stage_a": "ppl_synth_data.pipeline.stages.stage_a_envspec",
    "run_stage_b": "ppl_synth_data.pipeline.stages.stage_b_compile",
    "run_stage_c": "ppl_synth_data.pipeline.stages.stage_c_data",
    "run_stage_d": "ppl_synth_data.pipeline.stages.stage_d_prompts",
    "run_stage_e": "ppl_synth_data.pipeline.stages.stage_e_completions",
    "run_stage_f": "ppl_synth_data.pipeline.stages.stage_f_validate",
    "run_stage_g": "ppl_synth_data.pipeline.stages.stage_g_dataset",
}

__all__ = list(_STAGE_MAP)


def __getattr__(name: str) -> StageRunner:
    if name in _STAGE_MAP:
        import importlib

        mod = importlib.import_module(_STAGE_MAP[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

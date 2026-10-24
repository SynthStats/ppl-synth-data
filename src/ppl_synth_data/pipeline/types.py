from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ppl_synth_data.pipeline.split_policy import SplitPolicy, validate_visible_ratio


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    run_id: str
    run_dir: Path
    stage_dirs: dict[str, Path]


@dataclass
class StageResult:
    stage: str
    status: Literal["success", "completed_with_refusals", "skipped", "failed"]
    outputs: dict[str, Any] = field(default_factory=dict)
    message: str | None = None


StageRunner = Callable[[RunPaths], StageResult]


@dataclass
class PipelineRunConfig:
    """Groups the parameters of ``default_stage_runners`` into a single config object."""

    split_policy: SplitPolicy
    visible_ratio: float
    limit: int = 25
    backend_name: str | None = None
    use_batch: bool = True
    envspec_model: str | None = None
    prompt_model: str | None = None
    completion_model: str | None = None
    repair_model: str | None = None
    judge_model: str | None = None
    judge_backend_name: str | None = None
    seed: int = 0
    source_plan_path: Path | None = None
    repo_id: str | None = None
    hf_token: str | None = None
    private: bool = True
    revision: str | None = None
    commit_message: str | None = None
    skip_upload: bool = False
    target_language: str = "pymc"
    pymc_semantic_draws: int = 100
    pymc_semantic_tune: int = 100
    pymc_semantic_timeout_sec: float = 180.0
    acceptance_min_pymc_semantic_draws: int | None = None
    acceptance_min_pymc_semantic_tune: int | None = None

    def __post_init__(self) -> None:
        self.split_policy = SplitPolicy(self.split_policy)
        validate_visible_ratio(self.visible_ratio)

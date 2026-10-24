from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ppl_synth_data.config import Settings
from ppl_synth_data.pipeline.artifacts import run_manifest_payload, stage_is_resumable
from ppl_synth_data.pipeline.backends import (
    resolve_backend_name,
    resolve_judge_backend_name,
    resolve_stage_model,
)
from ppl_synth_data.pipeline.contracts import STAGE_ORDER, VALID_STAGE_STATUSES
from ppl_synth_data.pipeline.io import write_json
from ppl_synth_data.pipeline.types import PipelineRunConfig, RunPaths, StageResult, StageRunner

if TYPE_CHECKING:
    from ppl_synth_data.pipeline.stages.stage_a_envspec import run_stage_a as run_stage_a
    from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b as run_stage_b
    from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c as run_stage_c
    from ppl_synth_data.pipeline.stages.stage_d_prompts import run_stage_d as run_stage_d
    from ppl_synth_data.pipeline.stages.stage_e_completions import run_stage_e as run_stage_e
    from ppl_synth_data.pipeline.stages.stage_f_validate import run_stage_f as run_stage_f
    from ppl_synth_data.pipeline.stages.stage_g_dataset import run_stage_g as run_stage_g
    from ppl_synth_data.runtimes import get_runtime as get_runtime
    from ppl_synth_data.runtimes.base import TargetRuntime


@dataclass(frozen=True)
class _ResolvedRunnerConfig:
    backend_name: str
    envspec_model: str
    prompt_model: str
    completion_model: str
    repair_model: str
    judge_model: str
    judge_backend_name: str
    runtime: TargetRuntime
    settings: Settings


def run_all(
    paths: RunPaths,
    *,
    runners: dict[str, StageRunner],
    resume: bool = True,
    runtime_name: str | None = None,
) -> dict[str, StageResult]:
    results: dict[str, StageResult] = {}

    for stage in STAGE_ORDER:
        stage_dir = paths.stage_dirs[stage]
        manifest_path = stage_dir / "manifest.json"

        if resume:
            try:
                resumable = stage_is_resumable(
                    stage,
                    stage_dir,
                    requested_runtime=runtime_name,
                )
            except Exception as exc:
                error_payload = {
                    "status": "failed",
                    "stage": stage,
                    "error": f"resume check failed: {exc}",
                    "exception_type": type(exc).__name__,
                }
                write_json(manifest_path, error_payload)
                result = StageResult(
                    stage=stage,
                    status="failed",
                    outputs=error_payload,
                    message=str(exc),
                )
                results[stage] = result
                _mark_downstream_skipped(results, failed_stage=stage)
                break

            if resumable:
                results[stage] = StageResult(
                    stage=stage,
                    status="skipped",
                    outputs={"manifest": str(manifest_path)},
                    message="stage already marked successful",
                )
                continue

        try:
            result = runners[stage](paths)
        except Exception as exc:
            error_payload = {
                "status": "failed",
                "stage": stage,
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }
            write_json(manifest_path, error_payload)
            result = StageResult(
                stage=stage,
                status="failed",
                outputs=error_payload,
                message=str(exc),
            )

        if result.status not in VALID_STAGE_STATUSES:
            result = StageResult(
                stage=stage,
                status="failed",
                outputs={
                    "status": "failed",
                    "stage": stage,
                    "error": f"invalid stage status: {result.status}",
                },
                message=f"invalid stage status: {result.status}",
            )
            write_json(manifest_path, result.outputs)

        results[stage] = result
        if result.status == "failed":
            _mark_downstream_skipped(results, failed_stage=stage)
            break

    write_json(
        paths.run_dir / "run_manifest.json",
        run_manifest_payload(paths, results, runtime_name=runtime_name),
    )
    return results


def default_stage_runners(cfg: PipelineRunConfig) -> dict[str, StageRunner]:
    from ppl_synth_data.pipeline.stages.stage_a_envspec import run_stage_a
    from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b
    from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c
    from ppl_synth_data.pipeline.stages.stage_d_prompts import run_stage_d
    from ppl_synth_data.pipeline.stages.stage_e_completions import run_stage_e
    from ppl_synth_data.pipeline.stages.stage_f_validate import run_stage_f
    from ppl_synth_data.pipeline.stages.stage_g_dataset import run_stage_g
    from ppl_synth_data.runtimes import get_runtime

    runtime = get_runtime(cfg.target_language)
    resolved = _ResolvedRunnerConfig(
        backend_name=resolve_backend_name(cfg.backend_name),
        envspec_model=resolve_stage_model("envspec", cfg.envspec_model),
        prompt_model=resolve_stage_model("prompts", cfg.prompt_model),
        completion_model=resolve_stage_model("completions", cfg.completion_model),
        repair_model=resolve_stage_model("repair", cfg.repair_model),
        judge_model=resolve_stage_model("judge", cfg.judge_model),
        judge_backend_name=resolve_judge_backend_name(cfg.judge_backend_name),
        runtime=runtime,
        settings=Settings(),
    )

    return {
        "A": lambda paths: run_stage_a(
            paths,
            limit=cfg.limit,
            seed=cfg.seed,
            backend_name=resolved.backend_name,
            use_batch=cfg.use_batch,
            model=resolved.envspec_model,
            source_plan_path=cfg.source_plan_path,
            runtime_target=resolved.runtime.name,
            settings=resolved.settings,
        ),
        "B": lambda paths: run_stage_b(paths),
        "C": lambda paths: run_stage_c(
            paths,
            split_policy=cfg.split_policy,
            seed=cfg.seed,
            limit=cfg.limit,
            visible_ratio=cfg.visible_ratio,
        ),
        "D": lambda paths: run_stage_d(
            paths,
            limit=cfg.limit,
            backend_name=resolved.backend_name,
            use_batch=cfg.use_batch,
            model=resolved.prompt_model,
            runtime=resolved.runtime,
            settings=resolved.settings,
        ),
        "E": lambda paths: run_stage_e(
            paths,
            limit=cfg.limit,
            backend_name=resolved.backend_name,
            use_batch=cfg.use_batch,
            model=resolved.completion_model,
            runtime=resolved.runtime,
            settings=resolved.settings,
        ),
        "F": lambda paths: run_stage_f(
            paths,
            backend_name=resolved.backend_name,
            model=resolved.repair_model,
            judge_model=resolved.judge_model,
            judge_backend_name=resolved.judge_backend_name,
            runtime=resolved.runtime,
            settings=resolved.settings,
            pymc_semantic_draws=cfg.pymc_semantic_draws,
            pymc_semantic_tune=cfg.pymc_semantic_tune,
            pymc_semantic_timeout_sec=cfg.pymc_semantic_timeout_sec,
            acceptance_min_pymc_semantic_draws=(cfg.acceptance_min_pymc_semantic_draws),
            acceptance_min_pymc_semantic_tune=cfg.acceptance_min_pymc_semantic_tune,
        ),
        "G": lambda paths: run_stage_g(
            paths,
            repo_id=cfg.repo_id,
            hf_token=cfg.hf_token,
            private=cfg.private,
            revision=cfg.revision,
            commit_message=cfg.commit_message,
            skip_upload=cfg.skip_upload,
            runtime=resolved.runtime,
        ),
    }


def _mark_downstream_skipped(results: dict[str, StageResult], *, failed_stage: str) -> None:
    failed_index = STAGE_ORDER.index(failed_stage)
    for pending in STAGE_ORDER[failed_index + 1 :]:
        results[pending] = StageResult(
            stage=pending,
            status="skipped",
            outputs={},
            message=f"not run because stage {failed_stage} did not succeed",
        )

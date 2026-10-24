from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

import ppl_synth_data.cli as _cli
from ppl_synth_data.cli import _resolve_paths, run_app
from ppl_synth_data.cli.runtime_options import resolve_runtime
from ppl_synth_data.pipeline.split_policy import SplitPolicy


# Each CLI parameter below maps 1:1 to a PipelineRunConfig field.
# Typer requires explicit parameter declarations for CLI flags; there's no
# way to derive them from the dataclass automatically while preserving help
# text and type annotations. This duplication is intentional.
@run_app.command("all")
def run_all_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier")],
    split_policy: Annotated[
        SplitPolicy,
        typer.Option("--split-policy", help="Stage C split policy"),
    ],
    visible_ratio: Annotated[
        float,
        typer.Option("--visible-ratio", help="Visible data ratio for Stage C"),
    ],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Optional run root override"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Max items for A/D/E and env cap for C", min=1),
    ] = 25,
    seed: Annotated[int, typer.Option(help="Base random seed for stage C")] = 0,
    backend: Annotated[str | None, typer.Option(help="LLM backend for A/D/E/F")] = None,
    use_batch: Annotated[bool, typer.Option("--use-batch/--sync")] = True,
    envspec_model: Annotated[str | None, typer.Option("--envspec-model")] = None,
    prompt_model: Annotated[str | None, typer.Option("--prompt-model")] = None,
    completion_model: Annotated[str | None, typer.Option("--completion-model")] = None,
    repair_model: Annotated[str | None, typer.Option("--repair-model")] = None,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Model for leakage judge")
    ] = None,
    judge_backend: Annotated[
        str | None, typer.Option("--judge-backend", help="Backend for leakage judge")
    ] = None,
    source_plan: Annotated[
        Path | None,
        typer.Option(
            "--source-plan",
            help="Optional JSON/JSONL source-plan slots for Stage A source provenance",
        ),
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", help="HF dataset repo id"),
    ] = None,
    skip_upload: Annotated[
        bool,
        typer.Option("--skip-upload", help="Build local artifacts only"),
    ] = False,
    private: Annotated[
        bool,
        typer.Option("--private/--public", help="Hub visibility"),
    ] = True,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    commit_message: Annotated[str | None, typer.Option("--commit-message")] = None,
    pymc_semantic_draws: Annotated[
        int,
        typer.Option(
            "--pymc-semantic-draws",
            help="Posterior draws for Stage F PyMC semantic validation",
            min=1,
        ),
    ] = 100,
    pymc_semantic_tune: Annotated[
        int,
        typer.Option(
            "--pymc-semantic-tune",
            help="Tuning steps for Stage F PyMC semantic validation",
            min=1,
        ),
    ] = 100,
    pymc_semantic_timeout_sec: Annotated[
        float,
        typer.Option(
            "--pymc-semantic-timeout-sec",
            help="Timeout for each Stage F PyMC semantic validation subprocess",
            min=1.0,
        ),
    ] = 180.0,
    acceptance_min_pymc_semantic_draws: Annotated[
        int | None,
        typer.Option(
            "--acceptance-min-pymc-semantic-draws",
            help="Fail before Stage F if PyMC semantic draws are below this budget",
            min=1,
        ),
    ] = None,
    acceptance_min_pymc_semantic_tune: Annotated[
        int | None,
        typer.Option(
            "--acceptance-min-pymc-semantic-tune",
            help="Fail before Stage F if PyMC semantic tune steps are below this budget",
            min=1,
        ),
    ] = None,
    runtime: Annotated[
        str | None,
        typer.Option("--runtime", help="Target runtime: pymc or lazyppl"),
    ] = None,
) -> None:
    settings, paths = _resolve_paths(run_id, run_root)
    if not skip_upload and not repo_id:
        raise typer.BadParameter("--repo-id is required unless --skip-upload is set")
    runtime_name, _ = resolve_runtime(runtime, paths=paths)

    hf_token = settings.require_hf_token() if not skip_upload else None
    config = _cli.PipelineRunConfig(
        split_policy=split_policy,
        visible_ratio=visible_ratio,
        limit=limit,
        backend_name=backend,
        use_batch=use_batch,
        envspec_model=envspec_model,
        prompt_model=prompt_model,
        completion_model=completion_model,
        repair_model=repair_model,
        judge_model=judge_model,
        judge_backend_name=judge_backend,
        seed=seed,
        source_plan_path=source_plan,
        repo_id=repo_id,
        hf_token=hf_token,
        private=private,
        revision=revision,
        commit_message=commit_message,
        skip_upload=skip_upload,
        target_language=runtime_name,
        pymc_semantic_draws=pymc_semantic_draws,
        pymc_semantic_tune=pymc_semantic_tune,
        pymc_semantic_timeout_sec=pymc_semantic_timeout_sec,
        acceptance_min_pymc_semantic_draws=acceptance_min_pymc_semantic_draws,
        acceptance_min_pymc_semantic_tune=acceptance_min_pymc_semantic_tune,
    )
    runners = _cli.default_stage_runners(config)
    results = _cli.run_all(paths, runners=runners, resume=resume, runtime_name=runtime_name)
    summary = {stage: result.status for stage, result in results.items()}
    typer.echo(json.dumps(summary, indent=2))

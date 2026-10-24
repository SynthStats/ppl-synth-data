from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from ppl_synth_data.cli import _resolve_paths, stage_app
from ppl_synth_data.cli.runtime_options import resolve_runtime
from ppl_synth_data.pipeline.split_policy import SplitPolicy


@stage_app.command("f-declared")
def run_stage_f_declared_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier")],
    run_root: Annotated[Path, typer.Option("--run-root", help="Run root with compiled specs")],
) -> None:
    """Execute declared four-family golds on visible data without fitting or LLM calls."""
    from ppl_synth_data.pipeline.artifacts import resolve_run_paths
    from ppl_synth_data.pipeline.stages.stage_f_declared import run_stage_f_declared

    result = run_stage_f_declared(resolve_run_paths(run_root, run_id))
    typer.echo(json.dumps(result.outputs, indent=2))


if TYPE_CHECKING:
    from ppl_synth_data.pipeline.stages.stage_a_envspec import (
        run_stage_a as run_stage_a,  # noqa: F401
    )
    from ppl_synth_data.pipeline.stages.stage_b_compile import (
        run_stage_b as run_stage_b,  # noqa: F401
    )
    from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c as run_stage_c  # noqa: F401
    from ppl_synth_data.pipeline.stages.stage_d_prompts import (
        run_stage_d as run_stage_d,  # noqa: F401
    )
    from ppl_synth_data.pipeline.stages.stage_e_completions import (
        run_stage_e as run_stage_e,  # noqa: F401
    )
    from ppl_synth_data.pipeline.stages.stage_f_validate import (
        run_stage_f as run_stage_f,  # noqa: F401
    )
    from ppl_synth_data.pipeline.stages.stage_g_dataset import (
        run_stage_g as run_stage_g,  # noqa: F401
    )
    from ppl_synth_data.runtimes import get_runtime as get_runtime  # noqa: F401


@stage_app.command("a")
def run_stage_a_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Optional run root override"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Number of EnvSpecs to generate", min=1),
    ] = 25,
    seed: Annotated[int, typer.Option(help="Base seed for generation plan")] = 0,
    backend: Annotated[str | None, typer.Option(help="LLM backend (defaults from config)")] = None,
    model: Annotated[
        str | None, typer.Option(help="Model for stage A (defaults from config)")
    ] = None,
    source_plan: Annotated[
        Path | None,
        typer.Option(
            "--source-plan",
            help="Optional JSON/JSONL source-plan slots for source-provenance-aware EnvSpecs",
        ),
    ] = None,
    runtime_name: Annotated[
        str | None,
        typer.Option("--runtime", help="Filter source-plan rows by target runtime"),
    ] = None,
    use_batch: Annotated[bool, typer.Option("--use-batch/--sync")] = True,
) -> None:
    from ppl_synth_data.pipeline.stages.stage_a_envspec import run_stage_a

    _, paths = _resolve_paths(run_id, run_root)
    runtime_target = None
    if runtime_name is not None:
        runtime_target, _ = resolve_runtime(runtime_name, paths=paths)
    result = run_stage_a(
        paths,
        limit=limit,
        seed=seed,
        backend_name=backend,
        model=model,
        source_plan_path=source_plan,
        runtime_target=runtime_target,
        use_batch=use_batch,
    )
    typer.echo(json.dumps(result.outputs, indent=2))


@stage_app.command("b")
def run_stage_b_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Optional run root override"),
    ] = None,
) -> None:
    from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b

    _, paths = _resolve_paths(run_id, run_root)
    result = run_stage_b(paths)
    typer.echo(json.dumps(result.outputs, indent=2))


@stage_app.command("c")
def run_stage_c_cmd(
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
    seed: Annotated[int, typer.Option(help="Base seed")] = 0,
    limit: Annotated[
        int | None,
        typer.Option(help="Optional max env count", min=1),
    ] = None,
) -> None:
    from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c

    _, paths = _resolve_paths(run_id, run_root)
    result = run_stage_c(
        paths,
        split_policy=split_policy,
        seed=seed,
        limit=limit,
        visible_ratio=visible_ratio,
    )
    typer.echo(json.dumps(result.outputs, indent=2))


@stage_app.command("d")
def run_stage_d_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Optional run root override"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Optional max env count", min=1),
    ] = None,
    backend: Annotated[str | None, typer.Option(help="LLM backend (defaults from config)")] = None,
    model: Annotated[
        str | None, typer.Option(help="Model for stage D (defaults from config)")
    ] = None,
    use_batch: Annotated[bool, typer.Option("--use-batch/--sync")] = True,
    runtime_name: Annotated[
        str | None,
        typer.Option("--runtime", help="Target runtime: pymc or lazyppl"),
    ] = None,
) -> None:
    from ppl_synth_data.pipeline.stages.stage_d_prompts import run_stage_d

    _, paths = _resolve_paths(run_id, run_root)
    _, runtime = resolve_runtime(runtime_name, paths=paths)
    result = run_stage_d(
        paths,
        limit=limit,
        backend_name=backend,
        model=model,
        use_batch=use_batch,
        runtime=runtime,
    )
    typer.echo(json.dumps(result.outputs, indent=2))


@stage_app.command("e")
def run_stage_e_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Optional run root override"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Optional max prompt count", min=1),
    ] = None,
    backend: Annotated[str | None, typer.Option(help="LLM backend (defaults from config)")] = None,
    model: Annotated[
        str | None, typer.Option(help="Model for stage E (defaults from config)")
    ] = None,
    use_batch: Annotated[bool, typer.Option("--use-batch/--sync")] = True,
    runtime_name: Annotated[
        str | None,
        typer.Option("--runtime", help="Target runtime: pymc or lazyppl"),
    ] = None,
) -> None:
    from ppl_synth_data.pipeline.stages.stage_e_completions import run_stage_e

    _, paths = _resolve_paths(run_id, run_root)
    _, runtime = resolve_runtime(runtime_name, paths=paths)
    result = run_stage_e(
        paths,
        limit=limit,
        backend_name=backend,
        model=model,
        use_batch=use_batch,
        runtime=runtime,
    )
    typer.echo(json.dumps(result.outputs, indent=2))


@stage_app.command("f")
def run_stage_f_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Optional run root override"),
    ] = None,
    backend: Annotated[
        str | None, typer.Option(help="LLM backend for repairs (defaults from config)")
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Model for stage F repairs (defaults from config)")
    ] = None,
    judge_model: Annotated[
        str | None,
        typer.Option("--judge-model", help="Model for leakage judge (defaults from config)"),
    ] = None,
    judge_backend: Annotated[
        str | None,
        typer.Option("--judge-backend", help="Backend for leakage judge (defaults from config)"),
    ] = None,
    max_retries: Annotated[
        int,
        typer.Option(help="Validation repair retry budget", min=0),
    ] = 3,
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
    runtime_name: Annotated[
        str | None,
        typer.Option("--runtime", help="Target runtime: pymc or lazyppl"),
    ] = None,
) -> None:
    from ppl_synth_data.pipeline.stages.stage_f_validate import run_stage_f

    _, paths = _resolve_paths(run_id, run_root)
    _, runtime = resolve_runtime(runtime_name, paths=paths)
    result = run_stage_f(
        paths,
        backend_name=backend,
        model=model,
        judge_model=judge_model,
        judge_backend_name=judge_backend,
        max_retries=max_retries,
        runtime=runtime,
        pymc_semantic_draws=pymc_semantic_draws,
        pymc_semantic_tune=pymc_semantic_tune,
        pymc_semantic_timeout_sec=pymc_semantic_timeout_sec,
        acceptance_min_pymc_semantic_draws=acceptance_min_pymc_semantic_draws,
        acceptance_min_pymc_semantic_tune=acceptance_min_pymc_semantic_tune,
    )
    typer.echo(json.dumps(result.outputs, indent=2))


@stage_app.command("g")
def run_stage_g_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Optional run root override"),
    ] = None,
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", help="HF dataset repo id, e.g. org/name"),
    ] = None,
    skip_upload: Annotated[
        bool,
        typer.Option("--skip-upload", help="Build artifacts without pushing"),
    ] = False,
    private: Annotated[
        bool,
        typer.Option("--private/--public", help="Hub visibility"),
    ] = True,
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Optional target revision"),
    ] = None,
    commit_message: Annotated[
        str | None,
        typer.Option("--commit-message", help="Optional commit message"),
    ] = None,
    runtime_name: Annotated[
        str | None,
        typer.Option("--runtime", help="Target runtime: pymc or lazyppl"),
    ] = None,
) -> None:
    settings, paths = _resolve_paths(run_id, run_root)
    if not skip_upload and not repo_id:
        raise typer.BadParameter("--repo-id is required unless --skip-upload is set")

    from ppl_synth_data.pipeline.stages.stage_g_dataset import run_stage_g

    hf_token = settings.require_hf_token() if not skip_upload else None
    _, runtime = resolve_runtime(runtime_name, paths=paths)
    result = run_stage_g(
        paths,
        repo_id=repo_id,
        hf_token=hf_token,
        private=private,
        revision=revision,
        commit_message=commit_message,
        skip_upload=skip_upload,
        runtime=runtime,
    )
    typer.echo(json.dumps(result.outputs, indent=2))

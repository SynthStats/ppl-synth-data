from __future__ import annotations

from pathlib import Path

import typer

from ppl_synth_data.config import Settings
from ppl_synth_data.pipeline import (
    PipelineRunConfig,
    RunPaths,
    default_stage_runners,
    resolve_run_paths,
    run_all,
)

__all__ = [
    "PipelineRunConfig",
    "Settings",
    "default_stage_runners",
    "run_all",
]

app: typer.Typer = typer.Typer(help="ppl-synth-data CLI")
stage_app: typer.Typer = typer.Typer(help="Run single pipeline stages")
run_app: typer.Typer = typer.Typer(help="Run orchestrated pipeline flows")
audit_app: typer.Typer = typer.Typer(
    help="GPT-powered JSONL dataset audits via the OpenAI Batch API",
)

app.add_typer(stage_app, name="stage")
app.add_typer(run_app, name="run")
app.add_typer(audit_app, name="audit")


def _resolve_paths(run_id: str, run_root: Path | None) -> tuple[Settings, RunPaths]:
    settings = Settings()
    root = run_root if run_root is not None else Path(settings.run_root)
    try:
        paths = resolve_run_paths(root, run_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--run-id") from exc
    return settings, paths


# --- Side-effect imports for CLI command registration ---
# Import the command modules explicitly so their decorator side effects run
# without widening this package's public surface via wildcard re-exports.
from ppl_synth_data.cli import audit as _audit_module  # noqa: E402
from ppl_synth_data.cli import commands as _commands_module  # noqa: E402
from ppl_synth_data.cli import runs as _runs_module  # noqa: E402
from ppl_synth_data.cli import stages as _stages_module  # noqa: E402

_REGISTERED_CLI_MODULES = (
    _audit_module,
    _commands_module,
    _runs_module,
    _stages_module,
)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

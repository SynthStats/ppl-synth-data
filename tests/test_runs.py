from __future__ import annotations

from typer.testing import CliRunner

import ppl_synth_data.cli.runs as cli_runs
from ppl_synth_data.cli import app

runner = CliRunner()


def test_runs_module_registers_run_all_command() -> None:
    assert cli_runs.run_all_cmd.__name__ == "run_all_cmd"

    result = runner.invoke(app, ["run", "all", "--help"])

    assert result.exit_code == 0, result.output
    assert "--runtime" in result.stdout
    assert "--repo-id" in result.stdout

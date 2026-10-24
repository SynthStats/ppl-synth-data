from __future__ import annotations

import json

from typer.testing import CliRunner

import ppl_synth_data.cli.commands as cli_commands
from ppl_synth_data.cli import app

runner = CliRunner()


def test_commands_module_registers_doctor_command() -> None:
    assert cli_commands.doctor.__name__ == "doctor"

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "openai_api_key_set" in payload
    assert "hf_token_set" in payload

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ppl_synth_data.cli import app

pytestmark = pytest.mark.narrow_unit_not_acceptance

runner = CliRunner()


def test_doctor_redacts_secrets(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "sk-secret" not in result.stdout
    assert "hf-secret" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["openai_api_key_set"] is True
    assert payload["hf_token_set"] is True


def test_scan_leakage_works_without_repo_cwd(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Use a poisson model.", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["scan-leakage", str(prompt_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "poisson" in payload["hits"]

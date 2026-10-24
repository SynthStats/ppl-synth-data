from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from ppl_synth_data.cli import app

runner = CliRunner()


def _write_sample_data(tmp_path: Path) -> Path:
    record = {
        "prompt": "test prompt",
        "completion": (
            "<think>ok</think>\n"
            "<submit_program>\n"
            "def model(data):\n"
            "    import pymc as pm\n"
            "    with pm.Model() as m:\n"
            "        x = pm.Normal('x', 0, 1)\n"
            "    return m\n"
            "</submit_program>"
        ),
        "metadata": {"env_id": "test-env", "model_type": "generative", "domain": "test"},
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "val.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "test.jsonl").write_text("", encoding="utf-8")
    return tmp_path


def test_audit_prepare_dry_run(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_data(data_dir)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "audit",
            "prepare",
            "--data",
            str(data_dir),
            "--out",
            str(out_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Total entries: 1" in result.stdout
    assert "Dry run" in result.stdout
    assert not (out_dir / "batches.json").exists()


def test_audit_prepare_writes_batches(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_data(data_dir)

    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "audit",
            "prepare",
            "--data",
            str(data_dir),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "batches.json").exists()
    manifest = json.loads((out_dir / "batches.json").read_text(encoding="utf-8"))
    assert len(manifest) >= 1
    batch_file = out_dir / manifest[0]["file"]
    assert batch_file.exists()


def test_audit_prepare_does_not_import_runner_module(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_data(data_dir)

    out_dir = tmp_path / "out"
    sys.modules.pop("ppl_synth_data.audit.runner", None)

    result = runner.invoke(
        app,
        [
            "audit",
            "prepare",
            "--data",
            str(data_dir),
            "--out",
            str(out_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "ppl_synth_data.audit.runner" not in sys.modules


def test_audit_parse_requires_batches_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_data(data_dir)

    empty_out = tmp_path / "empty_out"
    empty_out.mkdir()

    result = runner.invoke(
        app,
        [
            "audit",
            "parse",
            "--out",
            str(empty_out),
            "--data",
            str(data_dir),
        ],
    )

    assert result.exit_code != 0
    combined = result.output + (str(result.exception) if result.exception else "")
    assert "batches.json" in combined


def test_audit_all_dry_run_stops_early(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_data(data_dir)

    result = runner.invoke(
        app,
        [
            "audit",
            "all",
            "--data",
            str(data_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Dry run" in result.stdout


def test_audit_runner_imports_cleanly() -> None:
    """Regression: verify audit.runner can be imported without errors."""
    from ppl_synth_data.audit import runner  # noqa: F401

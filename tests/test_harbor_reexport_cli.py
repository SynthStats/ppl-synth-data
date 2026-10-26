"""The package re-export CLI forwards explicit runtime and verifier controls."""

import json
from pathlib import Path

from typer.testing import CliRunner

from ppl_synth_data.cli import app
from ppl_synth_data.harbor import export_pymc


def test_reexport_cli_forwards_packages_and_runtime_options(monkeypatch, tmp_path):
    captured = {}

    def record(sources, out_dir, **kwargs):
        captured.update(sources=sources, out_dir=out_dir, **kwargs)
        return {"tasks_created": 2}

    monkeypatch.setattr(export_pymc, "reexport_harbor_tasks", record)
    result = CliRunner().invoke(
        app,
        [
            "reexport-harbor",
            "first",
            "second",
            "--out",
            str(tmp_path),
            "--validator-path",
            "validator.py",
            "--validator-sha256",
            "abc",
            "--attempt-id",
            "attempt-1",
            "--role-manifest",
            str(tmp_path / "ROLE-MANIFEST.json"),
            "--role-manifest-sha256",
            "d" * 64,
            "--exclude",
            "held",
            "--reference-workers",
            "2",
            "--allow-unlocked-reference-runtime",
            "--scorer-image",
            "sha256:image",
        ],
    )
    assert result.exit_code == 0, result.output
    # the CLI forwards both manifest arguments to the re-export. They are in the dict
    # comparison below, so a failure names which one did not arrive.
    assert json.loads(result.stdout) == {"tasks_created": 2}
    assert captured == {
        "sources": [Path("first"), Path("second")],
        "out_dir": tmp_path,
        "validator_path": Path("validator.py"),
        "validator_sha256": "abc",
        "excluded": ["held"],
        "reference_workers": 2,
        "allow_unlocked_reference_runtime": True,
        "scorer_image": "sha256:image",
        "role_manifest_path": tmp_path / "ROLE-MANIFEST.json",
        "role_manifest_sha256": "d" * 64,
        "attempt_id": "attempt-1",
    }

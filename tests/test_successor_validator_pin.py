"""Every exported task ships the reviewed successor verifier, byte for byte."""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path

from ppl_synth_data.harbor.export_pymc import export_harbor_tasks
from tests.test_harbor_export import _single_exported_task_dir, run_dir  # noqa: F401

SUCCESSOR_VALIDATOR_SHA256 = "232735577a24047b1743168741043668a734c48b5106dc3b13387d9ebe43cee3"


def test_packaged_validator_template_is_the_successor() -> None:
    template = resources.files("ppl_synth_data").joinpath("harbor/validator_template.py")

    assert hashlib.sha256(template.read_bytes()).hexdigest() == SUCCESSOR_VALIDATOR_SHA256


def test_exported_task_ships_the_successor_validator(
    run_dir: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    out = tmp_path / "harbor_out"
    export_harbor_tasks(
        run_dir,
        out,
        run_id="test-run",
        attempt_id="successor-pin-test",
        l6_draws=15,
        l6_tune=15,
        allow_unlocked_reference_runtime=True,
    )

    rendered = _single_exported_task_dir(out) / "tests" / "validator.py"

    assert hashlib.sha256(rendered.read_bytes()).hexdigest() == SUCCESSOR_VALIDATOR_SHA256

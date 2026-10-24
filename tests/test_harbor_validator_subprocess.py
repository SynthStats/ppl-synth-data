"""Run validator_template.py as a subprocess, matching Docker execution."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthstats_schemas import RewardArtifact

_VALIDATOR_REF = resources.files("ppl_synth_data").joinpath("harbor/validator_template.py")

# Worker mount and PID namespace isolation requires Linux.
pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="reducer role requires `unshare --mount --pid --fork` (Linux only)",
)


GOOD_CODE = """\
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        y = data["y"]
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
"""

BAD_CODE_POTENTIAL = """\
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        y = data["y"]
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        pm.Potential("hack", mu * 0)
    return m
"""


def _write_artifacts(tmp_path: Path, code: str) -> tuple[Path, Path, Path]:
    submission = tmp_path / "submission.py"
    submission.write_text(code, encoding="utf-8")

    df = pd.DataFrame({"x": np.linspace(0, 1, 10), "y": np.random.randn(10)})
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path, index=False)

    out_path = tmp_path / "reward.json"
    return submission, data_path, out_path


def _copy_validator(
    tmp_path: Path,
    *,
    l6_draws: int | None = None,
    l6_tune: int | None = None,
) -> Path:
    with resources.as_file(_VALIDATOR_REF) as validator:
        text = Path(validator).read_text(encoding="utf-8")
    if l6_draws is not None:
        text = text.replace("L6_DRAWS = 200", f"L6_DRAWS = {l6_draws}")
    if l6_tune is not None:
        text = text.replace("L6_TUNE = 200", f"L6_TUNE = {l6_tune}")
    copied = tmp_path / "validator.py"
    copied.write_text(text, encoding="utf-8")
    return copied


@pytest.mark.slow
def test_validator_subprocess_good_model(tmp_path: Path) -> None:
    submission, data_path, out_path = _write_artifacts(tmp_path, GOOD_CODE)
    holdout_path = tmp_path / "holdout.parquet"
    pd.DataFrame({"x": np.linspace(0, 1, 4), "y": np.random.randn(4)}).to_parquet(
        holdout_path, index=False
    )
    reference_path = tmp_path / "elpd_reference.json"
    reference_path.write_text(
        json.dumps({"gold_elpd": -3.0, "baseline_elpd": -10.0, "target_name": "y"})
    )
    # Real subprocess, real NUTS fit, small draws (not the 200/200 production
    # default) only to keep this test's wall-clock reasonable.
    validator = _copy_validator(tmp_path, l6_draws=25, l6_tune=25)
    result = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--role",
            "reducer",
            "--submission",
            str(submission),
            "--data",
            str(data_path),
            "--holdout-data",
            str(holdout_path),
            "--elpd-reference",
            str(reference_path),
            "--reward-artifact-out",
            str(out_path),
            "--harbor-rewards-out",
            str(tmp_path / "harbor-rewards.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["reward_source"] == "validation.pymc_elpd"
    assert payload["producer"] == "ppl_synth_data.harbor.validator_template"
    assert payload["diagnostics"]["elpd_status"] == "scored"
    RewardArtifact.model_validate(payload)


@pytest.mark.slow
def test_validator_subprocess_bad_model(tmp_path: Path) -> None:
    submission, data_path, out_path = _write_artifacts(tmp_path, BAD_CODE_POTENTIAL)
    validator = _copy_validator(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--role",
            "reducer",
            "--submission",
            str(submission),
            "--data",
            str(data_path),
            "--holdout-data",
            str(tmp_path / "missing-holdout.parquet"),
            "--elpd-reference",
            str(tmp_path / "missing-elpd-reference.json"),
            "--reward-artifact-out",
            str(out_path),
            "--harbor-rewards-out",
            str(tmp_path / "harbor-rewards.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 1
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["reward"] == 0.0
    assert payload["diagnostics"]["failed_layer"] == "L3_harbor_pymc_strict_profile"
    assert payload["reward_source"] == "validation.compilation"
    assert payload["diagnostics"]["failure_reason"] == "pm.Potential detected"
    RewardArtifact.model_validate(payload)

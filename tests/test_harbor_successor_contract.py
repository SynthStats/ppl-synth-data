"""Focused public-behaviour regressions, using real PyMC builds and sampling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppl_synth_data.harbor import validator_template as v

ROOT = Path(__file__).resolve().parent / "fixtures/successor_laws"

# The successor-law evaluation corpus is not distributed with this package.
if not (ROOT / "data/row_shape/line-0002/data.parquet").exists():
    pytest.skip("Evaluation fixtures are not distributed.", allow_module_level=True)


def split():
    folder = ROOT / "data/row_shape/line-0002"
    frames = [pd.read_parquet(folder / name) for name in ["data.parquet", "holdout.parquet"]]
    return [
        {"n": len(frame), **{c: frame[c].to_numpy() for c in frame.columns}} for frame in frames
    ]


def test_posterior_predictive_scale_repeats_exactly():
    """Removing the PPC seed must change min_pred_std on a repeat run."""
    visible, hidden = split()
    code = (ROOT / "data/row_shape/line-0002/program.py").read_text()
    a = v.check_l6_holdout_elpd(
        code, visible, hidden, target_name="response_time_ms", draws=40, tune=40
    )
    b = v.check_l6_holdout_elpd(
        code, visible, hidden, target_name="response_time_ms", draws=40, tune=40
    )
    assert a["outcome"] == b["outcome"] == "scored"
    assert a["min_pred_std"] is not None
    assert a["min_pred_std"] == b["min_pred_std"], (a["min_pred_std"], b["min_pred_std"])
    assert a["holdout_elpd"] == b["holdout_elpd"]


@pytest.mark.parametrize(
    "mean,code",
    [
        ('np.mean(data["y"]) + theta', "y_statistic_in_likelihood"),
        ('data["y"] + theta', "y_self_conditioned"),
        ('np.mean(data["x"]) + theta', "covariate_statistic_in_likelihood"),
    ],
)
def test_forbidden_value_flow_is_refused(mean, code):
    """Deleting any one static blocking flag admits a target/covariate leak."""
    program = f"""def model(data):
    with pm.Model() as m:
        theta = pm.Normal("theta", 0, 10)
        pm.Normal("obs", mu={mean}, sigma=1, observed=data["y"])
    return m
"""
    visible = {"n": 8, "x": np.arange(8.0), "y": np.array([1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 3.0, 4.0])}
    hidden = {"n": 3, "x": np.arange(3.0), "y": np.array([2.0, 1.0, 4.0])}
    result = v.check_l6_holdout_elpd(program, visible, hidden, target_name="y", draws=20, tune=20)
    assert result.get("refusal_code") == code, result


def test_total_size_through_keywords_is_refused():
    """Runtime inspection must catch total_size even when AST keywords hide it."""
    program = """def model(data):
    options = {"observed": data["y"], "total_size": 1}
    with pm.Model() as m:
        theta = pm.Normal("theta", 0, 10)
        pm.Normal("obs", mu=theta, sigma=1, **options)
    return m
"""
    visible = {"n": 8, "y": np.arange(8.0)}
    hidden = {"n": 3, "y": np.arange(3.0)}
    result = v.check_l6_holdout_elpd(program, visible, hidden, target_name="y", draws=20, tune=20)
    assert result.get("refusal_code") == "total_size_scaled_likelihood", result


def test_used_row_latent_has_an_explicit_refusal():
    """A real O9 used-latent program must not terminate in a bare TypeError."""
    folder = ROOT / "data/row_shape/line-0322"

    def data(name):
        frame = pd.read_parquet(folder / name)
        return {"n": len(frame), **{c: frame[c].to_numpy() for c in frame.columns}}

    target = json.loads((folder / "elpd_reference.json").read_text())["target_name"]
    result = v.check_l6_holdout_elpd(
        (folder / "program.py").read_text(),
        data("data.parquet"),
        data("holdout.parquet"),
        target_name=target,
        draws=20,
        tune=20,
    )
    assert result.get("refusal_code") == "row_latent_in_heldout_likelihood", result


def test_continuous_spike_on_integer_target_is_gated():
    """Integer-valued observations do not turn a Normal density into a mass."""
    program = """def model(data):
    with pm.Model() as m:
        shift = pm.Normal("shift", 0, 1)
        pm.Normal("obs", mu=data["x"] + shift, sigma=1e-6, observed=data["y"])
    return m
"""
    visible = {"n": 8, "x": np.arange(8.0), "y": np.arange(8)}
    hidden = {"n": 3, "x": np.arange(3.0), "y": np.arange(3)}
    result = v.check_l6_holdout_elpd(program, visible, hidden, target_name="y", draws=20, tune=20)
    assert result["outcome"] == "density_gate_failed", result

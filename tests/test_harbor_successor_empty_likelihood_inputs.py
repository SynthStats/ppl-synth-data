"""Likelihood scoring supports likelihoods with no posterior-dependent parameters."""

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from ppl_synth_data.harbor import validator_template as v

ROOT = Path(__file__).resolve().parent / "fixtures/successor_laws"


def test_unused_row_latent_with_fixed_likelihood_scores_the_exact_density():
    code = """def model(data):
    with pm.Model() as m:
        pm.Normal("row_effect", 0, 1, shape=data["n"])
        pm.Normal("obs", 0, np.float64(2), observed=data["y"])
    return m
"""
    visible = {"n": 8, "y": np.array([-2.0, -1.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])}
    hidden = {"n": 3, "y": np.array([-0.3, 2.7, 3.1])}
    result = v.check_l6_holdout_elpd(code, visible, hidden, target_name="y", draws=20, tune=20)
    assert result["outcome"] == "scored", result
    assert result["holdout_elpd"] == pytest.approx(
        float(norm.logpdf(hidden["y"], 0, 2).sum()), abs=1e-12
    )
    assert result["hidden_n_obs"] == 3

"""Latent-shape detection follows row indexing even when visible and hidden lengths coincide."""

from pathlib import Path

import numpy as np
import pytest

from ppl_synth_data.harbor import validator_template as v

ROOT = Path(__file__).resolve().parent / "fixtures/successor_laws"


@pytest.mark.parametrize("row_latent", [True, False])
def test_row_latent_is_distinguished_from_fixed_width_coefficients(row_latent):
    shape = 'data["n"]' if row_latent else "3"
    mean = "coefficient" if row_latent else 'pm.math.dot(data["x"], coefficient)'
    code = f"""def model(data):
    with pm.Model() as m:
        coefficient = pm.Normal("coefficient", 0, 1, shape={shape})
        pm.Normal("obs", mu={mean}, sigma=1, observed=data["y"])
    return m
"""
    visible = {"n": 3, "x": np.eye(3), "y": np.array([0.1, 0.8, 1.6])}
    hidden = {"n": 3, "x": np.eye(3) * 1.5, "y": np.array([0.2, 0.7, 1.4])}
    result = v.check_l6_holdout_elpd(code, visible, hidden, target_name="y", draws=20, tune=20)
    if row_latent:
        assert result.get("refusal_code") == "row_latent_in_heldout_likelihood", result
    else:
        assert result["outcome"] == "scored", result

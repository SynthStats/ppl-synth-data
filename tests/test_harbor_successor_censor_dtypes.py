"""Promoting integer sentinels must preserve other accepted observation dtypes."""

from pathlib import Path

import numpy as np
import pytest

from ppl_synth_data.harbor import validator_template as v

ROOT = Path(__file__).resolve().parent / "fixtures/successor_laws"


@pytest.mark.parametrize("kind", ["bool", "float32"])
def test_existing_boolean_and_float32_wrappers_still_bind(kind):
    if kind == "bool":
        code = """def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 2, 2)
        pm.Bernoulli("obs", p=p, observed=np.asarray(data["y"], dtype=bool))
    return m
"""
        values, held = [True, True, False, True], [False, True, False]
    else:
        code = """def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        pm.Normal("obs", mu=mu, sigma=2, observed=np.asarray(data["y"], dtype=np.float32))
    return m
"""
        values, held = [0.1, 0.4, 0.7, 0.3], [0.2, 0.3, 0.8]
    visible = {"n": 4, "y": np.array(values, dtype=kind)}
    hidden = {"n": 3, "y": np.array(held, dtype=kind)}
    result = v.check_l6_holdout_elpd(code, visible, hidden, target_name="y", draws=20, tune=20)
    assert result["outcome"] == "scored", result

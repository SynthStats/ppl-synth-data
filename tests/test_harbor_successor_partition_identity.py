"""Repeated target values must not conceal duplicated observation positions."""

from pathlib import Path

import numpy as np
import pytest

from ppl_synth_data.harbor import validator_template as v

ROOT = Path(__file__).resolve().parent / "fixtures/successor_laws"


@pytest.mark.parametrize("whole", [False, True])
def test_duplicate_positions_are_refused_even_when_the_shift_preserves_values(whole):
    first, second = ("y", "y[0]") if whole else ("y[:2]", "y[:2]")
    code = f"""def model(data):
    y = np.asarray(data["y"])
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        pm.Normal("a", mu=mu + 0.1 * data["n"], sigma=1, observed={first})
        pm.Normal("b", mu=mu + 0.1 * data["n"], sigma=1, observed={second})
    return m
"""
    y = [1.0, 2.0, 3.0, 1.0] if whole else [1.0, 2.0, 1.0, 2.0]
    visible = {"n": 4, "x": np.arange(4.0), "y": np.array(y)}
    hidden = {"n": 4, "x": np.arange(4.0), "y": np.array([2.0, 1.0, 2.0, 1.0])}
    structural = v.run_cascade(code, visible)
    assert structural["reward"] == 1, structural["diagnostics"]
    result = v.check_l6_holdout_elpd(code, visible, hidden, target_name="y", draws=20, tune=20)
    expected = "auxiliary_depends_on_target" if whole else "target_observed_more_than_once"
    assert result.get("refusal_code") == expected, result

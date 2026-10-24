"""The static blocker must analyze effective Python definitions and zero terms."""

from pathlib import Path

import pytest

from ppl_synth_data.harbor import validator_template as rules

ROOT = Path(__file__).resolve().parent / "fixtures/successor_laws"

# The successor-law evaluation corpus is not distributed with this package.
if not (ROOT / "data").exists():
    pytest.skip("Evaluation fixtures are not distributed.", allow_module_level=True)


def test_final_model_definition_is_the_executed_definition():
    code = (ROOT / "data/row_shape/line-1402/program.py").read_text()
    result = rules.analyze_program(code, target="positive_affect_score")
    assert "Y_SELF_CONDITIONED" not in result["flags"], result


def test_zero_weighted_covariate_statistic_does_not_change_likelihood():
    code = """def model(data):
    x = pm.Data("x", data["x"])
    theta = pm.Normal("theta", 0, 1)
    mu = theta + 0.0 * pm.math.sum(x) * 0.0
    pm.Normal("obs", mu=mu, sigma=1, observed=data["y"])
"""
    result = rules.analyze_program(code)
    assert "COVARIATE_STATISTIC_IN_LIKELIHOOD" not in result["flags"], result

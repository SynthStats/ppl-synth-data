"""Layer 3 on a data container that nothing reads, and layer 5 on a likelihood that is not finite.

Layer 6 scores held-out rows by building the model again on the held-out table and checking that
the observed variable equals those rows. So a program that declares `pm.Data` containers and then
computes with the raw arrays is scored correctly, and a container statement that nothing reads is
dead text. Layer 3 refuses when containers that the program reads reach no random variable: there
the program's own data route is cut.

Layer 5 compares the model's log-probabilities on the data and on a halved copy. Two values that
are both minus infinity say nothing about the data; such a program goes on to layer 6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ppl_synth_data.harbor.validator_template import (
    _halve_data,
    check_l3_harbor_pymc_strict_profile,
    check_l5_data_perturbation,
)

HEAD = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
"""
PRIORS = """\
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        b = pm.Normal("b", mu=0.0, sigma=1.0)
        s = pm.HalfNormal("s", sigma=1.0)
"""
TAIL = "    return m\n"

# The container is a bare statement; the likelihood reads the raw arrays.
BARE_CALL = (
    HEAD
    + '        pm.Data("y_obs", value=y)\n'
    + PRIORS
    + ('        pm.Normal("y", mu=a + b * x, sigma=s, observed=y)\n')
    + TAIL
)

# The container is bound to a name that the program never reads.
UNREAD_NAME = (
    HEAD
    + '        x_container = pm.Data("x", x)\n'
    + PRIORS
    + ('        pm.Normal("y", mu=a + b * x, sigma=s, observed=y)\n')
    + TAIL
)

# An unread container beside a likelihood without covariates: a weak model, for the later layers
# to score.
UNREAD_AND_NO_COVARIATE = (
    HEAD
    + '        pm.Data("x", x)\n'
    + PRIORS
    + ('        pm.Normal("y", mu=a, sigma=s, observed=y)\n')
    + TAIL
)

# The program reads its container into a predictor, and the likelihood never uses that predictor.
READ_BUT_CUT = (
    HEAD
    + '        x_container = pm.Data("x", x)\n'
    + PRIORS
    + (
        "        predictor = a + b * x_container\n"
        '        pm.Normal("y", mu=a, sigma=s, observed=y)\n'
    )
    + TAIL
)

# One container is used by the likelihood; another is a statement that nothing reads.
USED_AND_UNREAD = (
    HEAD
    + ('        x_container = pm.Data("x", x)\n        pm.Data("y_obs", value=y)\n')
    + PRIORS
    + '        pm.Normal("y", mu=a + b * x_container, sigma=s, observed=y)\n'
    + TAIL
)

# No name is bound to the container, yet the program reads it from the model object. It is not
# dead text: deleting the statement would break the model. The data rules apply to it as to any
# container.
FETCHED_INTO_LIKELIHOOD = (
    HEAD
    + '        pm.Data("x", x)\n'
    + PRIORS
    + ('        pm.Normal("y", mu=a + b * m["x"], sigma=s, observed=y)\n')
    + TAIL
)

FETCHED_INTO_PRIOR = (
    HEAD
    + (
        '        pm.Data("x", x)\n'
        '        a = pm.Normal("a", mu=m["x"].mean(), sigma=1.0)\n'
        '        b = pm.Normal("b", mu=0.0, sigma=1.0)\n'
        '        s = pm.HalfNormal("s", sigma=1.0)\n'
        '        pm.Normal("y", mu=a + b * x, sigma=s, observed=y)\n'
    )
    + TAIL
)


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    return {"n": 40, "y": 1.0 + 2.0 * x + rng.normal(scale=0.3, size=40), "x": x}


@pytest.mark.parametrize(
    "code",
    [BARE_CALL, UNREAD_NAME, UNREAD_AND_NO_COVARIATE, USED_AND_UNREAD],
    ids=["bare-call", "unread-name", "unread-and-no-covariate", "used-and-unread"],
)
def test_a_container_that_nothing_reads_is_not_a_refusal(reg_data, code):
    ok, reason = check_l3_harbor_pymc_strict_profile(code, reg_data)
    assert ok, reason


def test_containers_read_into_a_predictor_the_likelihood_ignores_are_refused(reg_data):
    ok, reason = check_l3_harbor_pymc_strict_profile(READ_BUT_CUT, reg_data)
    assert not ok
    assert "data variable 'x' not used by any random variable" in reason


def test_a_container_fetched_from_the_model_into_the_likelihood_passes(reg_data):
    ok, reason = check_l3_harbor_pymc_strict_profile(FETCHED_INTO_LIKELIHOOD, reg_data)
    assert ok, reason


def test_a_container_fetched_from_the_model_into_a_prior_is_still_refused(reg_data):
    ok, reason = check_l3_harbor_pymc_strict_profile(FETCHED_INTO_PRIOR, reg_data)
    assert not ok
    assert "data used in prior" in reason


FAKE_MODEL = """\
class _Model:
    def __init__(self, logps):
        self._logps = logps

    def point_logps(self):
        return dict(self._logps)


def model(data):
    return _Model({logps})
"""


def _l5(logps: str) -> tuple[bool, str]:
    data = {"n": 6, "t": np.arange(6.0), "x": np.arange(6.0)}
    return check_l5_data_perturbation(FAKE_MODEL.format(logps=logps), data)


@pytest.mark.parametrize("value", ["float('-inf')", "float('nan')", "float('inf')"])
def test_a_non_finite_observed_log_probability_is_not_evidence_of_ignoring_data(value):
    ok, reason = _l5(f'{{"a": -1.25, "obs": {value}}}')
    assert ok, reason
    assert "not finite" in reason


def test_identical_finite_log_probabilities_are_still_refused():
    ok, reason = _l5('{"a": -1.25, "obs": -3.5}')
    assert not ok
    assert reason == "model ignores data: point_logps identical under length perturbation"


def test_log_probabilities_that_change_with_the_data_pass():
    ok, reason = _l5('{"a": -1.25, "obs": -float(len(data["t"]))}')
    assert ok
    assert reason == "ok"


def test_halving_updates_n_whatever_the_target_is_called():
    halved = _halve_data({"n": 6, "response_time_ms": np.arange(6.0), "load": np.arange(6.0)})
    assert halved["n"] == 3
    assert len(halved["response_time_ms"]) == 3
    assert len(halved["load"]) == 3


def test_halving_a_table_whose_target_is_y_is_unchanged():
    halved = _halve_data({"n": 7, "y": np.arange(7.0), "x": np.arange(7.0)})
    assert halved["n"] == 3
    assert len(halved["y"]) == 3
    assert math.isclose(float(halved["x"][-1]), 2.0)

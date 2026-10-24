"""Route-A response law checks, independent of feature sampling and admission.

The fixed-input tests catch NB2-to-Poisson substitution and wrong dispersion.
The exact binomial bands share a 1e-6 false-alarm budget across 10 PMF cells.
Seeds and sample size are fixed before implementation; no seed retries.
"""

import importlib
import math

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError
from scipy import stats

from ppl_synth_data.specs import models


def response(distribution="nb2", **overrides):
    assert hasattr(models, "CountResponse"), "typed count-response contract is missing"
    payload = {
        "family": "regression_count",
        "intercept_log_rate": math.log(4),
        "coefficients": {"load": 0.25, "workers": -0.5},
        "count_distribution": distribution,
    }
    if distribution == "nb2":
        payload["nb2_alpha"] = 2.0
    payload.update(overrides)
    return TypeAdapter(models.CountResponse).validate_python(payload)


def kernel():
    return importlib.import_module("ppl_synth_data.envs.count_kernel")


def test_named_coefficients_use_raw_scale_and_ignore_mapping_order():
    law = response(intercept_log_rate=0.0)
    # eta = .25 * 8 - .5 * 2 = 1, without min-max normalization.
    assert kernel().count_mean(law, {"workers": 2, "load": 8}) == pytest.approx(math.e)


@pytest.mark.parametrize("distribution", ["poisson", "nb2"])
def test_conditional_count_law_against_independent_scipy_pmf(distribution):
    law = response(distribution)
    draws = kernel().sample_count(
        law, {"load": 0, "workers": 0}, np.random.default_rng(2026093010), size=20000
    )
    reference = stats.poisson(4) if distribution == "poisson" else stats.nbinom(2, 1 / 3)
    assert draws.dtype.kind in "iu"
    assert np.min(draws) >= 0
    for value in [0, 1, 2, 4, 8]:
        probability = reference.pmf(value)
        lower, upper = stats.binom.ppf([5e-8, 1 - 5e-8], len(draws), probability)
        assert lower <= np.count_nonzero(draws == value) <= upper
    # Independent first two moments: Poisson (4,4), NB2 (4,12).
    mean, variance, _, excess = reference.stats(moments="mvsk")
    z = stats.norm.isf(1e-7 / 2)
    assert abs(draws.mean() - mean) < z * np.sqrt(variance / len(draws))
    fourth = (excess + 3) * variance**2
    assert abs(draws.var() - variance) < z * np.sqrt((fourth - variance**2) / len(draws))


def test_nb2_replay_uses_only_supplied_rng():
    law = response()
    first = kernel().sample_count(
        law, {"load": 0, "workers": 0}, np.random.default_rng(91), size=64
    )
    second = kernel().sample_count(
        law, {"load": 0, "workers": 0}, np.random.default_rng(91), size=64
    )
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "changes",
    [
        {"nb2_alpha": 0},
        {"nb2_alpha": -1},
        {"nb2_alpha": float("inf")},
        {"nb2_alpha": None},
        {"nb2_alpha": True},
        {"count_distribution": "poisson", "nb2_alpha": 2},
        {"count_distribution": "poisson", "nb2_alpha": None},
        {"count_distribution": "other"},
        {"offset": 1},
        {"coefficients": {}},
        {"coefficients": {"load": float("nan")}},
    ],
)
def test_unsupported_or_invalid_response_is_refused(changes):
    with pytest.raises(ValidationError):
        response(**changes)


@pytest.mark.parametrize(
    "inputs",
    [
        {"load": 1},
        {"load": 1, "workers": 2, "extra": 0},
        {"load": float("nan"), "workers": 1},
        {"load": True, "workers": 1},
    ],
)
def test_query_must_match_named_finite_features(inputs):
    with pytest.raises(ValueError):
        kernel().count_mean(response(), inputs)


def test_extreme_log_mean_is_refused_without_clipping():
    with pytest.raises(ValueError, match="mean"):
        kernel().count_mean(response(intercept_log_rate=1000), {"load": 0, "workers": 0})

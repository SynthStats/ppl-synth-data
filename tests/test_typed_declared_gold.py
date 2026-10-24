"""Gold likelihoods must equal the declared laws on the raw feature scale."""

import ast

import numpy as np
import pymc as pm
import pytest
from scipy import stats

from ppl_synth_data.harbor.declared_gold import render_declared_gold
from ppl_synth_data.specs.models import FourFamilySpec


def typed_spec(distribution):
    return FourFamilySpec.model_validate(
        {
            "env_id": "gold-contract-check",
            "env_name": "gold_contract_check",
            "domain": "engineering",
            "scenario": {"intent": "Response to measured inputs", "source_row_sha256": "1" * 64},
            "generative_family": "regression_count",
            "features": [
                {"name": "raw_x", "law": {"dist": "normal", "mean": 2.0, "sd": 1.0}},
                {
                    "name": "group",
                    "law": {
                        "dist": "categorical",
                        "levels": ["base", "other"],
                        "probs": [0.5, 0.5],
                    },
                },
            ],
            "response": {
                "family": "regression_count",
                "intercept_log_rate": 0.4,
                "coefficients": {"raw_x": 0.7, "group=other": -0.3},
                "count_distribution": distribution,
                **({"nb2_alpha": 2.0} if distribution == "nb2" else {}),
            },
            "data_plan": {"n_observations": 30},
        }
    )


@pytest.mark.parametrize("distribution", ["poisson", "nb2"])
def test_typed_count_gold_matches_independent_pointwise_likelihood(distribution):
    spec = typed_spec(distribution)
    data = {
        "n": 3,
        "raw_x": np.array([1.0, 2.0, 3.0]),
        "group": np.array(["base", "other", "base"]),
        "y": np.array([0, 2, 4]),
    }
    code, audit = render_declared_gold(spec, data)
    # Stage F supplies pm and np and refuses imports in submitted programs.
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(ast.parse(code))
    )
    namespace = {"np": np, "pm": pm}
    exec(code, namespace)
    model = namespace["model"](data)
    point = model.initial_point()
    # The free intercept is centered at the visible means; the declared raw-scale point maps to it.
    centers = audit["prior_autoscale"]["centers"]
    point.update(
        intercept_centered=np.array(0.4 + 0.7 * centers[0] - 0.3 * centers[1]),
        coef_0=np.array(0.7),
        coef_1=np.array(-0.3),
    )
    means = np.exp(np.array([1.1, 1.5, 2.5]))
    if distribution == "nb2":
        point["alpha_log__"] = np.array(np.log(2.0))
        expected = stats.nbinom.logpmf(data["y"], 2.0, 2.0 / (2.0 + means))
    else:
        expected = stats.poisson.logpmf(data["y"], means)
    actual = model.compile_logp(vars=model.observed_RVs, sum=False)(point)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert audit["classification"] == "case-1-typed-values-fixed-skeleton"
    assert audit["coefficient_keys"] == ["raw_x", "group=other"]
    assert "source_row_sha256" not in code


@pytest.mark.parametrize(
    "family,distribution",
    [
        ("regression_continuous", None),
        ("binary_choice", None),
        ("zero_inflated_count", "poisson"),
        ("zero_inflated_count", "nb2"),
    ],
)
def test_remaining_gold_laws_match_scipy_with_fixed_count_raw_coding(family, distribution):
    payload = typed_spec("poisson").model_dump(mode="json")
    payload["generative_family"] = family
    payload["features"][1]["law"] = {"dist": "fixed_counts", "levels": [1, 2], "counts": [15, 15]}
    response = {"family": family, "coefficients": {"raw_x": 0.7, "group": -0.3}}
    if family == "regression_continuous":
        response.update(intercept=0.4, noise_sigma=0.6)
    elif family == "binary_choice":
        response.update(intercept=0.4, link="logit")
    else:
        response.update(
            intercept_log_rate=0.4, count_distribution=distribution, zero_inflation=0.25
        )
        if distribution == "nb2":
            response["nb2_alpha"] = 2.0
    payload["response"] = response
    y = np.array([0, 1, 0]) if family == "binary_choice" else np.array([0, 2, 4])
    data = {"n": 3, "raw_x": np.array([1.0, 2.0, 3.0]), "group": np.array([1, 2, 1]), "y": y}
    code, audit = render_declared_gold(FourFamilySpec.model_validate(payload), data)
    from ppl_synth_data.validation._pymc_semantics_harness import _unsafe_syntax_reason

    assert _unsafe_syntax_reason(code) is None
    namespace = {"np": np, "pm": pm}
    exec(code, namespace)
    model = namespace["model"](data)
    point = model.initial_point()
    centers = audit["prior_autoscale"]["centers"]
    point.update(
        intercept_centered=np.array(0.4 + 0.7 * centers[0] - 0.3 * centers[1]),
        coef_0=np.array(0.7),
        coef_1=np.array(-0.3),
    )
    eta = np.array([0.8, 1.2, 2.2])
    if family == "regression_continuous":
        point["sigma_log__"] = np.array(np.log(0.6))
        expected = stats.norm.logpdf(y, eta, 0.6)
    elif family == "binary_choice":
        expected = stats.bernoulli.logpmf(y, 1.0 / (1.0 + np.exp(-eta)))
    else:
        point["zero_inflation_logodds__"] = np.array(np.log(0.25 / 0.75))
        if distribution == "nb2":
            point["alpha_log__"] = np.array(np.log(2.0))
            base = stats.nbinom.pmf(y, 2.0, 2.0 / (2.0 + np.exp(eta)))
        else:
            base = stats.poisson.pmf(y, np.exp(eta))
        expected = np.log(0.75 * base + 0.25 * (y == 0))
    actual = model.compile_logp(vars=model.observed_RVs, sum=False)(point)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert audit["coefficient_keys"] == ["raw_x", "group"]

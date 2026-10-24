"""Contract checks for the separate EnvSpec-4F v0.1 input format."""

from __future__ import annotations

from copy import deepcopy
from math import exp
from pathlib import Path
from statistics import fmean, pvariance

import pytest
from pydantic import ValidationError

from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import load_env_instance
from ppl_synth_data.specs.models import FourFamilySpec


def count_spec() -> dict:
    return {
        "env_id": "4f-v0-pymc-train-023",
        "env_name": "four_family_count_023",
        "domain": "healthcare",
        "scenario": {
            "intent": "Count events across a measured predictor",
            "source_row_sha256": "a" * 64,
        },
        "generative_family": "regression_count",
        "features": [
            {
                "name": "exposure",
                "units": "hours",
                "law": {"dist": "normal", "mean": 2.0, "sd": 1.5},
            }
        ],
        "response": {
            "family": "regression_count",
            "intercept_log_rate": 0.4,
            "coefficients": {"exposure": 0.7},
            "count_distribution": "poisson",
        },
        "data_plan": {"n_observations": 30},
        "units": {"events": "count"},
        "allowed_hints": ["outcome is a count"],
        "forbidden_hints": ["coefficient values"],
    }


@pytest.mark.parametrize(
    ("family", "response"),
    [
        (
            "regression_continuous",
            {
                "family": "regression_continuous",
                "intercept": 1.0,
                "coefficients": {"exposure": 0.7},
                "noise_sigma": 0.5,
            },
        ),
        (
            "regression_count",
            {
                "family": "regression_count",
                "intercept_log_rate": 0.4,
                "coefficients": {"exposure": 0.7},
                "count_distribution": "nb2",
                "nb2_alpha": 3.0,
            },
        ),
        (
            "zero_inflated_count",
            {
                "family": "zero_inflated_count",
                "intercept_log_rate": 0.4,
                "coefficients": {"exposure": 0.7},
                "count_distribution": "poisson",
                "zero_inflation": 0.25,
            },
        ),
        (
            "binary_choice",
            {
                "family": "binary_choice",
                "intercept": -0.2,
                "coefficients": {"exposure": 0.7},
                "link": "logit",
            },
        ),
    ],
)
def test_four_response_families_round_trip_contract_fields(family: str, response: dict) -> None:
    payload = count_spec()
    payload["generative_family"] = family
    payload["response"] = response

    dumped = FourFamilySpec.model_validate(payload).model_dump(mode="json")

    assert dumped == payload


@pytest.mark.parametrize(
    ("law", "coefficient_key"),
    [
        ({"dist": "normal", "mean": -1.0, "sd": 2.0}, "month"),
        ({"dist": "discrete_uniform", "low": 1, "high": 12}, "month"),
        ({"dist": "bernoulli", "p": 0.3}, "month"),
        (
            {"dist": "categorical", "levels": ["jan", "feb", "mar"], "probs": [0.2, 0.3, 0.5]},
            "month=feb",
        ),
        (
            {"dist": "fixed_counts", "levels": ["jan", "feb", "mar"], "counts": [8, 10, 12]},
            "month=feb",
        ),
    ],
)
def test_input_laws_are_preserved(law: dict, coefficient_key: str) -> None:
    payload = count_spec()
    payload["features"] = [{"name": "month", "law": law}]
    coefficients = {coefficient_key: 0.2}
    if law["dist"] in {"categorical", "fixed_counts"}:
        coefficients["month=mar"] = -0.1
    payload["response"]["coefficients"] = coefficients

    assert (
        FourFamilySpec.model_validate(payload).model_dump(mode="json")["features"]
        == payload["features"]
    )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("top_extra", True),
        ("nested_extra", True),
        ("law_extra", True),
        ("family_mismatch", "binary_choice"),
        ("unknown_family", "categorical_choice"),
        ("coefficient_missing", {}),
        ("coefficient_extra", {"exposure": 0.7, "other": 1.0}),
        ("poisson_alpha", 2.0),
        ("normal_sd", 0.0),
        ("discrete_bounds", True),
        ("bernoulli_probability", 1.0),
        ("duplicate_feature", True),
    ],
)
def test_invalid_contract_declarations_are_refused(change: str, value: object) -> None:
    payload = count_spec()
    if change == "top_extra":
        payload["service_effect_sd"] = value
    elif change == "nested_extra":
        payload["response"]["feature_noise_scale"] = value
    elif change == "law_extra":
        payload["features"][0]["law"]["upper_bound"] = value
    elif change == "family_mismatch":
        payload["response"]["family"] = value
    elif change == "unknown_family":
        payload["generative_family"] = value
    elif change == "coefficient_missing" or change == "coefficient_extra":
        payload["response"]["coefficients"] = value
    elif change == "poisson_alpha":
        payload["response"]["nb2_alpha"] = value
    elif change == "normal_sd":
        payload["features"][0]["law"]["sd"] = value
    elif change == "discrete_bounds":
        payload["features"][0]["law"] = {"dist": "discrete_uniform", "low": 4, "high": 4}
    elif change == "bernoulli_probability":
        payload["features"][0]["law"] = {"dist": "bernoulli", "p": value}
    elif change == "duplicate_feature":
        payload["features"].append(deepcopy(payload["features"][0]))

    with pytest.raises(ValidationError):
        FourFamilySpec.model_validate(payload)


@pytest.mark.parametrize(
    "response_update",
    [
        {"count_distribution": "nb2"},
        {"count_distribution": "nb2", "nb2_alpha": 0.0},
    ],
)
def test_nb2_requires_positive_alpha(response_update: dict) -> None:
    payload = count_spec()
    payload["response"].update(response_update)
    with pytest.raises(ValidationError):
        FourFamilySpec.model_validate(payload)


@pytest.mark.parametrize(
    ("law", "n_observations", "second_fixed_counts"),
    [
        ({"dist": "categorical", "levels": ["a", "b"], "probs": [0.4]}, 30, False),
        ({"dist": "categorical", "levels": ["a", "b"], "probs": [0.4, 0.4]}, 30, False),
        ({"dist": "categorical", "levels": ["a", "a"], "probs": [0.5, 0.5]}, 30, False),
        ({"dist": "categorical", "levels": ["a", "b"], "probs": [0.0, 1.0]}, 30, False),
        ({"dist": "fixed_counts", "levels": ["a", "b"], "counts": [10]}, 30, False),
        ({"dist": "fixed_counts", "levels": ["a", "b"], "counts": [10, 10]}, 30, False),
        ({"dist": "fixed_counts", "levels": ["a", "b"], "counts": [0, 30]}, 30, False),
        ({"dist": "fixed_counts", "levels": ["a", "b"], "counts": [15, 15]}, 30, True),
    ],
)
def test_invalid_categorical_and_fixed_allocations_are_refused(
    law: dict, n_observations: int, second_fixed_counts: bool
) -> None:
    payload = count_spec()
    payload["data_plan"]["n_observations"] = n_observations
    payload["features"] = [{"name": "month", "law": law}]
    payload["response"]["coefficients"] = {"month=b": 0.2}
    if second_fixed_counts:
        payload["features"].append(
            {
                "name": "site",
                "law": {"dist": "fixed_counts", "levels": ["a", "b"], "counts": [15, 15]},
            }
        )
        payload["response"]["coefficients"]["site=b"] = 0.3

    with pytest.raises(ValidationError):
        FourFamilySpec.model_validate(payload)


def test_categorical_reference_level_has_no_coefficient() -> None:
    payload = count_spec()
    payload["features"] = [
        {
            "name": "month",
            "law": {"dist": "categorical", "levels": ["jan", "feb"], "probs": [0.4, 0.6]},
        }
    ]
    payload["response"]["coefficients"] = {"month=jan": 0.1, "month=feb": 0.2}

    with pytest.raises(ValidationError):
        FourFamilySpec.model_validate(payload)


@pytest.mark.parametrize("zero_inflation", [0.0, 1.0])
def test_zero_inflation_is_strictly_between_zero_and_one(zero_inflation: float) -> None:
    payload = count_spec()
    payload["generative_family"] = "zero_inflated_count"
    payload["response"]["family"] = "zero_inflated_count"
    payload["response"]["zero_inflation"] = zero_inflation

    with pytest.raises(ValidationError):
        FourFamilySpec.model_validate(payload)


def test_normal_predictor_compiles_and_draws_on_raw_scale(tmp_path: Path) -> None:
    payload = count_spec()
    module_path = TemplateCompiler().compile_envspec(
        FourFamilySpec.model_validate(payload), tmp_path
    )
    env = load_env_instance(
        module_path, env_name=payload["env_name"], seed=17, trusted_root=tmp_path
    )

    _, success = env.run_experiment("2.0")

    assert success is True
    assert env.get_df().iloc[0]["exposure"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("distribution", "alpha"),
    [("poisson", None), ("nb2", 2.0)],
)
def test_compiled_count_law_samples_at_raw_log_rate_and_replays(
    tmp_path: Path, distribution: str, alpha: float | None
) -> None:
    payload = count_spec()
    payload["response"]["count_distribution"] = distribution
    if alpha is not None:
        payload["response"]["nb2_alpha"] = alpha
    module_path = TemplateCompiler().compile_envspec(
        FourFamilySpec.model_validate(payload), tmp_path
    )
    env = load_env_instance(
        module_path, env_name=payload["env_name"], seed=23, trusted_root=tmp_path
    )

    def draw(n: int) -> list[int]:
        samples = []
        for _ in range(n):
            outcome, success = env.run_experiment("2.0")
            assert success is True
            assert isinstance(outcome, int) and outcome >= 0
            samples.append(outcome)
        return samples

    first = draw(2048)
    env.reset(seed=23)
    assert draw(32) == first[:32]

    # Raw x=2 implies exp(0.4 + 0.7*2) ~= 6.05. A min-max mapping
    # of this predictor would put the mean near 2, well outside this band.
    expected_mean = exp(0.4 + 0.7 * 2.0)
    assert abs(fmean(first) - expected_mean) < 0.6

    # At 2048 draws these broad moment bands absorb Monte Carlo error while
    # separating NB2 overdispersion from a Poisson sample at the same mean.
    expected_variance = expected_mean if alpha is None else expected_mean + expected_mean**2 / alpha
    variance_tolerance = 3.0 if alpha is None else 7.0
    assert abs(pvariance(first) - expected_variance) < variance_tolerance

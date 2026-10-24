"""Compare public response-law declarations with actual compiled draws.

Expected means and probabilities are analytical, conditional on the public
realized latent values. The tests never read the runtime's private attributes.
The worked domains deliberately have large offsets and spans.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import expit, softmax

from ppl_synth_data.envs.base import BaseEnv
from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import load_env_instance
from ppl_synth_data.specs.models import DataPlan, EnvSpec, InputType, IOSchema

PARAMETERS: dict[str, dict[str, Any]] = {
    "regression_continuous": {
        "intercept": 100.0,
        "coefficients": {"x": 4.0, "w": -2.0},
        "noise_sigma": 3.0,
    },
    "regression_count": {
        "intercept": 0.0,
        "coefficients": {"x": 0.5, "w": -0.25},
    },
    "binary_choice": {
        "intercept": -0.75,
        "coefficients": {"x": 1.5, "w": -0.5},
    },
    "categorical_choice": {
        "intercept": -0.75,
        "coefficients": {"x": 1.5, "w": -0.5},
        "n_categories": 3,
    },
    "mixture_continuous": {
        "coefficients": {"x": 4.0, "w": -2.0},
        "n_components": 2,
        "component_mean_lower": 80.0,
        "component_mean_upper": 120.0,
        "noise_sigma": 2.0,
    },
    "zero_inflated_count": {
        "intercept": 1.3862943611198906,
        "coefficients": {"x": 0.5, "w": -0.25},
        "zero_inflation": 0.4,
    },
    "time_series_level_seasonality": {
        "base_level": 100.0,
        "trend": 20.0,
        "season_period": 300.0,
        "season_amplitude": 3.0,
        "noise_sigma": 2.0,
    },
    "infection_process": {"population_n": 100, "rate_mean": 1.0, "rate_sigma": 0.0},
    "choice_with_discounting": {
        "epsilon": 0.1,
        "k_mean": 0.05,
        "k_sigma": 0.0,
        "choice_scale": 250.0,
    },
    "hierarchical_grouped": {"base_mean": 20.0, "noise_sigma": 2.0},
    "irt_binary_response": {
        "n_students": 3,
        "n_questions": 4,
        "ability_mean": 1.0,
        "ability_sigma": 0.5,
        "difficulty_mean": -0.5,
        "difficulty_sigma": 0.4,
        "discrimination_mean": 1.2,
    },
    "ode_predator_prey": {
        "alpha": 0.4,
        "beta": 0.02,
        "gamma": 0.5,
        "delta": 0.05,
        "prey_init": 10.0,
        "predator_init": 20.0,
        "noise_sigma": 1.0,
    },
    "spatial_signal_source": {
        "n_sources": 2,
        "dim": 2,
        "background": 10.0,
        "regularization_m": 2.0,
        "source_strength": 8.0,
        "noise_sigma": 0.5,
    },
    "survival_with_censoring": {
        "baseline_rate": 0.2,
        "covariate_effect": 0.0,
        "censor_time": 4.0,
        "n_subjects": 4,
    },
}

DISTRIBUTIONS = {
    "regression_continuous": "Normal",
    "regression_count": "Poisson",
    "binary_choice": "Bernoulli",
    "categorical_choice": "Categorical",
    "mixture_continuous": "NormalMixture",
    "zero_inflated_count": "ZeroInflatedPoisson",
    "time_series_level_seasonality": "Normal",
    "infection_process": "Binomial",
    "choice_with_discounting": "Bernoulli",
    "hierarchical_grouped": "CensoredNormal",
    "irt_binary_response": "Bernoulli",
    "ode_predator_prey": "RoundedCensoredNormal",
    "spatial_signal_source": "Normal",
    "survival_with_censoring": "CensoredExponential",
}

LINEAR = tuple(PARAMETERS)[:6]
STANDARDIZED = (*LINEAR, "time_series_level_seasonality", "infection_process")


def _spec(family: str) -> EnvSpec:
    domains = {"x": (100.0, 200.0), "w": (1000.0, 5000.0)}
    input_type = InputType.json
    grouping = []
    if family in {"time_series_level_seasonality", "infection_process"}:
        domains = {"t": (1000.0, 2000.0)}
        input_type = InputType.int if family == "time_series_level_seasonality" else InputType.float
    elif family == "choice_with_discounting":
        domains = {
            "immediate_reward": (100.0, 1000.0),
            "delayed_reward": (200.0, 2000.0),
            "delay_days": (10.0, 100.0),
        }
    elif family == "hierarchical_grouped":
        domains = {"group": (0.0, 3.0)}
        grouping = ["group"]
    elif family == "irt_binary_response":
        domains = {"student_id": (0.0, 2.0), "question_id": (0.0, 3.0)}
    elif family == "ode_predator_prey":
        domains = {"t": (1.0, 4.0)}
        input_type = InputType.float
    elif family == "spatial_signal_source":
        domains = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
    elif family == "survival_with_censoring":
        domains = {"subject_id": (0.0, 3.0)}
        input_type = InputType.int
    return EnvSpec(
        env_id=f"declaration_{family}",
        env_name=f"declaration_{family}",
        domain="engineering",
        complexity="simple",
        model_type="predictive",
        key_concepts=[family],
        generative_family=family,
        generative_params=PARAMETERS[family].copy(),
        io_schema=IOSchema(
            input_type=input_type,
            input_description="Declared corner probe inputs.",
            output_type="float",
            target_name="response",
            feature_names=list(domains),
            feature_domains=domains,
            feature_types={
                name: "integer"
                if name
                in {
                    "group",
                    "student_id",
                    "question_id",
                    "subject_id",
                }
                else "continuous"
                for name in domains
            },
        ),
        data_plan=DataPlan(n_observations=128, grouping=grouping),
    )


def _compile(spec: EnvSpec, out_dir: Path) -> BaseEnv:
    path = TemplateCompiler().compile_envspec(spec, out_dir)
    return load_env_instance(path, env_name=spec.env_name, seed=17, trusted_root=out_dir)


def _law(env: BaseEnv, spec: EnvSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    law = env.response_law
    assert law["family"] == spec.generative_family
    assert law["distribution"] == DISTRIBUTIONS[spec.generative_family]
    assert isinstance(law["link"], str) and law["link"].strip()
    assert set(law["parameters"]) == set(spec.generative_params)
    values = {}
    for name, parameter in law["parameters"].items():
        assert parameter["value"] == spec.generative_params[name]
        assert isinstance(parameter["units"], str) and parameter["units"].strip()
        values[name] = parameter["value"]
    standardization = law["standardization"]
    assert isinstance(standardization["method"], str) and standardization["method"]
    if spec.generative_family in STANDARDIZED:
        assert standardization["method"] == "minmax"
        declared_domains = {k: tuple(v) for k, v in standardization["domains"].items()}
        assert declared_domains == spec.io_schema.feature_domains
    else:
        assert standardization["method"] in {"none", "identity"}
    return law, values


def _corners(spec: EnvSpec):
    names = list(spec.io_schema.feature_domains)
    for bits in itertools.product((0, 1), repeat=len(names)):
        query = {
            name: spec.io_schema.feature_domains[name][bit]
            for name, bit in zip(names, bits, strict=True)
        }
        yield query, dict(zip(names, bits, strict=True))


def _draws(env: BaseEnv, query: dict[str, float] | str, n: int = 4096) -> np.ndarray:
    text = json.dumps(query) if isinstance(query, dict) else query
    samples = []
    for _ in range(n):
        result, accepted = env.run_experiment(text)
        assert accepted, result
        samples.append(result)
    return np.asarray(samples)


def _assert_moments(samples: np.ndarray, mean: float, variance: float) -> None:
    assert np.isfinite(samples).all()
    # Six standard errors allows independent RNG implementations without hiding
    # the large scale/link errors under examination.
    assert samples.mean() == pytest.approx(mean, abs=6 * math.sqrt(variance / len(samples)) + 1e-9)
    assert samples.var() == pytest.approx(variance, abs=0.2 * variance + 0.05)


@pytest.mark.parametrize("family", LINEAR)
def test_declared_linear_law_matches_draws_at_every_corner(tmp_path: Path, family: str) -> None:
    spec = _spec(family)
    env = _compile(spec, tmp_path)
    law, params = _law(env, spec)
    for query, z in _corners(spec):
        env.reset(seed=17)
        effect = sum(params["coefficients"][name] * value for name, value in z.items())
        samples = _draws(env, query)
        if family == "mixture_continuous":
            latents = law["latent_parameters"]
            centers = np.asarray(latents["component_centers"])
            weights = np.asarray(latents["mixture_weights"])
            # Two equal-width bins in [80,120] have midpoints 90 and 110.
            np.testing.assert_allclose(centers, [90.0, 110.0])
            assert weights.sum() == pytest.approx(1.0)
            mean = float(weights @ centers)
            variance = params["noise_sigma"] ** 2 + float(weights @ ((centers - mean) ** 2))
            _assert_moments(samples, mean + effect, variance)
            continue
        eta = params["intercept"] + effect
        if family == "regression_continuous":
            assert law["link"] == "identity"
            _assert_moments(samples, eta, params["noise_sigma"] ** 2)
        elif family == "regression_count":
            assert law["link"] == "log"
            assert (samples == np.floor(samples)).all() and (samples >= 0).all()
            _assert_moments(samples, math.exp(eta), math.exp(eta))
        elif family == "binary_choice":
            assert law["link"] == "logit"
            probability = float(expit(eta))
            assert set(samples) == {0, 1}
            _assert_moments(samples, probability, probability * (1 - probability))
        elif family == "categorical_choice":
            assert law["link"] == "softmax"
            weights = np.asarray(law["latent_parameters"]["category_weights"])
            assert len(weights) == params["n_categories"]
            probabilities = softmax(weights * eta)
            observed = np.bincount(samples.astype(int), minlength=params["n_categories"]) / len(
                samples
            )
            np.testing.assert_allclose(observed, probabilities, atol=0.04, rtol=0)
        else:
            rate, zero_fraction = math.exp(eta), params["zero_inflation"]
            mean = (1 - zero_fraction) * rate
            variance = (1 - zero_fraction) * rate * (1 + zero_fraction * rate)
            _assert_moments(samples, mean, variance)
            assert (samples == 0).mean() == pytest.approx(
                zero_fraction + (1 - zero_fraction) * math.exp(-rate),
                abs=0.04,
            )


def test_declared_time_law_uses_full_domain_trend_and_natural_period(tmp_path: Path) -> None:
    spec = _spec("time_series_level_seasonality")
    env = _compile(spec, tmp_path)
    law, params = _law(env, spec)
    assert "phase" not in type(env).__doc__.lower()
    assert "phase" not in law["parameters"]
    units = law["parameters"]["trend"]["units"].lower()
    assert all(word in units for word in ("full", "domain", "response"))
    for t, fraction in [(1000, 0.0), (1500, 0.5), (2000, 1.0)]:
        env.reset(seed=17)
        mean = (
            params["base_level"]
            + params["trend"] * fraction
            + params["season_amplitude"] * math.sin(2 * math.pi * t / params["season_period"])
        )
        _assert_moments(_draws(env, str(t)), mean, params["noise_sigma"] ** 2)


def test_declared_infection_hazard_is_over_the_full_time_domain(tmp_path: Path) -> None:
    spec = _spec("infection_process")
    env = _compile(spec, tmp_path)
    law, params = _law(env, spec)
    for name in ("rate_mean", "rate_sigma"):
        units = law["parameters"][name]["units"].lower()
        assert all(word in units for word in ("full", "domain", "hazard"))
    assert law["latent_parameters"]["theta"] == params["rate_mean"]
    # theta=1 gives probabilities 0, 1-exp(-1/2), 1-exp(-1), independent of seconds.
    for t, probability in [(1000, 0.0), (1500, 0.3934693402873666), (2000, 0.6321205588285577)]:
        samples = _draws(env, str(t))
        _assert_moments(
            samples,
            params["population_n"] * probability,
            params["population_n"] * probability * (1 - probability),
        )


def test_declared_discount_law_uses_commensurate_rewards_and_natural_days(tmp_path: Path) -> None:
    spec = _spec("choice_with_discounting")
    env = _compile(spec, tmp_path)
    law, params = _law(env, spec)
    k = law["latent_parameters"]["k"]
    assert k == params["k_mean"]
    assert "reward" in law["parameters"]["choice_scale"]["units"]
    for name in ("k_mean", "k_sigma"):
        assert "1/day" in law["parameters"][name]["units"]
    for query, _ in _corners(spec):
        difference = (
            query["delayed_reward"] / (1 + k * query["delay_days"]) - query["immediate_reward"]
        )
        probability = params["epsilon"] / 2 + (1 - params["epsilon"]) * float(
            expit(difference / params["choice_scale"])
        )
        samples = _draws(env, query)
        assert set(samples) == {0, 1}
        _assert_moments(samples, probability, probability * (1 - probability))


def test_declared_group_law_matches_each_latent_group_mean(tmp_path: Path) -> None:
    spec = _spec("hierarchical_grouped")
    env = _compile(spec, tmp_path)
    law, params = _law(env, spec)
    for group in (0, 3):
        samples = _draws(env, {"group": group})
        mean = params["base_mean"] + law["latent_parameters"]["group_effects"][str(group)]
        # These means are far enough above zero for censoring to have negligible
        # effect; the declaration still names the censoring distribution.
        assert mean > 12.0
        _assert_moments(samples, mean, params["noise_sigma"] ** 2)


def test_declared_irt_law_matches_first_and_last_ids(tmp_path: Path) -> None:
    spec = _spec("irt_binary_response")
    env = _compile(spec, tmp_path)
    law, params = _law(env, spec)
    latents = law["latent_parameters"]
    assert len(latents["abilities"]) == params["n_students"]
    assert len(latents["difficulties"]) == params["n_questions"]
    assert len(latents["discriminations"]) == params["n_questions"]
    for student_id, question_id in itertools.product((0, 2), (0, 3)):
        eta = latents["discriminations"][question_id] * (
            latents["abilities"][student_id] - latents["difficulties"][question_id]
        )
        probability = float(expit(eta))
        samples = _draws(env, {"student_id": student_id, "question_id": question_id})
        _assert_moments(samples, probability, probability * (1 - probability))


def test_declared_ode_law_preserves_analytic_equilibrium_at_time_corners(tmp_path: Path) -> None:
    spec = _spec("ode_predator_prey")
    env = _compile(spec, tmp_path)
    _, params = _law(env, spec)
    # alpha/beta=20 predators and gamma/delta=10 prey form an equilibrium.
    assert params["alpha"] / params["beta"] == params["predator_init"]
    assert params["gamma"] / params["delta"] == params["prey_init"]
    for t in ("1", "4"):
        samples = _draws(env, t, n=2048)
        assert samples.shape == (2048, 2)
        assert np.equal(samples, np.round(samples)).all()
        _assert_moments(samples[:, 0], 10.0, 1.0833333333)
        _assert_moments(samples[:, 1], 20.0, 1.0833333333)


def test_declared_spatial_law_matches_physical_distance_at_corners(tmp_path: Path) -> None:
    spec = _spec("spatial_signal_source")
    env = _compile(spec, tmp_path)
    law, params = _law(env, spec)
    sources = np.asarray(law["latent_parameters"]["source_positions"])
    strengths = np.asarray(law["latent_parameters"]["strengths"])
    assert sources.shape == (params["n_sources"], params["dim"])
    assert strengths.shape == (params["n_sources"],)
    for query, _ in _corners(spec):
        point = np.array([query["x"], query["y"]])
        squared_distances = ((point - sources) ** 2).sum(axis=1)
        mean = params["background"] + float(
            (strengths / (params["regularization_m"] + squared_distances)).sum()
        )
        _assert_moments(_draws(env, query), mean, params["noise_sigma"] ** 2)


def test_declared_survival_law_matches_censoring_at_subject_corners(tmp_path: Path) -> None:
    spec = _spec("survival_with_censoring")
    env = _compile(spec, tmp_path)
    law, params = _law(env, spec)
    # With covariate_effect=0, E[min(T,4)]=(1-exp(-0.8))/0.2,
    # and P(censored)=exp(-0.8). Reset resamples subjects; repeated queries do not.
    samples = {0: [], 3: []}
    for seed in range(4096):
        env.reset(seed=seed)
        for subject_id in samples:
            result = _draws(env, str(subject_id), n=2)
            np.testing.assert_array_equal(result[0], result[1])
            assert result[0, 0] == law["latent_parameters"]["times"][subject_id]
            assert result[0, 1] == law["latent_parameters"]["censored"][subject_id]
            samples[subject_id].append(result[0])
    for observations in samples.values():
        values = np.asarray(observations)
        assert ((values[:, 0] >= 0) & (values[:, 0] <= params["censor_time"])).all()
        assert values[:, 0].mean() == pytest.approx(2.7533551794, abs=0.13)
        assert values[:, 1].mean() == pytest.approx(0.4493289641, abs=0.04)

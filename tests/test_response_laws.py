"""Response laws on declared natural-scale inputs, through compiled environments.

The fixtures express natural-scale laws. The supported scenarios transport
those laws into the coefficient contract with independently worked constants.
The zero-inflated example uses a continuous input.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy.special import expit

from ppl_synth_data.envs.base import BaseEnv
from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import load_env_instance
from ppl_synth_data.pipeline.io import write_jsonl
from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b
from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c
from ppl_synth_data.pipeline.types import RunPaths
from ppl_synth_data.specs.models import DataPlan, EnvSpec, InputType, IOSchema

FIXTURES = Path(__file__).parent / "fixtures" / "response_laws"
DIRECT_LINEAR_FAMILIES = (
    "regression_continuous",
    "regression_count",
    "binary_choice",
    "categorical_choice",
    "mixture_continuous",
    "zero_inflated_count",
)


def _original_specs() -> dict[str, EnvSpec]:
    specs = [
        EnvSpec.model_validate_json(line)
        for line in (FIXTURES / "acceptance_specs.jsonl").read_text().splitlines()
    ]
    return {spec.generative_family: spec for spec in specs}


def _scenario(family: str) -> EnvSpec:
    if family in {"regression_continuous", "regression_count", "binary_choice"}:
        spec = _original_specs()[family]
        if family == "regression_continuous":
            # Original BP mean evaluated at all lower bounds is 129.74 mmHg.
            # Full-domain effects are 28.8, 27.6, 16.8 and -13.5 mmHg.
            spec.generative_params = {
                "intercept": 129.74,
                "coefficients": {
                    "age_years": 28.8,
                    "bmi_kg_m2": 27.6,
                    "daily_sodium_mg": 16.8,
                    "weekly_exercise_hours": -13.5,
                },
                "noise_sigma": 8.5,
            }
        elif family == "regression_count":
            spec.generative_params = {
                "intercept": 0.008,
                "coefficients": {
                    "spindle_speed_rpm": 0.432,
                    "feed_rate_mm_per_min": 0.88,
                    "coolant_flow_l_per_min": -0.63,
                    "tool_wear_hours": 1.26,
                },
            }
        else:
            spec.generative_params = {
                "intercept": -2.725,
                "coefficients": {
                    "annual_income_usd": -1.28,
                    "loan_amount_usd": 2.065,
                    "credit_score": -3.3,
                    "years_employed": -1.6,
                },
            }
        return spec

    params = {
        "categorical_choice": {
            "intercept": -0.5,
            "coefficients": {"age_years": 1.5, "daily_sodium_mg": -0.75},
            "n_categories": 3,
        },
        "mixture_continuous": {
            "coefficients": {"age_years": 10.0, "daily_sodium_mg": -4.0},
            "component_mean_lower": 90.0,
            "component_mean_upper": 110.0,
            "n_components": 2,
            "noise_sigma": 2.0,
        },
        "zero_inflated_count": {
            "intercept": 1.3862943611198906,
            "coefficients": {"age_years": 0.8, "daily_sodium_mg": -0.6},
            "zero_inflation": 0.4,
        },
    }
    return EnvSpec(
        env_id=f"response_law_{family}",
        env_name=f"response_law_{family}",
        domain="healthcare",
        complexity="simple",
        model_type="predictive",
        key_concepts=[family],
        io_schema=IOSchema(
            input_type=InputType.json,
            input_description="Independent continuous age in years and sodium intake in mg/day.",
            output_type="float",
            target_name="response",
            feature_names=["age_years", "daily_sodium_mg"],
            # Deliberately reverse the mapping order: names, not position, bind domains.
            feature_domains={"daily_sodium_mg": (1200.0, 5200.0), "age_years": (25.0, 85.0)},
            feature_types={"age_years": "continuous", "daily_sodium_mg": "continuous"},
        ),
        data_plan=DataPlan(n_observations=160),
        generative_family=family,
        generative_params=params[family],
    )


def _compile(spec: EnvSpec, out_dir: Path) -> BaseEnv:
    module = TemplateCompiler().compile_envspec(spec, out_dir=out_dir)
    return load_env_instance(module, env_name=spec.env_name, seed=0, trusted_root=out_dir)


def _draws(env: BaseEnv, query: dict[str, float] | str, n: int = 2048) -> np.ndarray:
    env.reset(seed=0)
    text = json.dumps(query) if isinstance(query, dict) else query
    values = []
    for _ in range(n):
        value, accepted = env.run_experiment(text)
        assert accepted, value
        values.append(value)
    if isinstance(query, dict):
        # The recorded data retain the actual input units, not internal z values.
        for name, value in query.items():
            assert env.get_df()[name].eq(value).all()
    return np.asarray(values)


def _lower_query(spec: EnvSpec) -> dict[str, float]:
    return {name: domain[0] for name, domain in spec.io_schema.feature_domains.items()}


def test_blood_pressure_preserves_mmhg_noise_and_each_clinical_effect(tmp_path: Path) -> None:
    spec = _scenario("regression_continuous")
    env = _compile(spec, tmp_path)
    lower = _lower_query(spec)
    base = _draws(env, lower)
    assert np.all((base > 60.0) & (base < 230.0))
    assert base.mean() == pytest.approx(129.74, abs=0.7)
    assert base.std() == pytest.approx(8.5, abs=0.5)
    # Resetting the real RNG supplies equal noise in both experiments; the paired
    # difference measures the response effect without mutating the noise law.
    for feature, effect in spec.generative_params["coefficients"].items():
        query = {**lower, feature: spec.io_schema.feature_domains[feature][1]}
        raised = _draws(env, query)
        assert np.all((raised > 60.0) & (raised < 230.0))
        np.testing.assert_allclose(raised - base, effect, rtol=0, atol=1e-12)


def test_bolt_counts_have_small_poisson_means_and_each_process_effect(tmp_path: Path) -> None:
    spec = _scenario("regression_count")
    env = _compile(spec, tmp_path)
    lower = _lower_query(spec)
    base = _draws(env, lower)
    assert base.mean() == pytest.approx(1.0080320855, abs=0.12)
    assert base.var() == pytest.approx(1.0080320855, abs=0.2)
    expected_means = {
        "spindle_speed_rpm": 1.5527072185,
        "feed_rate_mm_per_min": 2.4302642590,
        "coolant_flow_l_per_min": 0.5368696239,
        "tool_wear_hours": 3.5537379742,
    }
    for feature, expected_mean in expected_means.items():
        raised = _draws(env, {**lower, feature: spec.io_schema.feature_domains[feature][1]})
        assert np.all((raised >= 0) & (raised < 30))
        assert raised.mean() == pytest.approx(expected_mean, abs=0.2)
        effect = spec.generative_params["coefficients"][feature]
        assert (raised.mean() - base.mean()) * effect > 0


def test_default_probabilities_vary_with_every_financial_covariate(tmp_path: Path) -> None:
    spec = _scenario("binary_choice")
    env = _compile(spec, tmp_path)
    lower = _lower_query(spec)
    base = _draws(env, lower)
    assert set(base) == {0, 1}
    assert base.mean() == pytest.approx(0.0615, abs=0.02)
    expected_probabilities = {
        "annual_income_usd": 0.0179,
        "loan_amount_usd": 0.3407,
        "credit_score": 0.0024,
        "years_employed": 0.0131,
    }
    for feature, probability in expected_probabilities.items():
        raised = _draws(env, {**lower, feature: spec.io_schema.feature_domains[feature][1]})
        assert set(raised) <= {0, 1}
        assert raised.mean() == pytest.approx(probability, abs=0.02)
        effect = spec.generative_params["coefficients"][feature]
        assert (raised.mean() - base.mean()) * effect > 0


def test_categorical_probabilities_do_not_pin_and_vary_with_each_feature(tmp_path: Path) -> None:
    spec = _scenario("categorical_choice")
    env = _compile(spec, tmp_path)
    lower = _lower_query(spec)
    base = _draws(env, lower)
    assert set(base) == {0, 1, 2}
    base_frequencies = np.bincount(base, minlength=3) / len(base)
    for feature, domain in spec.io_schema.feature_domains.items():
        raised = _draws(env, {**lower, feature: domain[1]})
        assert set(raised) == {0, 1, 2}
        frequencies = np.bincount(raised, minlength=3) / len(raised)
        assert np.abs(frequencies - base_frequencies).sum() > 0.05


def test_mixture_keeps_response_centers_and_each_covariate_effect(tmp_path: Path) -> None:
    spec = _scenario("mixture_continuous")
    env = _compile(spec, tmp_path)
    lower = _lower_query(spec)
    base = _draws(env, lower)
    assert np.all((base > 70.0) & (base < 130.0))
    assert np.any(base < 100.0) and np.any(base > 100.0)
    for feature, effect in spec.generative_params["coefficients"].items():
        raised = _draws(env, {**lower, feature: spec.io_schema.feature_domains[feature][1]})
        assert np.all((raised > 70.0) & (raised < 130.0))
        np.testing.assert_allclose(raised - base, effect, rtol=0, atol=1e-12)


def test_continuous_input_zero_inflation_remains_a_mixture_and_varies(tmp_path: Path) -> None:
    spec = _scenario("zero_inflated_count")
    env = _compile(spec, tmp_path)
    lower = _lower_query(spec)
    base = _draws(env, lower, n=4096)
    # 40% structural zeros, otherwise Poisson(4): E[Y]=2.4 and Var[Y]=6.24.
    assert base.mean() == pytest.approx(2.4, abs=0.2)
    assert base.var() == pytest.approx(6.24, abs=0.8)
    assert (base == 0).mean() == pytest.approx(0.410989, abs=0.035)
    for feature, effect in spec.generative_params["coefficients"].items():
        raised = _draws(env, {**lower, feature: spec.io_schema.feature_domains[feature][1]})
        assert np.all((raised >= 0) & (raised < 40))
        assert (raised.mean() - base.mean()) * effect > 0


@pytest.mark.parametrize("family", DIRECT_LINEAR_FAMILIES)
def test_units_and_origins_do_not_change_the_response_law(tmp_path: Path, family: str) -> None:
    spec = _scenario(family)
    unit_spec = spec.model_copy(deep=True)
    unit_spec.io_schema.feature_domains = dict.fromkeys(spec.io_schema.feature_names, (0.0, 1.0))
    natural_env = _compile(spec, tmp_path / "natural")
    unit_env = _compile(unit_spec, tmp_path / "unit")
    for fraction in (0.0, 0.25, 0.5, 1.0):
        natural_query = {
            name: lo + fraction * (hi - lo)
            for name, (lo, hi) in spec.io_schema.feature_domains.items()
        }
        unit_query = dict.fromkeys(spec.io_schema.feature_names, fraction)
        np.testing.assert_allclose(
            _draws(natural_env, natural_query, n=128),
            _draws(unit_env, unit_query, n=128),
            rtol=0,
            atol=1e-12,
        )


@given(
    offset=st.integers(min_value=-10000, max_value=10000),
    width=st.integers(min_value=1, max_value=10000),
    numerator=st.integers(min_value=0, max_value=16),
)
@example(offset=0, width=1, numerator=1)
@settings(max_examples=20, deadline=None)
def test_domain_position_determines_linear_effect(offset: int, width: int, numerator: int) -> None:
    """Affine-domain property checked against a worked response at a unit fraction."""
    spec = _scenario("regression_continuous")
    spec.io_schema.feature_names = ["age_years"]
    spec.io_schema.feature_types = {"age_years": "continuous"}
    spec.io_schema.feature_domains = {"age_years": (float(offset), float(offset + width))}
    spec.generative_params = {
        "intercept": 100.0,
        "coefficients": {"age_years": 16.0},
        "noise_sigma": 0.0,
    }
    # Isolate the generated module for each Hypothesis example.
    with tempfile.TemporaryDirectory(prefix="response-law-property-") as directory:
        env = _compile(spec, Path(directory))
        query = json.dumps({"age_years": offset + width * numerator / 16})
        value, accepted = env.run_experiment(query)
    assert accepted
    assert value == pytest.approx(100.0 + numerator, abs=1e-12)


@pytest.mark.parametrize("family", DIRECT_LINEAR_FAMILIES)
@pytest.mark.parametrize("defect", ["missing", "extra", "infinite", "nan"])
def test_invalid_coefficient_contract_is_refused(tmp_path: Path, family: str, defect: str) -> None:
    spec = _scenario(family)
    coefficients = spec.generative_params["coefficients"]
    feature = spec.io_schema.feature_names[0]
    if defect == "missing":
        del coefficients[feature]
    elif defect == "extra":
        coefficients["undeclared_feature"] = 1.0
    else:
        coefficients[feature] = float("inf" if defect == "infinite" else "nan")
    with pytest.raises(ValueError, match=family + ".*coefficient"):
        _compile(spec, tmp_path)


@pytest.mark.parametrize("family", tuple(_original_specs()))
def test_legacy_provider_parameter_names_do_not_silently_become_defaults(
    tmp_path: Path, family: str
) -> None:
    with pytest.raises(ValueError, match=family + ".*coefficient"):
        _compile(_original_specs()[family], tmp_path)


def test_count_feature_appointments_still_fail_before_stage_c_draws(tmp_path: Path) -> None:
    spec = _original_specs()["zero_inflated_count"]
    # Supply a valid response law solely to reach the existing stage-C input-type
    # guard; the original count-valued prior_visits_12mo declaration is unchanged.
    spec.generative_params = {
        "intercept": 0.0,
        "coefficients": {name: 0.5 for name in spec.io_schema.feature_names},
        "zero_inflation": 0.4,
    }
    paths = RunPaths(
        run_root=tmp_path,
        run_id="appointments_type_refusal",
        run_dir=tmp_path / "appointments_type_refusal",
        stage_dirs={stage: tmp_path / "appointments_type_refusal" / stage for stage in "ABC"},
    )
    paths.stage_dirs["A"].mkdir(parents=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec.model_dump(mode="json")])
    run_stage_b(paths)
    with pytest.raises(
        RuntimeError,
        match="zero_inflated_count.*prior_visits_12mo.*unsupported type 'count'",
    ):
        run_stage_c(paths, split_policy="auto", visible_ratio=0.8, seed=0)
    assert not paths.stage_dirs["C"].exists()


def _time_spec(family: str) -> EnvSpec:
    return EnvSpec(
        env_id=f"scaled_{family}",
        env_name=f"scaled_{family}",
        domain="engineering",
        complexity="simple",
        model_type="predictive",
        key_concepts=[family],
        io_schema=IOSchema(
            input_type=(
                InputType.int if family == "time_series_level_seasonality" else InputType.float
            ),
            input_description="Elapsed time between 1000 and 2000 seconds.",
            output_type="float",
            target_name="response",
            feature_names=["t"],
            feature_types={"t": "continuous"},
            feature_domains={"t": (1000.0, 2000.0)},
        ),
        data_plan=DataPlan(n_observations=160),
        generative_family=family,
        generative_params=(
            {
                "base_level": 100.0,
                "trend": 20.0,
                "noise_sigma": 2.0,
                "season_amplitude": 0.0,
                "season_period": 12.0,
            }
            if family == "time_series_level_seasonality"
            else {"population_n": 100, "rate_mean": 1.0, "rate_sigma": 0.0}
        ),
    )


def test_time_trend_uses_declared_domain_without_rescaling_response_noise(tmp_path: Path) -> None:
    env = _compile(_time_spec("time_series_level_seasonality"), tmp_path)
    lower = _draws(env, "1000")
    upper = _draws(env, "2000")
    assert lower.mean() == pytest.approx(100.0, abs=0.2)
    assert upper.mean() == pytest.approx(120.0, abs=0.2)
    assert lower.std() == pytest.approx(2.0, abs=0.15)
    np.testing.assert_allclose(upper - lower, 20.0, rtol=0, atol=1e-12)


def test_infection_rate_uses_declared_time_domain_without_saturation(tmp_path: Path) -> None:
    env = _compile(_time_spec("infection_process"), tmp_path)
    lower = _draws(env, "1000")
    midpoint = _draws(env, "1500")
    upper = _draws(env, "2000")
    assert np.all(lower == 0)
    assert midpoint.mean() == pytest.approx(39.3469, abs=1.0)
    assert upper.mean() == pytest.approx(63.2121, abs=1.0)
    assert 0 < midpoint.mean() < upper.mean() < 100


@pytest.mark.parametrize(
    "family", (*DIRECT_LINEAR_FAMILIES, "time_series_level_seasonality", "infection_process")
)
def test_generated_environment_exposes_its_declared_process(tmp_path: Path, family: str) -> None:
    spec = _scenario(family) if family in DIRECT_LINEAR_FAMILIES else _time_spec(family)
    env = _compile(spec, tmp_path)
    # This verifies availability; a reader must check the declared mean, noise and
    # link against the actual process. Keyword presence cannot establish truth.
    assert type(env).__doc__ and type(env).__doc__.strip()


@pytest.mark.parametrize(
    ("family", "parameter", "invalid"),
    [
        ("regression_continuous", "intercept", "nan"),
        ("regression_continuous", "intercept", float("nan")),
        ("regression_continuous", "intercept", float("inf")),
        ("regression_continuous", "intercept", "invalid"),
        ("regression_continuous", "intercept", True),
        ("regression_continuous", "noise_sigma", float("nan")),
        ("regression_continuous", "noise_sigma", -1.0),
        ("regression_continuous", "noise_sigma", "1.0"),
        ("regression_continuous", "noise_sigma", False),
        ("regression_count", "intercept", float("inf")),
        ("binary_choice", "intercept", "nan"),
        ("categorical_choice", "intercept", float("nan")),
        ("categorical_choice", "n_categories", 2.5),
        ("categorical_choice", "n_categories", 0),
        ("categorical_choice", "n_categories", -1),
        ("categorical_choice", "n_categories", True),
        ("mixture_continuous", "n_components", 1.5),
        ("mixture_continuous", "n_components", 0),
        ("mixture_continuous", "n_components", -1),
        ("mixture_continuous", "n_components", "2"),
        ("mixture_continuous", "noise_sigma", float("inf")),
        ("mixture_continuous", "component_mean_lower", float("inf")),
        ("mixture_continuous", "component_mean_upper", float("nan")),
        ("zero_inflated_count", "intercept", float("nan")),
        ("zero_inflated_count", "zero_inflation", float("nan")),
        ("zero_inflated_count", "zero_inflation", float("inf")),
        ("zero_inflated_count", "zero_inflation", -0.1),
        ("zero_inflated_count", "zero_inflation", 1.1),
        ("zero_inflated_count", "zero_inflation", "0.4"),
        ("zero_inflated_count", "zero_inflation", False),
    ],
)
def test_invalid_scalar_response_parameter_is_refused_before_module_write(
    tmp_path: Path, family: str, parameter: str, invalid: object
) -> None:
    spec = _scenario(family)
    spec.generative_params[parameter] = invalid

    with pytest.raises(ValueError, match=family + ".*" + parameter):
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)

    assert not (tmp_path / f"{spec.env_name}.py").exists()


@pytest.mark.parametrize(
    ("family", "parameter", "valid"),
    [
        ("regression_continuous", "noise_sigma", 0.0),
        ("categorical_choice", "n_categories", 1),
        ("mixture_continuous", "n_components", 1),
        ("zero_inflated_count", "zero_inflation", 0.0),
        ("zero_inflated_count", "zero_inflation", 1.0),
    ],
)
def test_valid_scalar_response_parameter_boundaries_produce_finite_observations(
    tmp_path: Path, family: str, parameter: str, valid: float
) -> None:
    spec = _scenario(family)
    spec.generative_params[parameter] = valid
    env = _compile(spec, tmp_path)

    values = _draws(env, _lower_query(spec), n=64)

    assert np.isfinite(values).all()
    if parameter == "noise_sigma":
        np.testing.assert_allclose(values, 129.74, rtol=0, atol=1e-12)
    elif parameter == "n_categories" or (parameter == "zero_inflation" and valid == 1):
        assert np.all(values == 0)


EXPLICIT_OTHER_PARAMETERS = {
    "hierarchical_grouped": {"base_mean": 20.0, "noise_sigma": 1.0},
    "survival_with_censoring": {
        "n_subjects": 5,
        "baseline_rate": 0.1,
        "covariate_effect": 0.5,
        "censor_time": 10.0,
    },
    "ode_predator_prey": {
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 1.5,
        "delta": 0.075,
        "prey_init": 40.0,
        "predator_init": 9.0,
        "noise_sigma": 2.0,
    },
    "irt_binary_response": {
        "n_students": 10,
        "n_questions": 5,
        "ability_mean": 0.0,
        "ability_sigma": 1.0,
        "difficulty_mean": 0.0,
        "difficulty_sigma": 1.0,
        "discrimination_mean": 1.0,
    },
    "spatial_signal_source": {
        "n_sources": 2,
        "dim": 2,
        "background": 1.0,
        "regularization_m": 1.0,
        "source_strength": 10.0,
        "noise_sigma": 0.5,
    },
    "choice_with_discounting": {
        "epsilon": 0.1,
        "k_mean": 0.05,
        "k_sigma": 0.0,
        "choice_scale": 250.0,
    },
}
ALL_RESPONSE_FAMILIES = (
    *DIRECT_LINEAR_FAMILIES,
    "time_series_level_seasonality",
    "infection_process",
    *EXPLICIT_OTHER_PARAMETERS,
)


def _complete_spec(family: str) -> EnvSpec:
    if family in DIRECT_LINEAR_FAMILIES:
        return _scenario(family)
    spec = _time_spec(family)
    if family in EXPLICIT_OTHER_PARAMETERS:
        spec.generative_params = dict(EXPLICIT_OTHER_PARAMETERS[family])
    if family == "survival_with_censoring":
        spec.io_schema.feature_names = ["subject_id"]
        spec.io_schema.feature_domains = {"subject_id": (0.0, 4.0)}
        spec.io_schema.feature_types = {"subject_id": "integer"}
        spec.io_schema.input_type = InputType.int
    elif family == "hierarchical_grouped":
        spec.io_schema.feature_names = ["group"]
        spec.io_schema.feature_domains = {"group": (0.0, 3.0)}
        spec.io_schema.feature_types = {"group": "integer"}
        spec.io_schema.input_type = InputType.json
        spec.data_plan.grouping = ["group"]
    elif family == "irt_binary_response":
        spec.io_schema.feature_names = ["student_id", "question_id"]
        spec.io_schema.feature_domains = {"student_id": (0.0, 9.0), "question_id": (0.0, 4.0)}
        spec.io_schema.feature_types = {"student_id": "integer", "question_id": "integer"}
        spec.io_schema.input_type = InputType.json
    elif family == "spatial_signal_source":
        spec.io_schema.feature_names = ["x", "y"]
        spec.io_schema.feature_domains = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
        spec.io_schema.feature_types = {"x": "continuous", "y": "continuous"}
        spec.io_schema.input_type = InputType.json
    elif family == "choice_with_discounting":
        spec.io_schema.input_type = InputType.json
        spec.io_schema.feature_names = ["immediate_reward", "delayed_reward", "delay_days"]
        spec.io_schema.feature_domains = {
            "immediate_reward": (5.0, 1000.0),
            "delayed_reward": (50.0, 5000.0),
            "delay_days": (0.0, 365.0),
        }
        spec.io_schema.feature_types = dict.fromkeys(spec.io_schema.feature_names, "continuous")
    return spec


@pytest.mark.parametrize(
    ("family", "parameter"),
    [
        (family, parameter)
        for family in ALL_RESPONSE_FAMILIES
        for parameter in _complete_spec(family).generative_params
    ],
)
def test_missing_required_response_parameter_names_family_and_parameter(
    tmp_path: Path,
    family: str,
    parameter: str,
) -> None:
    spec = _complete_spec(family)
    del spec.generative_params[parameter]
    with pytest.raises(ValueError) as exc:
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)
    assert family in str(exc.value)
    assert parameter in str(exc.value)
    assert not (tmp_path / f"{spec.env_name}.py").exists()


@pytest.mark.parametrize("family", ALL_RESPONSE_FAMILIES)
def test_unknown_response_parameter_names_family_and_parameter(
    tmp_path: Path,
    family: str,
) -> None:
    spec = _complete_spec(family)
    spec.generative_params["misspelled_parameter"] = 0.1
    with pytest.raises(ValueError) as exc:
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)
    assert family in str(exc.value)
    assert "misspelled_parameter" in str(exc.value)


def test_discounting_domain_endpoints_only_bound_queries(tmp_path: Path) -> None:
    spec = _complete_spec("choice_with_discounting")
    wider_spec = spec.model_copy(deep=True)
    wider_spec.io_schema.feature_domains = {
        "immediate_reward": (0.0, 10000.0),
        "delayed_reward": (0.0, 50000.0),
        "delay_days": (0.0, 1000.0),
    }
    natural = _compile(spec, tmp_path / "natural")
    wider = _compile(wider_spec, tmp_path / "wider")
    for immediate, delayed, delay in [(100, 120, 30), (500, 1000, 7), (25, 5000, 0)]:
        query = {"immediate_reward": immediate, "delayed_reward": delayed, "delay_days": delay}
        np.testing.assert_array_equal(
            _draws(natural, query, n=256),
            _draws(wider, query, n=256),
        )


def test_discounting_corner_probabilities_reach_beyond_repair2_ceiling(tmp_path: Path) -> None:
    spec = _complete_spec("choice_with_discounting")
    env = _compile(spec, tmp_path)
    lower = _lower_query(spec)
    cases = [
        ({}, 0.540391),
        ({"immediate_reward": 1000.0}, 0.069693),
        ({"delayed_reward": 5000.0}, 0.95),
    ]
    probabilities = []
    for changed, expected in cases:
        values = _draws(env, {**lower, **changed}, n=8192)
        assert set(values) == {0, 1}
        assert values.mean() == pytest.approx(expected, abs=0.025)
        probabilities.append(values.mean())
    assert min(probabilities) < 0.1
    assert max(probabilities) > 0.9


@pytest.mark.parametrize(
    ("immediate", "delayed", "days", "expected"),
    [(100.0, 120.0, 30.0, 0.453368), (500.0, 1000.0, 7.0, 0.701342)],
)
def test_discounting_natural_reward_examples_have_midrange_probabilities(
    tmp_path: Path,
    immediate: float,
    delayed: float,
    days: float,
    expected: float,
) -> None:
    env = _compile(_complete_spec("choice_with_discounting"), tmp_path)
    samples = _draws(
        env,
        {
            "immediate_reward": immediate,
            "delayed_reward": delayed,
            "delay_days": days,
        },
        n=8192,
    )
    assert samples.mean() == pytest.approx(expected, abs=0.025)
    assert 0.2 < samples.mean() < 0.8


@pytest.mark.parametrize("invalid", [None, 0.0, -1.0, float("nan"), True])
def test_discounting_requires_positive_choice_scale(tmp_path: Path, invalid: object) -> None:
    spec = _complete_spec("choice_with_discounting")
    if invalid is None:
        del spec.generative_params["choice_scale"]
    else:
        spec.generative_params["choice_scale"] = invalid
    with pytest.raises(ValueError, match="choice_with_discounting.*choice_scale"):
        TemplateCompiler().compile_envspec(spec, tmp_path)


def test_discounting_reward_unit_conversion_also_converts_choice_scale(tmp_path: Path) -> None:
    spec = _complete_spec("choice_with_discounting")
    cents = spec.model_copy(deep=True)
    cents.generative_params["choice_scale"] *= 100
    for name in ("immediate_reward", "delayed_reward"):
        cents.io_schema.feature_domains[name] = tuple(
            endpoint * 100 for endpoint in cents.io_schema.feature_domains[name]
        )
    natural = _compile(spec, tmp_path / "natural")
    converted = _compile(cents, tmp_path / "cents")
    np.testing.assert_array_equal(
        _draws(natural, {"immediate_reward": 100, "delayed_reward": 120, "delay_days": 30}),
        _draws(converted, {"immediate_reward": 10000, "delayed_reward": 12000, "delay_days": 30}),
    )


@pytest.mark.parametrize("feature", ["immediate_reward", "delayed_reward", "delay_days"])
def test_discounting_requires_and_enforces_each_declared_domain(
    tmp_path: Path,
    feature: str,
) -> None:
    spec = _complete_spec("choice_with_discounting")
    env = _compile(spec, tmp_path)
    lo, hi = spec.io_schema.feature_domains[feature]
    result, accepted = env.run_experiment(json.dumps({**_lower_query(spec), feature: hi + 1}))
    assert not accepted
    assert "choice_with_discounting" in result
    assert feature in result and str(hi + 1) in result and str(lo) in result
    assert env.get_data() == []
    del spec.io_schema.feature_domains[feature]
    with pytest.raises(ValueError, match="choice_with_discounting.*" + feature):
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path / "missing")


def test_discounting_refuses_a_declared_negative_delay_domain(tmp_path: Path) -> None:
    spec = _complete_spec("choice_with_discounting")
    spec.io_schema.feature_domains["delay_days"] = (-7.0, 37.0)
    with pytest.raises(ValueError, match="choice_with_discounting.*delay_days"):
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)


def test_ode_refuses_a_declared_negative_time_domain(tmp_path: Path) -> None:
    spec = _complete_spec("ode_predator_prey")
    spec.io_schema.feature_domains["t"] = (-1.0, 1.0)
    with pytest.raises(ValueError, match="ode_predator_prey.*feature_domains.*'t'"):
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)
    assert not (tmp_path / f"{spec.env_name}.py").exists()


def test_ode_accepts_a_declared_zero_time_lower_bound(tmp_path: Path) -> None:
    spec = _complete_spec("ode_predator_prey")
    spec.io_schema.feature_domains["t"] = (0.0, 1.0)
    spec.generative_params["noise_sigma"] = 0.0
    env = _compile(spec, tmp_path)
    result, accepted = env.run_experiment("0")
    assert accepted, result
    assert result == (40, 9)


@pytest.mark.parametrize("family", ALL_RESPONSE_FAMILIES)
@pytest.mark.parametrize("target", [None, "", " "])
def test_required_target_name_never_becomes_y(
    tmp_path: Path,
    family: str,
    target: str | None,
) -> None:
    spec = _complete_spec(family)
    spec.io_schema.target_name = target
    with pytest.raises(ValueError, match=family + ".*target_name"):
        TemplateCompiler().compile_envspec(spec, tmp_path)
    assert not (tmp_path / f"{spec.env_name}.py").exists()


def test_required_grouping_never_becomes_group(tmp_path: Path) -> None:
    spec = _complete_spec("hierarchical_grouped")
    spec.data_plan.grouping = []
    with pytest.raises(ValueError, match="hierarchical_grouped.*grouping"):
        TemplateCompiler().compile_envspec(spec, tmp_path)


@pytest.mark.parametrize("defect", ["empty_features", "missing_domain", "zero_width", "inverted"])
def test_compiler_domain_refusals_name_family_and_parameter(tmp_path: Path, defect: str) -> None:
    spec = _scenario("regression_continuous")
    name = spec.io_schema.feature_names[0]
    parameter = name
    if defect == "empty_features":
        spec.io_schema.feature_names = []
        parameter = "feature_names"
    elif defect == "missing_domain":
        del spec.io_schema.feature_domains[name]
    else:
        spec.io_schema.feature_domains[name] = (5.0, 5.0 if defect == "zero_width" else 4.0)
    with pytest.raises(ValueError, match="regression_continuous.*" + parameter):
        TemplateCompiler().compile_envspec(spec, tmp_path)


@pytest.mark.parametrize("family", ALL_RESPONSE_FAMILIES)
def test_every_family_requires_and_enforces_declared_query_domains(
    tmp_path: Path,
    family: str,
) -> None:
    spec = _complete_spec(family)
    env = _compile(spec, tmp_path / "valid")
    name = spec.io_schema.feature_names[0]
    lo, hi = spec.io_schema.feature_domains[name]
    query = {**_lower_query(spec), name: hi + 1}
    text = json.dumps(query) if spec.io_schema.input_type == "json" else str(hi + 1)
    result, accepted = env.run_experiment(text)
    assert not accepted
    assert family in result and name in result
    assert str(lo) in result and str(hi) in result and str(hi + 1) in result
    assert env.get_data() == []
    del spec.io_schema.feature_domains[name]
    with pytest.raises(ValueError, match=family + ".*" + name):
        TemplateCompiler().compile_envspec(spec, tmp_path / "missing")


@pytest.mark.parametrize(
    ("family", "name", "valid", "outside", "count"),
    [
        ("irt_binary_response", "student_id", 2, 0, "n_students"),
        ("irt_binary_response", "question_id", 2, 0, "n_questions"),
        ("survival_with_censoring", "subject_id", 2, 0, "n_subjects"),
    ],
)
def test_identifier_domains_are_explicit_subsets_of_structural_indices(
    tmp_path: Path,
    family: str,
    name: str,
    valid: int,
    outside: int,
    count: str,
) -> None:
    spec = _complete_spec(family)
    spec.io_schema.feature_domains[name] = (1.0, 3.0)
    env = _compile(spec, tmp_path / "valid")
    for value, expected in [(valid, True), (outside, False), (1.5, False), (True, False)]:
        query = {**_lower_query(spec), name: value}
        text = json.dumps(query) if spec.io_schema.input_type == "json" else str(value)
        before = len(env.get_data())
        result, accepted = env.run_experiment(text)
        assert accepted is expected, result
        assert len(env.get_data()) == before + int(expected)
        if not expected:
            assert family in result and name in result
    spec.io_schema.feature_domains[name] = (0.0, float(spec.generative_params[count]))
    with pytest.raises(ValueError, match=family + ".*" + name):
        TemplateCompiler().compile_envspec(spec, tmp_path / "outside_structure")


@pytest.mark.parametrize("family", ["irt_binary_response", "survival_with_censoring"])
def test_one_subject_or_item_has_a_singleton_identifier_domain(tmp_path: Path, family: str) -> None:
    spec = _complete_spec(family)
    counts = (
        {"subject_id": "n_subjects"}
        if family == "survival_with_censoring"
        else {
            "student_id": "n_students",
            "question_id": "n_questions",
        }
    )
    for name, count in counts.items():
        spec.generative_params[count] = 1
        spec.io_schema.feature_domains[name] = (0.0, 0.0)
    # This is a real specification round trip, so schema and compiler both see it.
    spec = EnvSpec.model_validate(spec.model_dump(mode="json"))
    env = _compile(spec, tmp_path)
    query = dict.fromkeys(counts, 0)
    text = json.dumps(query) if family == "irt_binary_response" else "0"
    result, accepted = env.run_experiment(text)
    assert accepted, result


@pytest.mark.parametrize(
    "family",
    [
        "choice_with_discounting",
        "hierarchical_grouped",
        "irt_binary_response",
        "spatial_signal_source",
        "survival_with_censoring",
        "time_series_level_seasonality",
        "ode_predator_prey",
    ],
)
def test_shared_environment_api_preserves_query_order_rows_and_failure_state(
    tmp_path: Path,
    family: str,
) -> None:
    spec = _complete_spec(family)
    if family == "ode_predator_prey":
        spec.io_schema.feature_domains = {"t": (0.0, 4.0)}
    env = _compile(spec, tmp_path)
    query = _lower_query(spec)
    text = (
        json.dumps(dict(reversed(list(query.items()))))
        if spec.io_schema.input_type == "json"
        else str(next(iter(query.values())))
    )
    # The empty DataFrame already has the same schema as a populated one.
    columns = env.get_ordered_column_names()
    assert list(env.get_df().columns) == columns
    value, accepted = env.run_experiment(text)
    assert accepted, value
    if family == "survival_with_censoring":
        sid = int(query["subject_id"])
        expected = (sid, env.response_law["latent_parameters"]["covariates"][sid], *value)
        assert columns == ["subject_id", "covariate", "response", "censored"]
    else:
        response = value if isinstance(value, tuple) else (value,)
        expected = (*query.values(), *response)
    assert env.get_data() == [expected]
    assert tuple(env.get_df().itertuples(index=False, name=None))[0] == expected
    copied = env.get_data()
    copied.clear()
    result, accepted = env.run_experiment("invalid-input")
    assert not accepted and family in result
    for name in spec.io_schema.feature_names:
        assert repr(name) in result
    assert env.get_data() == [expected]
    env.reset(seed=0)
    assert env.get_data() == []
    assert list(env.get_df().columns) == columns


def test_irt_expit_preserves_corner_probability_and_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _compile(_complete_spec("irt_binary_response"), tmp_path)
    step_globals = env.step.__globals__
    assert step_globals.get("expit") is expit
    assert not hasattr(env, "_sigmoid")
    probabilities = []

    def record_expit(value: float) -> float:
        probability = float(expit(value))
        probabilities.append(probability)
        return probability

    # Observe the real expit output while retaining the real Bernoulli draws.
    monkeypatch.setitem(step_globals, "expit", record_expit)
    responses = _draws(env, {"student_id": 9, "question_id": 4}, n=32)
    # Recorded from the private sigmoid before its removal, with seed 0.
    np.testing.assert_allclose(probabilities, [0.493597220424702] * 32, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(
        responses,
        [
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
        ],
    )


@pytest.mark.parametrize(("time", "prey"), [(0.0, 40), (0.001, 44), (0.005, 66)])
def test_ode_uses_initial_state_at_zero_and_does_not_floor_small_times(
    tmp_path: Path,
    time: float,
    prey: int,
) -> None:
    spec = _complete_spec("ode_predator_prey")
    spec.io_schema.feature_domains = {"t": (0.0, 0.02)}
    spec.generative_params = {
        "alpha": 100.0,
        "beta": 0.0,
        "gamma": 0.0,
        "delta": 0.0,
        "prey_init": 40.0,
        "predator_init": 9.0,
        "noise_sigma": 0.0,
    }
    env = _compile(spec, tmp_path)
    # beta=gamma=delta=0 gives prey=40*exp(100*t), predator=9 exactly.
    result, accepted = env.run_experiment(str(time))
    assert accepted
    assert result == (prey, 9)
    assert env.get_data() == [(time, prey, 9)]


def test_spatial_sources_use_the_declared_coordinate_domains(tmp_path: Path) -> None:
    spec = _complete_spec("spatial_signal_source")
    spec.io_schema.feature_domains = {"x": (100.0, 1100.0), "y": (2000.0, 2200.0)}
    env = _compile(spec, tmp_path)
    for seed in (0, 17):
        env.reset(seed=seed)
        positions = np.asarray(env.response_law["latent_parameters"]["source_positions"])
        assert positions.shape == (2, 2)
        assert ((positions[:, 0] >= 100.0) & (positions[:, 0] <= 1100.0)).all()
        assert ((positions[:, 1] >= 2000.0) & (positions[:, 1] <= 2200.0)).all()


@pytest.mark.parametrize("family", ALL_RESPONSE_FAMILIES)
@pytest.mark.parametrize("method", ["validate_input", "run_experiment", "get_data", "get_df"])
def test_families_inherit_the_shared_query_and_observation_contract(
    tmp_path: Path,
    family: str,
    method: str,
) -> None:
    env = _compile(_complete_spec(family), tmp_path)
    assert getattr(type(env), method) is getattr(BaseEnv, method)


@pytest.mark.parametrize("family", ALL_RESPONSE_FAMILIES)
@pytest.mark.parametrize("defect", ["malformed", "missing", "boolean", "nonfinite", "nonnumeric"])
def test_query_refusals_name_family_and_offending_parameter(
    tmp_path: Path,
    family: str,
    defect: str,
) -> None:
    spec = _complete_spec(family)
    env = _compile(spec, tmp_path)
    name = spec.io_schema.feature_names[0]
    query = _lower_query(spec)
    if defect == "malformed":
        text = "invalid-input"
        parameter = "input"
    elif defect == "missing":
        del query[name]
        text = json.dumps(query) if spec.io_schema.input_type == "json" else ""
        parameter = name
    else:
        value = {"boolean": True, "nonfinite": float("inf"), "nonnumeric": "abc"}[defect]
        query[name] = value
        text = json.dumps(query) if spec.io_schema.input_type == "json" else str(value)
        parameter = name
    result, accepted = env.run_experiment(text)
    assert not accepted, result
    assert family in result
    assert parameter in result
    assert env.get_data() == []


@pytest.mark.parametrize("family", DIRECT_LINEAR_FAMILIES)
def test_hostile_feature_name_remains_a_literal_through_real_render_and_query(
    tmp_path: Path,
    family: str,
) -> None:
    spec = _scenario(family)
    hostile_name = 'x"); raise RuntimeError("unescaped feature name"); #'
    original_name = spec.io_schema.feature_names[0]
    spec.io_schema.feature_names[0] = hostile_name
    spec.io_schema.feature_domains[hostile_name] = spec.io_schema.feature_domains.pop(original_name)
    spec.io_schema.feature_types[hostile_name] = spec.io_schema.feature_types.pop(original_name)
    coefficients = spec.generative_params["coefficients"]
    coefficients[hostile_name] = coefficients.pop(original_name)
    env = _compile(spec, tmp_path)
    result, accepted = env.run_experiment(json.dumps(_lower_query(spec)))
    assert accepted, result
    assert np.isfinite(result)
    assert hostile_name in env.get_ordered_features()
    assert hostile_name in env.get_df().columns
    declared_effect = env.response_law["parameters"]["coefficients"]["value"][hostile_name]
    assert declared_effect == coefficients[hostile_name]


@pytest.mark.parametrize("family", ALL_RESPONSE_FAMILIES)
def test_compiler_refuses_declared_domain_that_has_no_query_feature(
    tmp_path: Path,
    family: str,
) -> None:
    spec = _complete_spec(family)
    spec.io_schema.feature_domains["unused_domain"] = (1.0, 2.0)
    with pytest.raises(ValueError, match=family + ".*feature_domains.*unused_domain"):
        TemplateCompiler().compile_envspec(spec, tmp_path)


@pytest.mark.parametrize(
    "family",
    [
        "choice_with_discounting",
        "hierarchical_grouped",
        "irt_binary_response",
        "spatial_signal_source",
        "survival_with_censoring",
        "time_series_level_seasonality",
        "infection_process",
        "ode_predator_prey",
    ],
)
@pytest.mark.parametrize("defect", ["missing", "extra"])
def test_structural_family_refuses_feature_names_it_cannot_query(
    tmp_path: Path,
    family: str,
    defect: str,
) -> None:
    spec = _complete_spec(family)
    if defect == "missing":
        spec.io_schema.feature_names.pop()
    else:
        spec.io_schema.feature_names.append("unused_feature")
        spec.io_schema.feature_domains["unused_feature"] = (1.0, 2.0)
        spec.io_schema.feature_types["unused_feature"] = "continuous"
    with pytest.raises(ValueError, match=family + ".*feature_names"):
        TemplateCompiler().compile_envspec(spec, tmp_path)


@pytest.mark.parametrize("family", DIRECT_LINEAR_FAMILIES)
def test_duplicate_feature_name_cannot_silently_double_its_effect(
    tmp_path: Path,
    family: str,
) -> None:
    spec = _scenario(family)
    spec.io_schema.feature_names.append(spec.io_schema.feature_names[0])
    with pytest.raises(ValueError, match=family + ".*feature_names"):
        TemplateCompiler().compile_envspec(spec, tmp_path)
    assert not (tmp_path / f"{spec.env_name}.py").exists()

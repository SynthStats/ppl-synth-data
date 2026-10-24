from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ppl_synth_data.envs.base import InputValidationError
from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import load_env_instance
from ppl_synth_data.pipeline.io import write_jsonl
from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b
from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c
from ppl_synth_data.pipeline.types import RunPaths
from ppl_synth_data.specs.models import DataPlan, EnvSpec, InputType, IOSchema

NUMERIC_FAMILIES = (
    "regression_continuous",
    "regression_count",
    "zero_inflated_count",
    "binary_choice",
    "categorical_choice",
    "mixture_continuous",
)


def _numeric_spec(family: str, domains: dict[str, tuple[float, float]]) -> EnvSpec:
    feature_names = list(reversed(domains))
    weights = (0.001, 0.00115)
    params = {
        "regression_continuous": {"intercept": 0.0, "noise_sigma": 1.0},
        "regression_count": {"intercept": 0.5},
        "zero_inflated_count": {"intercept": 0.5, "zero_inflation": 0.35},
        "binary_choice": {"intercept": 0.0},
        "categorical_choice": {"intercept": 0.0, "n_categories": 3},
        "mixture_continuous": {
            "n_components": 2,
            "noise_sigma": 1.0,
            "component_mean_lower": 0.0,
            "component_mean_upper": 10.0,
        },
        "infection_process": {"population_n": 100, "rate_mean": 0.05, "rate_sigma": 0.01},
        "ode_predator_prey": {
            "alpha": 1.0,
            "beta": 0.1,
            "gamma": 1.5,
            "delta": 0.075,
            "prey_init": 40.0,
            "predator_init": 9.0,
            "noise_sigma": 2.0,
        },
        "time_series_level_seasonality": {
            "base_level": 10.0,
            "trend": 0.1,
            "season_period": 12.0,
            "season_amplitude": 2.0,
            "noise_sigma": 1.0,
        },
        "survival_with_censoring": {
            "baseline_rate": 0.1,
            "covariate_effect": 0.5,
            "censor_time": 10.0,
            "n_subjects": 50,
        },
    }[family]
    if family in NUMERIC_FAMILIES:
        params["coefficients"] = dict(
            zip(feature_names, weights[: len(feature_names)], strict=True)
        )
    return EnvSpec(
        env_id=f"bounds_{family}",
        env_name=f"bounds_{family}",
        domain="healthcare",
        complexity="simple",
        model_type="predictive",
        key_concepts=[family],
        io_schema=IOSchema(
            input_type="float",
            input_description="Numeric inputs in their explicitly declared domains.",
            output_type="float",
            target_name="y",
            # Different map/name order checks that the domain follows its name.
            feature_names=feature_names,
            feature_types={name: "continuous" for name in domains},
            feature_domains=domains,
        ),
        data_plan=DataPlan(n_observations=24),
        generative_family=family,
        generative_params=params,
    )


def _compile_and_load(spec: EnvSpec, out_dir: Path):
    module = TemplateCompiler().compile_envspec(spec, out_dir=out_dir)
    return load_env_instance(module, env_name=spec.env_name, seed=0, trusted_root=out_dir)


def test_stage_b_and_c_materialize_covariates_on_disjoint_scales(tmp_path: Path) -> None:
    spec = EnvSpec(
        env_id="disjoint_feature_domains",
        env_name="disjoint_feature_domains",
        domain="healthcare",
        complexity="simple",
        model_type="predictive",
        key_concepts=["regression"],
        io_schema=IOSchema(
            input_type="json",
            input_description="BMI in 18-42 kg/m² and daily sodium in 1200-5200 mg.",
            output_type="float",
            target_name="systolic_bp",
            feature_names=["bmi_kg_m2", "daily_sodium_mg"],
            feature_types={"bmi_kg_m2": "continuous", "daily_sodium_mg": "continuous"},
            feature_domains={
                "bmi_kg_m2": (18.0, 42.0),
                "daily_sodium_mg": (1200.0, 5200.0),
            },
        ),
        data_plan=DataPlan(
            n_observations=24,
            covariates=["bmi_kg_m2", "daily_sodium_mg"],
        ),
        generative_family="regression_continuous",
        generative_params={
            "intercept": 90.0,
            "coefficients": {"bmi_kg_m2": 0.01, "daily_sodium_mg": 0.0115},
            "noise_sigma": 1.0,
        },
    )
    paths = RunPaths(
        run_root=tmp_path,
        run_id="bounds",
        run_dir=tmp_path / "bounds",
        stage_dirs={stage: tmp_path / "bounds" / stage for stage in ("A", "B", "C")},
    )
    paths.stage_dirs["A"].mkdir(parents=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec.model_dump(mode="json")])

    compiled = run_stage_b(paths)
    assert compiled.outputs["compiled_count"] == 1
    materialized = run_stage_c(paths, split_policy="auto", visible_ratio=0.75, seed=0)

    assert materialized.outputs["materialized_env_count"] == 1
    assert materialized.outputs["skipped_env_count"] == 0
    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / f"{spec.env_id}.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / f"{spec.env_id}.parquet")
    assert len(visible) == 18
    assert len(hidden) == 6
    data = pd.concat([visible, hidden], ignore_index=True)
    assert data["bmi_kg_m2"].between(18.0, 42.0).all()
    assert data["daily_sodium_mg"].between(1200.0, 5200.0).all()


def test_compiler_rejects_numeric_spec_without_declared_features(tmp_path: Path) -> None:
    spec = EnvSpec(
        env_id="missing_feature_names",
        env_name="missing_feature_names",
        domain="healthcare",
        complexity="simple",
        model_type="predictive",
        key_concepts=["regression"],
        io_schema=IOSchema(
            input_type="float",
            input_description="BMI in 18-42 kg/m².",
            output_type="float",
            target_name="systolic_bp",
            feature_names=[],
            feature_domains={},
        ),
        data_plan=DataPlan(n_observations=24),
        generative_family="regression_continuous",
    )

    with pytest.raises(ValueError, match="regression_continuous.*feature_names"):
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)


@pytest.mark.parametrize("family", NUMERIC_FAMILIES)
@pytest.mark.parametrize(
    "domains",
    [
        {"bmi_kg_m2": (18.0, 42.0)},
        {"bmi_kg_m2": (18.0, 42.0), "daily_sodium_mg": (1200.0, 5200.0)},
    ],
    ids=["single", "disjoint"],
)
def test_generated_numeric_environment_accepts_domain_endpoints(
    tmp_path: Path, family: str, domains: dict[str, tuple[float, float]]
) -> None:
    spec = _numeric_spec(family, domains)
    env = _compile_and_load(spec, tmp_path)

    for endpoint in (0, 1):
        payload = {name: bounds[endpoint] for name, bounds in domains.items()}
        query = json.dumps(payload) if len(payload) > 1 else str(payload["bmi_kg_m2"])
        result, accepted = env.run_experiment(query)
        assert accepted, result

    assert env.get_df()["bmi_kg_m2"].tolist() == [18.0, 42.0]
    if len(domains) > 1:
        assert env.get_df()["daily_sodium_mg"].tolist() == [1200.0, 5200.0]


@pytest.mark.parametrize("family", NUMERIC_FAMILIES)
@pytest.mark.parametrize(
    ("feature", "bad_value", "lower", "upper"),
    [
        ("bmi_kg_m2", 17.0, 18.0, 42.0),
        ("bmi_kg_m2", 43.0, 18.0, 42.0),
        ("daily_sodium_mg", 1199.0, 1200.0, 5200.0),
        ("daily_sodium_mg", 5201.0, 1200.0, 5200.0),
    ],
)
def test_generated_environment_rejection_names_feature_own_bounds_and_value(
    tmp_path: Path, family: str, feature: str, bad_value: float, lower: float, upper: float
) -> None:
    spec = _numeric_spec(family, {"bmi_kg_m2": (18.0, 42.0), "daily_sodium_mg": (1200.0, 5200.0)})
    env = _compile_and_load(spec, tmp_path)
    payload = {"bmi_kg_m2": 30.0, "daily_sodium_mg": 3200.0, feature: bad_value}

    result, accepted = env.run_experiment(json.dumps(payload))

    assert accepted is False
    for detail in (family, feature, str(lower), str(upper), str(bad_value)):
        assert detail in result
    assert env.get_data() == []


@pytest.mark.parametrize("family", NUMERIC_FAMILIES)
def test_generated_single_feature_rejection_names_its_bounds_and_value(
    tmp_path: Path, family: str
) -> None:
    env = _compile_and_load(_numeric_spec(family, {"bmi_kg_m2": (18.0, 42.0)}), tmp_path)

    result, accepted = env.run_experiment("43.0")

    assert accepted is False
    for detail in (family, "bmi_kg_m2", "18.0", "42.0", "43.0"):
        assert detail in result
    assert env.get_data() == []


@pytest.mark.parametrize("family", NUMERIC_FAMILIES)
def test_compiler_rejects_a_missing_feature_domain(tmp_path: Path, family: str) -> None:
    spec = _numeric_spec(family, {"bmi_kg_m2": (18.0, 42.0)})
    spec.io_schema.feature_names.append("daily_sodium_mg")

    with pytest.raises(ValueError, match=family + ".*daily_sodium_mg"):
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)


@pytest.mark.parametrize("domain", [(18.0, 42.0), (1200.0, 5200.0)])
def test_mixture_response_centers_are_independent_of_input_domain(
    tmp_path: Path, domain: tuple[float, float]
) -> None:
    spec = _numeric_spec("mixture_continuous", {"x": domain})
    spec.generative_params = {
        "n_components": 1,
        "component_mean_lower": 80.0,
        "component_mean_upper": 120.0,
        "coefficients": {"x": 0.0},
        "noise_sigma": 0.0,
    }
    env = _compile_and_load(spec, tmp_path)

    for value in domain:
        result, accepted = env.run_experiment(str(value))
        assert accepted is True
        assert result == 100.0


@pytest.mark.parametrize("family", ["infection_process", "ode_predator_prey"])
def test_fixed_time_feature_uses_its_declared_domain(tmp_path: Path, family: str) -> None:
    env = _compile_and_load(_numeric_spec(family, {"t": (110.0, 120.0)}), tmp_path)

    assert env.validate_input("110.0") == (110.0,)
    assert env.validate_input("120.0") == (120.0,)
    with pytest.raises(InputValidationError) as exc:
        env.validate_input("121.0")
    for detail in (family, "t", "110.0", "120.0", "121.0"):
        assert detail in str(exc.value)


@pytest.mark.parametrize(
    ("family", "feature"),
    [("time_series_level_seasonality", "t"), ("survival_with_censoring", "subject_id")],
)
def test_integer_feature_respects_narrower_declared_domain(
    tmp_path: Path, family: str, feature: str
) -> None:
    spec = _numeric_spec(family, {feature: (2.0, 4.0)})
    spec.io_schema.input_type = InputType.int
    env = _compile_and_load(spec, tmp_path)

    for query in ("2", "4"):
        result, accepted = env.run_experiment(query)
        assert accepted, result
    for query in ("1", "5"):
        result, accepted = env.run_experiment(query)
        assert accepted is False
        for detail in (family, feature, "2.0", "4.0", f"{query}.0"):
            assert detail in result
    assert env.get_df()[feature].tolist() == [2, 4]


def test_ode_experiment_returns_rejected_result_for_out_of_domain_time(tmp_path: Path) -> None:
    env = _compile_and_load(_numeric_spec("ode_predator_prey", {"t": (0.0, 1.0)}), tmp_path)

    result, accepted = env.run_experiment("2.0")

    assert accepted is False
    for detail in ("ode_predator_prey", "t", "0.0", "1.0", "2.0"):
        assert detail in result
    assert env.get_data() == []


@pytest.mark.parametrize("domain", [(-1.0, 3.0), (0.0, 5.0)])
def test_compiler_rejects_subject_domain_outside_generated_population(
    tmp_path: Path, domain: tuple[float, float]
) -> None:
    spec = _numeric_spec("survival_with_censoring", {"subject_id": domain})
    spec.io_schema.input_type = InputType.int
    spec.generative_params["n_subjects"] = 5

    with pytest.raises(ValueError, match="survival_with_censoring.*subject_id"):
        TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)


def test_generated_system_message_reports_each_feature_domain(tmp_path: Path) -> None:
    spec = _numeric_spec(
        "regression_continuous",
        {"bmi_kg_m2": (18.0, 42.0), "daily_sodium_mg": (1200.0, 5200.0)},
    )
    env = _compile_and_load(spec, tmp_path)

    message = env.generate_system_message()

    for detail in ("bmi_kg_m2", "18.0", "42.0", "daily_sodium_mg", "1200.0", "5200.0"):
        assert detail in message


def test_stage_b_and_c_materialize_single_feature_declared_as_json(tmp_path: Path) -> None:
    spec = _numeric_spec("regression_continuous", {"bmi_kg_m2": (18.0, 42.0)})
    spec.io_schema.input_type = InputType.json
    paths = RunPaths(
        run_root=tmp_path,
        run_id="single_json",
        run_dir=tmp_path / "single_json",
        stage_dirs={stage: tmp_path / "single_json" / stage for stage in ("A", "B", "C")},
    )
    paths.stage_dirs["A"].mkdir(parents=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec.model_dump(mode="json")])

    compiled = run_stage_b(paths)
    assert compiled.outputs["compiled_count"] == 1
    materialized = run_stage_c(paths, split_policy="auto", visible_ratio=0.75, seed=0)

    assert materialized.outputs["materialized_env_count"] == 1
    assert materialized.outputs["skipped_env_count"] == 0
    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / f"{spec.env_id}.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / f"{spec.env_id}.parquet")
    assert len(visible) + len(hidden) == 24
    data = pd.concat([visible, hidden], ignore_index=True)
    assert data["bmi_kg_m2"].between(18.0, 42.0).all()


@pytest.mark.parametrize(
    "bad_value",
    [True, None, "bad", float("nan"), float("inf"), 10**400],
    ids=["boolean", "null", "nonnumeric", "nan", "infinity", "overflowing_integer"],
)
def test_invalid_feature_value_rejection_includes_own_domain_and_value(
    tmp_path: Path, bad_value: object
) -> None:
    spec = _numeric_spec(
        "regression_continuous",
        {"bmi_kg_m2": (18.0, 42.0), "daily_sodium_mg": (1200.0, 5200.0)},
    )
    env = _compile_and_load(spec, tmp_path)
    payload = {"bmi_kg_m2": bad_value, "daily_sodium_mg": 3200.0}

    result, accepted = env.run_experiment(json.dumps(payload))

    assert accepted is False
    assert env.get_data() == []
    for detail in (spec.generative_family, "bmi_kg_m2", "18.0", "42.0", repr(bad_value)):
        assert detail in result


def test_scalar_infinity_rejection_includes_feature_domain_and_value(tmp_path: Path) -> None:
    spec = _numeric_spec("regression_continuous", {"bmi_kg_m2": (18.0, 42.0)})
    env = _compile_and_load(spec, tmp_path)

    result, accepted = env.run_experiment("inf")

    assert accepted is False
    assert env.get_data() == []
    for detail in (spec.generative_family, "bmi_kg_m2", "18.0", "42.0", "inf"):
        assert detail in result


def test_scalar_nonnumeric_rejection_includes_feature_domain_and_value(tmp_path: Path) -> None:
    spec = _numeric_spec("regression_continuous", {"bmi_kg_m2": (18.0, 42.0)})
    env = _compile_and_load(spec, tmp_path)

    result, accepted = env.run_experiment("bad")

    assert accepted is False
    assert env.get_data() == []
    for detail in (
        spec.generative_family,
        "single float",
        "bmi_kg_m2",
        "18.0",
        "42.0",
        repr("bad"),
    ):
        assert detail in result

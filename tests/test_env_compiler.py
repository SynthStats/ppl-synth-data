from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import snake_to_camel
from ppl_synth_data.specs.models import Complexity, DataPlan, EnvSpec, IOSchema


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generated_env", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCALAR_MULTI_FEATURE_FAMILIES = [
    "regression_continuous",
    "regression_count",
    "binary_choice",
    "categorical_choice",
    "mixture_continuous",
]


def _make_scalar_spec(family: str, feature_names: list[str]) -> EnvSpec:
    weights = {
        "regression_continuous": (1.0, 1.15, 1.3),
        "regression_count": (0.3, 0.345, 0.39),
        "binary_choice": (1.0, 1.15, 1.3),
        "categorical_choice": (1.0, 1.15, 1.3),
        "mixture_continuous": (0.5, 0.575, 0.65),
    }[family]
    params = {
        "regression_continuous": {"intercept": 0.0, "noise_sigma": 1.0},
        "regression_count": {"intercept": 0.5},
        "binary_choice": {"intercept": 0.0},
        "categorical_choice": {"intercept": 0.0, "n_categories": 3},
        "mixture_continuous": {
            "n_components": 2,
            "noise_sigma": 1.0,
            "component_mean_lower": -5.0,
            "component_mean_upper": 5.0,
        },
    }[family]
    params["coefficients"] = dict(zip(feature_names, weights[: len(feature_names)], strict=True))
    return EnvSpec(
        env_id=f"{family}_runtime_test_v1",
        env_name=f"{family}_runtime_test_v1",
        domain="healthcare",
        complexity=Complexity.medium,
        model_type="predictive",
        key_concepts=[family],
        io_schema=IOSchema(
            input_type="float",
            input_description="numeric features",
            output_type="float",
            target_name="target",
            feature_names=feature_names,
            feature_domains={name: (-5.0, 5.0) for name in feature_names},
        ),
        data_plan=DataPlan(n_observations=10, grouping=[], covariates=[]),
        generative_family=family,
        generative_params=params,
        prompt_guidance={},
    )


def test_compile_hierarchical_grouped(tmp_path: Path) -> None:
    spec = EnvSpec(
        env_id="healthcare_er_wait_v1",
        env_name="healthcare_er_wait_v1",
        domain="healthcare",
        complexity=Complexity.medium,
        model_type="predictive",
        key_concepts=["hierarchical_model"],
        io_schema=IOSchema(
            input_type="json",
            input_description="JSON with severity identifier from 0 to 3",
            output_type="float",
            target_name="wait_minutes",
            feature_names=["severity"],
            feature_types={"severity": "integer"},
            feature_domains={"severity": (0.0, 3.0)},
        ),
        data_plan=DataPlan(n_observations=100, grouping=["severity"], covariates=[]),
        generative_family="hierarchical_grouped",
        generative_params={"base_mean": 40.0, "noise_sigma": 8.0},
        prompt_guidance={"units": {"wait_minutes": "minutes"}},
    )

    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)

    mod = _load_module(module_path)
    cls = mod.HealthcareErWaitV1Env
    env = cls(seed=0)

    for method in [
        "reset",
        "step",
        "validate_input",
        "run_experiment",
        "get_data",
        "get_df",
        "generate_system_message",
    ]:
        assert hasattr(env, method), method


def test_generative_params_injection(tmp_path: Path) -> None:
    spec = EnvSpec(
        env_id="injection_test_v1",
        env_name="injection_test_v1",
        domain="healthcare",
        complexity=Complexity.medium,
        model_type="predictive",
        key_concepts=["hierarchical_model"],
        io_schema=IOSchema(
            input_type="json",
            input_description="JSON with group",
            output_type="float",
            target_name="y",
            feature_names=["group"],
        ),
        data_plan=DataPlan(n_observations=10, grouping=["group"], covariates=[]),
        generative_family="hierarchical_grouped",
        generative_params={
            "base_mean": "0.0); import os; os.system('echo pwned') #",
            "noise_sigma": 1.0,
        },
        prompt_guidance={},
    )
    compiler = TemplateCompiler()
    # Invalid parameters are now refused before any generated module can run.
    with pytest.raises(ValueError, match="hierarchical_grouped.*base_mean.*finite number"):
        compiler.compile_envspec(spec, out_dir=tmp_path)
    assert not (tmp_path / f"{spec.env_name}.py").exists()


def test_path_traversal_rejected(tmp_path: Path) -> None:
    spec = EnvSpec(
        env_id="bad",
        env_name="../escape",
        domain="healthcare",
        complexity=Complexity.medium,
        model_type="predictive",
        key_concepts=["hierarchical_model"],
        io_schema=IOSchema(
            input_type="json",
            input_description="x",
            output_type="float",
            target_name="wait_minutes",
            feature_names=["severity"],
        ),
        data_plan=DataPlan(n_observations=10, grouping=["severity"], covariates=[]),
        generative_family="hierarchical_grouped",
        generative_params={},
        prompt_guidance={},
    )
    compiler = TemplateCompiler()
    with pytest.raises(ValueError, match="hierarchical_grouped.*env_name"):
        compiler.compile_envspec(spec, out_dir=tmp_path)


def test_compile_envspec_accepts_safe_mixed_case_target_column(tmp_path: Path) -> None:
    spec = _make_scalar_spec("regression_continuous", ["age_years", "daily_sodium_g"])
    spec.io_schema.target_name = "systolic_bp_mmHg"
    compiler = TemplateCompiler()

    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)

    mod = _load_module(module_path)
    class_name = f"{snake_to_camel(spec.env_name)}Env"
    env = getattr(mod, class_name)(seed=0)
    _, success = env.run_experiment(
        json.dumps({"age_years": 1.2, "daily_sodium_g": 3.1}, ensure_ascii=False)
    )
    assert success is True
    assert "systolic_bp_mmHg" in env.get_df().columns


def test_compile_envspec_rejects_unsafe_target_column(tmp_path: Path) -> None:
    spec = _make_scalar_spec("regression_continuous", ["x"])
    spec.io_schema.target_name = "bad-name"
    compiler = TemplateCompiler()

    with pytest.raises(ValueError, match="regression_continuous.*target_name"):
        compiler.compile_envspec(spec, out_dir=tmp_path)


@pytest.mark.parametrize("family", SCALAR_MULTI_FEATURE_FAMILIES)
def test_scalar_templates_support_multi_feature_json_inputs(tmp_path: Path, family: str) -> None:
    spec = _make_scalar_spec(family, ["x1", "x2", "x3"])
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)

    mod = _load_module(module_path)
    class_name = f"{snake_to_camel(spec.env_name)}Env"
    env = getattr(mod, class_name)(seed=0)

    result, success = env.run_experiment(
        json.dumps({"x1": 1.0, "x2": -0.5, "x3": 0.25}, ensure_ascii=False)
    )

    assert success is True
    df = env.get_df()
    assert df.shape == (1, 4)
    assert list(df.columns) == ["x1", "x2", "x3", "target"]
    assert pytest.approx(float(df.iloc[0]["x1"]), rel=0, abs=1e-12) == 1.0
    assert result is not None


@pytest.mark.parametrize("family", SCALAR_MULTI_FEATURE_FAMILIES)
def test_scalar_templates_keep_single_feature_float_input(tmp_path: Path, family: str) -> None:
    spec = _make_scalar_spec(family, ["x"])
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)

    mod = _load_module(module_path)
    class_name = f"{snake_to_camel(spec.env_name)}Env"
    env = getattr(mod, class_name)(seed=0)

    _, success = env.run_experiment("1.25")
    assert success is True

    df = env.get_df()
    assert df.shape == (1, 2)
    assert list(df.columns) == ["x", "target"]


@pytest.mark.parametrize("family", SCALAR_MULTI_FEATURE_FAMILIES)
def test_scalar_templates_reject_nonfinite_multi_feature_inputs(
    tmp_path: Path,
    family: str,
) -> None:
    spec = _make_scalar_spec(family, ["x1", "x2", "x3"])
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)

    mod = _load_module(module_path)
    class_name = f"{snake_to_camel(spec.env_name)}Env"
    env = getattr(mod, class_name)(seed=0)

    result, success = env.run_experiment(
        json.dumps({"x1": float("nan"), "x2": 1.0, "x3": 0.25}, ensure_ascii=False)
    )

    assert success is False
    assert family in result and "x1" in result
    assert "finite" in str(result).lower()
    assert env.get_data() == []


@pytest.mark.parametrize("family", SCALAR_MULTI_FEATURE_FAMILIES)
def test_scalar_templates_reject_boolean_multi_feature_inputs(tmp_path: Path, family: str) -> None:
    spec = _make_scalar_spec(family, ["x1", "x2", "x3"])
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)

    mod = _load_module(module_path)
    class_name = f"{snake_to_camel(spec.env_name)}Env"
    env = getattr(mod, class_name)(seed=0)

    result, success = env.run_experiment(
        json.dumps({"x1": True, "x2": 1.0, "x3": 0.25}, ensure_ascii=False)
    )

    assert success is False
    assert family in result and "x1" in result
    assert "numeric" in str(result).lower()
    assert env.get_data() == []


@pytest.mark.parametrize("family", SCALAR_MULTI_FEATURE_FAMILIES)
@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf"])
def test_scalar_templates_reject_nonfinite_single_feature_inputs(
    tmp_path: Path,
    family: str,
    bad_value: str,
) -> None:
    spec = _make_scalar_spec(family, ["x"])
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)

    mod = _load_module(module_path)
    class_name = f"{snake_to_camel(spec.env_name)}Env"
    env = getattr(mod, class_name)(seed=0)

    result, success = env.run_experiment(bad_value)

    assert success is False
    assert family in result and "x" in result
    assert "finite" in str(result).lower()
    assert env.get_data() == []


def test_compiled_env_inherits_validate_input_from_base(tmp_path: Path) -> None:
    from ppl_synth_data.envs.base import BaseEnv

    spec = _make_scalar_spec("regression_continuous", ["x1", "x2"])
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)

    mod = _load_module(module_path)
    class_name = f"{snake_to_camel(spec.env_name)}Env"
    env_cls = getattr(mod, class_name)

    assert env_cls.validate_input is BaseEnv.validate_input
    assert env_cls.run_experiment is BaseEnv.run_experiment
    assert env_cls.get_data is BaseEnv.get_data
    assert env_cls.get_df is BaseEnv.get_df
    assert env_cls.get_ordered_features is BaseEnv.get_ordered_features

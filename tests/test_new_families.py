"""Tests for the 5 new generative family templates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.specs.models import Complexity, DataPlan, EnvSpec, IOSchema


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generated_env", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- ode_predator_prey ---


def _make_ode_spec() -> EnvSpec:
    return EnvSpec(
        env_id="predator_prey_test_v1",
        env_name="predator_prey_test_v1",
        domain="ecology",
        complexity=Complexity.medium,
        model_type="predictive",
        key_concepts=["ode", "lotka_volterra"],
        io_schema=IOSchema(
            input_type="float",
            input_description="time",
            output_type="float",
            target_name="population",
            feature_names=["t"],
            feature_domains={"t": (0.0, 50.0)},
        ),
        data_plan=DataPlan(n_observations=20),
        generative_family="ode_predator_prey",
        generative_params={
            "alpha": 1.0,
            "beta": 0.1,
            "gamma": 1.5,
            "delta": 0.075,
            "prey_init": 40.0,
            "predator_init": 9.0,
            "noise_sigma": 2.0,
        },
        prompt_guidance={},
    )


def test_ode_predator_prey_compiles_and_runs(tmp_path: Path) -> None:
    spec = _make_ode_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.PredatorPreyTestV1Env(seed=42)

    result, success = env.run_experiment("5.0")
    assert success is True
    prey, pred = result
    assert isinstance(prey, int)
    assert isinstance(pred, int)
    assert prey >= 0
    assert pred >= 0


def test_ode_predator_prey_deterministic(tmp_path: Path) -> None:
    spec = _make_ode_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)

    env1 = mod.PredatorPreyTestV1Env(seed=42)
    r1, _ = env1.run_experiment("5.0")

    env2 = mod.PredatorPreyTestV1Env(seed=42)
    r2, _ = env2.run_experiment("5.0")

    assert r1 == r2


# --- irt_binary_response ---


def _make_irt_spec() -> EnvSpec:
    return EnvSpec(
        env_id="irt_test_v1",
        env_name="irt_test_v1",
        domain="psychology",
        complexity=Complexity.medium,
        model_type="inferential",
        key_concepts=["irt", "2pl"],
        io_schema=IOSchema(
            input_type="json",
            input_description="student and question ids",
            output_type="int",
            target_name="correct",
            feature_names=["student_id", "question_id"],
            feature_types={"student_id": "integer", "question_id": "integer"},
            feature_domains={"student_id": (0.0, 9.0), "question_id": (0.0, 4.0)},
        ),
        data_plan=DataPlan(n_observations=50, grouping=["student_id"], covariates=["question_id"]),
        generative_family="irt_binary_response",
        generative_params={
            "n_students": 10,
            "n_questions": 5,
            "ability_mean": 0.0,
            "ability_sigma": 1.0,
            "difficulty_mean": 0.0,
            "difficulty_sigma": 1.0,
            "discrimination_mean": 1.0,
        },
        prompt_guidance={},
    )


def test_irt_compiles_and_runs(tmp_path: Path) -> None:
    spec = _make_irt_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.IrtTestV1Env(seed=42)

    result, success = env.run_experiment(json.dumps({"student_id": 0, "question_id": 0}))
    assert success is True
    assert result in (0, 1)


def test_irt_rejects_invalid_ids(tmp_path: Path) -> None:
    spec = _make_irt_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.IrtTestV1Env(seed=42)

    result, success = env.run_experiment(json.dumps({"student_id": 999, "question_id": 0}))
    assert success is False
    assert "irt_binary_response" in result and "student_id" in result


# --- infection_process ---


def _make_infection_spec() -> EnvSpec:
    return EnvSpec(
        env_id="infection_test_v1",
        env_name="infection_test_v1",
        domain="healthcare",
        complexity=Complexity.simple,
        model_type="predictive",
        key_concepts=["infection", "binomial"],
        io_schema=IOSchema(
            input_type="float",
            input_description="time",
            output_type="int",
            target_name="infected",
            feature_names=["t"],
            feature_domains={"t": (0.01, 100.0)},
        ),
        data_plan=DataPlan(n_observations=20),
        generative_family="infection_process",
        generative_params={
            "population_n": 100,
            "rate_mean": 0.05,
            "rate_sigma": 0.01,
        },
        prompt_guidance={},
    )


def test_infection_compiles_and_runs(tmp_path: Path) -> None:
    spec = _make_infection_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.InfectionTestV1Env(seed=42)

    result, success = env.run_experiment("10.0")
    assert success is True
    assert isinstance(result, int)
    assert 0 <= result <= 100


def test_infection_increases_with_time(tmp_path: Path) -> None:
    spec = _make_infection_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.InfectionTestV1Env(seed=42)

    early_counts = [env.step((1.0,)) for _ in range(50)]
    env.reset(seed=42)
    late_counts = [env.step((50.0,)) for _ in range(50)]

    # on average, later times should have more infections
    assert sum(late_counts) > sum(early_counts)


# --- spatial_signal_source ---


def _make_spatial_spec() -> EnvSpec:
    return EnvSpec(
        env_id="spatial_test_v1",
        env_name="spatial_test_v1",
        domain="engineering",
        complexity=Complexity.complex,
        model_type="inferential",
        key_concepts=["spatial", "inverse_distance"],
        io_schema=IOSchema(
            input_type="json",
            input_description="x,y coordinates",
            output_type="float",
            target_name="intensity",
            feature_names=["x", "y"],
            feature_types={"x": "continuous", "y": "continuous"},
            feature_domains={"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        ),
        data_plan=DataPlan(n_observations=30, covariates=["x", "y"]),
        generative_family="spatial_signal_source",
        generative_params={
            "n_sources": 2,
            "dim": 2,
            "background": 1.0,
            "regularization_m": 1.0,
            "source_strength": 10.0,
            "noise_sigma": 0.5,
        },
        prompt_guidance={},
    )


def test_spatial_compiles_and_runs(tmp_path: Path) -> None:
    spec = _make_spatial_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.SpatialTestV1Env(seed=42)

    result, success = env.run_experiment(json.dumps({"x": 0.0, "y": 0.0}))
    assert success is True
    assert isinstance(result, float)


def test_spatial_rejects_missing_coords(tmp_path: Path) -> None:
    spec = _make_spatial_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.SpatialTestV1Env(seed=42)

    result, success = env.run_experiment(json.dumps({"x": 0.0}))
    assert success is False
    assert "spatial_signal_source" in result and "y" in result


# --- choice_with_discounting ---


def _make_discount_spec() -> EnvSpec:
    return EnvSpec(
        env_id="discount_test_v1",
        env_name="discount_test_v1",
        domain="economics",
        complexity=Complexity.medium,
        model_type="inferential",
        key_concepts=["temporal_discounting", "choice"],
        io_schema=IOSchema(
            input_type="json",
            input_description="choice scenario",
            output_type="int",
            target_name="choice",
            feature_names=["immediate_reward", "delayed_reward", "delay_days"],
            feature_domains={
                "immediate_reward": (0.0, 20.0),
                "delayed_reward": (0.0, 40.0),
                "delay_days": (0.0, 60.0),
            },
        ),
        data_plan=DataPlan(
            n_observations=40,
            covariates=["immediate_reward", "delayed_reward", "delay_days"],
        ),
        generative_family="choice_with_discounting",
        generative_params={
            "epsilon": 0.05,
            "k_mean": 0.05,
            "k_sigma": 0.02,
            "choice_scale": 10.0,
        },
        prompt_guidance={},
    )


def test_discount_compiles_and_runs(tmp_path: Path) -> None:
    spec = _make_discount_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.DiscountTestV1Env(seed=42)

    result, success = env.run_experiment(
        json.dumps({"immediate_reward": 10.0, "delayed_reward": 20.0, "delay_days": 30.0})
    )
    assert success is True
    assert result in (0, 1)


def test_discount_rejects_negative_delay(tmp_path: Path) -> None:
    spec = _make_discount_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.DiscountTestV1Env(seed=42)

    result, success = env.run_experiment(
        json.dumps({"immediate_reward": 10.0, "delayed_reward": 20.0, "delay_days": -5.0})
    )
    assert success is False
    assert "choice_with_discounting" in result and "delay_days" in result


def test_discount_df_shape(tmp_path: Path) -> None:
    spec = _make_discount_spec()
    compiler = TemplateCompiler()
    module_path = compiler.compile_envspec(spec, out_dir=tmp_path)
    mod = _load_module(module_path)
    env = mod.DiscountTestV1Env(seed=42)

    for _ in range(5):
        env.run_experiment(
            json.dumps({"immediate_reward": 10.0, "delayed_reward": 20.0, "delay_days": 30.0})
        )
    df = env.get_df()
    assert df.shape == (5, 4)
    assert list(df.columns) == ["immediate_reward", "delayed_reward", "delay_days", "choice"]


# --- registry tests ---


def test_core_families_in_registry() -> None:
    from ppl_synth_data.specs.registry import SUPPORTED_FAMILIES

    for fam in [
        "ode_predator_prey",
        "irt_binary_response",
        "infection_process",
        "spatial_signal_source",
        "choice_with_discounting",
        "zero_inflated_count",
    ]:
        assert fam in SUPPORTED_FAMILIES


def test_every_registered_family_has_a_template_file() -> None:
    """Every registered family has a renderable .j2 template."""
    from pathlib import Path

    from ppl_synth_data.envs.compiler import _FAMILY_TEMPLATE
    from ppl_synth_data.specs.registry import SUPPORTED_FAMILIES

    template_dir = (
        Path(__file__).resolve().parents[1] / "src" / "ppl_synth_data" / "envs" / "templates"
    )
    for fam in SUPPORTED_FAMILIES:
        assert fam in _FAMILY_TEMPLATE, f"family {fam!r} has no compiler template mapping"
        template_path = template_dir / _FAMILY_TEMPLATE[fam]
        assert template_path.exists(), f"family {fam!r} template {template_path.name} missing"

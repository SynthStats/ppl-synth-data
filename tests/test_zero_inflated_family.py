"""Check the zero_inflated_count family through registry, template, simulator and verifier.

The simulator produces zero-inflated data, the PyMC program passes validation, and the family
has its own structural diversity mode.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.specs.models import Complexity, DataPlan, EnvSpec, IOSchema
from ppl_synth_data.validation.diversity import program_shape


def _spec() -> EnvSpec:
    return EnvSpec(
        env_id="defect_zero_inflated_v1",
        env_name="defect_zero_inflated_v1",
        domain="manufacturing",
        complexity=Complexity.medium,
        model_type="inferential",
        key_concepts=["count", "zero_inflation", "excess_zeros"],
        io_schema=IOSchema(
            input_type="float",
            input_description="line load index",
            output_type="int",
            target_name="defects",
            feature_names=["load"],
            feature_domains={"load": (0.0, 10.0)},
        ),
        data_plan=DataPlan(n_observations=60),
        generative_family="zero_inflated_count",
        generative_params={
            "intercept": 0.8,
            "coefficients": {"load": 4.0},
            "zero_inflation": 0.45,
        },
        prompt_guidance={},
    )


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("zi_env", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_zero_inflated_compiles_and_simulates_excess_zeros(tmp_path: Path) -> None:
    module_path = TemplateCompiler().compile_envspec(_spec(), out_dir=tmp_path)
    mod = _load(module_path)
    env = mod.DefectZeroInflatedV1Env(seed=7)

    samples = [env.step((float(x),)) for x in np.linspace(0.0, 5.0, 200)]
    assert all(isinstance(s, int) and s >= 0 for s in samples)
    zero_frac = np.mean([s == 0 for s in samples])
    # The structural-zero probability is 0.45, so the observed zero fraction must
    # be materially above what a plain Poisson at these rates would give.
    assert zero_frac > 0.35


def test_zero_inflated_program_shape_is_distinct():
    zi_code = """
def model(data):
    x = np.asarray(data["load"]); y = np.asarray(data["defects"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1); b = pm.Normal("b", 0, 1)
        psi = pm.Beta("psi", 1, 1)
        pm.ZeroInflatedPoisson("obs", psi=psi, mu=pm.math.exp(a * x + b), observed=y)
    return m
"""
    plain_code = """
def model(data):
    x = np.asarray(data["load"]); y = np.asarray(data["defects"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1); b = pm.Normal("b", 0, 1)
        pm.Poisson("obs", mu=pm.math.exp(a * x + b), observed=y)
    return m
"""
    zi = program_shape(zi_code)
    plain = program_shape(plain_code)
    assert zi is not None and plain is not None
    assert "zero_inflated" in zi.structural_flags
    assert zi.key() != plain.key()  # a genuinely distinct structural mode


def test_zero_inflated_fixture_preserves_original_natural_load_effect(tmp_path: Path) -> None:
    mod = _load(TemplateCompiler().compile_envspec(_spec(), out_dir=tmp_path))
    env = mod.DefectZeroInflatedV1Env(seed=17)
    values = np.array([env.step((10.0,)) for _ in range(8192)])
    # The original process had log-rate 0.8 + 0.4*load and structural-zero mass .45.
    # At load=10 this is .55*exp(4.8) = 66.83073 counts, not .55*exp(1.2).
    assert values.mean() == pytest.approx(66.83073, abs=3.0)


@pytest.mark.slow
def test_zero_inflated_model_passes_verifier(tmp_path: Path) -> None:
    from synthstats_schemas import CandidateKey

    from ppl_synth_data.validation.pymc_selection import gate_and_score_pymc

    mod = _load(TemplateCompiler().compile_envspec(_spec(), out_dir=tmp_path))
    env = mod.DefectZeroInflatedV1Env(seed=3)
    loads = np.linspace(0.0, 5.0, 50)
    y = np.array([env.step((float(x),)) for x in loads])
    visible = {"load": loads[:38], "defects": y[:38]}
    hidden = {"load": loads[38:], "defects": y[38:]}

    code = """
def model(data):
    x = np.asarray(data["load"], dtype=float)
    y = np.asarray(data["defects"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 2)
        b = pm.Normal("b", 0, 2)
        psi = pm.Beta("psi", 2, 2)
        mu = pm.math.exp(pm.math.clip(a * x + b, -10, 10))
        pm.ZeroInflatedPoisson("defects", psi=psi, mu=mu, observed=y)
    return m
"""
    cand = gate_and_score_pymc(
        code,
        visible_data=visible,
        hidden_data=hidden,
        candidate_key=CandidateKey(task_id="zi", family="test", candidate_idx=0),
        fitness_draws=100,
        fitness_tune=100,
    )
    assert cand.gate.eligible is True
    # Count target: the Gaussian degeneracy gate is correctly SKIPPED (no
    # min_predictive_scale gate emitted for an integer-valued target).
    assert all(g.name != "min_predictive_scale" for g in cand.gate.density_gates)
    assert cand.score is not None and cand.score.holdout_n == 12

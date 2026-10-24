"""Real compiled response laws, with SciPy distribution checks.

Statistical policy fixed before execution: each distribution check uses a
two-sided false-rejection probability of 1e-6. Fixed seeds are never retried.
These are law checks, not parameter recovery or production admission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import load_env_instance
from ppl_synth_data.specs.models import FourFamilySpec


def compile_response(tmp_path: Path, response: dict):
    spec = FourFamilySpec.model_validate(
        {
            "env_id": "round14-response-law",
            "env_name": "round14_response_law",
            "domain": "synthetic",
            "scenario": {"intent": "Test a declared response law", "source_row_sha256": "c" * 64},
            "generative_family": response["family"],
            "features": [{"name": "signal", "law": {"dist": "normal", "mean": 20.0, "sd": 3.0}}],
            "response": response,
            "data_plan": {"n_observations": 100},
        }
    )
    path = TemplateCompiler().compile_envspec(spec, tmp_path)
    return load_env_instance(path, env_name=spec.env_name, seed=14001, trusted_root=tmp_path)


def draw(env, query: str, n: int = 2048):
    values = []
    for _ in range(n):
        value, success = env.run_experiment(query)
        assert success, value
        values.append(value)
    return np.asarray(values)


def test_gaussian_raw_scale_noise_and_replay(tmp_path: Path) -> None:
    env = compile_response(
        tmp_path,
        {
            "family": "regression_continuous",
            "intercept": 7.0,
            "coefficients": {"signal": -0.4},
            "noise_sigma": 2.5,
        },
    )
    values = draw(env, "20.0")
    # At raw x=20 the independently worked mean is -1, SD=2.5.
    assert stats.kstest(values, stats.norm(loc=-1.0, scale=2.5).cdf).pvalue > 1e-6
    assert np.issubdtype(values.dtype, np.floating)
    env.reset(seed=14001)
    assert np.array_equal(draw(env, "20.0", 32), values[:32])
    assert "non-negative integer" not in env.generate_system_message()


@pytest.mark.parametrize(
    ("intercept", "probability"),
    [(1.0, 0.2689414213699951), (-1000.0, 0.0), (1000.0, 1.0)],
)
def test_binary_logit_raw_scale_and_extremes(
    tmp_path: Path,
    intercept: float,
    probability: float,
) -> None:
    env = compile_response(
        tmp_path,
        {
            "family": "binary_choice",
            "intercept": intercept,
            "coefficients": {"signal": -0.1},
            "link": "logit",
        },
    )
    values = draw(env, "20.0")
    assert set(values) <= {0, 1}
    assert np.issubdtype(values.dtype, np.integer)
    assert stats.binomtest(int(values.sum()), len(values), probability).pvalue > 1e-6
    env.reset(seed=14001)
    assert np.array_equal(draw(env, "20.0", 32), values[:32])
    assert "binary" in env.generate_system_message().lower()


@pytest.mark.parametrize(
    ("distribution", "inflation"),
    [("poisson", 0.35), ("nb2", 0.35), ("poisson", 0.99)],
)
def test_zero_inflated_count_component_law_and_no_clipping(
    tmp_path: Path,
    distribution: str,
    inflation: float,
) -> None:
    response = {
        "family": "zero_inflated_count",
        "intercept_log_rate": 0.0,
        "coefficients": {"signal": 0.06931471805599453},
        "count_distribution": distribution,
        "zero_inflation": inflation,
    }
    if distribution == "nb2":
        response["nb2_alpha"] = 2.0
    env = compile_response(tmp_path, response)
    values = draw(env, "20.0")
    assert np.issubdtype(values.dtype, np.integer)
    assert (values >= 0).all()
    # Independently worked component mean=4, NB2 shape=2, success probability=1/3.
    reference = stats.poisson(4.0) if distribution == "poisson" else stats.nbinom(2.0, 1.0 / 3.0)
    for boundary in (0, 1, 4, 8):
        probability = inflation + (1.0 - inflation) * reference.cdf(boundary)
        observed = int((values <= boundary).sum())
        assert stats.binomtest(observed, len(values), probability).pvalue > 1e-6
    env.reset(seed=14001)
    assert np.array_equal(draw(env, "20.0", 32), values[:32])

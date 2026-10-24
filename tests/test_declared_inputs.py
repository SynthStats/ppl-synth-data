"""Observable laws for EnvSpec-4F v0.1 predictor draws and count queries."""

from __future__ import annotations

import json
from collections import Counter
from math import exp
from pathlib import Path
from statistics import fmean, pstdev

import pytest

from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import load_env_instance
from ppl_synth_data.specs.models import FourFamilySpec


def _spec(features: list[dict], coefficients: dict[str, float], *, n: int = 4000) -> FourFamilySpec:
    """A fresh synthetic declaration, unrelated to any source scenario row."""
    return FourFamilySpec.model_validate(
        {
            "env_id": "declared-input-test",
            "env_name": "declared_input_test",
            "domain": "synthetic",
            "scenario": {
                "intent": "Exercise a declared predictor law",
                "source_row_sha256": "a" * 64,
            },
            "generative_family": "regression_count",
            "features": features,
            "response": {
                "family": "regression_count",
                "intercept_log_rate": 0.2,
                "coefficients": coefficients,
                "count_distribution": "poisson",
            },
            "data_plan": {"n_observations": n},
        }
    )


def _sample(spec: FourFamilySpec, *, seed: int) -> list[dict]:
    # Import at the public seam so a missing implementation reports each law red.
    from ppl_synth_data.envs.declared_inputs import sample_inputs

    rows = sample_inputs(spec, seed=seed)
    assert len(rows) == spec.data_plan.n_observations
    assert all(set(row) == {feature.name for feature in spec.features} for row in rows)
    return rows


def test_normal_draws_have_declared_raw_moments_and_unbounded_tails() -> None:
    spec = _spec(
        [{"name": "signal", "law": {"dist": "normal", "mean": -2.5, "sd": 3.0}}], {"signal": 0.3}
    )

    values = [row["signal"] for row in _sample(spec, seed=11)]

    assert all(isinstance(value, float) for value in values)
    assert abs(fmean(values) - (-2.5)) < 0.18
    assert abs(pstdev(values) - 3.0) < 0.18
    assert min(values) < -8.5
    assert max(values) > 3.5


def test_discrete_uniform_draws_include_both_integer_endpoints() -> None:
    spec = _spec(
        [{"name": "month", "law": {"dist": "discrete_uniform", "low": -2, "high": 2}}],
        {"month": 0.3},
    )

    counts = Counter(row["month"] for row in _sample(spec, seed=12))

    assert set(counts) == {-2, -1, 0, 1, 2}
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in counts)
    assert all(abs(count - 800) < 110 for count in counts.values())


def test_bernoulli_draws_have_declared_probability() -> None:
    spec = _spec([{"name": "exposed", "law": {"dist": "bernoulli", "p": 0.23}}], {"exposed": 0.3})

    values = [row["exposed"] for row in _sample(spec, seed=13)]

    assert set(values) == {0, 1}
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in values)
    assert abs(fmean(values) - 0.23) < 0.035


def test_categorical_draws_follow_declared_level_frequencies() -> None:
    levels = ["control", "low", "high"]
    spec = _spec(
        [
            {
                "name": "arm",
                "law": {"dist": "categorical", "levels": levels, "probs": [0.1, 0.25, 0.65]},
            }
        ],
        {"arm=low": 0.3, "arm=high": 0.5},
    )

    counts = Counter(row["arm"] for row in _sample(spec, seed=14))

    assert set(counts) == set(levels)
    assert abs(counts["control"] / 4000 - 0.1) < 0.025
    assert abs(counts["low"] / 4000 - 0.25) < 0.035
    assert abs(counts["high"] / 4000 - 0.65) < 0.035


def test_fixed_counts_allocate_exactly_and_shuffle_reproducibly() -> None:
    spec = _spec(
        [
            {
                "name": "arm",
                "law": {
                    "dist": "fixed_counts",
                    "levels": ["control", "low", "high"],
                    "counts": [11, 19, 30],
                },
            }
        ],
        {"arm=low": 0.3, "arm=high": 0.5},
        n=60,
    )

    first = _sample(spec, seed=15)
    replay = _sample(spec, seed=15)
    other = _sample(spec, seed=16)

    assert first == replay
    assert first != other
    assert Counter(row["arm"] for row in first) == {"control": 11, "low": 19, "high": 30}
    assert Counter(row["arm"] for row in other) == {"control": 11, "low": 19, "high": 30}
    assert [row["arm"] for row in first] != ["control"] * 11 + ["low"] * 19 + ["high"] * 30


def test_seed_replays_and_identically_declared_features_have_independent_streams() -> None:
    law = {"dist": "normal", "mean": 1.5, "sd": 0.7}
    spec = _spec(
        [{"name": "first", "law": law}, {"name": "second", "law": law}],
        {"first": 0.3, "second": -0.2},
        n=256,
    )

    first = _sample(spec, seed=17)

    assert first == _sample(spec, seed=17)
    assert first != _sample(spec, seed=18)
    assert [row["first"] for row in first] != [row["second"] for row in first]


@pytest.mark.parametrize("law", ["categorical", "fixed_counts"])
def test_compiled_count_runtime_uses_first_level_as_zero_effect(tmp_path: Path, law: str) -> None:
    if law == "categorical":
        input_law = {"dist": law, "levels": ["control", "treated"], "probs": [0.5, 0.5]}
    else:
        input_law = {"dist": law, "levels": ["control", "treated"], "counts": [1000, 1000]}
    spec = _spec([{"name": "arm", "law": input_law}], {"arm=treated": 1.5}, n=2000)
    module = TemplateCompiler().compile_envspec(spec, tmp_path)
    env = load_env_instance(module, env_name=spec.env_name, seed=23, trusted_root=tmp_path)

    def draw(level: str) -> list[int]:
        outcomes = []
        for _ in range(768):
            outcome, success = env.run_experiment(json.dumps({"arm": level}))
            assert success is True
            assert isinstance(outcome, int) and outcome >= 0
            outcomes.append(outcome)
        return outcomes

    control = draw("control")
    treated = draw("treated")

    assert abs(fmean(control) - exp(0.2)) < 0.20
    assert abs(fmean(treated) - exp(0.2 + 1.5)) < 0.45
    assert set(env.get_df()["arm"]) == {"control", "treated"}

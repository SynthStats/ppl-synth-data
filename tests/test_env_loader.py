"""Tests for pipeline.env_loader: snake_to_camel, seed_for_env, build_query, load_env_instance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppl_synth_data.pipeline.env_loader import (
    build_query,
    load_env_instance,
    seed_for_env,
    snake_to_camel,
)
from ppl_synth_data.specs.models import (
    Complexity,
    DataPlan,
    EnvSpec,
    IOSchema,
)

# snake_to_camel.


@pytest.mark.parametrize(
    "input_val, expected",
    [
        ("hello_world", "HelloWorld"),
        ("single", "Single"),
        ("a_b_c", "ABC"),
        ("already", "Already"),
        ("with-dashes", "WithDashes"),
        ("mixed_case-and_dash", "MixedCaseAndDash"),
        ("multiple___underscores", "MultipleUnderscores"),
        ("leading_trailing_", "LeadingTrailing"),
        ("with spaces too", "WithSpacesToo"),
    ],
)
def test_snake_to_camel(input_val: str, expected: str) -> None:
    assert snake_to_camel(input_val) == expected


def test_snake_to_camel_empty_string() -> None:
    assert snake_to_camel("") == ""


# seed_for_env.


def test_seed_for_env_is_deterministic() -> None:
    a = seed_for_env(42, "env_alpha")
    b = seed_for_env(42, "env_alpha")
    assert a == b


def test_seed_for_env_differs_by_env_id() -> None:
    a = seed_for_env(42, "env_alpha")
    b = seed_for_env(42, "env_beta")
    assert a != b


def test_seed_for_env_differs_by_base_seed() -> None:
    a = seed_for_env(1, "env_alpha")
    b = seed_for_env(2, "env_alpha")
    assert a != b


def test_seed_for_env_fits_in_32_bits() -> None:
    result = seed_for_env(9999, "some_env")
    assert 0 <= result < 2**32


# build_query helpers.


def _make_spec(
    input_type: str,
    feature_names: list[str] | None = None,
    grouping: list[str] | None = None,
    covariates: list[str] | None = None,
    feature_domains: dict[str, tuple[float, float]] | None = None,
    env_id: str = "build_query_test_v1",
) -> EnvSpec:
    return EnvSpec(
        env_id=env_id,
        env_name=env_id,
        domain="testing",
        complexity=Complexity.simple,
        model_type="predictive",
        key_concepts=["test"],
        io_schema=IOSchema(
            input_type=input_type,
            input_description="test input",
            output_type="float",
            target_name="y",
            feature_names=feature_names or [],
            feature_types={name: "continuous" for name in feature_names or []},
            feature_domains=feature_domains or {},
        ),
        data_plan=DataPlan(
            n_observations=10,
            grouping=grouping or [],
            covariates=covariates or [],
        ),
        generative_family="regression_continuous",
        generative_params={},
        prompt_guidance={},
    )


# build_query: int.


def test_build_query_int_returns_digit_string() -> None:
    spec = _make_spec("int")
    result = build_query(spec, idx=7, rng=np.random.default_rng(0))
    assert result == "7"


def test_build_query_int_wraps_at_10() -> None:
    spec = _make_spec("int")
    assert build_query(spec, idx=13, rng=np.random.default_rng(0)) == "3"


# build_query: json.


def test_build_query_json_uses_grouping_and_covariates() -> None:
    spec = _make_spec("json", grouping=["region"], covariates=["age"])
    result = build_query(spec, idx=5, rng=np.random.default_rng(0))
    payload = json.loads(result)
    assert "region" in payload
    assert "age" in payload
    assert payload["region"] == "region_1"
    assert payload["age"] == "age_1"


def test_build_query_json_uses_numeric_feature_names_when_declared() -> None:
    spec = _make_spec(
        "json",
        feature_names=["age_years", "bmi", "medication_adherence_score"],
        grouping=["region"],
        covariates=["age"],
        feature_domains={
            "age_years": (18.0, 90.0),
            "bmi": (15.0, 45.0),
            "medication_adherence_score": (0.0, 1.0),
        },
    )
    result = build_query(spec, idx=5, rng=np.random.default_rng(0))
    payload = json.loads(result)

    assert list(payload) == ["age_years", "bmi", "medication_adherence_score"]
    assert all(isinstance(value, float) for value in payload.values())
    assert 18.0 <= payload["age_years"] <= 90.0
    assert 15.0 <= payload["bmi"] <= 45.0
    assert 0.0 <= payload["medication_adherence_score"] <= 1.0


def test_build_query_json_numeric_feature_without_domain_raises() -> None:
    spec = _make_spec("json", feature_names=["age_years"])
    with pytest.raises(RuntimeError, match="feature_domains"):
        build_query(spec, idx=0, rng=np.random.default_rng(0))


def test_build_query_json_defaults_to_group_field() -> None:
    spec = _make_spec("json", grouping=[], covariates=[])
    result = build_query(spec, idx=2, rng=np.random.default_rng(0))
    payload = json.loads(result)
    assert "group" in payload
    assert payload["group"] == "group_2"


# build_query: none.


def test_build_query_none_returns_empty_string() -> None:
    spec = _make_spec("none")
    assert build_query(spec, idx=0, rng=np.random.default_rng(0)) == ""


def test_build_query_none_ignores_idx() -> None:
    spec = _make_spec("none")
    assert build_query(spec, idx=999, rng=np.random.default_rng(0)) == ""


# build_query: string.


def test_build_query_string_returns_idx_as_str() -> None:
    spec = _make_spec("string")
    assert build_query(spec, idx=42, rng=np.random.default_rng(0)) == "42"


# build_query: float (single feature, already tested in pipeline/).


def test_build_query_float_single_feature_is_string_float() -> None:
    spec = _make_spec("float", feature_names=["x"], feature_domains={"x": (0.0, 9.9)})
    result = build_query(spec, idx=3, rng=np.random.default_rng(0))
    parsed = float(result)
    assert 0.0 <= parsed <= 9.9


def test_build_query_float_single_feature_without_domain_raises() -> None:
    spec = _make_spec("float", feature_names=["x"])
    with pytest.raises(RuntimeError, match="feature_domains"):
        build_query(spec, idx=0, rng=np.random.default_rng(0))


# build_query: float (multi-feature).


def test_build_query_float_multi_feature_json_keys_match_feature_names() -> None:
    spec = _make_spec(
        "float",
        feature_names=["a", "b"],
        feature_domains={"a": (0.0, 10.0), "b": (-5.0, 5.0)},
    )
    result = build_query(spec, idx=1, rng=np.random.default_rng(0))
    payload = json.loads(result)
    assert list(payload.keys()) == ["a", "b"]
    assert all(isinstance(v, float) for v in payload.values())
    assert 0.0 <= payload["a"] <= 10.0
    assert -5.0 <= payload["b"] <= 5.0


def test_build_query_float_multi_feature_without_domain_raises() -> None:
    spec = _make_spec("float", feature_names=["a", "b"], feature_domains={"a": (0.0, 10.0)})
    with pytest.raises(RuntimeError, match="feature_domains"):
        build_query(spec, idx=0, rng=np.random.default_rng(0))


# build_query: stage C design loop (rotated-covariates regression).


def test_stage_c_design_loop_draws_independent_covariates_not_rotated_indices() -> None:
    """Stage C draws independent covariates through one seeded RNG shared across observations.

    The covariate columns must not be fixed rotations of one row-index grid.
    """
    domains = {"a": (0.0, 10.0), "b": (0.0, 10.0), "c": (0.0, 10.0)}
    spec = _make_spec(
        "float",
        feature_names=["a", "b", "c"],
        feature_domains=domains,
        env_id="rotated_covariates_regression_env",
    )
    n_obs = 120
    base_seed = 42
    env_seed = seed_for_env(base_seed, spec.env_id)

    def _generate_rows(seed: int) -> list[dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [json.loads(build_query(spec, idx, rng=rng)) for idx in range(n_obs)]

    rows = _generate_rows(env_seed)
    columns: dict[str, list[float]] = {name: [row[name] for row in rows] for name in domains}

    # (a) no column shares its sorted value set with another under a constant
    # grid-index rotation: i.e. the covariates are not one grid variable
    # under three labels.
    for name_a, name_b in [("a", "b"), ("a", "c"), ("b", "c")]:
        assert sorted(columns[name_a]) != sorted(columns[name_b]), (
            f"{name_a!r} and {name_b!r} share a sorted value set: still one rotated grid"
        )

    # (b) pairwise correlation between independently-drawn covariates is low.
    for name_a, name_b in [("a", "b"), ("a", "c"), ("b", "c")]:
        corr = np.corrcoef(columns[name_a], columns[name_b])[0, 1]
        assert abs(corr) < 0.5, f"|corr({name_a}, {name_b})| = {abs(corr)} >= 0.5"

    # (c) reproducible from base_seed and env_id: a fresh generator seeded the
    # same way produces the identical design.
    rows_again = _generate_rows(seed_for_env(base_seed, spec.env_id))
    assert rows == rows_again

    # Sanity: still respects the declared per-feature domain.
    for name, (lo, hi) in domains.items():
        assert all(lo <= v <= hi for v in columns[name])


#, load_env_instance ---------------------------------------------------------


def test_load_env_instance_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="compiled env module missing"):
        load_env_instance(tmp_path / "does_not_exist.py", env_name="test", seed=0)


def test_load_env_instance_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.py"
    real.write_text("class TestEnv:\n  pass\n", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(real)

    with pytest.raises(RuntimeError, match="refusing to execute symlinked"):
        load_env_instance(link, env_name="test", seed=0)


def test_load_env_instance_succeeds_for_valid_module(tmp_path: Path) -> None:
    module = tmp_path / "valid_module.py"
    module.write_text(
        "class GoodEnv:\n  def __init__(self, seed):\n    self.seed = seed\n",
        encoding="utf-8",
    )
    env = load_env_instance(module, env_name="good", seed=42)
    assert env.seed == 42


def test_load_env_instance_reports_instantiation_failure(tmp_path: Path) -> None:
    module = tmp_path / "broken.py"
    module.write_text(
        "class BrokenEnv:\n  def __init__(self, seed):\n    raise ValueError('boom')\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="failed to instantiate"):
        load_env_instance(module, env_name="broken", seed=0)


def test_load_env_instance_within_trusted_root(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    module = trusted / "safe.py"
    module.write_text(
        "class SafeEnv:\n  def __init__(self, seed):\n    self.seed = seed\n",
        encoding="utf-8",
    )
    env = load_env_instance(module, env_name="safe", seed=7, trusted_root=trusted)
    assert env.seed == 7

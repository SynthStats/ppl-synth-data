"""Feature-query contracts at the generated environment and stage C boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import build_query, load_env_instance
from ppl_synth_data.pipeline.io import write_jsonl
from ppl_synth_data.pipeline.stages import stage_c_data
from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b
from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c
from ppl_synth_data.pipeline.types import RunPaths
from ppl_synth_data.specs.models import DataPlan, EnvSpec, InputType, IOSchema


def _spec(input_type: InputType, feature_names: list[str]) -> EnvSpec:
    weights = (1.0, 1.15)
    return EnvSpec(
        env_id="query_feature_contract",
        env_name="query_feature_contract",
        domain="testing",
        complexity="simple",
        model_type="predictive",
        key_concepts=["continuous covariates"],
        io_schema=IOSchema(
            input_type=input_type,
            input_description="Continuous covariates in a small positive domain.",
            output_type="float",
            target_name="y",
            feature_names=feature_names,
            feature_domains={name: (0.00001, 0.00002) for name in feature_names},
            feature_types={name: "continuous" for name in feature_names},
        ),
        data_plan=DataPlan(n_observations=24, covariates=feature_names),
        generative_family="regression_continuous",
        generative_params={
            "intercept": 1.0,
            "coefficients": dict(zip(feature_names, weights[: len(feature_names)], strict=True)),
            "noise_sigma": 0.1,
        },
    )


@pytest.mark.parametrize("input_type", [InputType.float, InputType.json])
@pytest.mark.parametrize("feature_names", [["concentration"], ["concentration", "dose"]])
def test_tiny_domain_query_round_trips_through_environment(
    tmp_path: Path, input_type: InputType, feature_names: list[str]
) -> None:
    spec = _spec(input_type, feature_names)
    module = TemplateCompiler().compile_envspec(spec, out_dir=tmp_path)
    env = load_env_instance(module, env_name=spec.env_name, seed=0, trusted_root=tmp_path)
    rng = np.random.default_rng(42)

    for idx in range(spec.data_plan.n_observations):
        query = build_query(spec, idx, rng=rng)
        result, accepted = env.run_experiment(query)
        assert accepted, f"Environment rejected serialized query {query!r}: {result}"

    received = env.get_df()
    assert len(received) == spec.data_plan.n_observations
    for feature in feature_names:
        assert received[feature].between(0.00001, 0.00002).all()
        assert received[feature].nunique() == spec.data_plan.n_observations


def _compile_specs(tmp_path: Path, specs: list[EnvSpec]) -> RunPaths:
    paths = RunPaths(
        run_root=tmp_path,
        run_id="query_contract",
        run_dir=tmp_path / "query_contract",
        stage_dirs={stage: tmp_path / "query_contract" / stage for stage in ("A", "B", "C")},
    )
    paths.stage_dirs["A"].mkdir(parents=True)
    write_jsonl(
        paths.stage_dirs["A"] / "envspecs.jsonl", [s.model_dump(mode="json") for s in specs]
    )
    compiled = run_stage_b(paths)
    assert compiled.outputs["compiled_count"] == len(specs)
    assert compiled.outputs["skipped_env_count"] == 0
    return paths


@pytest.mark.parametrize(
    "feature_type", ["binary", "integer", "categorical", "derived", "dependent", "unknown"]
)
@pytest.mark.parametrize("input_type", [InputType.float, InputType.json])
@pytest.mark.parametrize("feature_names", [["concentration"], ["concentration", "dose"]])
def test_stage_c_refuses_unsupported_feature_type(
    tmp_path: Path,
    feature_type: str,
    input_type: InputType,
    feature_names: list[str],
) -> None:
    spec = _spec(input_type, feature_names)
    unsupported_feature = feature_names[-1]
    spec.io_schema.feature_types[unsupported_feature] = feature_type
    paths = _compile_specs(tmp_path, [spec])

    with pytest.raises(RuntimeError) as exc:
        run_stage_c(paths, split_policy="auto", visible_ratio=0.75, seed=0)

    assert unsupported_feature in str(exc.value)
    assert feature_type in str(exc.value)


@pytest.mark.parametrize("input_type", [InputType.float, InputType.json])
@pytest.mark.parametrize(
    ("feature_names", "feature_types", "missing_feature"),
    [
        (["concentration"], {}, "concentration"),
        (["concentration", "dose"], {}, "concentration"),
        (["concentration", "dose"], {"concentration": "continuous"}, "dose"),
    ],
)
def test_stage_c_refuses_sampled_feature_without_explicit_type(
    tmp_path: Path,
    input_type: InputType,
    feature_names: list[str],
    feature_types: dict[str, str],
    missing_feature: str,
) -> None:
    spec = _spec(input_type, feature_names)
    spec.io_schema.feature_types = feature_types
    paths = _compile_specs(tmp_path, [spec])

    with pytest.raises(RuntimeError, match=missing_feature):
        run_stage_c(paths, split_policy="auto", visible_ratio=0.75, seed=0)


@pytest.mark.parametrize("invalid_first", [False, True])
@pytest.mark.parametrize("unsupported_type", ["binary", None])
def test_stage_c_does_not_skip_invalid_feature_contract_in_mixed_run(
    tmp_path: Path, invalid_first: bool, unsupported_type: str | None
) -> None:
    valid = _spec(InputType.json, ["concentration", "dose"])
    invalid = valid.model_copy(deep=True)
    invalid.env_id = "invalid_query_contract"
    invalid.env_name = "invalid_query_contract"
    if unsupported_type is None:
        del invalid.io_schema.feature_types["dose"]
    else:
        invalid.io_schema.feature_types["dose"] = unsupported_type
    specs = [invalid, valid] if invalid_first else [valid, invalid]
    paths = _compile_specs(tmp_path, specs)

    with pytest.raises(RuntimeError, match="dose"):
        run_stage_c(paths, split_policy="auto", visible_ratio=0.75, seed=0)


@pytest.mark.parametrize("input_type", list(InputType))
def test_query_encoding_cannot_bypass_explicit_unsupported_feature_type(
    input_type: InputType,
) -> None:
    spec = _spec(input_type, ["concentration", "dose"])
    spec.io_schema.feature_types["dose"] = "binary"

    with pytest.raises(RuntimeError) as exc:
        build_query(spec, 0, rng=np.random.default_rng(42))

    assert "dose" in str(exc.value)
    assert "binary" in str(exc.value)


@pytest.mark.parametrize("input_type", [InputType.int, InputType.string, InputType.none])
def test_query_refuses_continuous_feature_with_incompatible_encoding(
    input_type: InputType,
) -> None:
    spec = _spec(input_type, ["concentration"])

    with pytest.raises(RuntimeError) as exc:
        build_query(spec, 0, rng=np.random.default_rng(42))

    assert "concentration" in str(exc.value)
    assert input_type.value in str(exc.value)


def _zero_effect_binary_spec(n_observations: int) -> EnvSpec:
    return EnvSpec(
        env_id="zero_effect_binary",
        env_name="zero_effect_binary",
        domain="testing",
        complexity="simple",
        model_type="predictive",
        key_concepts=["binary outcome"],
        io_schema=IOSchema(
            input_type=InputType.float,
            input_description="One continuous covariate on the unit interval.",
            output_type="int",
            target_name="y",
            feature_names=["x"],
            feature_domains={"x": (0.0, 1.0)},
            feature_types={"x": "continuous"},
        ),
        data_plan=DataPlan(n_observations=n_observations, covariates=["x"]),
        generative_family="binary_choice",
        generative_params={"intercept": 0.0, "coefficients": {"x": 0.0}},
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stage_c_draws_outcomes_independently_of_the_covariates(tmp_path: Path, seed: int) -> None:
    # With no effect of x, y ~ Bernoulli(0.5) on both halves of the domain. An outcome
    # stream that repeats the covariate stream instead gives y = 1 exactly when x > 0.5.
    spec = _zero_effect_binary_spec(200)
    paths = _compile_specs(tmp_path, [spec])

    run_stage_c(paths, split_policy="random_row", visible_ratio=0.75, seed=seed)

    data_dir = paths.stage_dirs["C"] / "data"
    rows = pd.concat(
        [
            pd.read_parquet(data_dir / f"{spec.env_id}.parquet"),
            pd.read_parquet(data_dir / "hidden" / f"{spec.env_id}.parquet"),
        ]
    )
    assert len(rows) == 200
    upper = rows["x"] > 0.5
    assert abs(rows.loc[upper, "y"].mean() - rows.loc[~upper, "y"].mean()) < 0.3


@pytest.mark.narrow_unit_not_acceptance
def test_stage_c_split_does_not_reuse_the_outcome_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds: dict[str, int] = {}
    load_env = stage_c_data.load_env_instance
    split = stage_c_data.split_visible_hidden

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        seeds["outcome"] = kwargs["seed"]
        return load_env(*args, **kwargs)

    def recording_split(*args: Any, **kwargs: Any) -> Any:
        seeds["split"] = kwargs["seed"]
        return split(*args, **kwargs)

    monkeypatch.setattr(stage_c_data, "load_env_instance", recording_load)
    monkeypatch.setattr(stage_c_data, "split_visible_hidden", recording_split)
    paths = _compile_specs(tmp_path, [_zero_effect_binary_spec(40)])

    run_stage_c(paths, split_policy="random_row", visible_ratio=0.75, seed=0)

    assert seeds["split"] != seeds["outcome"]

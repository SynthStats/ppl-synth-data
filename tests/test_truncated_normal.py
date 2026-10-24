"""Observable v0.2 contract, sampling and pure consumer checks."""

import ast
import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import kstest, truncnorm

from ppl_synth_data.envs.declared_inputs import sample_inputs
from ppl_synth_data.specs.models import FourFamilySpec


def payload(law=None, n=5000):
    return {
        "env_id": "truncated-normal-test",
        "env_name": "truncated_normal_test",
        "domain": "synthetic",
        "scenario": {
            "intent": "Measure a response to bounded raw inputs",
            "source_row_sha256": "a" * 64,
        },
        "generative_family": "regression_continuous",
        "features": [
            {
                "name": "dose",
                "law": law
                or {"dist": "truncated_normal", "mean": 0.5, "sd": 2.0, "low": 0.0, "high": 1.0},
            }
        ],
        "response": {
            "family": "regression_continuous",
            "intercept": 1.0,
            "coefficients": {"dose": 3.0},
            "noise_sigma": 0.1,
        },
        "data_plan": {"n_observations": n},
    }


@pytest.mark.parametrize(
    "bounds",
    [
        {"low": 0.0},
        {"high": 1.0},
        {"low": 0, "high": 1},
        {"low": None, "high": 1.0},
        {"low": 0.0, "high": None},
        {"low": 0.5, "high": 1.0},
    ],
)
def test_valid_numeric_one_or_two_sided_bounds_round_trip(bounds):
    law = {"dist": "truncated_normal", "mean": 0.5, "sd": 2.0, **bounds}
    spec = FourFamilySpec.model_validate(payload(law))
    assert spec.model_dump(mode="json")["features"][0]["law"] == law
    assert not spec.features[0].uses_reference_coding


@pytest.mark.parametrize(
    "update,pointer",
    [
        ({"sd": 0.0}, "/features/0/law/sd"),
        ({"sd": -1.0}, "/features/0/law/sd"),
        ({"low": None, "high": None}, "/features/0/law/low"),
        ({"low": 1.0, "high": 0.0}, "/features/0/law/high"),
        ({"low": 0.5, "high": 0.5}, "/features/0/law/high"),
        ({"mean": -0.1}, "/features/0/law/mean"),
        ({"mean": 1.1}, "/features/0/law/mean"),
        ({"low": "0"}, "/features/0/law/low"),
        ({"low": True}, "/features/0/law/low"),
        ({"high": float("inf")}, "/features/0/law/high"),
    ],
)
def test_invalid_law_refuses_with_json_pointer(update, pointer, tmp_path):
    from ppl_synth_data.specs.io import read_compile_inputs

    raw = payload()
    raw["features"][0]["law"].update(update)
    with pytest.raises(ValidationError):
        FourFamilySpec.model_validate(raw)
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps(raw) + "\n")
    specs, refusals = read_compile_inputs(source)
    assert specs == []
    assert pointer in {r["json_pointer"] for r in refusals[0]["refusals"]}


def test_coefficients_use_feature_name_only():
    raw = payload()
    assert FourFamilySpec.model_validate(raw).response.coefficients == {"dose": 3.0}
    raw["response"]["coefficients"] = {"dose=0": 3.0}
    with pytest.raises(ValidationError, match="/response/coefficients"):
        FourFamilySpec.model_validate(raw)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.0, None), (None, 1.0)])
def test_real_draws_have_truncated_moments_support_and_no_boundary_atoms(bounds):
    low, high = bounds
    law = {"dist": "truncated_normal", "mean": 0.5, "sd": 2.0, "low": low, "high": high}
    spec = FourFamilySpec.model_validate(payload(law))
    values = np.array([row["dose"] for row in sample_inputs(spec, seed=901)])
    a = -np.inf if low is None else (low - 0.5) / 2.0
    b = np.inf if high is None else (high - 0.5) / 2.0
    law_reference = truncnorm(a, b, loc=0.5, scale=2.0)
    expected_mean, expected_var = law_reference.stats(moments="mv")
    assert np.isfinite(values).all()
    if low is not None:
        assert np.all(values > low)
    if high is not None:
        assert np.all(values < high)
    assert abs(values.mean() - expected_mean) < 6 * np.sqrt(expected_var / len(values))
    assert abs(values.std() - np.sqrt(expected_var)) < 0.06
    assert kstest(values, law_reference.cdf).pvalue > 1e-5


def test_seed_replay_and_independent_feature_streams():
    raw = payload()
    raw["features"].append({"name": "second", "law": deepcopy(raw["features"][0]["law"])})
    raw["response"]["coefficients"]["second"] = -0.2
    spec = FourFamilySpec.model_validate(raw)
    rows = sample_inputs(spec, seed=902)
    assert rows == sample_inputs(spec, seed=902)
    assert rows != sample_inputs(spec, seed=903)
    first, second = np.array([[r["dose"], r["second"]] for r in rows]).T
    assert not np.array_equal(first, second)
    assert abs(np.corrcoef(first, second)[0, 1]) < 0.06
    one = FourFamilySpec.model_validate(payload())
    assert first.tolist() == [r["dose"] for r in sample_inputs(one, seed=902)]


def test_v02_schema_accepts_new_and_existing_laws_and_refuses_unbounded_law():
    schema_path = Path(__file__).parent / "fixtures/declared_specs/envspec-4f-v0.2.schema.json"
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload(), schema)
    jsonschema.validate(payload({"dist": "normal", "mean": 0.5, "sd": 2.0}), schema)
    invalid = payload()
    invalid["features"][0]["law"].update(low=None, high=None)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_real_response_gold_and_stage_d_accept_raw_bounded_feature():
    from ppl_synth_data.envs.declared_response import DeclaredResponseEnv
    from ppl_synth_data.harbor.declared_gold import render_declared_gold
    from ppl_synth_data.pipeline.stages.stage_d_prompts import build_prompt_record
    from ppl_synth_data.runtimes import get_runtime

    spec = FourFamilySpec.model_validate(payload())
    env = DeclaredResponseEnv(spec.model_dump(mode="json"), seed=904)
    draws = [env.step(env.validate_input("0.8")) for _ in range(1000)]
    assert abs(np.mean(draws) - 3.4) < 0.02
    visible = {
        "n": 5,
        "dose": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "y": np.array([1.3, 1.9, 2.5, 3.1, 3.7]),
    }
    code, audit = render_declared_gold(spec, visible)
    ast.parse(code)
    assert audit["coefficient_keys"] == ["dose"]
    assert "np.asarray(data['dose'])" in code
    record = build_prompt_record(
        request_id="bounded-prompt",
        spec=spec,
        prompt_text="Fit the response using dose.",
        runtime=get_runtime("pymc"),
    )
    assert record.env_id == spec.env_id


@pytest.mark.parametrize("query", ["0.0", "1.0"])
def test_real_response_query_accepts_closed_bound_endpoints(query):
    from ppl_synth_data.envs.declared_response import DeclaredResponseEnv

    env = DeclaredResponseEnv(payload(), seed=905)
    assert env.validate_input(query) == (float(query),)


@pytest.mark.parametrize("query", ["-0.01", "1.01"])
def test_real_response_query_rejects_outside_bounds(query):
    from ppl_synth_data.envs.base import InputValidationError
    from ppl_synth_data.envs.declared_response import DeclaredResponseEnv

    env = DeclaredResponseEnv(payload(), seed=905)
    with pytest.raises(InputValidationError, match="/features/0/law"):
        env.validate_input(query)


@pytest.mark.parametrize("appendix", ["_pymc_data_schema_appendix", "_lazy_ppl_schema_appendix"])
def test_real_stage_b_c_bounded_data_and_stage_d_visible_appendix(tmp_path, appendix):
    import pandas as pd

    from ppl_synth_data.pipeline.artifacts import resolve_run_paths
    from ppl_synth_data.pipeline.io import read_jsonl, write_jsonl
    from ppl_synth_data.pipeline.stages import stage_d_prompts as stage_d
    from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b
    from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c

    raw = payload(n=100)
    paths = resolve_run_paths(tmp_path, "bounded-stage-integration")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [raw])
    assert run_stage_b(paths).status == "success"
    assert (
        run_stage_c(paths, seed=20260930, split_policy="random_row", visible_ratio=0.6).status
        == "success"
    )
    spec = FourFamilySpec.model_validate(raw)
    data_dir = paths.stage_dirs["C"] / "data"
    visible = pd.read_parquet(data_dir / f"{spec.env_id}.parquet")
    hidden = pd.read_parquet(data_dir / "hidden" / f"{spec.env_id}.parquet")
    assert (len(visible), len(hidden)) == (60, 40)
    for table in (visible, hidden):
        assert set(table.columns) == {"dose", "y"}
        assert table["dose"].between(0.0, 1.0, inclusive="neither").all()
        assert np.isfinite(table["y"]).all()
    combined = pd.concat([visible, hidden], ignore_index=True)
    assert len(combined) == 100
    data_ref = read_jsonl(paths.stage_dirs["C"] / "data_refs.jsonl")[0]
    text = getattr(stage_d, appendix)(spec=spec, data_ref=data_ref, run_dir=paths.run_dir)
    assert "`dose`" in text and "`y`" in text
    assert "n=60" in text
    assert "n=100" not in text and "n=40" not in text
    assert spec.scenario.source_row_sha256 not in text
    assert "hidden" not in text

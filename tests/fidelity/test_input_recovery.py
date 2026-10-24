"""declared input laws checked against real sampler episodes."""

import copy
import json
import math
import os
import zlib
from collections import Counter
from pathlib import Path

import pytest

SOURCE = Path(
    os.environ.get(
        "FIDELITY_SPECS",
        Path(__file__).resolve().parents[1] / "fixtures/declared_specs/selected-120.jsonl",
    )
)
SPECS = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]


def selected_law(dist):
    for spec in SPECS:
        for feature in spec["features"]:
            law = feature["law"]
            if law["dist"] != dist:
                continue
            if dist == "categorical" and len(set(law["probs"])) == 1:
                continue
            if dist == "fixed_counts" and len(set(law["counts"])) == 1:
                continue
            return spec, feature
    raise AssertionError(f"selected120 lacks required {dist} design")


def actual_episodes(spec, wrapper=None):
    from ppl_synth_data.envs.declared_inputs import sample_inputs
    from ppl_synth_data.specs.models import FourFamilySpec

    # The same fixed seed rule and observation floor as response recovery;
    # no search for a seed that makes a statistical assertion pass.
    seed = 260930 + zlib.crc32(spec["env_id"].encode())
    episodes = []
    for episode in range(math.ceil(20000 / spec["data_plan"]["n_observations"])):
        typed = FourFamilySpec.model_validate(spec)
        episodes.append((wrapper or sample_inputs)(typed, seed=seed + episode))
    return episodes


def check(spec, episodes):
    from input_recovery import check_input_laws
    from recovery_run import FALSE_ALARM_BUDGET, parameter_budget

    result = check_input_laws(spec, episodes, FALSE_ALARM_BUDGET / parameter_budget(SPECS))
    assert type(result["passed"]) is bool
    assert isinstance(result["checks"], list) and result["checks"]
    for item in result["checks"]:
        assert isinstance(item["name"], str)
        assert type(item["passed"]) is bool
        if "pvalue" in item:
            assert 0 <= item["pvalue"] <= 1
    return result


@pytest.mark.parametrize(
    "dist,expected",
    [
        ("normal", 2),
        ("bernoulli", 1),
        ("categorical", 1),
        ("discrete_uniform", 1),
        ("fixed_counts", 0),
    ],
)
def test_input_budget_reserves_each_stochastic_law(dist, expected):
    from input_recovery import input_check_count

    _, feature = selected_law(dist)
    assert input_check_count({"features": [feature]}) == expected


def test_response_and_input_checks_share_one_corpus_budget():
    from recovery_run import FALSE_ALARM_BUDGET, parameter_budget

    response_count = sum(
        1
        + len(s["response"]["coefficients"])
        + (s["response"]["family"] == "regression_continuous")
        + (s["response"].get("count_distribution") == "nb2")
        + (s["response"]["family"] == "zero_inflated_count")
        for s in SPECS
    )
    input_count = sum(
        {
            "normal": 2,
            "truncated_normal": 3,
            "bernoulli": 1,
            "categorical": 1,
            "discrete_uniform": 1,
            "fixed_counts": 0,
        }[f["law"]["dist"]]
        for s in SPECS
        for f in s["features"]
    )
    assert FALSE_ALARM_BUDGET == 0.01
    assert parameter_budget(SPECS) == response_count + input_count


@pytest.mark.parametrize(
    "dist", ["normal", "bernoulli", "categorical", "discrete_uniform", "fixed_counts"]
)
def test_real_declared_input_law_recovers(dist):
    spec, _ = selected_law(dist)
    result = check(spec, actual_episodes(spec))
    assert result["passed"], result
    assert all(item["passed"] for item in result["checks"]), result


@pytest.mark.parametrize(
    "mutation",
    [
        "normal_mean",
        "normal_sd",
        "categorical_probs",
        "uniform_high_exclusive",
        "fixed_counts_permuted",
    ],
)
def test_real_sampler_input_mutations_are_rejected(mutation):
    from ppl_synth_data.envs.declared_inputs import sample_inputs
    from ppl_synth_data.specs.models import FourFamilySpec

    dist = {
        "normal_mean": "normal",
        "normal_sd": "normal",
        "categorical_probs": "categorical",
        "uniform_high_exclusive": "discrete_uniform",
        "fixed_counts_permuted": "fixed_counts",
    }[mutation]
    spec, feature = selected_law(dist)

    def corrupted_sampler(typed, *, seed):
        altered = typed.model_dump()
        law = next(f["law"] for f in altered["features"] if f["name"] == feature["name"])
        if mutation == "normal_mean":
            law["mean"] += 2 * law["sd"]
        elif mutation == "normal_sd":
            law["sd"] *= 2
        elif mutation == "uniform_high_exclusive":
            law["high"] -= 1
        else:
            values = law["probs" if dist == "categorical" else "counts"]
            low, high = values.index(min(values)), values.index(max(values))
            values[low], values[high] = values[high], values[low]
        return sample_inputs(FourFamilySpec.model_validate(altered), seed=seed)

    result = check(spec, actual_episodes(spec, corrupted_sampler))
    printed = copy.deepcopy(result)
    for item in printed["checks"]:
        if "failures" in item:
            item["failure_count"] = len(item["failures"])
            item["failures"] = item["failures"][:2]
    print(
        json.dumps(
            {"env_id": spec["env_id"], "mutation": mutation, "result": printed}, sort_keys=True
        )
    )
    assert not result["passed"], result
    assert any(
        not item["passed"] and item["name"].startswith(feature["name"] + ":")
        for item in result["checks"]
    ), result


def test_fixed_counts_rejects_compensating_corruption_in_each_episode():
    spec, feature = selected_law("fixed_counts")
    episodes = actual_episodes(spec)[:2]
    corrupted = copy.deepcopy(episodes)
    name = feature["name"]
    left, right = feature["law"]["levels"][:2]
    next(row for row in corrupted[0] if row[name] == left)[name] = right
    next(row for row in corrupted[1] if row[name] == right)[name] = left
    # Pooling conceals the defect: each episode must retain its own allocation.
    assert Counter(row[name] for rows in episodes for row in rows) == Counter(
        row[name] for rows in corrupted for row in rows
    )
    allocation_spec = copy.deepcopy(spec)
    allocation_spec["features"] = [feature]
    result = check(allocation_spec, corrupted)
    print(
        json.dumps(
            {
                "env_id": spec["env_id"],
                "mutation": "fixed_counts_compensating_swaps",
                "result": result,
            },
            sort_keys=True,
        )
    )
    assert any(
        item["name"] == name + ":fixed_counts" and not item["passed"] for item in result["checks"]
    ), result
    for episode in corrupted:
        result = check(allocation_spec, [episode])
        assert any(
            item["name"] == name + ":fixed_counts" and not item["passed"]
            for item in result["checks"]
        ), result


def test_numeric_fixed_counts_design_uses_raw_values():
    from recovery_mle import design_columns, design_matrix

    spec = {
        "features": [{"name": "dose", "law": {"dist": "fixed_counts", "levels": [0, 50, 100]}}],
        "response": {"coefficients": {"dose": 0.2}},
    }
    assert design_columns(spec) == [("dose", "dose", None)]
    assert design_matrix(spec, [{"dose": 0}, {"dose": 50}, {"dose": 100}]).tolist() == [
        [1.0, 0.0],
        [1.0, 50.0],
        [1.0, 100.0],
    ]

"""Parameter-recovery checks for wide uniform input laws."""

import copy

import pytest
from test_input_recovery import SPECS, actual_episodes, check


@pytest.mark.parametrize(
    "env_id,feature_name",
    [
        ("4f-v01-fwd-168", "origin_population"),
        ("4f-v01-fwd-082", "average_daily_traffic"),
    ],
)
def test_wide_discrete_uniform_real_spec_recovers(env_id, feature_name):
    spec = next(spec for spec in SPECS if spec["env_id"] == env_id)
    result = check(spec, actual_episodes(spec))
    assert result["passed"], result
    frequency = next(
        item for item in result["checks"] if item["name"] == feature_name + ":frequencies"
    )
    assert frequency["passed"], frequency


@pytest.mark.parametrize("bad_value", [19999, 200001, 20000.0, True])
def test_wide_uniform_rejects_invalid_support(bad_value):
    spec = next(spec for spec in SPECS if spec["env_id"] == "4f-v01-fwd-168")
    episodes = actual_episodes(spec)
    episodes[0][0]["origin_population"] = bad_value
    result = check(spec, episodes)
    assert not result["passed"]
    support = next(item for item in result["checks"] if item["name"] == "origin_population:support")
    assert not support["passed"]


def test_unequal_integer_bins_use_exact_support_masses():
    from input_recovery import check_input_laws

    from ppl_synth_data.envs.declared_inputs import sample_inputs
    from ppl_synth_data.specs.models import FourFamilySpec

    spec = copy.deepcopy(next(spec for spec in SPECS if spec["env_id"] == "4f-v01-fwd-168"))
    spec["data_plan"]["n_observations"] = 20
    spec["features"][0]["law"] = {"dist": "discrete_uniform", "low": 10, "high": 16}
    episode = sample_inputs(FourFamilySpec.model_validate(spec), seed=260930)
    # The other feature laws are outside this small arithmetic check.
    spec["features"] = spec["features"][:1]
    result = check_input_laws(spec, [episode], alpha=0.01)
    frequency = next(item for item in result["checks"] if item["name"].endswith(":frequencies"))
    # Seven integers split into three bins: two, two, and three integers.
    assert frequency["bins"] == [[10, 11], [12, 13], [14, 16]]
    assert frequency["expected"] == pytest.approx([40 / 7, 40 / 7, 60 / 7])
    assert frequency["observed"] == [
        sum(10 <= row["origin_population"] <= 11 for row in episode),
        sum(12 <= row["origin_population"] <= 13 for row in episode),
        sum(14 <= row["origin_population"] <= 16 for row in episode),
    ]

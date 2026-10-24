"""Independent input recovery rejects the clipping and dropped-bound mutations."""

import sys
from copy import deepcopy
from pathlib import Path

# Keep direct fidelity-file invocation equivalent to repository-root collection.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from ppl_synth_data.envs.declared_inputs import sample_inputs
from ppl_synth_data.specs.models import FourFamilySpec
from tests.fidelity.input_recovery import check_input_laws, input_check_count
from tests.fidelity.recovery_mle import design_columns, design_matrix
from tests.test_truncated_normal import payload

POSITIVE_LAWS = [
    {"dist": "truncated_normal", "mean": 0.5, "sd": 2.0, "low": low, "high": high}
    for low, high in [(0.0, 1.0), (0.0, None), (None, 1.0)]
]


def shared_alpha():
    from recovery_run import FALSE_ALARM_BUDGET, parameter_budget

    from tests.fidelity.test_input_recovery import SPECS

    return FALSE_ALARM_BUDGET / parameter_budget([*SPECS, *[payload(law) for law in POSITIVE_LAWS]])


def test_truncated_law_allocates_three_stochastic_checks():
    assert input_check_count(payload()) == 3


def test_independent_response_design_uses_raw_values():
    raw = payload()
    assert design_columns(raw) == [("dose", "dose", None)]
    np.testing.assert_array_equal(
        design_matrix(raw, [{"dose": 0.2}, {"dose": 0.8}]), [[1.0, 0.2], [1.0, 0.8]]
    )


@pytest.mark.parametrize("law", POSITIVE_LAWS)
def test_real_sampler_passes_pooled_truncated_input_recovery(law):
    raw = payload(law)
    low, high = law["low"], law["high"]
    spec = FourFamilySpec.model_validate(raw)
    episodes = [sample_inputs(spec, seed=seed) for seed in (910, 911, 912, 913)]
    alpha = shared_alpha()
    result = check_input_laws(raw, episodes, alpha)
    assert result["passed"], result
    stochastic = [c for c in result["checks"] if "pvalue" in c]
    assert len(stochastic) == 3
    assert all(c["alpha"] == alpha for c in stochastic)
    from scipy.stats import truncnorm

    a = -np.inf if low is None else (low - 0.5) / 2.0
    b = np.inf if high is None else (high - 0.5) / 2.0
    mean, var = truncnorm(a, b, loc=0.5, scale=2.0).stats(moments="mv")
    moments = {c["name"]: c for c in stochastic}
    assert moments["dose:mean"]["truth"] == pytest.approx(float(mean))
    assert moments["dose:sd"]["truth"] == pytest.approx(float(np.sqrt(var)))


@pytest.mark.parametrize("mutation", ["clip", "drop_bound"])
def test_real_normal_sampler_mutations_are_rejected(mutation):
    declared = payload()
    altered = deepcopy(declared)
    altered["features"][0]["law"] = {"dist": "normal", "mean": 0.5, "sd": 2.0}
    normal = FourFamilySpec.model_validate(altered)
    episodes = [sample_inputs(normal, seed=seed) for seed in (910, 911, 912, 913)]
    if mutation == "clip":
        for episode in episodes:
            for row in episode:
                row["dose"] = float(np.clip(row["dose"], 0.0, 1.0))
    result = check_input_laws(declared, episodes, shared_alpha())
    assert not result["passed"], result
    assert any(not c["passed"] for c in result["checks"])

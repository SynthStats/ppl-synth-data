"""Check distribution contracts in the real scoring runtime without posterior fitting."""

import json

import numpy as np
import pymc as pm
import pytensor
import pytest
from pymc.logprob.utils import ParameterValueError
from pymc.variational.minibatch_rv import MinibatchRandomVariable


@pytest.mark.heavy_lane
def test_mixture_logp_rejects_non_normalized_weights():
    assert (pm.__version__, pytensor.__version__, np.__version__) == ("5.28.5", "2.38.3", "2.4.6")
    components = pm.Normal.dist(mu=[-1.0, 1.0], sigma=1.0)
    valid = float(pm.logp(pm.Mixture.dist(w=[0.5, 0.5], comp_dists=components), 0.0).eval())
    assert np.isfinite(valid)
    results = {"normalized_logp": valid, "invalid": []}
    for weights in ([1.0, 1.0], [0.2, 0.2]):
        with pytest.raises(ParameterValueError) as exc:
            pm.logp(pm.Mixture.dist(w=weights, comp_dists=components), 0.0).eval()
        with pm.Model() as model:
            pm.Mixture(
                "obs",
                w=weights,
                comp_dists=pm.Normal.dist(mu=[-1.0, 1.0], sigma=1.0),
                observed=np.array([0.0]),
            )
        compiled_logp = float(model.compile_logp()({}))
        assert compiled_logp == -np.inf
        results["invalid"].append(
            {
                "weights": weights,
                "direct_exception": type(exc.value).__name__,
                "compiled_logp": str(compiled_logp),
            }
        )
    print(json.dumps(results, sort_keys=True))


def test_total_size_observation_has_minibatch_random_variable_op():
    with pm.Model() as model:
        obs = pm.Normal("obs", 0.0, 1.0, observed=np.array([0.0, 1.0]), total_size=10)
    assert obs in model.observed_RVs
    assert isinstance(obs.owner.op, MinibatchRandomVariable)
    with pm.Model():
        plain = pm.Normal("plain", 0.0, 1.0, observed=np.array([0.0, 1.0]))
    assert not isinstance(plain.owner.op, MinibatchRandomVariable)
    print(
        json.dumps(
            {
                "observed_names": [rv.name for rv in model.observed_RVs],
                "op_class": type(obs.owner.op).__name__,
                "op_module": type(obs.owner.op).__module__,
                "plain_op_class": type(plain.owner.op).__name__,
            },
            sort_keys=True,
        )
    )

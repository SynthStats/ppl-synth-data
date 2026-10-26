"""An artificial half-size covariance must not floor a valid equal split.

The diagnostic probe's non-square Cholesky error leaves the real split scoreable.
"""

import json
import math

import numpy as np

from ppl_synth_data.harbor import validator_template as validator

CODE = """
import numpy as np
import pymc as pm

def model(data):
    chol = np.linalg.cholesky(data["covariance"])
    with pm.Model() as m:
        beta = pm.MvNormal("beta", mu=np.zeros(chol.shape[0]), chol=chol,
                           shape=chol.shape[0])
        pm.Normal("obs", mu=pm.math.dot(data["x"], beta), sigma=1.0,
                  observed=data["y"])
    return m
"""


def test_equal_split_n_by_n_covariance_probe_failure_still_scores():
    visible = {
        "n": 6,
        "covariance": np.eye(6),
        "x": np.eye(6),
        "y": np.array([-1.0, 0.0, 1.0, -0.5, 0.5, 1.5]),
    }
    hidden = {
        "n": 6,
        "covariance": np.eye(6),
        "x": np.eye(6),
        "y": np.array([-0.8, 0.2, 1.2, -0.4, 0.4, 1.4]),
    }
    result = validator.check_l6_holdout_elpd(
        CODE, visible, hidden, target_name="y", draws=200, tune=200
    )
    print(json.dumps(result, sort_keys=True))
    assert result["outcome"] == "scored", result
    assert math.isfinite(result["holdout_elpd"]), result
    assert result["hidden_n_obs"] == 6, result

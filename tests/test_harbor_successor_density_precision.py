"""Characterize the real PyMC constant-scale precision used by likelihood scoring."""

import numpy as np
import pymc as pm
from scipy.stats import norm


def test_constant_normal_scale_precision():
    y = np.array([-0.3, 2.7, 3.1])
    integer_scale = pm.Normal.dist(0, 2)
    double_scale = pm.Normal.dist(0, np.float64(2))
    a = pm.logp(integer_scale, y).eval()
    b = pm.logp(double_scale, y).eval()
    reference = norm.logpdf(y, 0, 2)
    print(
        {
            "integer_scale_dtype": integer_scale.owner.inputs[-1].dtype,
            "double_scale_dtype": double_scale.owner.inputs[-1].dtype,
            "integer_scale_elpd": a.sum(),
            "double_scale_elpd": b.sum(),
            "scipy_elpd": reference.sum(),
            "integer_delta": (a - reference).sum(),
        }
    )
    np.testing.assert_allclose(b, reference, rtol=0, atol=1e-12)

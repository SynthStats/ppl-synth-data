"""NumPy allowlist and observed-variable selection in the gate harness.

Supported NumPy operations execute in the harness. LOO selects the target
by name when a model has multiple observed variables.
"""

import arviz as az
import numpy as np
import pytest

from ppl_synth_data.validation import _pymc_semantics_harness as harness

TEACHER_FUNCTIONS = (
    "floor",
    "ceil",
    "quantile",
    "full",
    "argsort",
    "argmin",
    "ptp",
    "unique",
    "float32",
)
# Names that read or write files, load objects, map memory, reach C, or hand back
# uninitialised memory (which, in the
# process that also holds HIDDEN_DATA, is a possible read of held-out values).
NEVER = {
    "load",
    "loadtxt",
    "save",
    "savez",
    "savetxt",
    "fromfile",
    "tofile",
    "frombuffer",
    "fromregex",
    "genfromtxt",
    "memmap",
    "ctypeslib",
    "lib",
    "empty",
    "empty_like",
    "DataSource",
    "set_printoptions",
    "seterr",
    "random",
}


@pytest.mark.parametrize("name", TEACHER_FUNCTIONS)
def test_teacher_numpy_function_is_allowed(name):
    assert getattr(harness.ALLOWED_NUMPY, name) is getattr(np, name)


def test_every_allowed_numpy_name_is_a_pure_computation():
    for name, value in vars(harness.ALLOWED_NUMPY).items():
        assert name not in NEVER, name
        module = getattr(value, "__module__", "numpy") or "numpy"
        assert not any(
            part in module
            for part in ("npyio", "ctypeslib", "memmap", "format", "testing", "random")
        ), name


def _idata(log_likelihood: dict) -> az.InferenceData:
    rng = np.random.default_rng(0)
    return az.from_dict(posterior={"mu": rng.normal(size=(2, 50))}, log_likelihood=log_likelihood)


def test_loo_covers_two_observed_variables():
    rng = np.random.default_rng(1)
    y, y_initial = rng.normal(-1.0, 0.1, size=(2, 50, 20)), rng.normal(-2.0, 0.1, size=(2, 50, 1))
    loo = harness._loo_all_observed(_idata({"y": y, "y_initial": y_initial}))
    assert np.asarray(loo.pareto_k).size == 21
    single = az.loo(_idata({"all": np.concatenate([y, y_initial], axis=-1)}), pointwise=True)
    assert loo.elpd_loo == pytest.approx(single.elpd_loo)


def test_loo_with_one_observed_variable_is_unchanged():
    y = np.random.default_rng(2).normal(-1.0, 0.1, size=(2, 50, 20))
    idata = _idata({"y": y})
    assert harness._loo_all_observed(idata).elpd_loo == pytest.approx(
        az.loo(idata, pointwise=True).elpd_loo
    )

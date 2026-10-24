"""Environment-only preflight: real imports and exact scoring pins."""

import importlib.metadata

import pytest


@pytest.mark.heavy_lane
def test_scoring_pins_and_real_runtime_imports():
    import arviz
    import numpy
    import pymc
    import pytensor

    assert (pymc.__version__, pytensor.__version__, numpy.__version__) == (
        "5.28.5",
        "2.38.3",
        "2.4.6",
    )
    assert arviz.__version__ == "0.23.4"
    print(
        {
            p: importlib.metadata.version(p)
            for p in ["pymc", "pytensor", "numpy", "arviz", "scipy", "pandas", "pyarrow"]
        }
    )

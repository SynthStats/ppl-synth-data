from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ppl_synth_data.pipeline.data_summary import summarize_numeric


def test_empty_list():
    assert summarize_numeric([]) == {"n": 0}


def test_single_value():
    result = summarize_numeric([5.0])
    assert result["n"] == 1
    assert result["mean"] == 5.0
    assert result["std"] == 0.0
    assert result["median"] == 5.0
    assert result["q25"] == 5.0
    assert result["q75"] == 5.0


def test_known_stats():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = summarize_numeric(values)
    assert result["n"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["std"] == pytest.approx(np.std(values, ddof=0))
    assert result["median"] == 3.0
    assert result["q25"] == pytest.approx(np.quantile(values, 0.25))
    assert result["q75"] == pytest.approx(np.quantile(values, 0.75))
    assert result["p90"] == pytest.approx(np.quantile(values, 0.90))
    assert result["p99"] == pytest.approx(np.quantile(values, 0.99))


def test_all_values_are_float():
    result = summarize_numeric([1, 2, 3])
    for key in ("mean", "std", "median", "q25", "q75", "p90", "p99"):
        assert isinstance(result[key], float)


def test_dataset_summarize_module_removed() -> None:
    assert importlib.util.find_spec("ppl_synth_data.dataset.summarize") is None


def test_data_summary_module_removed_from_generated_data_namespace() -> None:
    # A clean checkout has no ppl_synth_data.data package, and find_spec raises for a
    # missing parent instead of returning None; both mean the module is gone.
    try:
        spec = importlib.util.find_spec("ppl_synth_data.data.summarize")
    except ModuleNotFoundError:
        spec = None
    assert spec is None

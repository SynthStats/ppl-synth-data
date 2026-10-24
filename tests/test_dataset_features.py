from __future__ import annotations

from datasets import Features

from ppl_synth_data.dataset.features import (
    pymc_sft_features,
    pymc_with_data_features,
    sft_features,
    with_data_features,
)


def test_sft_returns_features():
    f = pymc_sft_features()
    assert isinstance(f, Features)


def test_sft_has_15_keys():
    assert len(pymc_sft_features()) == 15


def test_with_data_returns_features():
    f = pymc_with_data_features()
    assert isinstance(f, Features)


def test_with_data_has_23_keys():
    assert len(pymc_with_data_features()) == 23
    assert "parsed_program" in pymc_with_data_features()


def test_with_data_is_superset_of_sft():
    sft_keys = set(pymc_sft_features())
    wd_keys = set(pymc_with_data_features())
    assert sft_keys.issubset(wd_keys)


def test_sft_features_parameterized_pymc():
    f = sft_features("pymc")
    assert "pymc_version" in f
    assert f.type == pymc_sft_features().type


def test_sft_features_parameterized_lazyppl():
    f = sft_features("lazyppl")
    assert "lazyppl_version" in f
    assert "pymc_version" not in f
    assert len(f) == 15


def test_with_data_features_parameterized_lazyppl():
    f = with_data_features("lazyppl")
    assert "lazyppl_version" in f
    assert "pymc_version" not in f
    assert "parsed_program" in f
    assert len(f) == 23


def test_parameterized_with_data_is_superset_of_sft():
    for rt_name in ("pymc", "lazyppl"):
        sft_keys = set(sft_features(rt_name))
        wd_keys = set(with_data_features(rt_name))
        assert sft_keys.issubset(wd_keys)

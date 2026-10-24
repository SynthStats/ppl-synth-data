from __future__ import annotations

import subprocess

import numpy as np
import pytest

import ppl_synth_data.validation.pymc_runtime as pymc_runtime
from ppl_synth_data.validation.pymc_runtime import runtime_compile_check
from ppl_synth_data.validation.sample_data import serialize_sample_data


def test_serialize_scalar_values() -> None:
    data = {"n": 10, "label": "test"}
    result = serialize_sample_data(data)
    assert '"n": 10' in result
    assert '"label": "test"' in result
    assert result.startswith("{")
    assert result.endswith("}")


def test_serialize_numpy_arrays() -> None:
    data = {"y": np.array([1.0, 2.0, 3.0])}
    result = serialize_sample_data(data)
    assert "np.array" in result
    assert "[1.0, 2.0, 3.0]" in result


def test_serialize_mixed() -> None:
    data = {"n": 5, "X": np.array([[1, 2], [3, 4]])}
    result = serialize_sample_data(data)
    assert '"n": 5' in result
    assert "np.array" in result


def test_serialize_empty() -> None:
    result = serialize_sample_data({})
    assert result == "{}"


def test_runtime_compile_check_rejects_missing_sample_data() -> None:
    result = runtime_compile_check("def model(data):\n    return None\n", sample_data=None)

    assert result.ok is False
    assert "sample_data is required" in result.message


@pytest.mark.narrow_unit_not_acceptance
def test_runtime_compile_check_omits_preexec_on_windows(monkeypatch) -> None:
    observed_has_preexec: list[bool] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        observed_has_preexec.append("preexec_fn" in kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="OK\n", stderr="")

    monkeypatch.setattr(pymc_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(pymc_runtime, "run_with_process_group_timeout", fake_run)

    result = runtime_compile_check(
        "def model(data):\n    return None\n",
        sample_data={"x": np.array([1.0])},
    )

    assert result.ok is True
    assert observed_has_preexec == [False]

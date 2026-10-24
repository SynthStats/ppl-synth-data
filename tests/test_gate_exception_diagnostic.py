"""Visible-phase exception text remains a diagnostic.

The reason stays sanitized. Visible-data exceptions populate Score.diagnostic;
held-out exceptions omit their text because it can contain held-out values.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ppl_synth_data.gates import reward_variance as gate
from ppl_synth_data.validation.pymc_semantics import score_pymc_semantics

# Allow cold PyMC imports within the diagnostic test timeout.
COLD_START_TIMEOUT_SEC = 300


def _row() -> dict[str, object]:
    return {
        "row_id": "r1",
        "runtime": "pymc",
        "split": "train",
        "data_source": "fixture",
        "structural_class": None,
        "prompt": 'Observed data JSON: {"y": [1]}',
        "visible_data": {"y": [1]},
        "heldout_data": {"y": [2]},
        "dataset": gate.DATASET,
        "revision": "a" * 40,
    }


def _settings() -> gate.VerifierSettings:
    return gate.VerifierSettings(
        data_shape="lists",
        pymc_draws=50,
        pymc_tune=50,
        pymc_timeout_sec=60,
        lazyppl_timeout_sec=600,
        lazyppl_harbor_image="lazyppl-harbor:test",
    )


def _score_with(monkeypatch, result: SimpleNamespace) -> gate.Score:
    from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime
    from ppl_synth_data.validation import pymc_semantics

    monkeypatch.setattr(PyMCRuntime, "extract_code", lambda self, text: "model")
    monkeypatch.setattr(pymc_semantics, "score_pymc_semantics", lambda *args, **kwargs: result)
    return gate.score_one(_row(), {"row_id": "r1", "sample_idx": 0, "text": "program"}, _settings())


def test_the_harness_keeps_visible_phase_exception_text_in_details_only():
    result = score_pymc_semantics(
        "def model(data):\n    raise ValueError('SENSITIVE_MARKER')\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=COLD_START_TIMEOUT_SEC,
    )
    assert result.message == "model(visible_data) raised: ValueError"
    assert result.details["exception"] == "ValueError: SENSITIVE_MARKER"


def test_the_gate_copies_the_diagnostic_and_keeps_the_reason(monkeypatch):
    score = _score_with(
        monkeypatch,
        SimpleNamespace(
            ok=False,
            metric_name=None,
            metric_value=None,
            status="critical",
            message="model(visible_data) raised: ValueError",
            details={"exception": "ValueError: SENSITIVE_MARKER"},
        ),
    )
    assert score.reason == "model(visible_data) raised: ValueError"
    assert score.diagnostic == "ValueError: SENSITIVE_MARKER"


def test_a_result_without_an_exception_leaves_the_diagnostic_empty(monkeypatch):
    score = _score_with(
        monkeypatch,
        SimpleNamespace(
            ok=False,
            metric_name=None,
            metric_value=None,
            status="critical",
            message="posterior sampling failed: RuntimeError",
        ),
    )
    assert score.diagnostic is None


def test_a_held_out_phase_failure_never_fills_the_diagnostic_at_the_gate(monkeypatch):
    score = _score_with(
        monkeypatch,
        SimpleNamespace(
            ok=False,
            metric_name="train_elpd_loo",
            metric_value=-3.0,
            status="critical",
            message="holdout log-likelihood scoring failed: ValueError",
            details={"exception": "ValueError: HELDOUT_MARKER"},
        ),
    )
    assert score.diagnostic is None
    assert "HELDOUT_MARKER" not in score.reason


@pytest.mark.slow
def test_a_held_out_phase_failure_carries_no_exception_text_from_the_harness():
    # Needs real PyMC sampling in the harness subprocess, which compiles C through
    # PyTensor: it passes on a Linux host
    # and fails on this project's Macs with "posterior sampling failed: CompileError".
    code = (
        "def model(data):\n"
        "    y = np.asarray(data['y'], dtype=float)\n"
        "    if y.size == 3:\n"
        "        raise ValueError('HELDOUT_MARKER')\n"
        "    with pm.Model() as m:\n"
        "        mu = pm.Normal('mu', 0.0, 10.0)\n"
        "        pm.Normal('obs', mu, 1.0, observed=y)\n"
        "    return m\n"
    )
    result = score_pymc_semantics(
        code,
        visible_data={"y": np.array([0.1, -0.2, 0.3, 0.0, 0.5])},
        hidden_data={"y": np.array([0.2, 0.1, -0.1])},
        timeout_sec=COLD_START_TIMEOUT_SEC,
    )
    assert result.message.startswith("holdout log-likelihood scoring failed")
    assert "HELDOUT_MARKER" not in result.message
    assert "HELDOUT_MARKER" not in json.dumps(result.details)
    assert "exception" not in result.details

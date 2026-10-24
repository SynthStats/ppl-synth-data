"""Three likelihood screens required before a LazyPPL program becomes an SFT target or RL reference.

A program that compiles and returns a finite reward can still have an unsound likelihood. For one
program and one
posterior, from the harness's per-row log-likelihood matrix (draws x rows), with lpd_i = logsumexp_s
log_lik[s, i] - log S:
  1. row-ignoring: the SD of lpd_i across rows is 0 up to rounding (`logLikRow st _ =`);
  2. floor reached: any entry sits at log(1e-300), the clamp value; a sound likelihood never gives a
  real observation
     probability 1e-300, so one such entry marks a defect for review (the shares are reported);
  3. not a probability: an entry above 0 (beyond rounding) when the outcome is discrete (all values
  whole numbers).
score.py computes the numbers next to the matrix; the gate applies the thresholds, since only it
knows the outcome values.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from ppl_synth_data.gates import likelihood_screens

SCORE_PY = Path(__file__).resolve().parents[1] / "docker" / "lazyppl-harbor" / "app" / (
                'score.py'
            )
FLOOR = math.log(1e-300)
KEYS = {"screen_lpd_sd", "screen_floor_entry_share", "screen_floor_row_share", (
                'screen_max_log_lik'
            )}


def _metrics(matrix):
    spec = importlib.util.spec_from_file_location("lazyppl_score", SCORE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.likelihood_screen_metrics(np.asarray(matrix, dtype=float))


def _noisy(values, draws=40, seed=0):
    return np.array(values, dtype=float)[None, :] + np.random.default_rng(seed).normal(
        0, 0.01, size=(draws, len(values))
    )


def test_score_py_reports_the_screen_numbers():
    m = _metrics(_noisy([-1.0, -2.0, -0.5]))
    assert set(m) == KEYS
    assert (
        m["screen_lpd_sd"] > 0
        and m["screen_floor_row_share"] == 0.0
        and m["screen_max_log_lik"] < 0
    )


def test_a_row_ignoring_likelihood_has_zero_spread_of_lpd():
    constant = np.tile(np.linspace(-3, -1, 40)[:, None], (1, 5))  # varies by draw, never by row
    m = _metrics(constant)
    assert m["screen_lpd_sd"] == pytest.approx(0.0, abs=1e-12)
    assert likelihood_screens.flags(m, outcome=[1.2, 3.4, 0.5, 2.2, 1.1]) == ["row_ignoring"]


def test_more_than_half_the_rows_at_the_floor_is_flagged():
    matrix = np.full((40, 4), FLOOR)
    matrix[:, 3] = -1.0 + np.random.default_rng(1).normal(0, 0.01, size=40)
    m = _metrics(matrix)
    assert m["screen_floor_row_share"] == pytest.approx(0.75) and m[
        "screen_floor_entry_share"
    ] == pytest.approx(0.75)
    assert "floor_reached" in likelihood_screens.flags(m, outcome=[1.0, 2.5, 3.5, 0.2])


def test_half_the_rows_at_the_floor_is_flagged():
    matrix = np.full((40, 4), FLOOR)
    matrix[:, 2:] = _noisy([-1.0, -2.0], seed=2)
    m = _metrics(matrix)
    assert m["screen_floor_row_share"] == pytest.approx(0.5)
    assert "floor_reached" in likelihood_screens.flags(m, outcome=[1.0, 2.5, 3.5, 0.2])


def test_a_positive_log_value_on_a_discrete_outcome_is_not_a_probability():
    m = _metrics(_noisy([0.4, -1.0, -2.0]))
    assert likelihood_screens.flags(m, outcome=[0, 3, 1]) == ["positive_log_prob_discrete"]


def test_a_positive_log_density_on_a_continuous_outcome_is_allowed():
    assert (
        likelihood_screens.flags(_metrics(_noisy([0.4, -1.0, -2.0])), outcome=[0.12, 3.5, 1.7])
        == []
    )


def test_rounding_noise_around_zero_is_not_a_positive_log_value():
    matrix = np.random.default_rng(3).choice([2.2e-16, -2.2e-16], size=(40, 3))
    assert "positive_log_prob_discrete" not in likelihood_screens.flags(
        _metrics(matrix), outcome=[0, 1, 1]
    )


def test_a_sound_likelihood_passes_every_screen():
    assert (
        likelihood_screens.flags(_metrics(_noisy([-1.1, -2.3, -0.7, -1.9])), outcome=[2, 5, 1, 4])
        == []
    )


def test_one_row_cannot_show_row_ignoring():
    m = _metrics(_noisy([-1.0]))
    assert m["screen_lpd_sd"] is None
    assert likelihood_screens.flags(m, outcome=[2]) == []


def test_missing_screen_numbers_raise_instead_of_passing():
    with pytest.raises(KeyError):
        likelihood_screens.flags({}, outcome=[1.0])


def test_the_gate_keeps_the_screen_numbers_on_a_scored_lazyppl_program(monkeypatch):
    from types import SimpleNamespace

    from ppl_synth_data.gates import reward_variance as gate
    from ppl_synth_data.runtimes.lazyppl_runtime import LazyPPLRuntime
    from ppl_synth_data.validation import lazyppl_semantics
    from tests.test_reward_variance_gate import _row, _verifier_settings

    numbers = {
        "screen_lpd_sd": 0.0,
        "screen_floor_entry_share": 0.1,
        "screen_floor_row_share": 0.0,
        "screen_max_log_lik": -0.5,
    }
    monkeypatch.setattr(LazyPPLRuntime, "extract_code", lambda self, text: "model")
    monkeypatch.setattr(
        lazyppl_semantics,
        "score_lazyppl_semantics",
        lambda *a, **k: SimpleNamespace(
            ok=True,
            details={
                "reward_artifact": {
                    "reward_source": "validation.lazyppl_elpd",
                    "training_log_reward": -1.0,
                },
                **numbers,
            },
            metric_name="avg_holdout_logp",
            metric_value=-3.0,
            status="pass",
            message="scored",
        ),
    )
    score = gate.score_one(
        _row("lazy", runtime="lazyppl"),
        {"row_id": "lazy", "sample_idx": 0, "text": "program"},
        _verifier_settings(),
    )
    assert score.screens == numbers


def test_one_entry_at_the_floor_is_flagged():
    matrix = _noisy([-1.0, -2.0, -0.5])
    matrix[7, 1] = FLOOR
    m = _metrics(matrix)
    assert m["screen_floor_entry_share"] > 0 and m["screen_floor_row_share"] == 0.0
    assert likelihood_screens.flags(m, outcome=[1.5, 2.5, 0.5]) == ["floor_reached"]

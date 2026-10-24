"""Reward-hack regression corpus + adapter integration for PyMC selection.

Fast unit tests cover the static integrity gates directly. One end-to-end test
runs real PyMC subprocesses to prove a healthy model is selected while the
degenerate / unnormalized hacks are rejected.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import default_rng
from synthstats_schemas import CandidateKey, SelectionMode

from ppl_synth_data.validation.pymc_selection import (
    _ast_safety_gate,
    _no_unnormalized_likelihood_gate,
    gate_and_score_pymc,
)
from ppl_synth_data.validation.selection import select_for_task

# --- programs --------------------------------------------------------------

GOOD = """
def model(data):
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("obs", mu=a * x + b, sigma=sigma, observed=y)
    return m
"""

POTENTIAL_HACK = """
def model(data):
    y = np.asarray(data["y"])
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 5)
        pm.Potential("ll", -0.5 * ((y - mu) ** 2))
    return m
"""

DEGENERATE_HACK = """
def model(data):
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 0.001)
        pm.Normal("obs", mu=a * x + b, sigma=sigma, observed=y)
    return m
"""

UNSAFE_HACK = """
def model(data):
    y = np.asarray(data["y"])
    leaked = eval("1 + 1")
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 5)
        pm.Normal("obs", mu=mu, sigma=1.0, observed=y)
    return m
"""


# --- fast static-gate unit tests -------------------------------------------


def test_potential_call_fails_static_gate():
    g = _no_unnormalized_likelihood_gate(POTENTIAL_HACK)
    assert g.passed is False
    assert "Potential" in g.detail


def test_clean_code_passes_static_likelihood_gate():
    assert _no_unnormalized_likelihood_gate(GOOD).passed is True


def test_eval_fails_ast_safety_gate():
    assert _ast_safety_gate(UNSAFE_HACK).passed is False


def test_clean_code_passes_ast_safety_gate():
    assert _ast_safety_gate(GOOD).passed is True


# --- end-to-end (real PyMC) ------------------------------------------------


def _key(idx: int) -> CandidateKey:
    return CandidateKey(task_id="corpus", family="test", candidate_idx=idx)


@pytest.mark.slow
def test_corpus_good_selected_unnormalized_rejected():
    # Shared noisy task: a healthy model competes with an unnormalized-likelihood
    # hack. The good model is selected; the Potential hack is rejected.
    rng = default_rng(0)
    x_vis = np.linspace(0.0, 6.0, 14)
    y_vis = 2.0 * x_vis + 1.0 + rng.normal(0, 0.6, x_vis.size)
    x_hid = np.linspace(6.5, 10.0, 8)  # >= MIN_HOLDOUT_FOR_SELECTION so GOOD is rankable
    y_hid = 2.0 * x_hid + 1.0 + rng.normal(0, 0.6, x_hid.size)
    visible = {"x": x_vis, "y": y_vis}
    hidden = {"x": x_hid, "y": y_hid}

    good = gate_and_score_pymc(
        GOOD,
        visible_data=visible,
        hidden_data=hidden,
        candidate_key=_key(0),
        fitness_draws=80,
        fitness_tune=80,
    )
    potential = gate_and_score_pymc(
        POTENTIAL_HACK,
        visible_data=visible,
        hidden_data=hidden,
        candidate_key=_key(1),
        fitness_draws=80,
        fitness_tune=80,
    )

    assert good.gate.eligible is True
    assert good.score is not None and good.score.holdout_n == 8

    assert potential.gate.eligible is False
    assert any(
        g.name == "no_unnormalized_likelihood" and not g.passed
        for g in potential.gate.integrity_gates
    )

    sel = select_for_task(
        [good, potential],
        runtime="pymc",
        source_plan_signature="regression_continuous/medium",
        rng=rng,
    )
    assert sel.winner_key is not None and sel.winner_key.candidate_idx == 0
    modes = {r.candidate_key.candidate_idx: r.selection_mode for r in sel.records}
    assert modes[0] is SelectionMode.selected
    assert modes[1] is SelectionMode.rejected


@pytest.mark.slow
def test_sigma_to_zero_hack_caught_by_density_gate():
    # The sigma->0 hack only WINS when it can fit the held-out points sharply
    # (near-noiseless data). There it posts a degenerately-high logp via a near-
    # zero predictive scale, exactly what the density gate must reject.
    x_vis = np.linspace(0.0, 6.0, 14)
    y_vis = 2.0 * x_vis + 1.0  # noiseless: the tiny-sigma model fits perfectly
    x_hid = np.linspace(6.5, 10.0, 8)
    y_hid = 2.0 * x_hid + 1.0
    visible = {"x": x_vis, "y": y_vis}
    hidden = {"x": x_hid, "y": y_hid}

    degenerate = gate_and_score_pymc(
        DEGENERATE_HACK,
        visible_data=visible,
        hidden_data=hidden,
        candidate_key=_key(0),
        fitness_draws=80,
        fitness_tune=80,
    )

    # The fitness pass SUCCEEDED (cheap gate still passed): the rejection is the
    # density gate firing on a genuinely high-scoring degenerate model, NOT a
    # timeout/crash that would retroactively fail the crash gate.
    assert degenerate.gate.cheap_gate.passed is True
    assert degenerate.gate.eligible is False
    assert any(
        g.name == "min_predictive_scale" and not g.passed for g in degenerate.gate.density_gates
    )
    # And the implied scale is genuinely tiny (the model fit the holdout sharply).
    scale_gate = next(g for g in degenerate.gate.density_gates if g.name == "min_predictive_scale")
    assert "min_pred_std=" in scale_gate.detail


@pytest.mark.slow
def test_no_hidden_data_is_gate_only_eligible_unscored():
    # A clean model with NO holdout: passes gates, but with no held-out data it
    # cannot be scored or density-checked, so it is eligible-but-unscored and can
    # only ever be gate_only downstream (never selected).
    x_vis = np.linspace(0.0, 6.0, 12)
    y_vis = 2.0 * x_vis + 1.0
    cand = gate_and_score_pymc(
        GOOD,
        visible_data={"x": x_vis, "y": y_vis},
        hidden_data=None,
        candidate_key=_key(0),
        fitness_draws=80,
        fitness_tune=80,
    )
    assert cand.gate.eligible is True
    assert cand.gate.density_gates == ()  # no holdout -> no density signal
    assert cand.score is None

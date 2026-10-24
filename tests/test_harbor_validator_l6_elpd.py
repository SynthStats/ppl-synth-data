"""Acceptance tests for the PyMC Harbor L6 held-out-ELPD reward.

These tests cover the reward contract. All PyMC fits here are real NUTS runs with small
draw counts for test speed; check_l6_holdout_elpd and sampling are not replaced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from synthstats_schemas import RewardArtifact

from ppl_synth_data.harbor.validator_template import (
    L6_TIMEOUT_EXIT_CODE,
    check_l6_holdout_elpd,
    exit_code_for_full_cascade,
    run_full_cascade,
)

_DRAWS = 40
_TUNE = 40


@pytest.fixture()
def reg_split():
    rng = np.random.default_rng(0)
    n_visible, n_hidden = 24, 8
    x_visible = rng.normal(size=n_visible)
    x_hidden = rng.normal(size=n_hidden)
    theta_true, beta_true = 1.5, 2.0
    y_visible = theta_true + beta_true * x_visible + rng.normal(scale=0.3, size=n_visible)
    y_hidden = theta_true + beta_true * x_hidden + rng.normal(scale=0.3, size=n_hidden)
    visible = {"n": n_visible, "x": x_visible, "y": y_visible}
    hidden = {"n": n_hidden, "x": x_hidden, "y": y_hidden}
    return visible, hidden


@pytest.fixture()
def near_noiseless_split():
    """A near-zero-noise linear trend with a held-out point sitting exactly
    on it. With sigma free (not artificially collapsed by the prior), the
    posterior legitimately concentrates sigma near-zero, and the "lucky"
    held-out point gets a spuriously enormous held-out log density; the
    real selection-gaming signature the density gate exists to catch
    (density_gates.py docstring: "a near-zero residual scale that
    concentrates probability mass on the 2-3 held-out points").
    """
    theta_true, beta_true = 1.5, 2.0
    x_visible = np.linspace(-2, 2, 20)
    y_visible = (
        theta_true + beta_true * x_visible + np.random.default_rng(0).normal(scale=1e-4, size=20)
    )
    visible = {"n": 20, "x": x_visible, "y": y_visible}
    x_hidden = np.array([0.5, -1.3, 1.9, 0.0])
    y_hidden = theta_true + beta_true * x_hidden
    hidden = {"n": 4, "x": x_hidden, "y": y_hidden}
    return visible, hidden


INTERCEPT_ONLY = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    with pm.Model() as m:
        theta = pm.Normal("theta", mu=0.0, sigma=5.0)
        s = pm.HalfNormal("s", sigma=2.0)
        pm.Normal("y", mu=theta, sigma=s, observed=y)
    return m
"""

ZERO_SENSITIVITY = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        theta = pm.Normal("theta", mu=0.0, sigma=5.0)
        s = pm.HalfNormal("s", sigma=2.0)
        mu = theta + 0.0 * pm.math.sum(xd) * 0.0
        pm.Normal("y", mu=mu, sigma=s, observed=y)
    return m
"""

SHAPE_ONLY = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        theta = pm.Normal("theta", mu=0.0, sigma=5.0)
        s = pm.HalfNormal("s", sigma=2.0)
        mu = pm.math.zeros_like(xd) + theta
        pm.Normal("y", mu=mu, sigma=s, observed=y)
    return m
"""

GENUINE_FIT = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        theta = pm.Normal("theta", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=5.0)
        s = pm.HalfNormal("s", sigma=2.0)
        pm.Normal("y", mu=theta + beta * xd, sigma=s, observed=y)
    return m
"""

VARIANCE_COLLAPSE = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        theta = pm.Normal("theta", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=5.0)
        s = pm.HalfNormal("s", sigma=1e-6)
        pm.Normal("y", mu=theta + beta * xd, sigma=s, observed=y)
    return m
"""

POTENTIAL_HACK = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    with pm.Model() as m:
        theta = pm.Normal("theta", mu=0.0, sigma=5.0)
        s = pm.HalfNormal("s", sigma=2.0)
        pm.Normal("y", mu=theta, sigma=s, observed=y)
        pm.Potential("hack", theta * 10.0)
    return m
"""

DEAD_DATA_MODEL = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        pm.Data("x", x)
        theta = pm.Normal("theta", mu=0.0, sigma=5.0)
        s = pm.HalfNormal("s", sigma=2.0)
        pm.Normal("y", mu=theta, sigma=s, observed=y)
    return m
"""

UNSAMPLABLE_MODEL = """\
def model(data):
    # A strictly-positive-support likelihood (Exponential) fed the raw
    # (real-valued, negative-including) target: logp is -inf at the observed
    # data for EVERY parameter value, so NUTS init genuinely raises
    # SamplingError("Bad initial energy"): a real, not simulated,
    # sampling failure. x is still wired in (zero-weighted) so the model
    # passes L3's data-usage check.
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        lam = pm.HalfNormal("lam", sigma=1.0) + 0.0 * pm.math.sum(xd) * 0.0
        pm.Exponential("y", lam=lam, observed=y)
    return m
"""

GROUP_HOLDOUT_SHAPE_MISMATCH_MODEL = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    group_raw = np.asarray(data["group"]).astype(str)
    labels, idx = np.unique(group_raw, return_inverse=True)
    n_groups = labels.shape[0]
    with pm.Model() as m:
        mu_overall = pm.Normal("mu_overall", mu=0.0, sigma=2.0)
        tau = pm.HalfNormal("tau", sigma=1.0)
        group_offset = pm.Normal("group_offset", mu=0.0, sigma=1.0, shape=n_groups)
        mu = mu_overall + tau * group_offset[idx]
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    return m
"""


def _log_reward(visible, hidden, code, *, baseline, gold_elpd=None):
    result = run_full_cascade(
        code, visible, holdout_data=hidden, gold_elpd=gold_elpd, baseline=baseline, target_name="y"
    )
    assert result["reward_source"] == "validation.pymc_elpd", result
    assert result["diagnostics"]["elpd_status"] == "scored", result["diagnostics"]
    return result["training_log_reward"], result


class TestDegenerateModelRewards:
    """Degenerate == intercept-only, never inflated, never beats genuine fit."""

    def test_zero_sensitivity_and_shape_only_match_intercept_only(self, reg_split):
        visible, hidden = reg_split
        baseline_result = check_l6_holdout_elpd(
            INTERCEPT_ONLY, visible, hidden, target_name="y", draws=_DRAWS, tune=_TUNE
        )
        assert baseline_result["outcome"] == "scored"
        baseline = baseline_result["holdout_elpd"]

        log_reward_intercept, _ = _log_reward(visible, hidden, INTERCEPT_ONLY, baseline=baseline)
        log_reward_zero, _ = _log_reward(visible, hidden, ZERO_SENSITIVITY, baseline=baseline)
        log_reward_shape, _ = _log_reward(visible, hidden, SHAPE_ONLY, baseline=baseline)

        # Prediction-identical submissions score near-identically under the
        # primary baseline (their own holdout_elpd == the baseline itself).
        assert abs(log_reward_intercept) < 1e-2
        assert abs(log_reward_zero - log_reward_intercept) < 1e-2
        assert abs(log_reward_shape - log_reward_intercept) < 1e-2

        log_reward_genuine, _ = _log_reward(visible, hidden, GENUINE_FIT, baseline=baseline)
        assert log_reward_genuine > log_reward_intercept
        assert log_reward_genuine > log_reward_zero
        assert log_reward_genuine > log_reward_shape


class TestTargetIsolationCache:
    """L6 rejects a target cache shared between model builds.

    Each build executes in a new namespace, and submitted code receives censored held-out data.
    The cache-based submission must reach the -20.0 reward floor.
    """

    def test_cross_call_cache_replay_is_floored_not_rewarded(self, reg_split):
        visible, hidden = reg_split
        cache_cheat = """\
_seen = {}
def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    n = len(y)
    if n not in _seen:
        _seen[n] = y.copy()
    with pm.Model() as m:
        xd = pm.Data("x", x)
        b = pm.Normal("b", 0.0, 1.0)
        mu = _seen[n] + 0.0 * xd * b
        pm.Normal("y", mu=mu, sigma=0.05 * float(np.std(_seen[n])), observed=y)
    return m
"""
        result = run_full_cascade(
            cache_cheat,
            visible,
            holdout_data=hidden,
            gold_elpd=-2.0,
            baseline=-10.0,
            target_name="y",
        )
        assert result["training_log_reward"] == -20.0
        assert not (
            result["reward_source"] == "validation.pymc_elpd"
            and result["diagnostics"].get("elpd_status") == "scored"
            and result["training_log_reward"] > -20.0
        )


class TestModuleAttributeIsolation:
    """A module attribute cannot expose the true held-out target to submitted code.

    A submission can stash a value on the shared ``np`` module object and replay it on a later
    build. A per-call exec namespace alone does not isolate that module object. The scorer
    therefore builds the model on censored data and transplants the true target directly into
    rvs_to_values, without executing the submission on true data. The existing self-check
    rejects a likelihood pinned to the stashed value: transplanting the observed RV has no
    effect on that likelihood. This test exercises that structural protection.
    """

    def test_module_attribute_stash_is_floored_not_rewarded(self, reg_split):
        visible, hidden = reg_split
        stash_cheat = """\
def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    n = len(y)
    try:
        store = np._synthstash
    except AttributeError:
        store = {}
        np._synthstash = store
    if n not in store:
        store[n] = y.copy()
    with pm.Model() as m:
        xd = pm.Data("x", x)
        b = pm.Normal("b", 0.0, 1.0)
        mu = store[n] + 0.0 * xd * b
        pm.Normal("y", mu=mu, sigma=0.05 * float(np.std(store[n])), observed=y)
    return m
"""
        result = run_full_cascade(
            stash_cheat,
            visible,
            holdout_data=hidden,
            gold_elpd=-2.0,
            baseline=-10.0,
            target_name="y",
        )
        assert result["training_log_reward"] == -20.0
        assert not (
            result["reward_source"] == "validation.pymc_elpd"
            and result["diagnostics"].get("elpd_status") == "scored"
            and result["training_log_reward"] > -20.0
        )


class TestGoldBeatsBaseline:
    def test_genuine_fit_strictly_beats_intercept_only_with_margin(self, reg_split):
        visible, hidden = reg_split
        baseline_result = check_l6_holdout_elpd(
            INTERCEPT_ONLY, visible, hidden, target_name="y", draws=_DRAWS, tune=_TUNE
        )
        baseline = baseline_result["holdout_elpd"]
        gold_result = check_l6_holdout_elpd(
            GENUINE_FIT,
            visible,
            hidden,
            target_name="y",
            draws=_DRAWS,
            tune=_TUNE,
        )
        assert gold_result["outcome"] == "scored"
        gold_elpd = gold_result["holdout_elpd"]

        log_reward_intercept, _ = _log_reward(
            visible, hidden, INTERCEPT_ONLY, baseline=baseline, gold_elpd=gold_elpd
        )
        log_reward_gold, _ = _log_reward(
            visible, hidden, GENUINE_FIT, baseline=baseline, gold_elpd=gold_elpd
        )
        assert log_reward_gold > log_reward_intercept + 0.3
        # Confirm deterministic scoring under seed 0 within the same process.
        log_reward_gold_2, _ = _log_reward(
            visible, hidden, GENUINE_FIT, baseline=baseline, gold_elpd=gold_elpd
        )
        assert log_reward_gold == pytest.approx(log_reward_gold_2)


class TestVarianceCollapseBlocked:
    def test_sigma_collapse_floors_at_density_gate(self, near_noiseless_split):
        # A free-sigma model fit on near-noiseless data legitimately
        # concentrates its posterior sigma near zero; a held-out point that
        # sits exactly on the trend then gets a spuriously enormous held-out
        # log density; the density gate must floor this, not reward it.
        visible, hidden = near_noiseless_split
        result = run_full_cascade(
            GENUINE_FIT,
            visible,
            holdout_data=hidden,
            gold_elpd=None,
            baseline=-100.0,
            target_name="y",
        )
        assert result["training_log_reward"] == -20.0
        assert result["diagnostics"]["elpd_status"] == "density_gate_failed"
        assert result["diagnostics"]["failed_density_gate"] == "min_predictive_scale"
        assert result["reward_source"] == "validation.pymc_elpd"
        assert result["evidence_kind"] == "verifier"
        RewardArtifact.model_validate(result)


class TestStructuralGate:
    def test_potential_hack_never_reaches_l6(self, reg_split):
        visible, hidden = reg_split
        result = run_full_cascade(
            POTENTIAL_HACK,
            visible,
            holdout_data=hidden,
            gold_elpd=None,
            baseline=-100.0,
            target_name="y",
        )
        assert result["reward_source"] == "validation.compilation"
        assert result["training_log_reward"] == -20.0
        assert "elpd_status" not in result["diagnostics"]

    def test_unread_data_reaches_l6(self, reg_split):
        visible, hidden = reg_split
        result = run_full_cascade(
            DEAD_DATA_MODEL,
            visible,
            holdout_data=hidden,
            gold_elpd=None,
            baseline=-100.0,
            target_name="y",
        )
        assert result["reward_source"] == "validation.pymc_elpd"
        assert result["diagnostics"]["elpd_status"] == "scored"
        assert math.isfinite(result["metrics"]["holdout_elpd"])


class TestExplicitFailureOutcomes:
    def test_missing_holdout_is_a_fault_not_a_silent_pass(self, reg_split):
        visible, _hidden = reg_split
        result = run_full_cascade(
            GENUINE_FIT, visible, holdout_data=None, gold_elpd=None, baseline=None, target_name="y"
        )
        assert result["reward_source"] == "heuristic.pymc_elpd_error"
        assert result["diagnostics"]["elpd_status"] == "missing_holdout"
        assert result["training_log_reward"] == -20.0
        assert result["reward"] == 0.0

    def test_unsamplable_model_is_a_fault(self, reg_split):
        # UNSAMPLABLE_MODEL's zero-weighted x usage trips L5's data-ignoring
        # heuristic (both point_logps come back -inf regardless of
        # perturbation), so it never reaches L6 through the full cascade --
        # exercise check_l6_holdout_elpd directly instead, isolating the L6
        # sampling-failure branch itself (the run_full_cascade wiring from an
        # L6 fault outcome to reward_source=heuristic.pymc_elpd_error is
        # already proven end-to-end by
        # test_group_holdout_shape_mismatch_is_a_fault_never_a_crash_or_nan
        # below).
        visible, hidden = reg_split
        result = check_l6_holdout_elpd(
            UNSAMPLABLE_MODEL,
            visible,
            hidden,
            target_name="y",
            draws=_DRAWS,
            tune=_TUNE,
        )
        assert result["outcome"] == "sampling_failed"
        assert "Bad initial energy" in result["error"] or "SamplingError" in result["error"]

    def test_group_holdout_shape_mismatch_is_a_fault_never_a_crash_or_nan(self):
        """group_holdout can present held-out rows with group levels unseen at
        fit time. A model that indexes groups positionally (shape=n_groups,
        no coords/dims) hits a genuine PyTensor shape mismatch when scoring
        such holdout rows. This must route through the model-fault path,
        never crash the verifier process and never leak a NaN-poisoned
        score into the RewardArtifact.
        """
        rng = np.random.default_rng(0)
        visible_groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
        visible = {"n": 12, "group": visible_groups, "y": rng.normal(size=12)}
        hidden_groups = np.array(["d", "e", "f", "g", "h"])
        hidden = {"n": 5, "group": hidden_groups, "y": rng.normal(size=5)}

        result = run_full_cascade(
            GROUP_HOLDOUT_SHAPE_MISMATCH_MODEL,
            visible,
            holdout_data=hidden,
            gold_elpd=-5.0,
            baseline=-10.0,
            target_name="y",
        )
        assert result["reward_source"] == "heuristic.pymc_elpd_error"
        assert result["evidence_kind"] == "heuristic"
        assert result["diagnostics"]["elpd_status"] == "holdout_failed"
        assert result["training_log_reward"] == -20.0
        assert math.isfinite(result["training_log_reward"])
        assert exit_code_for_full_cascade(result) == 1
        assert exit_code_for_full_cascade(result) != L6_TIMEOUT_EXIT_CODE
        RewardArtifact.model_validate(result)

    def test_timeout_is_a_distinct_infra_fault_exit_code(self, monkeypatch):
        import gc

        import ppl_synth_data.harbor.validator_template as validator_template

        # Exercise the real one-second deadline during NUTS inference.
        # Pause cyclic garbage collection to keep the signal exception in
        # the inference context rather than a profiling callback.
        rng = np.random.default_rng(0)
        n_visible, n_hidden = 3000, 50
        x_visible = rng.normal(size=n_visible)
        x_hidden = rng.normal(size=n_hidden)
        theta_true, beta_true = 1.5, 2.0
        y_visible = theta_true + beta_true * x_visible + rng.normal(scale=0.3, size=n_visible)
        y_hidden = theta_true + beta_true * x_hidden + rng.normal(scale=0.3, size=n_hidden)
        visible = {"n": n_visible, "x": x_visible, "y": y_visible}
        hidden = {"n": n_hidden, "x": x_hidden, "y": y_hidden}

        monkeypatch.setattr(validator_template, "L6_HOLDOUT_ELPD_TIMEOUT_S", 1)
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            result = run_full_cascade(
                GENUINE_FIT,
                visible,
                holdout_data=hidden,
                gold_elpd=None,
                baseline=-100.0,
                target_name="y",
            )
        finally:
            if gc_was_enabled:
                gc.enable()
        assert result["reward_source"] == "heuristic.pymc_elpd_error"
        assert result["diagnostics"]["elpd_status"] == "timeout"
        assert result["training_log_reward"] == -20.0
        assert exit_code_for_full_cascade(result) == L6_TIMEOUT_EXIT_CODE


class TestContractValidity:
    @pytest.mark.parametrize(
        "code,baseline,gold_elpd",
        [
            (GENUINE_FIT, -100.0, None),
            (VARIANCE_COLLAPSE, -100.0, None),
            (POTENTIAL_HACK, -100.0, None),
            (UNSAMPLABLE_MODEL, -100.0, None),
        ],
    )
    def test_every_outcome_validates_against_pinned_schema(
        self, reg_split, code, baseline, gold_elpd
    ):
        visible, hidden = reg_split
        result = run_full_cascade(
            code,
            visible,
            holdout_data=hidden,
            gold_elpd=gold_elpd,
            baseline=baseline,
            target_name="y",
        )
        RewardArtifact.model_validate(result)

    def test_missing_holdout_artifact_also_validates(self, reg_split):
        visible, _hidden = reg_split
        result = run_full_cascade(
            GENUINE_FIT, visible, holdout_data=None, gold_elpd=None, baseline=None, target_name="y"
        )
        RewardArtifact.model_validate(result)


class TestDeterminism:
    def test_same_submission_seed_0_is_deterministic_across_calls(self, reg_split):
        visible, hidden = reg_split
        baseline_result = check_l6_holdout_elpd(
            INTERCEPT_ONLY, visible, hidden, target_name="y", draws=_DRAWS, tune=_TUNE
        )
        baseline = baseline_result["holdout_elpd"]

        result_a = run_full_cascade(
            GENUINE_FIT,
            visible,
            holdout_data=hidden,
            gold_elpd=None,
            baseline=baseline,
            target_name="y",
        )
        result_b = run_full_cascade(
            GENUINE_FIT,
            visible,
            holdout_data=hidden,
            gold_elpd=None,
            baseline=baseline,
            target_name="y",
        )
        assert result_a["training_log_reward"] == pytest.approx(result_b["training_log_reward"])
        assert result_a["metrics"]["avg_holdout_logp"] == pytest.approx(
            result_b["metrics"]["avg_holdout_logp"]
        )

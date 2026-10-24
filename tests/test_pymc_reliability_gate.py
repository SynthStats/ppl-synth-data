"""Tests for the sampler-reliability gate on the PyMC semantic scorer.

Two layers. The pure layer pins the decision rule (RefineStat's "at least five
of seven diagnostics pass" (arXiv:2509.01082)) against diagnostics handed in
directly, so the expected verdicts come from that rule and not from re-running
the scorer. The integration layer runs real PyMC fits: a well-behaved linear
regression, which must clear the gate, and the centred parameterisation of the
eight-schools model, the textbook funnel geometry that NUTS cannot explore
without reparameterisation, which must not.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppl_synth_data.evidence import SemanticStatus
from ppl_synth_data.validation.pymc_semantics import (
    MIN_PASSING_RELIABILITY_CHECKS,
    RELIABILITY_CHECK_NAMES,
    SamplerDiagnostics,
    evaluate_sampler_reliability,
    psis_tail_length,
    score_pymc_semantics,
)

# Diagnostics from a fit that is healthy on every axis. Each value clears the
# published threshold the gate cites, with room to spare.
_HEALTHY = {
    "chains": 2,
    "draws_per_chain": 500,
    "split_r_hat_max": 1.002,
    "bfmi_min": 0.95,
    "ess_bulk_min": 640.0,
    "ess_tail_min": 580.0,
    "divergences": 0,
    "pareto_k_max": 0.31,
    "elpd_loo": -41.2,
}


def _diagnostics(**overrides: object) -> SamplerDiagnostics:
    return SamplerDiagnostics(**{**_HEALTHY, **overrides})  # type: ignore[arg-type]


@pytest.mark.narrow_unit_not_acceptance
def test_healthy_fit_passes_every_check() -> None:
    verdict = evaluate_sampler_reliability(_diagnostics())

    assert verdict.passed is True
    assert verdict.passed_count == len(RELIABILITY_CHECK_NAMES)
    assert verdict.failing_names == ()
    assert tuple(check.name for check in verdict.checks) == RELIABILITY_CHECK_NAMES


@pytest.mark.narrow_unit_not_acceptance
def test_five_of_seven_is_the_boundary() -> None:
    # RefineStat counts a program when at least five diagnostics pass. Two
    # failures leave five, which is reliable; three leave four, which is not.
    two_failures = _diagnostics(split_r_hat_max=1.4, divergences=17)
    three_failures = _diagnostics(split_r_hat_max=1.4, divergences=17, bfmi_min=0.05)

    assert MIN_PASSING_RELIABILITY_CHECKS == 5
    assert evaluate_sampler_reliability(two_failures).passed_count == 5
    assert evaluate_sampler_reliability(two_failures).passed is True
    assert evaluate_sampler_reliability(three_failures).passed_count == 4
    assert evaluate_sampler_reliability(three_failures).passed is False


@pytest.mark.narrow_unit_not_acceptance
@pytest.mark.parametrize(
    ("overrides", "expected_failure"),
    [
        ({"split_r_hat_max": 1.02}, "split_r_hat"),
        ({"bfmi_min": 0.29}, "bfmi"),
        ({"ess_bulk_min": 199.0}, "ess_bulk"),
        ({"ess_tail_min": 199.0}, "ess_tail"),
        ({"divergences": 1}, "no_divergences"),
        ({"pareto_k_max": 0.71}, "pareto_k"),
        ({"elpd_loo": float("-inf")}, "finite_elpd"),
    ],
)
def test_each_threshold_fails_its_own_check(
    overrides: dict[str, object], expected_failure: str
) -> None:
    verdict = evaluate_sampler_reliability(_diagnostics(**overrides))

    assert verdict.failing_names == (expected_failure,)


@pytest.mark.narrow_unit_not_acceptance
def test_ess_threshold_scales_with_chain_count() -> None:
    # The published criterion is per chain, so four chains need four times the
    # effective sample size that one chain needs.
    one_chain = _diagnostics(chains=1, ess_bulk_min=150.0, ess_tail_min=150.0)
    four_chains = _diagnostics(chains=4, ess_bulk_min=150.0, ess_tail_min=150.0)

    assert evaluate_sampler_reliability(one_chain).failing_names == ()
    assert evaluate_sampler_reliability(four_chains).failing_names == (
        "ess_bulk",
        "ess_tail",
    )


@pytest.mark.narrow_unit_not_acceptance
def test_uncomputable_diagnostic_fails_its_check_and_says_why() -> None:
    # An absent diagnostic is never evidence of reliability.
    verdict = evaluate_sampler_reliability(
        _diagnostics(
            split_r_hat_max=None,
            errors={"split_r_hat": "KeyError: posterior"},
        )
    )
    check = next(c for c in verdict.checks if c.name == "split_r_hat")

    assert check.passed is False
    assert "unavailable" in check.detail
    assert "KeyError: posterior" in check.detail


@pytest.mark.narrow_unit_not_acceptance
def test_non_finite_pareto_k_below_21_draws_is_named_as_a_draw_budget_artifact() -> None:
    # ArviZ reports an infinite Pareto-k whenever the PSIS tail holds four or
    # fewer draws, which is every model at 20 posterior draws or fewer, correct
    # or degenerate. The check still fails: there is no evidence either way.
    # The detail distinguishes that from a genuine tail failure.
    starved = evaluate_sampler_reliability(
        _diagnostics(chains=1, draws_per_chain=15, pareto_k_max=float("inf"))
    )
    well_fed = evaluate_sampler_reliability(
        _diagnostics(chains=2, draws_per_chain=500, pareto_k_max=float("inf"))
    )
    starved_detail = next(c for c in starved.checks if c.name == "pareto_k").detail
    well_fed_detail = next(c for c in well_fed.checks if c.name == "pareto_k").detail

    assert "pareto_k" in starved.failing_names
    assert "carries no information about the model" in starved_detail
    assert "carries no information about the model" not in well_fed_detail


@pytest.mark.narrow_unit_not_acceptance
def test_psis_tail_length_crosses_arviz_fit_floor_at_21_draws() -> None:
    # arviz.stats.stats._psislw: "if tail_len <= 4: k = np.inf".
    assert psis_tail_length(15) == 3
    assert psis_tail_length(20) == 4
    assert psis_tail_length(21) == 5
    assert psis_tail_length(1000) == 95


@pytest.mark.slow
@pytest.mark.real_boundary
def test_real_well_behaved_fit_clears_the_gate() -> None:
    code = """
def model(data):
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    with pm.Model() as m:
        alpha = pm.Normal("alpha", 0.0, 5.0)
        beta = pm.Normal("beta", 0.0, 5.0)
        sigma = pm.HalfNormal("sigma", 5.0)
        pm.Normal("obs", mu=alpha + beta * x, sigma=sigma, observed=y)
    return m
"""
    rng = np.random.default_rng(7)
    x_vis = rng.normal(size=40)
    x_hid = rng.normal(size=10)
    visible_data = {"x": x_vis, "y": 1.3 + 2.1 * x_vis + rng.normal(0.0, 0.7, x_vis.size)}
    hidden_data = {"x": x_hid, "y": 1.3 + 2.1 * x_hid + rng.normal(0.0, 0.7, x_hid.size)}

    result = score_pymc_semantics(
        code,
        visible_data=visible_data,
        hidden_data=hidden_data,
        draws=500,
        tune=500,
        timeout_sec=300,
    )

    assert result.ok is True
    reliability = result.details["reliability"]
    diagnostics = result.details["sampler_diagnostics"]
    assert diagnostics["errors"] == {}, "every diagnostic must be measurable on a real fit"
    # Split R-hat is a between-chain statistic: a finite value here is what
    # proves the fit ran more than one chain.
    assert diagnostics["chains"] >= 2
    assert np.isfinite(diagnostics["split_r_hat_max"])
    assert reliability["failing"] == []
    assert reliability["passed"] is True
    assert result.status == SemanticStatus.pass_
    # The score is reported as measured, not withheld or adjusted by the gate.
    assert result.metric_name == "avg_holdout_logp"
    assert np.isfinite(result.metric_value)


@pytest.mark.slow
@pytest.mark.real_boundary
def test_real_funnel_geometry_fails_the_gate_on_a_sampling_pathology() -> None:
    # Eight schools in the centred parameterisation: theta ~ Normal(mu, tau)
    # with tau weakly identified gives the funnel NUTS cannot explore, so the
    # fit shows divergent transitions and inflated R-hat however good its
    # predictive score looks. Data is the standard Rubin (1981) set.
    code = """
def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0.0, sigma=5.0)
        tau = pm.HalfCauchy("tau", beta=5.0)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("obs", mu=theta, sigma=np.asarray(data["sigma"]),
                  observed=np.asarray(data["y"]))
    return m
"""
    visible_data = {
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }

    result = score_pymc_semantics(
        code,
        visible_data=visible_data,
        hidden_data=None,
        draws=200,
        tune=200,
        timeout_sec=300,
    )

    assert result.ok is True, result.message
    reliability = result.details["reliability"]
    diagnostics = result.details["sampler_diagnostics"]
    assert diagnostics["errors"] == {}, "the verdict must rest on measured diagnostics"
    assert reliability["passed"] is False
    # Fails because the sampler misbehaved, not because a diagnostic went
    # missing: at least one of the sampling-quality checks is among the
    # failures.
    assert {"no_divergences", "split_r_hat", "ess_bulk", "ess_tail"} & set(reliability["failing"])
    assert result.status == SemanticStatus.flag
    assert "unreliable sampling" in result.message


@pytest.mark.slow
@pytest.mark.real_boundary
def test_all_seven_diagnostics_are_recorded_without_hidden_data() -> None:
    # Recording is unconditional: a run with no held-out slice still carries
    # every diagnostic, because diagnostics we record but do not act on are
    # still evidence.
    code = """
def model(data):
    y = np.asarray(data["y"])
    with pm.Model() as m:
        mu = pm.Normal("mu", 0.0, 10.0)
        sigma = pm.HalfNormal("sigma", 10.0)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
"""
    rng = np.random.default_rng(3)
    visible_data = {"y": rng.normal(5.0, 2.0, 30)}

    result = score_pymc_semantics(
        code,
        visible_data=visible_data,
        hidden_data=None,
        draws=200,
        tune=200,
        timeout_sec=300,
    )

    assert result.ok is True
    diagnostics = result.details["sampler_diagnostics"]
    for key in (
        "split_r_hat_max",
        "bfmi_min",
        "ess_bulk_min",
        "ess_tail_min",
        "divergences",
        "pareto_k_max",
        "elpd_loo",
    ):
        assert diagnostics[key] is not None, key
    recorded = tuple(check["name"] for check in result.details["reliability"]["checks"])
    assert recorded == RELIABILITY_CHECK_NAMES
    assert result.metric_name == "train_elpd_loo"

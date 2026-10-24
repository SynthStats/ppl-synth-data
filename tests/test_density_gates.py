"""Tests for the density-degeneracy gates (anti-reward-hack)."""

from __future__ import annotations

from ppl_synth_data.validation.density_gates import (
    MAX_EXTREME_PROB_FRACTION,
    MIN_PP_SPREAD_FRACTION,
    MIN_PRED_SCALE_FRACTION,
    DensityDiagnostics,
    evaluate_density_gates,
)


def _names(gates) -> set[str]:
    return {g.name for g in gates}


def _failed(gates) -> set[str]:
    return {g.name for g in gates if not g.passed}


def test_healthy_continuous_model_passes_all():
    d = DensityDiagnostics(data_std=2.0, min_pred_std=1.5, pp_spread=1.8)
    gates = evaluate_density_gates(d)
    assert _names(gates) == {"min_predictive_scale", "predictive_not_collapsed"}
    assert _failed(gates) == set()


def test_near_zero_sigma_hack_is_caught():
    # Predictive scale 0.005% of data spread -> sigma->0 / too-perfect hack.
    d = DensityDiagnostics(data_std=2.0, min_pred_std=1e-4, pp_spread=1.8)
    gates = evaluate_density_gates(d)
    assert "min_predictive_scale" in _failed(gates)


def test_min_pred_scale_threshold_boundary():
    data_std = 2.0
    thresh = MIN_PRED_SCALE_FRACTION * data_std
    passing = DensityDiagnostics(data_std=data_std, min_pred_std=thresh, pp_spread=1.0)
    failing = DensityDiagnostics(data_std=data_std, min_pred_std=thresh * 0.5, pp_spread=1.0)
    assert "min_predictive_scale" not in _failed(evaluate_density_gates(passing))
    assert "min_predictive_scale" in _failed(evaluate_density_gates(failing))


def test_dirac_posterior_predictive_is_caught():
    # Posterior-predictive spread collapsed to a point mass.
    d = DensityDiagnostics(data_std=2.0, min_pred_std=1.5, pp_spread=1e-5)
    gates = evaluate_density_gates(d)
    assert "predictive_not_collapsed" in _failed(gates)


def test_degenerate_binary_probabilities_are_caught():
    # Almost every predicted probability sits near 0 or 1 on a noisy binary task.
    d = DensityDiagnostics(data_std=0.5, pred_prob_extreme_frac=0.995)
    gates = evaluate_density_gates(d)
    assert "probability_non_degenerate" in _failed(gates)


def test_reasonable_binary_probabilities_pass():
    d = DensityDiagnostics(data_std=0.5, pred_prob_extreme_frac=MAX_EXTREME_PROB_FRACTION - 0.1)
    gates = evaluate_density_gates(d)
    assert _failed(gates) == set()


def test_gates_skipped_when_inputs_unavailable():
    # No continuous predictive stats, no binary probs -> no gates emitted (the
    # scorer simply did not supply them); nothing to fail on.
    d = DensityDiagnostics(data_std=2.0)
    assert evaluate_density_gates(d) == ()


def test_zero_data_std_skips_scale_gates():
    # Degenerate reference (constant data) -> scale-relative gates are skipped
    # rather than dividing by zero; probability gate still applies.
    d = DensityDiagnostics(
        data_std=0.0, min_pred_std=1e-9, pp_spread=1e-9, pred_prob_extreme_frac=0.5
    )
    gates = evaluate_density_gates(d)
    assert _names(gates) == {"probability_non_degenerate"}


def test_pp_spread_uses_its_own_fraction_constant():
    data_std = 4.0
    thresh = MIN_PP_SPREAD_FRACTION * data_std
    failing = DensityDiagnostics(data_std=data_std, pp_spread=thresh * 0.5)
    assert "predictive_not_collapsed" in _failed(evaluate_density_gates(failing))

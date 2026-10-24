"""Adapter: run one PyMC candidate program through gates + fitness scoring.

Composes the existing, subprocess-isolated scorer (``score_pymc_semantics``),
static safety checks (``static_code``), and the density-degeneracy gates
(``density_gates``) into a ``ScoredCandidate`` for the selection decision core.

Safety: untrusted generated code is executed ONLY inside
``score_pymc_semantics``'s resource-limited subprocess (build + sample). The
integrity gates here are STATIC (AST-level, no execution), so nothing dangerous
runs in-process.

Given a program and visible/hidden data, this entry point returns the integrity
verdict and per-held-out-row fitness vector for selection and reward scoring.

Integrity checks here inspect the AST. Harbor's model-execution checks for
fabricated data and length perturbation are outside this static integrity pass.
"""

from __future__ import annotations

import ast
from typing import Any

from synthstats_schemas import CandidateKey, GateOutcome

from ppl_synth_data.validation.density_gates import (
    DensityDiagnostics,
    evaluate_density_gates,
)
from ppl_synth_data.validation.pymc_semantics import score_pymc_semantics
from ppl_synth_data.validation.selection import GateReport, HoldoutScore, ScoredCandidate
from ppl_synth_data.validation.static_code import StaticCodeError, validate_python_safety

# Unnormalized / custom likelihoods bypass the normalization the held-out logp
# assumes, so a program using them can post an arbitrary score; the Harbor
# strict profile forbids them and so do we.
_FORBIDDEN_LIKELIHOOD_CALLS: tuple[str, ...] = ("Potential", "CustomDist", "DensityDist")


def _ast_safety_gate(code: str) -> GateOutcome:
    try:
        validate_python_safety(code)
    except StaticCodeError as exc:
        return GateOutcome(name="ast_safety", passed=False, detail=str(exc))
    return GateOutcome(name="ast_safety", passed=True, detail="ok")


def _no_unnormalized_likelihood_gate(code: str) -> GateOutcome:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return GateOutcome(name="no_unnormalized_likelihood", passed=False, detail=str(exc))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else (func.id if isinstance(func, ast.Name) else None)
            )
            if name in _FORBIDDEN_LIKELIHOOD_CALLS:
                found.add(name)
    if found:
        return GateOutcome(
            name="no_unnormalized_likelihood",
            passed=False,
            detail=f"forbidden likelihood calls: {sorted(found)}",
        )
    return GateOutcome(name="no_unnormalized_likelihood", passed=True, detail="ok")


def _density_diagnostics(details: dict[str, Any]) -> DensityDiagnostics:
    return DensityDiagnostics(
        data_std=float(details.get("data_std", 0.0)),
        min_pred_std=details.get("implied_min_pred_std"),
        pp_spread=None,
        pred_prob_extreme_frac=None,
    )


def gate_and_score_pymc(
    code: str,
    *,
    visible_data: dict[str, Any],
    hidden_data: dict[str, Any] | None,
    candidate_key: CandidateKey,
    gate_draws: int = 10,
    gate_tune: int = 10,
    fitness_draws: int = 500,
    fitness_tune: int = 500,
    timeout_sec: int = 120,
) -> ScoredCandidate:
    """Gate one candidate and, if eligible, score its held-out fitness.

    First a cheap ``gate_draws`` crash gate plus static integrity checks reject
    crashers and exploiters before any expensive run. Only survivors get the
    high-draw ``fitness_draws`` pass, and the density-degeneracy gate is computed
    from THAT pass: at ``gate_draws`` NUTS has not converged, so the degeneracy
    signal (a near-zero implied predictive scale) is unreliable: it only
    manifests once the model has actually been fit.
    """
    cheap = score_pymc_semantics(
        code,
        visible_data=visible_data,
        hidden_data=hidden_data,
        draws=gate_draws,
        tune=gate_tune,
        timeout_sec=timeout_sec,
    )
    cheap_gate = GateOutcome(name="crash_gate", passed=bool(cheap.ok), detail=cheap.message)
    integrity_gates = (
        _ast_safety_gate(code),
        _no_unnormalized_likelihood_gate(code),
    )
    prelim_ok = cheap_gate.passed and all(g.passed for g in integrity_gates)

    density_gates: tuple[GateOutcome, ...] = ()
    fit_pointwise: tuple[float, ...] | None = None
    if prelim_ok and hidden_data is not None:
        fit = score_pymc_semantics(
            code,
            visible_data=visible_data,
            hidden_data=hidden_data,
            draws=fitness_draws,
            tune=fitness_tune,
            timeout_sec=timeout_sec,
        )
        if fit.ok:
            density_gates = evaluate_density_gates(_density_diagnostics(fit.details))
            pointwise = fit.details.get("holdout_pointwise")
            if pointwise:
                fit_pointwise = tuple(float(v) for v in pointwise)
        else:
            # The high-draw pass itself failed (e.g. timed out): treat as a
            # crash so the candidate is not silently promoted.
            cheap_gate = GateOutcome(
                name="crash_gate", passed=False, detail=f"fitness pass failed: {fit.message}"
            )

    gate = GateReport(
        cheap_gate=cheap_gate,
        integrity_gates=integrity_gates,
        density_gates=density_gates,
    )

    score: HoldoutScore | None = None
    if gate.eligible and fit_pointwise is not None:
        score = HoldoutScore(pointwise=fit_pointwise, draws=fitness_draws)

    return ScoredCandidate(key=candidate_key, gate=gate, score=score)

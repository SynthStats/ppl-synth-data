"""Density-degeneracy gates for the selection verifier.

Structural integrity gates (Harbor L0-L5) catch programs that misuse the data (``pm.Potential``,
data-dependent priors, fabricated observed arrays). They do NOT catch a structurally-honest
model that still GAMES the held-out score by being degenerate: a near-zero residual scale that
concentrates probability mass on the 2-3 held-out points scores very high ``avg_holdout_logp``
yet models nothing. Because selection takes the argmax of held-out logp, it actively seeks such
programs out, addressing a dangerous failure mode for ELPD-guided selection (see Vehtari's
CV FAQ on small-sample selection overfitting).

These gates are PURE: they interpret a ``DensityDiagnostics`` summary produced by the scorer
(one PyMC subprocess pass) into pass/fail ``GateOutcome``s. They run INSIDE the eligibility
check, before any candidate is ranked, so a degenerate program is rejected rather than promoted
to SFT ground truth.

Thresholds are deliberately hard-coded domain constants, not parameters: they encode a single
judgement ("a model whose predictive scale is below 1% of the data's own spread is degenerate"),
and exposing them as knobs would invite silent reward-hacking by tuning them.
"""

from __future__ import annotations

from dataclasses import dataclass

from synthstats_schemas import GateOutcome

# A model's predictive noise scale must be at least this fraction of the data's
# own spread; below it the model is a near-Dirac that fits the few held-out
# points by collapsing variance (the sigma -> 0 / "too perfect" hack).
MIN_PRED_SCALE_FRACTION: float = 0.01

# The posterior-predictive distribution must itself have spread at least this
# fraction of the data spread; below it the predictive is effectively a point
# mass (overconfident / Dirac posterior predictive).
MIN_PP_SPREAD_FRACTION: float = 0.01

# For binary / bounded-probability outcomes, at most this fraction of predicted
# probabilities may sit in the extreme tails (near 0 or near 1). A model that
# predicts ~0/1 for almost every point on a genuinely noisy binary task is
# degenerate (it has learned a near-deterministic rule that games logp).
MAX_EXTREME_PROB_FRACTION: float = 0.98


@dataclass(frozen=True)
class DensityDiagnostics:
    """Degeneracy summary produced by the scorer for one candidate program.

    Fields are Optional because they depend on the likelihood family: continuous
    outcomes carry ``min_pred_std``; binary/bounded outcomes carry
    ``pred_prob_extreme_frac``. A gate is simply skipped when its input is
    unavailable.
    """

    data_std: float  # std of the observed target on visible data (the scale reference)
    min_pred_std: float | None = None  # min per-point posterior-predictive std (continuous)
    pp_spread: float | None = None  # std of posterior-predictive draws (predictive spread)
    pred_prob_extreme_frac: float | None = None  # fraction of predicted probs near 0 or 1


def evaluate_density_gates(d: DensityDiagnostics) -> tuple[GateOutcome, ...]:
    """Interpret density diagnostics into pass/fail gates.

    Returns one GateOutcome per applicable check. A candidate is density-eligible
    iff every returned gate passed (the selection decision core ANDs them).
    """
    gates: list[GateOutcome] = []

    if d.min_pred_std is not None and d.data_std > 0.0:
        thresh = MIN_PRED_SCALE_FRACTION * d.data_std
        passed = d.min_pred_std >= thresh
        gates.append(
            GateOutcome(
                name="min_predictive_scale",
                passed=passed,
                detail=(
                    f"min_pred_std={d.min_pred_std:.4g} "
                    f"threshold={thresh:.4g} (>= {MIN_PRED_SCALE_FRACTION:g} * data_std)"
                ),
            )
        )

    if d.pp_spread is not None and d.data_std > 0.0:
        thresh = MIN_PP_SPREAD_FRACTION * d.data_std
        passed = d.pp_spread >= thresh
        gates.append(
            GateOutcome(
                name="predictive_not_collapsed",
                passed=passed,
                detail=(
                    f"pp_spread={d.pp_spread:.4g} "
                    f"threshold={thresh:.4g} (>= {MIN_PP_SPREAD_FRACTION:g} * data_std)"
                ),
            )
        )

    if d.pred_prob_extreme_frac is not None:
        passed = d.pred_prob_extreme_frac <= MAX_EXTREME_PROB_FRACTION
        gates.append(
            GateOutcome(
                name="probability_non_degenerate",
                passed=passed,
                detail=(
                    f"extreme_prob_frac={d.pred_prob_extreme_frac:.4g} "
                    f"max={MAX_EXTREME_PROB_FRACTION:g}"
                ),
            )
        )

    return tuple(gates)

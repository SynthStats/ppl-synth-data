"""Statistically sound candidate selection for multi-family best-of-N generation.

This is the decision core of the selection verifier. It is deliberately PURE and INJECTABLE: it
takes already-computed gate outcomes and held-out scores and decides each candidate's
disposition. The heavy PyMC / LazyPPL scoring (``score_pymc_semantics``, ``run_cascade``) plugs
in around it via thin adapters, so the same logic serves three call sites:

1. offline SFT-data selection (best-of-N -> one kept row);
2. on-policy distillation grading of a student's own rollouts during training;
3. GFlowNet TB/SubTB reward.

The selection rules are:

- eligibility before ranking: a candidate that failed any gate never enters the
  fitness ranking (and can never be ``selected``);
- argmax only when the held-out split is large enough to rank
  (``holdout_n >= MIN_HOLDOUT_FOR_SELECTION``); below that, keep eligible
  candidates ``gate_only`` rather than ranking on 2-3 points of noise;
- the winner must beat the runner-up by a bootstrap CI that excludes zero
  (paired resampling over held-out rows), not a Monte-Carlo standard error;
- if no eligible candidate clearly wins, do not force a single pick; keep the
  eligible candidates ``gate_only`` (the "average/stack, do not select one"
  result for scarce data).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng
from synthstats_schemas import (
    MIN_HOLDOUT_FOR_SELECTION,
    CandidateKey,
    FitnessRecord,
    GateOutcome,
    SelectionMode,
    SelectionRecord,
    SelectionRuntime,
)

# Held-out fitness metric name (per-observation average log predictive density).
HOLDOUT_METRIC: str = "avg_holdout_logp"

# Bootstrap settings for the ranking margin.
_DEFAULT_N_BOOT: int = 2000
_DEFAULT_ALPHA: float = 0.05

# A winner must beat the runner-up by BOTH a bootstrap CI excluding zero AND this
# minimum absolute gap in per-observation nats. At small held-out n the bootstrap
# CI alone is a weak signal (Vehtari's CV-FAQ: the elpd-difference SE is only
# reliable when |diff| > 4 and n > 100), so this magnitude floor is the effective
# guard below ~100 observations. It is a deliberate, conservative heuristic: a gap
# under ~0.5 nats/obs is not a substantively meaningful difference between two
# probabilistic programs, so we decline to pick one and keep both gate_only.
MIN_WINNER_GAP_NATS: float = 0.5


@dataclass(frozen=True)
class GateReport:
    """Outcome of running one candidate through every gate layer."""

    cheap_gate: GateOutcome
    integrity_gates: tuple[GateOutcome, ...]
    density_gates: tuple[GateOutcome, ...]

    @property
    def eligible(self) -> bool:
        return (
            self.cheap_gate.passed
            and all(g.passed for g in self.integrity_gates)
            and all(g.passed for g in self.density_gates)
        )


@dataclass(frozen=True)
class HoldoutScore:
    """Per-held-out-row average log predictive density, plus the draw budget.

    ``pointwise`` is one value per held-out observation (already averaged over
    posterior draws via log-mean-exp). Keeping the per-row vector lets us
    bootstrap the ranking margin over the held-out rows, the dominant source
    of selection noise at small n, which a Monte-Carlo SE over draws ignores.
    """

    pointwise: tuple[float, ...]
    draws: int

    @property
    def holdout_n(self) -> int:
        return len(self.pointwise)

    @property
    def point_estimate(self) -> float:
        return float(np.mean(self.pointwise))


@dataclass(frozen=True)
class ScoredCandidate:
    key: CandidateKey
    gate: GateReport
    score: HoldoutScore | None  # None whenever the gate rejected the candidate


@dataclass(frozen=True)
class TaskSelection:
    """Result of selecting among one task's candidate population."""

    records: tuple[SelectionRecord, ...]  # one per candidate (audit + kept rows)
    winner_key: CandidateKey | None  # the selected/kept-best candidate, if any


def _record(
    cand: ScoredCandidate,
    *,
    runtime: SelectionRuntime,
    mode: SelectionMode,
    source_plan_signature: str,
    fitness: FitnessRecord | None,
    winner_reason: str,
) -> SelectionRecord:
    return SelectionRecord(
        runtime=runtime,
        candidate_key=cand.key,
        selection_mode=mode,
        eligible=cand.gate.eligible,
        cheap_gate=cand.gate.cheap_gate,
        integrity_gates=cand.gate.integrity_gates,
        density_gates=cand.gate.density_gates,
        fitness=fitness,
        source_plan_signature=source_plan_signature,
        winner_reason=winner_reason,
    )


def _bootstrap_indices(n: int, n_boot: int, rng: Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def _own_ci(pointwise: np.ndarray, boot_idx: np.ndarray, alpha: float) -> tuple[float, float]:
    means = pointwise[boot_idx].mean(axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def _paired_margin_lower_bound(
    winner: np.ndarray, runner_up: np.ndarray, boot_idx: np.ndarray, alpha: float
) -> float:
    """Lower bound of the (1-alpha) CI on (winner - runner_up), paired over rows.

    Both candidates are scored on the SAME held-out rows, so we resample row
    indices once and apply them to both in a paired bootstrap. A lower bound > 0
    means the winner robustly beats the runner-up given split uncertainty.
    """
    if winner.shape != runner_up.shape:
        raise ValueError(
            "paired bootstrap requires equal-length held-out scores; "
            f"got winner {winner.shape} vs runner_up {runner_up.shape}"
        )
    diffs = (winner[boot_idx] - runner_up[boot_idx]).mean(axis=1)
    return float(np.quantile(diffs, alpha / 2.0))


def select_for_task(
    candidates: Sequence[ScoredCandidate],
    *,
    runtime: SelectionRuntime,
    source_plan_signature: str,
    rng: Generator | None = None,
    n_boot: int = _DEFAULT_N_BOOT,
    alpha: float = _DEFAULT_ALPHA,
) -> TaskSelection:
    """Decide the disposition of every candidate produced for one task.

    Returns one SelectionRecord per candidate. Ineligible candidates are
    ``rejected``. Eligible candidates are ranked only when their held-out split
    is adequate AND a robust winner exists; otherwise they are kept
    ``gate_only``. At most one candidate is ``selected``.
    """
    if rng is None:
        rng = default_rng(0)

    records: list[SelectionRecord] = []
    eligible = [c for c in candidates if c.gate.eligible]
    rejected = [c for c in candidates if not c.gate.eligible]

    for c in rejected:
        records.append(
            _record(
                c,
                runtime=runtime,
                mode=SelectionMode.rejected,
                source_plan_signature=source_plan_signature,
                fitness=None,
                winner_reason="failed an integrity or density gate",
            )
        )

    # Eligible candidates can only be ranked if they carry a held-out score of
    # adequate size. Anything below threshold (or unscored) is kept gate_only.
    rankable = [
        c
        for c in eligible
        if c.score is not None and c.score.holdout_n >= MIN_HOLDOUT_FOR_SELECTION
    ]
    unrankable = [c for c in eligible if c not in rankable]

    winner_key: CandidateKey | None = None

    if rankable:
        ranked = sorted(rankable, key=lambda c: c.score.point_estimate, reverse=True)  # type: ignore[union-attr]
        best = ranked[0]
        best_pw = np.asarray(best.score.pointwise, dtype=float)  # type: ignore[union-attr]
        boot_idx = _bootstrap_indices(best_pw.size, n_boot, rng)
        ci_low, ci_high = _own_ci(best_pw, boot_idx, alpha)

        robust_winner = True
        runner_up_margin: float | None = None
        if len(ranked) >= 2:
            runner = ranked[1]
            runner_pw = np.asarray(runner.score.pointwise, dtype=float)  # type: ignore[union-attr]
            runner_up_margin = _paired_margin_lower_bound(best_pw, runner_pw, boot_idx, alpha)
            point_gap = best.score.point_estimate - runner.score.point_estimate  # type: ignore[union-attr]
            # Robust only if the paired CI on (best - runner_up) excludes zero AND
            # the gap is substantively large. At small held-out n the CI alone is a
            # weak signal, so the magnitude floor is the effective guard.
            robust_winner = runner_up_margin > 0.0 and point_gap >= MIN_WINNER_GAP_NATS

        if robust_winner:
            winner_key = best.key
            records.append(
                _record(
                    best,
                    runtime=runtime,
                    mode=SelectionMode.selected,
                    source_plan_signature=source_plan_signature,
                    fitness=FitnessRecord(
                        metric=HOLDOUT_METRIC,
                        draws=best.score.draws,  # type: ignore[union-attr]
                        holdout_n=best.score.holdout_n,  # type: ignore[union-attr]
                        n_splits=1,
                        point_estimate=best.score.point_estimate,  # type: ignore[union-attr]
                        bootstrap_ci_low=min(ci_low, best.score.point_estimate),  # type: ignore[union-attr]
                        bootstrap_ci_high=max(ci_high, best.score.point_estimate),  # type: ignore[union-attr]
                        runner_up_margin=runner_up_margin,
                    ),
                    winner_reason=(
                        "highest held-out logp; beats runner-up by a bootstrap CI excluding 0"
                    ),
                )
            )
            # The non-winning rankable candidates are eligible but not kept.
            for c in ranked[1:]:
                records.append(
                    _record(
                        c,
                        runtime=runtime,
                        mode=SelectionMode.gate_only,
                        source_plan_signature=source_plan_signature,
                        fitness=None,
                        winner_reason="eligible runner-up; not the selected winner",
                    )
                )
        else:
            # No robust winner: keep all rankable eligible candidates gate_only
            # rather than forcing a noisy argmax (scarce-data: do not select one).
            for c in ranked:
                records.append(
                    _record(
                        c,
                        runtime=runtime,
                        mode=SelectionMode.gate_only,
                        source_plan_signature=source_plan_signature,
                        fitness=None,
                        winner_reason="eligible; no candidate robustly beats the runner-up",
                    )
                )

    for c in unrankable:
        records.append(
            _record(
                c,
                runtime=runtime,
                mode=SelectionMode.gate_only,
                source_plan_signature=source_plan_signature,
                fitness=None,
                winner_reason=(
                    "eligible; held-out split too small to rank "
                    f"(need holdout_n >= {MIN_HOLDOUT_FOR_SELECTION})"
                ),
            )
        )

    return TaskSelection(records=tuple(records), winner_key=winner_key)

"""Tests for the statistically sound selection decision core."""

from __future__ import annotations

from numpy.random import default_rng
from synthstats_schemas import CandidateKey, GateOutcome, SelectionMode

from ppl_synth_data.validation.selection import (
    GateReport,
    HoldoutScore,
    ScoredCandidate,
    select_for_task,
)

RUNTIME = "pymc"
SIG = "count/complex/zero_inflated"


def _jit(center: float, n: int = 10) -> list[float]:
    """Deterministic small spread around a center, for held-out score fixtures."""
    return [center + ((i % 3) - 1) * 0.05 for i in range(n)]


def _gate(passed: bool) -> GateReport:
    def mk(n: str, p: bool) -> GateOutcome:
        return GateOutcome(name=n, passed=p, detail="")

    if passed:
        return GateReport(
            cheap_gate=mk("crash_gate", True),
            integrity_gates=(mk("l3_strict", True), mk("l4_fabricated", True)),
            density_gates=(mk("min_predictive_variance", True),),
        )
    return GateReport(
        cheap_gate=mk("crash_gate", True),
        integrity_gates=(mk("l3_strict", False),),
        density_gates=(mk("min_predictive_variance", True),),
    )


def _cand(idx: int, *, eligible: bool, holdout=None, draws: int = 500) -> ScoredCandidate:
    score = None
    if holdout is not None:
        score = HoldoutScore(pointwise=tuple(holdout), draws=draws)
    return ScoredCandidate(
        key=CandidateKey(task_id="t1", family="glm-5.2", candidate_idx=idx),
        gate=_gate(eligible),
        score=score,
    )


def _by_idx(sel) -> dict[int, SelectionMode]:
    return {r.candidate_key.candidate_idx: r.selection_mode for r in sel.records}


def test_ineligible_is_rejected_never_winner():
    rng = default_rng(0)
    bad = _cand(0, eligible=False, holdout=[-0.5] * 10)
    sel = select_for_task([bad], runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert sel.winner_key is None
    rec = sel.records[0]
    assert rec.selection_mode is SelectionMode.rejected
    assert rec.eligible is False
    assert rec.fitness is None


# High-variance per-row scores so the bootstrap CI is genuinely exercised (a
# constant per-row difference would collapse the CI to a point and hide a
# quantile-direction or paired/unpaired bug).
_HV = [-1.0, -5.0, -2.0, -8.0, -3.0, -9.0, -4.0, -7.0, -6.0, -10.0]


def _shifted(base: list[float], shifts: list[float]) -> list[float]:
    return [v + s for v, s in zip(base, shifts, strict=True)]


def test_clear_winner_is_selected_with_robust_margin():
    rng = default_rng(1)
    # Real per-row variance; winner beats runner by ~2.5 nats with genuine spread.
    win_pw = [-0.3, -0.5, -0.4, -0.6, -0.5, -0.45, -0.55, -0.4, -0.5, -0.6]
    run_pw = [-2.8, -3.2, -3.0, -2.9, -3.1, -3.0, -2.95, -3.05, -3.0, -3.1]
    winner = _cand(0, eligible=True, holdout=win_pw)
    runner = _cand(1, eligible=True, holdout=run_pw)
    sel = select_for_task([runner, winner], runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert sel.winner_key is not None
    assert sel.winner_key.candidate_idx == 0
    modes = _by_idx(sel)
    assert modes[0] is SelectionMode.selected
    assert modes[1] is SelectionMode.gate_only  # eligible runner-up, not kept as winner
    win = next(r for r in sel.records if r.selection_mode is SelectionMode.selected)
    assert win.fitness is not None
    assert win.fitness.metric == "avg_holdout_logp"
    assert win.fitness.runner_up_margin is not None and win.fitness.runner_up_margin > 0.0
    assert win.fitness.holdout_n == 10


def test_paired_bootstrap_required_not_unpaired():
    # Winner has a small CONSISTENT per-row advantage over a high-variance shared
    # base. The paired difference is always positive (-> paired CI excludes 0, so
    # selected), but the marginals overlap massively, so an UNPAIRED bootstrap
    # would put the CI across 0 and fail to select. Asserting `selected` proves
    # the margin is computed paired, not on independent resamples.
    rng = default_rng(11)
    advs = [0.7, 0.9, 0.8, 0.85, 0.75, 0.95, 0.7, 0.9, 0.8, 0.85]  # mean ~0.82 >= MIN gap
    winner = _cand(0, eligible=True, holdout=_shifted(_HV, advs))
    runner = _cand(1, eligible=True, holdout=list(_HV))
    sel = select_for_task([winner, runner], runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert sel.winner_key is not None and sel.winner_key.candidate_idx == 0
    assert _by_idx(sel)[0] is SelectionMode.selected


def test_magnitude_gate_blocks_tiny_but_consistent_gap():
    # A consistent but tiny advantage (~0.2 nats) clears the paired CI (excludes
    # 0) yet is below MIN_WINNER_GAP_NATS, so no winner is declared: a sub-0.5-nat
    # gap is not a substantively meaningful difference between two programs.
    rng = default_rng(12)
    advs = [0.15, 0.25, 0.2, 0.18, 0.22, 0.2, 0.17, 0.23, 0.19, 0.21]  # mean ~0.2 < MIN gap
    winner = _cand(0, eligible=True, holdout=_shifted(_HV, advs))
    runner = _cand(1, eligible=True, holdout=list(_HV))
    sel = select_for_task([winner, runner], runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert sel.winner_key is None
    modes = _by_idx(sel)
    assert modes[0] is SelectionMode.gate_only
    assert modes[1] is SelectionMode.gate_only


def test_eligible_without_score_is_gate_only():
    # Passed every gate but the fitness pass produced no held-out vector
    # (score=None): unrankable, kept gate_only, never selected.
    rng = default_rng(13)
    c = _cand(0, eligible=True, holdout=None)
    sel = select_for_task([c], runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert sel.winner_key is None
    assert sel.records[0].selection_mode is SelectionMode.gate_only
    assert sel.records[0].eligible is True


def test_no_robust_winner_keeps_all_gate_only():
    rng = default_rng(2)
    # Per-row differences straddle zero (b alternates above/below a), so the
    # paired bootstrap CI on (a - b) includes 0: no robust winner.
    base = _jit(-1.0)
    straddle = [v + (0.1 if i % 2 == 0 else -0.1) for i, v in enumerate(base)]
    a = _cand(0, eligible=True, holdout=base)
    b = _cand(1, eligible=True, holdout=straddle)
    sel = select_for_task([a, b], runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert sel.winner_key is None
    modes = _by_idx(sel)
    assert modes[0] is SelectionMode.gate_only
    assert modes[1] is SelectionMode.gate_only


def test_small_holdout_is_gate_only_not_ranked():
    rng = default_rng(3)
    # Holdout of 3 rows is below MIN_HOLDOUT_FOR_SELECTION (8): never ranked.
    a = _cand(0, eligible=True, holdout=[-0.5, -0.4, -0.6])
    b = _cand(1, eligible=True, holdout=[-5.0, -5.1, -4.9])
    sel = select_for_task([a, b], runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert sel.winner_key is None  # cannot rank on 3 points even though b looks far worse
    modes = _by_idx(sel)
    assert modes[0] is SelectionMode.gate_only
    assert modes[1] is SelectionMode.gate_only


def test_single_eligible_with_adequate_holdout_is_selected_without_runner_up():
    rng = default_rng(4)
    only = _cand(0, eligible=True, holdout=[-0.5] * 10)
    sel = select_for_task([only], runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert sel.winner_key is not None
    rec = sel.records[0]
    assert rec.selection_mode is SelectionMode.selected
    assert rec.fitness is not None
    assert rec.fitness.runner_up_margin is None  # no runner-up


def test_mixed_population_partitions_correctly():
    rng = default_rng(5)
    rejected = _cand(0, eligible=False, holdout=[-0.1] * 10)  # would "win" but ineligible
    winner = _cand(1, eligible=True, holdout=[-0.5] * 10)
    runner = _cand(2, eligible=True, holdout=[-4.0] * 10)
    sel = select_for_task(
        [rejected, winner, runner], runtime=RUNTIME, source_plan_signature=SIG, rng=rng
    )
    modes = _by_idx(sel)
    assert modes[0] is SelectionMode.rejected  # ineligible never wins despite best score
    assert modes[1] is SelectionMode.selected
    assert modes[2] is SelectionMode.gate_only
    assert sel.winner_key.candidate_idx == 1


def test_all_records_emitted_one_per_candidate():
    rng = default_rng(6)
    cands = [_cand(i, eligible=(i % 2 == 0), holdout=[-1.0] * 10) for i in range(4)]
    sel = select_for_task(cands, runtime=RUNTIME, source_plan_signature=SIG, rng=rng)
    assert len(sel.records) == 4
    assert {r.candidate_key.candidate_idx for r in sel.records} == {0, 1, 2, 3}

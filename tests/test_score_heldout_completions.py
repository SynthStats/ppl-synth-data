"""Tests for the one-shot completion scorer.

Two layers. The fast layer covers arm parsing, aggregation, the paired bootstrap estimator, and
the preflight wiring path. The layer marked ``slow`` runs the real verifier cascade and the real
PyMC semantic scorer over real completions and real sample data from the sealed corpus under
``artifacts/runs/``; those tests skip when that corpus is not on the machine, because it is not
committed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import score_heldout_completions as shc  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_SEALED_RUN = _REPO / "artifacts" / "runs" / "evaluation-reference"


def _score(task_id: str, *, passed: bool, logp: float | None) -> shc.CompletionScore:
    return shc.CompletionScore(task_id, True, True, True, "pass", logp, passed, "")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sealed_gold_task() -> tuple[str, dict[str, Any], dict[str, Any], str]:
    """One real gold task from the sealed corpus: id, visible, hidden, completion.

    The corpus lives under a gitignored ``artifacts/`` tree, so a machine without
    it skips rather than fails. Sample data comes from the same parquet pair the
    producer pipeline scored, read through pandas, and handed over as plain lists
    the way a frozen JSON eval set delivers them.
    """
    validated_path = _SEALED_RUN / "stage_F_validation" / "validated.jsonl"
    refs_path = _SEALED_RUN / "stage_C_data" / "data_refs.jsonl"
    if not validated_path.exists() or not refs_path.exists():
        pytest.skip(f"sealed corpus not on this machine: {_SEALED_RUN}")
    pd = pytest.importorskip("pandas")

    refs = {row["env_id"]: row for row in _jsonl(refs_path)}
    gold = next(
        row
        for row in _jsonl(validated_path)
        if isinstance(row.get("parsed_program"), str) and row["parsed_program"].strip()
    )
    ref = refs[gold["env_id"]]
    visible = {k: v.tolist() for k, v in pd.read_parquet(_SEALED_RUN / ref["data_ref"]).items()}
    hidden = {
        k: v.tolist() for k, v in pd.read_parquet(_SEALED_RUN / ref["hidden_data_ref"]).items()
    }
    return gold["id"], visible, hidden, gold["raw_completion"]


def test_parse_arm_args_orders_arms() -> None:
    arms = shc.parse_arm_args(["base=a.jsonl", "sft=b.jsonl", "sft_rescore=c.jsonl"])
    assert list(arms.keys()) == ["base", "sft", "sft_rescore"]
    assert arms["sft"] == Path("b.jsonl")


def test_parse_arm_args_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="label=path"):
        shc.parse_arm_args(["bad-spec"])


def test_paired_bootstrap_ci_clear_positive() -> None:
    a = {f"t{i}": float(i) for i in range(30)}
    b = {f"t{i}": float(i) + 2.0 for i in range(30)}
    ci = shc.paired_bootstrap_ci(a=a, b=b, n_boot=1000, seed=7)
    assert ci["n"] == 30
    assert ci["point"] == pytest.approx(2.0)
    assert ci["ci_low"] > 0.0


def test_paired_bootstrap_ci_empty_intersection_is_nan() -> None:
    ci = shc.paired_bootstrap_ci(a={"x": 1.0}, b={"y": 2.0}, n_boot=10, seed=0)
    assert ci["n"] == 0
    assert ci["point"] != ci["point"]  # NaN


def test_paired_bootstrap_ci_skips_none_values() -> None:
    a = {"t0": 1.0, "t1": None}
    b = {"t0": 2.0, "t1": 5.0}
    ci = shc.paired_bootstrap_ci(a=a, b=b, n_boot=100, seed=0)
    assert ci["n"] == 1


def test_aggregate_pass_at_1_and_logp() -> None:
    scores = [
        shc.CompletionScore("a", True, True, True, "pass", -1.0, True, ""),
        shc.CompletionScore("b", True, False, True, "critical", None, False, ""),
    ]
    agg = shc.aggregate(scores)
    assert agg["verifier_semantic_pass@1"] == pytest.approx(0.5)
    assert agg["mean_holdout_logp"] == pytest.approx(-1.0)
    assert agg["n_with_holdout_logp"] == 1


def test_build_comparison_emits_pairwise_contrasts() -> None:
    def scores(label_pass: bool) -> list[shc.CompletionScore]:
        return [
            shc.CompletionScore(f"t{i}", True, True, True, "pass", float(i), label_pass, "")
            for i in range(5)
        ]

    arm_scores = {"base": scores(False), "sft": scores(True), "sft_rescore": scores(True)}
    comparison = shc.build_comparison(arm_scores, ordered_labels=["base", "sft", "sft_rescore"])
    contrasts = [c["contrast"] for c in comparison["contrasts"]]
    assert contrasts == ["sft - base", "sft_rescore - sft"]


def test_load_eval_data_requires_visible_and_hidden(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"task_id": "a", "eval_data": {"visible": {}}}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="visible"):
        shc.load_eval_data(path)


def test_preflight_does_not_write_report(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            json.dumps(
                {"task_id": f"t{i}", "eval_data": {"visible": {"y": [1.0]}, "hidden": {"y": [2.0]}}}
            )
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    arms = {}
    for label in ("base", "sft"):
        comp = tmp_path / f"{label}.jsonl"
        comp.write_text(
            "\n".join(json.dumps({"task_id": f"t{i}", "completion": "x"}) for i in range(3)) + "\n",
            encoding="utf-8",
        )
        arms[label] = comp
    report = tmp_path / "report.html"

    result = shc.preflight(eval_set_path=eval_path, arms=arms)
    assert result["dry_run"] is True
    assert result["eval_task_count"] == 3
    assert result["checked_completion_count"] == 6
    assert result["arm_completion_counts"] == {"base": 3, "sft": 3}
    assert not report.exists()


def test_dry_run_rejects_unknown_task_id(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"task_id": "t0", "eval_data": {"visible": {}, "hidden": {}}}) + "\n",
        encoding="utf-8",
    )
    comp = tmp_path / "base.jsonl"
    comp.write_text(json.dumps({"task_id": "GHOST", "completion": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(KeyError, match="GHOST"):
        shc.preflight(eval_set_path=eval_path, arms={"base": comp})


# --------------------------------------------------------------------------- #
# The estimator: is it really paired?
# --------------------------------------------------------------------------- #
def test_bootstrap_resamples_pairs_not_arms() -> None:
    """The discriminating property between a paired and an unpaired bootstrap.

    Both arms share a large per-task level (0, 100, ..., 3900) and differ by a
    constant 1.0 on every task. Resampling the pairs sees a single repeated
    difference, so the interval collapses to zero width. Resampling each arm
    independently would draw two unrelated task subsets, and the interval would
    inherit the 3900-wide between-task spread instead.
    """
    a = {f"t{i}": 100.0 * i for i in range(40)}
    b = {f"t{i}": 100.0 * i + 1.0 for i in range(40)}
    ci = shc.paired_bootstrap_ci(a=a, b=b, n_boot=2000, seed=1)
    between_task_spread = max(a.values()) - min(a.values())
    assert ci["point"] == pytest.approx(1.0)
    assert ci["ci_high"] - ci["ci_low"] == pytest.approx(0.0)
    assert between_task_spread == pytest.approx(3900.0)


def test_bootstrap_on_identical_arms_contains_zero() -> None:
    arm = {f"t{i}": float(i) - 5.0 for i in range(30)}
    ci = shc.paired_bootstrap_ci(a=arm, b=dict(arm), n_boot=1000, seed=2)
    assert ci["n"] == 30
    assert ci["point"] == pytest.approx(0.0)
    assert ci["ci_low"] <= 0.0 <= ci["ci_high"]


def test_bootstrap_interval_brackets_point_and_stays_inside_the_data() -> None:
    """A bootstrap mean cannot fall outside the observed differences."""
    rng = random.Random(11)
    diffs = [rng.gauss(0.0, 1.0) for _ in range(200)]
    a = {f"t{i}": 0.0 for i in range(200)}
    b = {f"t{i}": diffs[i] for i in range(200)}
    ci = shc.paired_bootstrap_ci(a=a, b=b, n_boot=4000, seed=3)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert min(diffs) < ci["ci_low"]
    assert ci["ci_high"] < max(diffs)
    left, right = ci["point"] - ci["ci_low"], ci["ci_high"] - ci["point"]
    assert left == pytest.approx(right, rel=0.35)


def test_bootstrap_interval_narrows_with_the_square_root_of_n() -> None:
    """Sixteen times the tasks should roughly quarter the half-width."""
    half_widths = {}
    for n in (25, 400):
        rng = random.Random(5)
        a = {f"t{i}": 0.0 for i in range(n)}
        b = {f"t{i}": rng.gauss(0.5, 1.0) for i in range(n)}
        ci = shc.paired_bootstrap_ci(a=a, b=b, n_boot=3000, seed=4)
        half_widths[n] = (ci["ci_high"] - ci["ci_low"]) / 2.0
    assert half_widths[25] / half_widths[400] == pytest.approx(4.0, rel=0.25)


def test_bootstrap_is_deterministic_per_seed_and_point_is_seed_free() -> None:
    """Only the interval carries resampling noise; the point estimate is the data."""
    rng = random.Random(23)
    a = {f"t{i}": 0.0 for i in range(200)}
    b = {f"t{i}": rng.gauss(0.0, 1.0) for i in range(200)}
    first = shc.paired_bootstrap_ci(a=a, b=b, n_boot=2000, seed=7)
    again = shc.paired_bootstrap_ci(a=a, b=b, n_boot=2000, seed=7)
    other = shc.paired_bootstrap_ci(a=a, b=b, n_boot=2000, seed=8)
    assert first == again
    assert other["point"] == pytest.approx(first["point"])
    assert (other["ci_low"], other["ci_high"]) != (first["ci_low"], first["ci_high"])


def test_bootstrap_at_one_pair_reports_a_zero_width_interval_clear_of_zero() -> None:
    """A single paired task gives a zero-width interval and a paired-count badge."""
    ci = shc.paired_bootstrap_ci(a={"t0": 0.0}, b={"t0": 3.0}, n_boot=500, seed=0)
    assert ci["n"] == 1
    assert (ci["point"], ci["ci_low"], ci["ci_high"]) == (3.0, 3.0, 3.0)
    assert not ci["ci_low"] <= 0.0 <= ci["ci_high"]
    assert "paired n=1" in shc._ci_badge(ci)


# --------------------------------------------------------------------------- #
# Task membership in per-arm aggregates and paired contrasts
# --------------------------------------------------------------------------- #
def test_repeated_task_id_counts_once_in_the_contrast_and_twice_in_the_aggregate() -> None:
    """Aggregates count score entries; paired contrasts count distinct task identifiers."""
    base = [_score("t0", passed=False, logp=-3.0), _score("t1", passed=False, logp=-3.0)]
    sft = [
        _score("t0", passed=True, logp=-1.0),
        _score("t0", passed=False, logp=-9.0),
        _score("t1", passed=True, logp=-1.0),
    ]
    summary = shc.aggregate(sft)
    contrast = shc.build_comparison({"base": base, "sft": sft}, ordered_labels=["base", "sft"])[
        "contrasts"
    ][0]

    assert summary["n"] == 3
    assert summary["verifier_semantic_pass@1"] == pytest.approx(2 / 3)
    assert contrast["pass@1"]["n"] == 2
    assert contrast["holdout_logp"]["point"] == pytest.approx(-2.0)


def test_holdout_logp_contrast_keeps_only_tasks_both_arms_scored() -> None:
    """The logp contrast includes tasks scored by both arms; pass@1 includes every task.

    In this example, eight of ten tasks are excluded from the paired logp contrast and none from
    the pass@1 contrast.
    """
    base = [_score(f"t{i}", passed=True, logp=None if i < 8 else -2.0) for i in range(10)]
    sft = [_score(f"t{i}", passed=True, logp=-1.0) for i in range(10)]
    contrast = shc.build_comparison({"base": base, "sft": sft}, ordered_labels=["base", "sft"])[
        "contrasts"
    ][0]
    assert contrast["pass@1"]["n"] == 10
    assert contrast["holdout_logp"]["n"] == 2


def test_report_shows_the_paired_count_and_the_scored_task_count() -> None:
    base = [_score(f"t{i}", passed=False, logp=None if i else -2.0) for i in range(4)]
    sft = [_score(f"t{i}", passed=True, logp=-1.0) for i in range(4)]
    arm_scores = {"base": base, "sft": sft}
    ordered = ["base", "sft"]
    html = shc.render_html(
        eval_set_path=Path("eval.jsonl"),
        arm_summaries={label: shc.aggregate(s) for label, s in arm_scores.items()},
        comparison=shc.build_comparison(arm_scores, ordered_labels=ordered),
        ordered_labels=ordered,
    )
    assert "paired n=4" in html  # pass@1 keeps all four tasks
    assert "paired n=1" in html  # holdout logp keeps only the task both arms scored
    assert "tasks scored" in html


# --------------------------------------------------------------------------- #
# Real verifier path over the sealed corpus
# --------------------------------------------------------------------------- #
def test_score_arm_rejects_a_completion_outside_the_eval_set(tmp_path: Path) -> None:
    """The mismatch guard fires before any verifier runs."""
    completions = tmp_path / "arm.jsonl"
    completions.write_text(
        json.dumps({"task_id": "not-in-eval", "completion": "x"}) + "\n", encoding="utf-8"
    )
    eval_by_id = {"t0": shc.EvalTaskData("t0", {"y": [1.0]}, {"y": [2.0]})}
    with pytest.raises(KeyError, match="not-in-eval"):
        shc.score_arm(completions_path=completions, eval_by_id=eval_by_id)


def test_cascade_data_conversion_arrays_sequences_and_leaves_scalars_alone() -> None:
    converted = shc._as_cascade_data({"y": [1.0, 2.0, 3.0], "x": (4, 5), "n": 3, "label": "a"})
    assert hasattr(converted["y"], "tolist")
    assert hasattr(converted["x"], "tolist")
    assert converted["n"] == 3
    assert converted["label"] == "a"


@pytest.mark.slow
@pytest.mark.real_boundary
def test_cascade_verdict_lives_under_diagnostics_and_depends_on_column_type() -> None:
    """The cascade verdict is nested in diagnostics, and its perturbation operates on arrays."""
    from ppl_synth_data.harbor.validator_template import run_cascade
    from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime

    _task_id, visible, _hidden, completion = _sealed_gold_task()
    code = PyMCRuntime().extract_code(completion)
    assert code is not None

    as_lists = run_cascade(code, visible)
    as_arrays = run_cascade(code, shc._as_cascade_data(visible))
    assert "failed_layer" not in as_arrays
    assert as_lists["diagnostics"]["failed_layer"] == "L5_data_perturbation"
    assert as_arrays["diagnostics"]["failed_layer"] is None


@pytest.mark.slow
@pytest.mark.real_boundary
def test_score_one_over_a_real_gold_completion() -> None:
    """Exercise the validation gates with import-bearing and import-free model programs."""
    from ppl_synth_data.validation.pymc_semantics import score_pymc_semantics

    task_id, visible, hidden, completion = _sealed_gold_task()
    score = shc.score_one(
        completion_text=completion,
        eval_data=shc.EvalTaskData(task_id, visible, hidden),
    )
    assert (score.format_pass, score.runtime_build_pass, score.harbor_safety_pass) == (
        True,
        True,
        True,
    )
    assert score.semantic_status == "critical"
    assert score.holdout_logp is None
    assert score.verifier_semantic_pass is False
    assert "import statements are not allowed" in score.notes

    from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime

    code = PyMCRuntime().extract_code(completion)
    assert code is not None
    without_imports = "\n".join(
        line for line in code.splitlines() if not line.startswith(("import ", "from "))
    )
    rescored = score_pymc_semantics(without_imports, visible_data=visible, hidden_data=hidden)
    assert rescored.metric_name == "avg_holdout_logp"
    assert rescored.metric_value is not None
    assert -10.0 < rescored.metric_value < 0.0


@pytest.mark.slow
@pytest.mark.real_boundary
def test_run_writes_a_report_from_real_scores(tmp_path: Path) -> None:
    """End to end: real eval set, real completions, real verifiers, real report."""
    task_id, visible, hidden, completion = _sealed_gold_task()
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"task_id": task_id, "eval_data": {"visible": visible, "hidden": hidden}})
        + "\n",
        encoding="utf-8",
    )
    from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime

    empty_model = PyMCRuntime().replace_code(completion, "def model(data):\n    return None\n")

    arms = {}
    for label, text in (("base", empty_model), ("sft", completion)):
        path = tmp_path / f"{label}.jsonl"
        path.write_text(
            json.dumps({"task_id": task_id, "completion": text}) + "\n", encoding="utf-8"
        )
        arms[label] = path
    report = tmp_path / "report.html"

    result = shc.run(eval_set_path=eval_path, arms=arms, report_path=report)

    assert result["dry_run"] is False
    assert result["eval_task_count"] == 1
    assert result["summaries"]["base"]["n"] == 1
    assert result["summaries"]["sft"]["harbor_safety_pass_rate"] == pytest.approx(1.0)
    assert result["summaries"]["base"]["harbor_safety_pass_rate"] == pytest.approx(0.0)
    contrast = result["comparison"]["contrasts"][0]
    assert contrast["contrast"] == "sft - base"
    assert contrast["pass@1"]["n"] == 1
    html = report.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert 'meta charset="utf-8"' in html
    assert "prefers-color-scheme: dark" in html
    assert "paired n=1" in html

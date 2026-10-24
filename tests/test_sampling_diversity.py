"""Tests for diversity enforcement and coverage reporting."""

from __future__ import annotations

from ppl_synth_data.specs.registry import SUPPORTED_FAMILIES
from ppl_synth_data.specs.sampling import (
    COMPLEXITIES,
    DOMAINS,
    build_generation_plan,
    coverage_report,
    sample_domain_family,
)


def test_sample_domain_family_returns_valid() -> None:
    for seed in range(20):
        domain, family = sample_domain_family(seed)
        assert domain in DOMAINS
        assert family in SUPPORTED_FAMILIES


def test_domains_includes_new_entries() -> None:
    for d in ("psychology", "ecology", "engineering", "economics"):
        assert d in DOMAINS


def test_build_generation_plan_covers_all_family_complexity_pairs() -> None:
    plan = build_generation_plan(target_count=200)
    seen = set()
    for entry in plan:
        seen.add((entry["family"], entry["complexity"]))
    for fam in SUPPORTED_FAMILIES:
        for cx in COMPLEXITIES:
            assert (fam, cx.value) in seen, f"missing ({fam}, {cx.value})"


def test_build_generation_plan_reaches_target_count() -> None:
    plan = build_generation_plan(target_count=2000)
    assert len(plan) == 2000


def test_build_generation_plan_minimum_equals_family_complexity_product() -> None:
    min_count = len(SUPPORTED_FAMILIES) * len(COMPLEXITIES)
    plan = build_generation_plan(target_count=min_count)
    assert len(plan) == min_count


def test_coverage_report_no_gaps() -> None:
    plan = build_generation_plan(target_count=200)
    records = [
        {"generative_family": e["family"], "complexity": e["complexity"], "domain": e["domain"]}
        for e in plan
    ]
    report = coverage_report(records)
    assert report["n_gaps"] == 0
    assert report["total"] == 200


def test_coverage_report_detects_gaps() -> None:
    records = [
        {"generative_family": "regression_continuous", "complexity": "simple", "domain": "biology"},
    ]
    report = coverage_report(records)
    assert report["n_gaps"] > 0
    assert report["total"] == 1

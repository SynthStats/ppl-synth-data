"""Tests for the contract-backed reward source catalog used by eval schemas."""

from __future__ import annotations

import pytest
from synthstats_schemas import (
    EvidenceKind,
    RewardSourceNotRegisteredError,
    TruthOwner,
    ensure_reward_source_registered,
    get_reward_source,
    list_reward_sources,
)


def test_contract_catalog_exposes_only_explicit_sources() -> None:
    names = set(list_reward_sources())
    assert "validation.pymc_elpd" in names
    assert "validation.lazyppl_elpd" in names
    assert "validation.compilation" in names
    assert "heuristic.llm_critic" not in names
    assert "unknown" not in names
    assert "validation.elpd" not in names
    assert "validation.harbor_pymc_strict_profile" not in names


def test_verifier_source_has_verifier_provenance() -> None:
    spec = get_reward_source("validation.lazyppl_elpd")
    assert spec.evidence_kind is EvidenceKind.verifier
    assert spec.truth_owner is TruthOwner.verifier
    assert "ppl_synth_data.docker.lazyppl_harbor.score" in spec.allowed_producers


def test_heuristic_source_has_heuristic_provenance() -> None:
    spec = get_reward_source("heuristic.judge_exception")
    assert spec.evidence_kind is EvidenceKind.heuristic
    assert spec.truth_owner is TruthOwner.heuristic


def test_unregistered_source_raises() -> None:
    with pytest.raises(RewardSourceNotRegisteredError, match="not registered"):
        ensure_reward_source_registered("definitely.not.registered.xyz")


def test_prefixes_do_not_imply_provenance() -> None:
    with pytest.raises(RewardSourceNotRegisteredError, match="not registered"):
        ensure_reward_source_registered("validation.brand_new_scorer")

"""Tests for the agentic AI-scientist environment slice.

The fast tests exercise the structural tool-call parser and the env state
machine (real compiled infection_process env, real queries) without OpenAI or
PyMC. The `slow` test drives a known-good program through the real held-out
log-likelihood verifier and asserts a contract-valid RewardArtifact.
"""

from __future__ import annotations

import pytest
from synthstats_schemas.reward import EvidenceKind, RewardArtifact

from ppl_synth_data.agentic.domain import InfectionProcessDomain
from ppl_synth_data.agentic.run_episode import _parse_action

_GOOD_PROGRAM = """
def model(data):
    t = np.asarray(data["t"], dtype=float)
    infected = np.asarray(data["infected"], dtype=float)
    N = 100
    with pm.Model() as m:
        theta = pm.HalfNormal("theta", sigma=0.1)
        p = 1.0 - pm.math.exp(-theta * t)
        p = pm.math.clip(p, 1e-6, 1 - 1e-6)
        pm.Binomial("infected", n=N, p=p, observed=infected)
    return m
"""


def test_parse_action_strict_json():
    action = _parse_action('{"tool": "observe_data", "arguments": {"t": 3.0}}')
    assert action == {"tool": "observe_data", "arguments": {"t": 3.0}}


def test_parse_action_embedded_object():
    raw = 'Here is my move:\n{"tool": "observe_data", "arguments": {"t": 7.5}} done'
    action = _parse_action(raw)
    assert action["tool"] == "observe_data"
    assert action["arguments"]["t"] == 7.5


def test_parse_action_rejects_unknown_tool():
    with pytest.raises(ValueError):
        _parse_action('{"tool": "drop_table", "arguments": {}}')


def test_parse_action_rejects_non_json():
    with pytest.raises(ValueError):
        _parse_action("I would like to observe at t=3 please")


def test_domain_reset_and_real_query():
    domain = InfectionProcessDomain(seed=7, turn_budget=4)
    try:
        payload = domain.reset()
        assert payload["data_columns"] == ["t", "infected"]
        assert payload["target_name"] == "infected"
        # reset seeds two grounding observations from real env queries.
        assert len(payload["visible_data"]["t"]) == 2

        obs = domain.observe_data({"t": 9.0})
        assert obs.ok
        assert isinstance(obs.infected, int)
        assert 0 <= obs.infected <= 100
        assert obs.n_visible == 3
        assert domain.turns_used == 1
    finally:
        domain.close()


def test_domain_turn_budget_enforced():
    domain = InfectionProcessDomain(seed=8, turn_budget=1)
    try:
        domain.reset()
        domain.observe_data({"t": 5.0})
        with pytest.raises(RuntimeError):
            domain.observe_data({"t": 6.0})
    finally:
        domain.close()


@pytest.mark.slow
def test_submit_program_scores_with_real_verifier():
    domain = InfectionProcessDomain(seed=9, turn_budget=6, verifier_draws=30, verifier_tune=30)
    try:
        domain.reset()
        domain.observe_data({"t": 4.0})
        domain.observe_data({"t": 8.0})
        domain.observe_data({"t": 20.0})

        result = domain.submit_program(_GOOD_PROGRAM)

        assert result.semantic_status in {"pass", "flag"}
        assert result.metric_name == "avg_holdout_logp"
        assert result.metric_value is not None

        artifact = result.reward_artifact
        assert isinstance(artifact, RewardArtifact)
        assert artifact.reward_source == "validation.pymc_semantic_holdout_logp"
        assert artifact.evidence_kind is EvidenceKind.verifier
        assert artifact.producer == "ppl_synth_data.validation.pymc_semantics"
        assert artifact.training_log_reward >= -20.0
        # Re-validate the serialized artifact against the contract.
        RewardArtifact.model_validate(artifact.model_dump(mode="json"))

        # Terminal: a second submit must be rejected.
        with pytest.raises(RuntimeError):
            domain.submit_program(_GOOD_PROGRAM)
    finally:
        domain.close()

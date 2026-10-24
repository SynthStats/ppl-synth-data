"""Tests for schemas.trajectory Pydantic models.

Covers: frozen-ness, forbid-extra, reward-provenance validator (the
structural-reward-provenance invariant), length-alignment validators on Turn
and Episode, JSON Schema emission, and round-trip serialisation.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from synthstats_schemas import EvidenceKind, TruthOwner, get_reward_source

from ppl_synth_data.schemas.trajectory import (
    SCHEMA_VERSION,
    ActionType,
    BeliefOwner,
    BeliefSnapshot,
    BeliefSpec,
    Episode,
    EvalTaskSpec,
    GoalSpec,
    Message,
    MetricBreakdown,
    MetricSchedule,
    MetricSpec,
    Observation,
    RewardRecord,
    Step,
    StopReason,
    Trajectory,
    Trial,
    Turn,
)


def _make_reward(source: str = "validation.compilation") -> RewardRecord:
    """Build a reward that matches the registry's classification for `source`."""
    spec = get_reward_source(source)
    return RewardRecord(
        total=1.0,
        components={"formatting": 0.3, "compilation": 0.7},
        reward_source=source,
        evidence_kind=spec.evidence_kind,
        truth_owner=spec.truth_owner,
        producer="tests.unit",
    )


def _make_turn(idx: int = 0, n_tokens: int = 3) -> Turn:
    return Turn(
        turn_index=idx,
        messages_input=[Message(role="system", content="sys")],
        assistant_message=Message(role="assistant", content="ok"),
        token_ids=list(range(n_tokens)),
        token_logprobs=[-0.1] * n_tokens,
        loss_mask=[True] * n_tokens,
        parsed_action_type=ActionType.submit_program,
    )


def _make_task_spec() -> EvalTaskSpec:
    return EvalTaskSpec(
        task_id="task-abc",
        domain="dugongs",
        env_spec_id="envspec-dugongs-001",
        env_spec_digest="sha256:aaaa",
        compiled_env_code_digest="sha256:bbbb",
        goal=GoalSpec(
            goal_id="identify_growth_params",
            goal_type="parameter_identification",
            target={"params": ["alpha", "beta"]},
        ),
        belief_spec=BeliefSpec(
            model_family="pymc",
            sampler_config={"draws": 1000, "tune": 1000},
            sampler_config_digest="sha256:cccc",
        ),
        metric_specs=[
            MetricSpec(
                metric_id="eig-analytic",
                metric_name="eig_analytic",
                schedule=MetricSchedule.online,
                reward_source="validation.compilation",
                evidence_kind=EvidenceKind.verifier,
                truth_owner=TruthOwner.verifier,
            )
        ],
        visible_data_ref="s3://bucket/dugongs/visible.parquet",
        hidden_data_ref="s3://bucket/dugongs/hidden.parquet",
        prompt_template_hash="sha256:dddd",
    )


def _make_observation(step_idx: int = 0, turn_idx: int = 0) -> Observation:
    return Observation(
        observation_id=f"obs-{step_idx}",
        step_index=step_idx,
        triggering_turn_index=turn_idx,
        payload={"y": 2.5},
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        source="env.ground_truth",
    )


def _make_trajectory() -> Trajectory:
    task = _make_task_spec()
    turn = _make_turn(0, 3)
    obs = _make_observation(0, 0)
    step = Step(
        step_index=0,
        triggering_turn_index=0,
        observation_id=obs.observation_id,
        env_state_digest_before="sha256:pre",
        env_state_digest_after="sha256:post",
        wall_seconds=0.02,
    )
    episode = Episode(
        episode_id="ep-001",
        turns=[turn],
        steps=[step],
        observations=[obs],
        belief_snapshots=[],
        belief_updates=[],
        tool_calls=[],
        metric_breakdowns=[],
        transition_rewards=[0.0],
        turn_discounts=[1.0],
    )
    trial = Trial(
        trial_id="trial-001",
        submitted_program="import pymc as pm\n# ...",
        language="pymc",
        sandbox_backend="harbor",
        reward=_make_reward(),
    )
    return Trajectory(
        trajectory_id="traj-001",
        task_spec=task,
        episode=episode,
        trial=trial,
        seed=42,
        created_at_iso="2026-04-22T00:00:00Z",
    )


class TestRewardProvenanceIsStructural:
    def test_unregistered_reward_source_raises_on_construction(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RewardRecord(
                total=0.0,
                reward_source="definitely.not.registered.phantom",
                evidence_kind=EvidenceKind.heuristic,
                truth_owner=TruthOwner.heuristic,
                producer="tests.unit",
            )
        # Pydantic wraps the registry error
        assert "not registered" in str(excinfo.value).lower()

    def test_evidence_kind_mismatch_with_registry_raises(self) -> None:
        """validation.compilation is verifier; heuristic mismatch must raise."""
        with pytest.raises(ValidationError) as excinfo:
            RewardRecord(
                total=0.0,
                reward_source="validation.compilation",
                evidence_kind=EvidenceKind.heuristic,  # mismatch
                truth_owner=TruthOwner.heuristic,
                producer="tests.unit",
            )
        assert "mismatch" in str(excinfo.value).lower()

    def test_matching_provenance_constructs_cleanly(self) -> None:
        r = _make_reward("validation.compilation")
        assert r.evidence_kind is EvidenceKind.verifier
        assert r.truth_owner is TruthOwner.verifier

    def test_heuristic_source_requires_heuristic_kind(self) -> None:
        with pytest.raises(ValidationError, match="mismatch"):
            RewardRecord(
                total=0.0,
                reward_source="heuristic.no_submission",
                evidence_kind=EvidenceKind.verifier,
                truth_owner=TruthOwner.verifier,
                producer="tests.unit",
            )

    def test_metric_breakdown_also_enforces_registration(self) -> None:
        with pytest.raises(ValidationError, match="not registered"):
            MetricBreakdown(
                metric_id="m-0",
                metric_name="custom",
                turn_index=0,
                value=1.0,
                schedule=MetricSchedule.online,
                reward_source="unregistered.metric",
                evidence_kind=EvidenceKind.heuristic,
                truth_owner=TruthOwner.heuristic,
                producer="tests.unit",
            )

    def test_metric_spec_also_enforces_registration(self) -> None:
        with pytest.raises(ValidationError, match="not registered"):
            MetricSpec(
                metric_id="m-0",
                metric_name="custom",
                schedule=MetricSchedule.online,
                reward_source="unregistered.spec",
                evidence_kind=EvidenceKind.heuristic,
                truth_owner=TruthOwner.heuristic,
            )


class TestTurnLengthAlignment:
    def test_mismatched_token_logprobs_raises(self) -> None:
        with pytest.raises(ValidationError, match="token_logprobs length"):
            Turn(
                turn_index=0,
                messages_input=[],
                assistant_message=Message(role="assistant", content="x"),
                token_ids=[1, 2, 3],
                token_logprobs=[-0.1, -0.2],  # wrong length
                loss_mask=[True, True, True],
            )

    def test_mismatched_loss_mask_raises(self) -> None:
        with pytest.raises(ValidationError, match="loss_mask length"):
            Turn(
                turn_index=0,
                messages_input=[],
                assistant_message=Message(role="assistant", content="x"),
                token_ids=[1, 2, 3],
                token_logprobs=[-0.1, -0.2, -0.3],
                loss_mask=[True, True],  # wrong length
            )

    def test_empty_eos_logprobs_is_allowed(self) -> None:
        t = _make_turn(0, 3)
        assert t.eos_logprobs == []

    def test_eos_logprobs_length_enforced_when_nonempty(self) -> None:
        with pytest.raises(ValidationError, match="eos_logprobs length"):
            Turn(
                turn_index=0,
                messages_input=[],
                assistant_message=Message(role="assistant", content="x"),
                token_ids=[1, 2, 3],
                token_logprobs=[-0.1, -0.2, -0.3],
                loss_mask=[True, True, True],
                eos_logprobs=[-5.0, -5.0],  # wrong length
            )


class TestEpisodeAlignment:
    def test_transition_rewards_must_match_turn_count(self) -> None:
        with pytest.raises(ValidationError, match="transition_rewards length"):
            Episode(
                episode_id="ep-0",
                turns=[_make_turn(0, 2), _make_turn(1, 2)],
                steps=[],
                observations=[],
                belief_snapshots=[],
                belief_updates=[],
                tool_calls=[],
                metric_breakdowns=[],
                transition_rewards=[0.0],  # only 1, not 2
                turn_discounts=[1.0, 1.0],
            )


class TestFrozenAndForbidExtra:
    def test_trajectory_is_frozen(self) -> None:
        traj = _make_trajectory()
        with pytest.raises(ValidationError):
            traj.seed = 99  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            Message(role="user", content="x", bogus_field="nope")  # type: ignore[call-arg]


class TestSchemaVersionAndSerialisation:
    def test_schema_version_on_trajectory(self) -> None:
        traj = _make_trajectory()
        assert traj.schema_version == SCHEMA_VERSION

    def test_trajectory_roundtrip_json(self) -> None:
        traj = _make_trajectory()
        dumped = traj.model_dump(mode="json")
        restored = Trajectory.model_validate(dumped)
        assert restored == traj

    def test_trajectory_json_schema_emits(self) -> None:
        schema = Trajectory.model_json_schema()
        assert schema["title"] == "Trajectory"
        # Required top-level fields are present
        required = set(schema.get("required", []))
        assert {"trajectory_id", "task_spec", "episode", "trial", "seed"} <= required
        # Schema version appears
        json.dumps(schema)  # must be JSON-serialisable

    def test_belief_owner_values(self) -> None:
        assert BeliefOwner.environment_gold.value == "environment_gold"
        assert BeliefOwner.agent_candidate.value == "agent_candidate"
        assert BeliefOwner.verifier_reconstruction.value == "verifier_reconstruction"

    def test_belief_snapshot_captures_cache_key_triple(self) -> None:
        snap = BeliefSnapshot(
            snapshot_id="snap-0",
            turn_index=0,
            belief_owner=BeliefOwner.environment_gold,
            model_family="pymc",
            summary={"mean": 1.0},
            posterior_digest=None,
            content_digest="sha256:content",
            seed=42,
            sampler_config_digest="sha256:sampler",
            compute_seconds=1.5,
        )
        # The triple must all be present; this encodes the cache-key rule.
        assert snap.content_digest
        assert snap.seed == 42
        assert snap.sampler_config_digest


class TestTerminalStates:
    def test_trial_with_error_stop_reason(self) -> None:
        trial = Trial(
            trial_id="trial-err",
            submitted_program=None,
            language="pymc",
            reward=_make_reward("heuristic.no_submission"),
        )
        assert trial.submitted_program is None
        assert trial.reward.evidence_kind is EvidenceKind.heuristic

    def test_stop_reason_values(self) -> None:
        assert StopReason.completed.value == "completed"
        assert StopReason.error.value == "error"


class TestHeuristicTruthTagging:
    def test_task_spec_defaults_to_verifier(self) -> None:
        ts = _make_task_spec()
        assert ts.evidence_kind is EvidenceKind.verifier
        assert ts.truth_owner is TruthOwner.verifier

    def test_task_spec_can_declare_heuristic_truth(self) -> None:
        """emotion / moral_machines domains must be able to declare heuristic class."""
        metric_spec = MetricSpec(
            metric_id="emotion-m0",
            metric_name="emotion_scored",
            schedule=MetricSchedule.terminal,
            reward_source="heuristic.llm_rubric",
            evidence_kind=EvidenceKind.heuristic,
            truth_owner=TruthOwner.heuristic,
        )
        ts = EvalTaskSpec(
            task_id="emotion-task-0",
            domain="emotion",
            env_spec_id="envspec-emotion-001",
            env_spec_digest="sha256:e1",
            compiled_env_code_digest="sha256:e2",
            goal=GoalSpec(
                goal_id="classify_emotion",
                goal_type="parameter_identification",
                target={"panelists": 10},
            ),
            belief_spec=BeliefSpec(
                model_family="llm_simulated",
                sampler_config={},
                sampler_config_digest="sha256:e3",
            ),
            metric_specs=[metric_spec],
            visible_data_ref="s3://bucket/emotion/visible",
            hidden_data_ref="s3://bucket/emotion/hidden",
            prompt_template_hash="sha256:e4",
            evidence_kind=EvidenceKind.heuristic,
            truth_owner=TruthOwner.heuristic,
        )
        assert ts.evidence_kind is EvidenceKind.heuristic

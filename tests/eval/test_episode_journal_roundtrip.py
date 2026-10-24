"""Roundtrip tests for EpisodeJournal + events_to_trajectory.

Verifies: append -> read -> project reproduces the source trajectory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ppl_synth_data.eval.persistence import (
    EpisodeJournal,
    FsyncMode,
    events_to_trajectory,
)
from ppl_synth_data.eval.persistence.projectors import ProjectionError
from ppl_synth_data.schemas.trajectory import (
    BeliefOwner,
    BeliefSnapshot,
    BeliefUpdate,
    MetricBreakdown,
    MetricSchedule,
    Step,
    StopReason,
    ToolCallRecord,
    Trajectory,
)
from tests.eval.test_trajectory_schema import (
    _make_observation,
    _make_reward,
    _make_task_spec,
    _make_turn,
)


@pytest.fixture
def journal(tmp_path: Path) -> EpisodeJournal:
    return EpisodeJournal.for_episode(tmp_path, "ep-001", fsync_mode=FsyncMode.never)


def _emit_complete_episode(
    journal: EpisodeJournal,
    *,
    n_turns: int = 2,
    with_trial: bool = True,
) -> Trajectory:
    """Emit a full sequence of events and return the projected trajectory."""
    task_spec = _make_task_spec()
    journal.append(
        {
            "event_type": "episode_started",
            "episode_id": "ep-001",
            "trajectory_id": "traj-001",
            "task_spec": task_spec.model_dump(),
            "seed": 42,
        }
    )

    for t in range(n_turns):
        turn = _make_turn(t, 3)
        journal.append({"event_type": "turn_recorded", "turn": turn.model_dump()})

        obs = _make_observation(t, t)
        journal.append({"event_type": "observation_recorded", "observation": obs.model_dump()})
        step = Step(
            step_index=t,
            triggering_turn_index=t,
            observation_id=obs.observation_id,
            env_state_digest_before=f"sha256:pre{t}",
            env_state_digest_after=f"sha256:post{t}",
            wall_seconds=0.02,
        )
        journal.append({"event_type": "step_recorded", "step": step.model_dump()})

    if with_trial:
        journal.append(
            {
                "event_type": "trial_submitted",
                "trial_id": "trial-001",
                "submitted_program": "import pymc as pm\n# ...",
                "language": "pymc",
            }
        )
        reward = _make_reward()
        journal.append(
            {
                "event_type": "trial_scored",
                "trial_id": "trial-001",
                "reward": reward.model_dump(),
                "sandbox_backend": "harbor",
                "verifier_stdout": "",
                "verifier_stderr": "",
                "exit_code": 0,
                "wall_seconds": 1.2,
                "artefacts": {},
            }
        )

    journal.append(
        {
            "event_type": "episode_finalised",
            "episode_id": "ep-001",
            "terminal_transition_rewards": [0.0] * n_turns,
            "terminal_turn_discounts": [1.0] * n_turns,
            "terminal_stop_reason": StopReason.completed.value,
            "created_at_iso": "2026-04-22T00:00:00Z",
            "dependency_digests": {},
            "run_metadata": {},
        }
    )

    trajectory = events_to_trajectory(journal.read_events())
    journal.finalise(trajectory)
    return trajectory


class TestRoundtrip:
    def test_append_read_project_roundtrip(self, journal: EpisodeJournal) -> None:
        trajectory = _emit_complete_episode(journal, n_turns=2)
        assert trajectory.trajectory_id == "traj-001"
        assert len(trajectory.episode.turns) == 2
        assert len(trajectory.episode.observations) == 2
        assert len(trajectory.episode.steps) == 2
        assert trajectory.trial.reward.total == 1.0
        assert trajectory.trial.reward.reward_source == "validation.compilation"

    def test_sequence_numbers_are_monotonic(self, journal: EpisodeJournal) -> None:
        _emit_complete_episode(journal, n_turns=1)
        events = journal.read_events()
        sequences = [e.sequence for e in events]
        assert sequences == list(range(len(sequences)))

    def test_finalised_flag_set_after_finalise(self, journal: EpisodeJournal) -> None:
        assert not journal.finalised
        _emit_complete_episode(journal, n_turns=1)
        assert journal.finalised

    def test_trajectory_json_parses_as_trajectory(self, journal: EpisodeJournal) -> None:
        trajectory = _emit_complete_episode(journal, n_turns=2)
        import json

        on_disk = Trajectory.model_validate(
            json.loads(journal.paths.trajectory_path.read_text(encoding="utf-8"))
        )
        assert on_disk == trajectory

    def test_belief_and_tool_and_metric_events_roundtrip(self, journal: EpisodeJournal) -> None:
        task_spec = _make_task_spec()
        journal.append(
            {
                "event_type": "episode_started",
                "episode_id": "ep-001",
                "trajectory_id": "traj-001",
                "task_spec": task_spec.model_dump(),
                "seed": 42,
            }
        )
        turn = _make_turn(0, 2)
        journal.append({"event_type": "turn_recorded", "turn": turn.model_dump()})

        snap = BeliefSnapshot(
            snapshot_id="snap-0",
            turn_index=0,
            belief_owner=BeliefOwner.environment_gold,
            model_family="pymc",
            summary={"mean": 1.0},
            posterior_digest=None,
            content_digest="sha256:content-0",
            seed=42,
            sampler_config_digest="sha256:sampler-0",
            compute_seconds=1.5,
        )
        journal.append({"event_type": "belief_snapshot_recorded", "snapshot": snap.model_dump()})

        snap2 = snap.model_copy(update={"snapshot_id": "snap-1"})
        journal.append({"event_type": "belief_snapshot_recorded", "snapshot": snap2.model_dump()})
        upd = BeliefUpdate(
            update_id="upd-0",
            turn_index=0,
            belief_owner=BeliefOwner.environment_gold,
            prior_snapshot_id="snap-0",
            posterior_snapshot_id="snap-1",
            observation_ids=["obs-0"],
            compute_seconds=0.8,
        )
        journal.append({"event_type": "belief_update_recorded", "update": upd.model_dump()})

        tc = ToolCallRecord(
            tool_call_id="tc-0",
            turn_index=0,
            tool_name="query_experiment",
            arguments={"x": 0.5},
            result={"y": 2.5},
            latency_seconds=0.03,
        )
        journal.append({"event_type": "tool_call_recorded", "tool_call": tc.model_dump()})

        mb = MetricBreakdown(
            metric_id="m-0",
            metric_name="eig_mc",
            turn_index=0,
            value=0.42,
            schedule=MetricSchedule.online,
            reward_source="validation.compilation",
            evidence_kind=_make_reward().evidence_kind,
            truth_owner=_make_reward().truth_owner,
            producer="tests.unit",
        )
        journal.append({"event_type": "metric_recorded", "metric": mb.model_dump()})

        journal.append(
            {
                "event_type": "trial_submitted",
                "trial_id": "trial-001",
                "submitted_program": "code",
                "language": "pymc",
            }
        )
        journal.append(
            {
                "event_type": "trial_scored",
                "trial_id": "trial-001",
                "reward": _make_reward().model_dump(),
                "sandbox_backend": "harbor",
            }
        )
        journal.append(
            {
                "event_type": "episode_finalised",
                "episode_id": "ep-001",
                "terminal_transition_rewards": [0.0],
                "terminal_turn_discounts": [1.0],
                "terminal_stop_reason": "completed",
                "created_at_iso": "2026-04-22T00:00:00Z",
            }
        )
        trajectory = events_to_trajectory(journal.read_events())
        assert len(trajectory.episode.belief_snapshots) == 2
        assert len(trajectory.episode.belief_updates) == 1
        assert len(trajectory.episode.tool_calls) == 1
        assert len(trajectory.episode.metric_breakdowns) == 1


class TestProjectionErrors:
    def test_empty_stream_raises(self) -> None:
        with pytest.raises(ProjectionError, match="empty"):
            events_to_trajectory([])

    def test_missing_start_event_raises(self, journal: EpisodeJournal) -> None:
        journal.append(
            {
                "event_type": "episode_finalised",
                "episode_id": "ep",
                "terminal_transition_rewards": [],
                "terminal_turn_discounts": [],
                "terminal_stop_reason": "error",
                "created_at_iso": "2026-04-22T00:00:00Z",
            }
        )
        with pytest.raises(ProjectionError, match="no episode_started"):
            events_to_trajectory(journal.read_events())

    def test_missing_finalise_raises(self, journal: EpisodeJournal) -> None:
        task_spec = _make_task_spec()
        journal.append(
            {
                "event_type": "episode_started",
                "episode_id": "ep",
                "trajectory_id": "traj",
                "task_spec": task_spec.model_dump(),
                "seed": 1,
            }
        )
        with pytest.raises(ProjectionError, match="episode_finalised"):
            events_to_trajectory(journal.read_events())

    def test_scored_without_submitted_raises(self, journal: EpisodeJournal) -> None:
        task_spec = _make_task_spec()
        journal.append(
            {
                "event_type": "episode_started",
                "episode_id": "ep",
                "trajectory_id": "traj",
                "task_spec": task_spec.model_dump(),
                "seed": 1,
            }
        )
        journal.append(
            {
                "event_type": "trial_scored",
                "trial_id": "trial",
                "reward": _make_reward().model_dump(),
                "sandbox_backend": "harbor",
            }
        )
        journal.append(
            {
                "event_type": "episode_finalised",
                "episode_id": "ep",
                "terminal_transition_rewards": [],
                "terminal_turn_discounts": [],
                "terminal_stop_reason": "completed",
                "created_at_iso": "2026-04-22T00:00:00Z",
            }
        )
        with pytest.raises(ProjectionError, match="scored without"):
            events_to_trajectory(journal.read_events())

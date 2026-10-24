"""Idempotence and determinism of projectors.

Projecting the same event stream twice MUST yield equal Trajectories. This
is the invariant that makes the WAL swappable with any downstream backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ppl_synth_data.eval.persistence import (
    EpisodeJournal,
    FsyncMode,
    events_to_trajectory,
    trajectory_to_trace_record,
    trajectory_to_trial_record,
)
from tests.eval.test_episode_journal_roundtrip import _emit_complete_episode


@pytest.fixture
def finalised_journal(tmp_path: Path) -> EpisodeJournal:
    journal = EpisodeJournal.for_episode(tmp_path, "ep-001", fsync_mode=FsyncMode.never)
    _emit_complete_episode(journal, n_turns=3)
    return journal


class TestTrajectoryProjectionIdempotent:
    def test_project_twice_equal(self, finalised_journal: EpisodeJournal) -> None:
        events = finalised_journal.read_events()
        first = events_to_trajectory(events)
        second = events_to_trajectory(events)
        assert first == second

    def test_project_survives_reread(self, finalised_journal: EpisodeJournal) -> None:
        first_events = finalised_journal.read_events()
        second_events = finalised_journal.read_events()
        first = events_to_trajectory(first_events)
        second = events_to_trajectory(second_events)
        assert first == second

    def test_trajectory_is_hashable_equivalent_under_dump(
        self, finalised_journal: EpisodeJournal
    ) -> None:
        events = finalised_journal.read_events()
        first = events_to_trajectory(events).model_dump(mode="json")
        second = events_to_trajectory(events).model_dump(mode="json")
        import json

        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


class TestLegacyViewProjections:
    def test_trace_record_projection_idempotent(self, finalised_journal: EpisodeJournal) -> None:
        traj = events_to_trajectory(finalised_journal.read_events())
        first = trajectory_to_trace_record(traj)
        second = trajectory_to_trace_record(traj)
        assert first == second

    def test_trace_record_includes_every_message(self, finalised_journal: EpisodeJournal) -> None:
        """TraceStep is per-message, not per-turn (matches platform schema)."""
        traj = events_to_trajectory(finalised_journal.read_events())
        trace = trajectory_to_trace_record(traj)
        expected_message_count = sum(
            len(turn.messages_input) + 1  # +1 for assistant_message
            for turn in traj.episode.turns
        )
        assert len(trace.steps) == expected_message_count

    def test_trial_record_carries_reward_provenance(
        self, finalised_journal: EpisodeJournal
    ) -> None:
        traj = events_to_trajectory(finalised_journal.read_events())
        trial_record = trajectory_to_trial_record(traj)
        metadata = trial_record.metadata
        assert metadata["reward_source"] == traj.trial.reward.reward_source
        assert metadata["evidence_kind"] == traj.trial.reward.evidence_kind.value
        assert metadata["truth_owner"] == traj.trial.reward.truth_owner.value
        assert metadata["producer"] == traj.trial.reward.producer

    def test_trial_record_reward_is_total(self, finalised_journal: EpisodeJournal) -> None:
        traj = events_to_trajectory(finalised_journal.read_events())
        trial_record = trajectory_to_trial_record(traj)
        assert trial_record.reward == traj.trial.reward.total

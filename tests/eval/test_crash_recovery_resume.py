"""Crash-recovery: the journal survives torn writes and reveals durable state.

Three realistic crash points:
1. Mid-write (partial line in episode.events.jsonl): journal must skip the
   partial tail, not reject the whole file.
2. Events durable but no trajectory.json: finalised=False; caller resumes.
3. Events durable AND trajectory.json present: finalised=True; replay matches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ppl_synth_data.eval.persistence import (
    EpisodeJournal,
    FsyncMode,
    events_to_trajectory,
    replay_episode,
)
from ppl_synth_data.eval.persistence.projectors import ProjectionError
from ppl_synth_data.eval.persistence.replay import (
    ReplayMismatchError,
    replay_prefix,
)
from ppl_synth_data.schemas.trajectory import Trajectory
from tests.eval.test_episode_journal_roundtrip import _emit_complete_episode


@pytest.fixture
def fresh_journal(tmp_path: Path) -> EpisodeJournal:
    return EpisodeJournal.for_episode(tmp_path, "ep-001", fsync_mode=FsyncMode.never)


class TestPartialLineTolerance:
    def test_journal_skips_trailing_partial_line(
        self, fresh_journal: EpisodeJournal, tmp_path: Path
    ) -> None:
        _emit_complete_episode(fresh_journal, n_turns=2)
        # Simulate a torn write by appending a partial line (no newline).
        with open(fresh_journal.paths.events_path, "a", encoding="utf-8") as fh:
            fh.write('{"event_type": "turn_recorded", "sequence": 999, "tim')

        # Re-opening the journal recomputes next_sequence from durable events
        # only; the partial line is invisible.
        reopened = EpisodeJournal(fresh_journal.paths, fsync_mode=FsyncMode.never)
        durable_events = reopened.read_events()
        # All full lines survived
        assert len(durable_events) > 0
        assert durable_events[-1].event_type == "episode_finalised"

    def test_next_sequence_recovers_from_partial_tail(self, fresh_journal: EpisodeJournal) -> None:
        _emit_complete_episode(fresh_journal, n_turns=1)
        last_seq = fresh_journal.next_sequence - 1
        with open(fresh_journal.paths.events_path, "a", encoding="utf-8") as fh:
            fh.write('{"event_type": "turn_recorded", "seque')

        reopened = EpisodeJournal(fresh_journal.paths, fsync_mode=FsyncMode.never)
        # The partial line did not advance the sequence counter.
        assert reopened.next_sequence == last_seq + 1


class TestRecoveryState:
    def test_fresh_journal_not_finalised(self, fresh_journal: EpisodeJournal) -> None:
        events, finalised = fresh_journal.crash_recover()
        assert events == []
        assert finalised is False

    def test_partial_stream_not_finalised(self, fresh_journal: EpisodeJournal) -> None:
        # Emit some events but never finalise
        from ppl_synth_data.eval.persistence.journal import EpisodeStartedEvent  # noqa: F401
        from tests.eval.test_trajectory_schema import _make_task_spec

        task_spec = _make_task_spec()
        fresh_journal.append(
            {
                "event_type": "episode_started",
                "episode_id": "ep-001",
                "trajectory_id": "traj-001",
                "task_spec": task_spec.model_dump(),
                "seed": 42,
            }
        )
        events, finalised = fresh_journal.crash_recover()
        assert len(events) == 1
        assert finalised is False
        # Projecting an incomplete stream is an error, not silent.
        with pytest.raises(ProjectionError, match="episode_finalised"):
            events_to_trajectory(events)

    def test_finalised_journal_replays_equal(self, fresh_journal: EpisodeJournal) -> None:
        original = _emit_complete_episode(fresh_journal, n_turns=2)
        replayed = replay_episode(fresh_journal)
        assert replayed == original

    def test_replay_detects_tampered_trajectory_json(self, fresh_journal: EpisodeJournal) -> None:
        _emit_complete_episode(fresh_journal, n_turns=1)
        # Tamper: rewrite trajectory.json with a different trajectory_id.
        import json

        data = json.loads(fresh_journal.paths.trajectory_path.read_text(encoding="utf-8"))
        data["trajectory_id"] = "tampered-id"
        fresh_journal.paths.trajectory_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ReplayMismatchError):
            replay_episode(fresh_journal)


class TestReplayPrefix:
    def test_replay_prefix_empty(self) -> None:
        assert replay_prefix([]) == (-1, 0)

    def test_replay_prefix_reports_last_sequence(self, fresh_journal: EpisodeJournal) -> None:
        _emit_complete_episode(fresh_journal, n_turns=1)
        events = fresh_journal.read_events()
        last_seq, n = replay_prefix(events)
        assert n == len(events)
        assert last_seq == events[-1].sequence


class TestNonMonotonicSequenceDetection:
    def test_corrupt_non_monotonic_raises_on_open(
        self, fresh_journal: EpisodeJournal, tmp_path: Path
    ) -> None:
        _emit_complete_episode(fresh_journal, n_turns=1)
        # Corrupt by appending a valid turn_recorded event with an
        # out-of-order sequence (0, when the stream already has seq >= some n).
        from tests.eval.test_trajectory_schema import _make_turn

        turn = _make_turn(0, 2)
        import json

        line = (
            json.dumps(
                {
                    "event_type": "turn_recorded",
                    "sequence": 0,  # out of order: already seen in the stream
                    "timestamp_iso": "2026-04-22T00:00:00Z",
                    "turn": turn.model_dump(mode="json"),
                }
            )
            + "\n"
        )
        with open(fresh_journal.paths.events_path, "a", encoding="utf-8") as fh:
            fh.write(line)
        with pytest.raises(RuntimeError, match="non-monotonic"):
            EpisodeJournal(fresh_journal.paths, fsync_mode=FsyncMode.never)


class TestFinalisedMarkerOrdering:
    def test_trajectory_json_exists_when_marker_exists(self, fresh_journal: EpisodeJournal) -> None:
        _emit_complete_episode(fresh_journal, n_turns=1)
        assert fresh_journal.finalised
        assert fresh_journal.paths.trajectory_path.exists()

    def test_trajectory_json_validates_under_schema(self, fresh_journal: EpisodeJournal) -> None:
        original = _emit_complete_episode(fresh_journal, n_turns=1)
        import json

        disk = Trajectory.model_validate(
            json.loads(fresh_journal.paths.trajectory_path.read_text(encoding="utf-8"))
        )
        assert disk == original

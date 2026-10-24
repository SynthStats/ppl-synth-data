"""Deterministic replay from the event journal.

The WAL lets a crashed or resumed episode come back to life: read the durable events, rebuild
the in-memory state, and continue from the next expected action. This module covers the read-
only replay path; re-executing an LLM/agent rollout deterministically belongs to the
orchestrator, which consults the seed ledger.

Two guarantees:
1. events_to_trajectory(events) is idempotent and deterministic.
2. A replayed Trajectory equals the originally-finalised one, field-by-field.
   ReplayMismatchError is raised on any divergence so silent corruption is
   impossible.
"""

from __future__ import annotations

from ppl_synth_data.eval.persistence.journal import (
    EpisodeEventUnion,
    EpisodeJournal,
)
from ppl_synth_data.eval.persistence.projectors import events_to_trajectory
from ppl_synth_data.schemas.trajectory import Trajectory


class ReplayMismatchError(RuntimeError):
    """Raised when replay produces a Trajectory != the finalised one."""


def replay_episode(journal: EpisodeJournal) -> Trajectory:
    """Reconstruct the Trajectory from a journal's durable events.

    Raises if the journal has no finalised trajectory.json yet AND the event
    stream is incomplete.
    """
    events = journal.read_events()
    trajectory = events_to_trajectory(events)
    if journal.finalised:
        _assert_matches_crystallised(journal, trajectory)
    return trajectory


def _assert_matches_crystallised(journal: EpisodeJournal, replayed: Trajectory) -> None:
    """Post-finalisation invariant: replayed == on-disk trajectory.json.

    If this fires, the journal or trajectory.json has been tampered with; a
    caller should treat the episode as corrupt and quarantine it.
    """
    import json

    disk = Trajectory.model_validate(
        json.loads(journal.paths.trajectory_path.read_text(encoding="utf-8"))
    )
    if disk != replayed:
        raise ReplayMismatchError(
            f"replay mismatch for episode {replayed.episode.episode_id}: "
            f"the event-derived Trajectory differs from trajectory.json. "
            f"The journal and the crystallised view have diverged; quarantine the run."
        )


def replay_prefix(events: list[EpisodeEventUnion]) -> tuple[int, int]:
    """Return (last_sequence, event_count) for recovery bookkeeping.

    Useful for an orchestrator deciding from which sequence to resume. An
    empty stream returns (-1, 0).
    """
    if not events:
        return -1, 0
    return events[-1].sequence, len(events)


__all__ = [
    "ReplayMismatchError",
    "replay_episode",
    "replay_prefix",
]

"""Event-sourced persistence: single source of truth for trajectories.

The local event journal is authoritative. Parquet, leaderboard, ATIF, OTel, TraceRecord,
TrialRecord; everything else is a deterministic projection from the journal. This is load-
bearing for the replay, observability-swap, and enterprise-isolation requirements.

Layout per episode:

    <run_dir>/<episode_id>/
        episode.events.jsonl     # append-only, ordered, fsync per write
        trial.events.jsonl       # terminal-trial events (may be absent)
        artifact.events.jsonl    # heavy-artefact pointers (may be absent)
        trajectory.json          # crystallised view; exists iff finalised
        .finalised               # sentinel marker; presence implies atomic commit

Crash-recovery semantics: on startup we read episode.events.jsonl; the last event with a valid
JSON line bounds the recoverable state. Any trailing partial line is ignored (journal reader is
tolerant of short tails).
"""

from ppl_synth_data.eval.persistence.journal import (
    EpisodeEvent,
    EpisodeFinalisedEvent,
    EpisodeJournal,
    EpisodeStartedEvent,
    FsyncMode,
    JournalCorruptionError,
    JournalPaths,
    MetricRecordedEvent,
    ObservationRecordedEvent,
    StepRecordedEvent,
    ToolCallRecordedEvent,
    TrialScoredEvent,
    TrialSubmittedEvent,
    TurnRecordedEvent,
)
from ppl_synth_data.eval.persistence.projectors import (
    events_to_trajectory,
    trajectory_to_trace_record,
    trajectory_to_trial_record,
)
from ppl_synth_data.eval.persistence.replay import (
    ReplayMismatchError,
    replay_episode,
)

__all__ = [
    "EpisodeEvent",
    "EpisodeJournal",
    "EpisodeStartedEvent",
    "EpisodeFinalisedEvent",
    "FsyncMode",
    "JournalCorruptionError",
    "JournalPaths",
    "MetricRecordedEvent",
    "ObservationRecordedEvent",
    "ReplayMismatchError",
    "StepRecordedEvent",
    "ToolCallRecordedEvent",
    "TrialScoredEvent",
    "TrialSubmittedEvent",
    "TurnRecordedEvent",
    "events_to_trajectory",
    "trajectory_to_trace_record",
    "trajectory_to_trial_record",
    "replay_episode",
]

"""Append-only episode event journal.

Events are emitted in causal order by the orchestrator and persisted via
EpisodeJournal.append(). The journal is tolerant of trailing-partial lines
(common after a crash between fsync and close) so recovery never loses the
durable prefix.

Design rules:
- One event per JSONL line. No multi-line records.
- Sequence numbers are monotonic, assigned by the journal (caller passes
  event payloads; the journal stamps sequence + timestamp).
- fsync per append (configurable via EpisodeJournal.fsync_mode for tests).
- No in-memory cache of events that rivals the file; reload is always a
  file read.
- Events are Pydantic discriminated union; downstream projectors dispatch on
  event_type.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ppl_synth_data.schemas.trajectory import (
    BeliefSnapshot,
    BeliefUpdate,
    EvalTaskSpec,
    MetricBreakdown,
    Observation,
    RewardRecord,
    Step,
    ToolCallRecord,
    Trajectory,
    Turn,
)

JOURNAL_SCHEMA_VERSION: Final[str] = "1.0.0"


class FsyncMode(StrEnum):
    """Durability policy for append()."""

    per_write = "per_write"  # fsync after every event (default; safest)
    per_batch = "per_batch"  # caller manages, via journal.flush_sync()
    never = "never"  # tests only


class JournalCorruptionError(RuntimeError):
    """Raised when a complete journal line contains invalid JSON.

    An incomplete final line is skipped. A line terminated by a newline must
    decode as JSON so that replay preserves event order and sequence gaps.
    """


# -----------------------------------------------------------------------------
# Event records
# -----------------------------------------------------------------------------

_EVENT_CONFIG: Final[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class _BaseEvent(BaseModel):
    model_config = _EVENT_CONFIG
    sequence: int
    timestamp_iso: str


class EpisodeStartedEvent(_BaseEvent):
    event_type: Literal["episode_started"] = "episode_started"
    episode_id: str
    trajectory_id: str
    task_spec: EvalTaskSpec
    seed: int


class TurnRecordedEvent(_BaseEvent):
    event_type: Literal["turn_recorded"] = "turn_recorded"
    turn: Turn


class StepRecordedEvent(_BaseEvent):
    event_type: Literal["step_recorded"] = "step_recorded"
    step: Step


class ObservationRecordedEvent(_BaseEvent):
    event_type: Literal["observation_recorded"] = "observation_recorded"
    observation: Observation


class BeliefSnapshotRecordedEvent(_BaseEvent):
    event_type: Literal["belief_snapshot_recorded"] = "belief_snapshot_recorded"
    snapshot: BeliefSnapshot


class BeliefUpdateRecordedEvent(_BaseEvent):
    event_type: Literal["belief_update_recorded"] = "belief_update_recorded"
    update: BeliefUpdate


class ToolCallRecordedEvent(_BaseEvent):
    event_type: Literal["tool_call_recorded"] = "tool_call_recorded"
    tool_call: ToolCallRecord


class MetricRecordedEvent(_BaseEvent):
    event_type: Literal["metric_recorded"] = "metric_recorded"
    metric: MetricBreakdown


class TrialSubmittedEvent(_BaseEvent):
    event_type: Literal["trial_submitted"] = "trial_submitted"
    trial_id: str
    submitted_program: str | None
    language: str


class TrialScoredEvent(_BaseEvent):
    event_type: Literal["trial_scored"] = "trial_scored"
    trial_id: str
    reward: RewardRecord
    sandbox_backend: str | None
    verifier_stdout: str = ""
    verifier_stderr: str = ""
    exit_code: int = 0
    wall_seconds: float = 0.0
    artefacts: dict[str, Any] = Field(default_factory=dict)


class EpisodeFinalisedEvent(_BaseEvent):
    event_type: Literal["episode_finalised"] = "episode_finalised"
    episode_id: str
    terminal_transition_rewards: list[float]
    terminal_turn_discounts: list[float]
    terminal_stop_reason: str
    created_at_iso: str
    dependency_digests: dict[str, str] = Field(default_factory=dict)
    run_metadata: dict[str, Any] = Field(default_factory=dict)


EpisodeEventUnion = (
    EpisodeStartedEvent
    | TurnRecordedEvent
    | StepRecordedEvent
    | ObservationRecordedEvent
    | BeliefSnapshotRecordedEvent
    | BeliefUpdateRecordedEvent
    | ToolCallRecordedEvent
    | MetricRecordedEvent
    | TrialSubmittedEvent
    | TrialScoredEvent
    | EpisodeFinalisedEvent
)

EpisodeEvent = Annotated[EpisodeEventUnion, Field(discriminator="event_type")]

_EVENT_ADAPTER: Final[TypeAdapter] = TypeAdapter(EpisodeEvent)


# -----------------------------------------------------------------------------
# Journal
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class JournalPaths:
    """Per-episode on-disk layout."""

    episode_dir: Path

    @property
    def events_path(self) -> Path:
        return self.episode_dir / "episode.events.jsonl"

    @property
    def trajectory_path(self) -> Path:
        return self.episode_dir / "trajectory.json"

    @property
    def finalised_marker(self) -> Path:
        return self.episode_dir / ".finalised"

    def ensure(self) -> None:
        self.episode_dir.mkdir(parents=True, exist_ok=True)


class EpisodeJournal:
    """Append-only durable log of events for a single episode.

    Use `append()` during rollout and `read_events()` during recovery or for
    projecting into Trajectory / TraceRecord / TrialRecord views.

    The journal does NOT hold an in-memory cache of the entire event stream;
    each `read_events()` call re-reads the file. That is deliberate: the file
    is the source of truth and any in-memory shadow invites drift.
    """

    def __init__(
        self,
        paths: JournalPaths,
        *,
        fsync_mode: FsyncMode = FsyncMode.per_write,
    ) -> None:
        self._paths = paths
        self._fsync_mode = fsync_mode
        self._paths.ensure()
        self._next_sequence: int = self._compute_next_sequence()

    @classmethod
    def for_episode(
        cls,
        run_dir: Path,
        episode_id: str,
        *,
        fsync_mode: FsyncMode = FsyncMode.per_write,
    ) -> EpisodeJournal:
        return cls(
            JournalPaths(episode_dir=run_dir / episode_id),
            fsync_mode=fsync_mode,
        )

    @property
    def paths(self) -> JournalPaths:
        return self._paths

    @property
    def finalised(self) -> bool:
        return self._paths.finalised_marker.exists()

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    # --- write -----------------------------------------------------------------

    def append(self, event_payload: dict[str, Any] | EpisodeEventUnion) -> EpisodeEventUnion:
        """Append one event. Returns the fully-stamped event for inspection.

        If `event_payload` is a dict, the journal fills in sequence + timestamp.
        If it's already an event instance, we assert its sequence matches and
        its timestamp is non-empty (caller responsibility).
        """
        if isinstance(event_payload, dict):
            stamped = dict(event_payload)
            stamped["sequence"] = self._next_sequence
            stamped["timestamp_iso"] = _now_iso()
            event = _EVENT_ADAPTER.validate_python(stamped)
        else:
            if event_payload.sequence != self._next_sequence:
                raise ValueError(
                    f"sequence mismatch: journal expects {self._next_sequence}, "
                    f"caller passed {event_payload.sequence}"
                )
            event = event_payload

        line = _EVENT_ADAPTER.dump_json(event).decode("utf-8")
        if "\n" in line:
            raise RuntimeError("serialized event contains newline; journal invariant violated")

        with open(self._paths.events_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            if self._fsync_mode is FsyncMode.per_write:
                fh.flush()
                os.fsync(fh.fileno())

        self._next_sequence += 1
        return event

    def flush_sync(self) -> None:
        """Force an fsync on the event file (no-op if file doesn't exist yet)."""
        if not self._paths.events_path.exists():
            return
        fd = os.open(self._paths.events_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    # --- read ------------------------------------------------------------------

    def read_events(self) -> list[EpisodeEventUnion]:
        """Read all durable events. Trailing partial lines are skipped."""
        return list(self._iter_events())

    def _iter_events(self) -> Iterator[EpisodeEventUnion]:
        path = self._paths.events_path
        if not path.exists():
            return
        with open(path, "rb") as fh:
            for raw_line in fh:
                if not raw_line.endswith(b"\n"):
                    # Trailing partial line from a crashed write; skip.
                    continue
                line = raw_line.decode("utf-8").rstrip("\n")
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    # A fully-terminated line that does not decode is a real
                    # corruption signal (disk error, concurrent writer, or
                    # tampering); not a torn tail. Raise loudly rather than
                    # silently drop subsequent events: a gap would break
                    # sequence-based invariants in `events_to_trajectory`.
                    raise JournalCorruptionError(
                        f"corrupt event line in {path}: {exc}. "
                        f"Journal is not recoverable; quarantine the episode."
                    ) from exc
                yield _EVENT_ADAPTER.validate_python(parsed)

    def _compute_next_sequence(self) -> int:
        last = -1
        for event in self._iter_events():
            if event.sequence <= last:
                raise RuntimeError(
                    f"non-monotonic sequence in {self._paths.events_path}: "
                    f"last={last} got={event.sequence}"
                )
            last = event.sequence
        return last + 1

    # --- finalise --------------------------------------------------------------

    def finalise(self, trajectory: Trajectory) -> Path:
        """Atomically write trajectory.json and the finalised marker.

        Callers should have already appended an EpisodeFinalisedEvent to the
        journal (the event captures the intent-to-finalise; this method crystallises
        the projection). The order is intentional: event first, crystallisation
        second, so recovery after a mid-finalise crash sees the intent and can
        replay deterministic projection.
        """
        tmp_path = self._paths.trajectory_path.with_suffix(".json.tmp")
        data = trajectory.model_dump(mode="json")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, self._paths.trajectory_path)

        # Finalised sentinel AFTER trajectory.json so its presence implies the
        # projection exists.
        self._paths.finalised_marker.write_text(
            json.dumps({"finalised_at_iso": _now_iso(), "trajectory_id": trajectory.trajectory_id}),
            encoding="utf-8",
        )
        return self._paths.trajectory_path

    def crash_recover(self) -> tuple[list[EpisodeEventUnion], bool]:
        """Return (durable events, finalised?). No side effects.

        If finalised is True, trajectory.json is guaranteed to exist and be
        the authoritative crystallised view.
        """
        return self.read_events(), self.finalised


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_event_id() -> str:
    """Convenience UUID4 generator for callers that want string IDs."""
    return str(uuid.uuid4())


__all__ = [
    "BeliefSnapshotRecordedEvent",
    "BeliefUpdateRecordedEvent",
    "EpisodeEvent",
    "EpisodeEventUnion",
    "EpisodeFinalisedEvent",
    "EpisodeJournal",
    "EpisodeStartedEvent",
    "FsyncMode",
    "JOURNAL_SCHEMA_VERSION",
    "JournalCorruptionError",
    "JournalPaths",
    "MetricRecordedEvent",
    "ObservationRecordedEvent",
    "StepRecordedEvent",
    "ToolCallRecordedEvent",
    "TrialScoredEvent",
    "TrialSubmittedEvent",
    "TurnRecordedEvent",
    "new_event_id",
]

"""Deterministic projections from the event journal to views.

Every function here is pure: same event list -> same output. This is the
invariant that makes the journal swappable with any downstream backend.

Three canonical projections:
- events_to_trajectory(events) -> Trajectory (the crystallised authoritative view)
- trajectory_to_trace_record(traj) -> TraceRecord (synthstats_schemas.trace view)
- trajectory_to_trial_record(traj) -> TrialRecord (synthstats_schemas.trace view)

Additional projections (Parquet row, OTel spans, Langfuse trace, SkyRL
payload) live in their own modules but all read from Trajectory, never from
raw events. This keeps the projection layer a fan-out, not a DAG.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ppl_synth_data.eval.persistence.journal import (
    BeliefSnapshotRecordedEvent,
    BeliefUpdateRecordedEvent,
    EpisodeEventUnion,
    EpisodeFinalisedEvent,
    EpisodeStartedEvent,
    MetricRecordedEvent,
    ObservationRecordedEvent,
    StepRecordedEvent,
    ToolCallRecordedEvent,
    TrialScoredEvent,
    TrialSubmittedEvent,
    TurnRecordedEvent,
)
from ppl_synth_data.schemas.trajectory import (
    Episode,
    StopReason,
    Trajectory,
    Trial,
)

if TYPE_CHECKING:
    from synthstats_schemas.trace import TraceRecord, TrialRecord


class ProjectionError(ValueError):
    """Raised when the event stream is internally inconsistent."""


def events_to_trajectory(events: Sequence[EpisodeEventUnion]) -> Trajectory:
    """Project a durable event stream to a Trajectory.

    Events are consumed in order. The EpisodeFinalisedEvent must be present
    and must be the last event; its payload carries the terminal metadata
    needed to close the Episode.

    Raises ProjectionError if the stream is incomplete, out of order, or
    internally inconsistent (e.g. trial scored but never submitted).
    """
    if not events:
        raise ProjectionError("empty event stream")

    started: EpisodeStartedEvent | None = None
    finalised: EpisodeFinalisedEvent | None = None
    turns = []
    steps = []
    observations = []
    belief_snapshots = []
    belief_updates = []
    tool_calls = []
    metric_breakdowns = []
    trial_submitted: TrialSubmittedEvent | None = None
    trial_scored: TrialScoredEvent | None = None

    for event in events:
        if isinstance(event, EpisodeStartedEvent):
            if started is not None:
                raise ProjectionError("multiple episode_started events")
            started = event
        elif isinstance(event, TurnRecordedEvent):
            turns.append(event.turn)
        elif isinstance(event, StepRecordedEvent):
            steps.append(event.step)
        elif isinstance(event, ObservationRecordedEvent):
            observations.append(event.observation)
        elif isinstance(event, BeliefSnapshotRecordedEvent):
            belief_snapshots.append(event.snapshot)
        elif isinstance(event, BeliefUpdateRecordedEvent):
            belief_updates.append(event.update)
        elif isinstance(event, ToolCallRecordedEvent):
            tool_calls.append(event.tool_call)
        elif isinstance(event, MetricRecordedEvent):
            metric_breakdowns.append(event.metric)
        elif isinstance(event, TrialSubmittedEvent):
            if trial_submitted is not None:
                raise ProjectionError("multiple trial_submitted events")
            trial_submitted = event
        elif isinstance(event, TrialScoredEvent):
            if trial_scored is not None:
                raise ProjectionError("multiple trial_scored events")
            trial_scored = event
        elif isinstance(event, EpisodeFinalisedEvent):
            finalised = event
        else:  # pragma: no cover - Pydantic union is closed
            raise ProjectionError(f"unknown event type: {type(event).__name__}")

    if started is None:
        raise ProjectionError("no episode_started event")
    if finalised is None:
        raise ProjectionError("no episode_finalised event; stream is incomplete")
    if trial_scored is not None and trial_submitted is None:
        raise ProjectionError("trial_scored without trial_submitted")

    # Align transition_rewards and turn_discounts with the turns we saw.
    transition_rewards = list(finalised.terminal_transition_rewards)
    turn_discounts = list(finalised.terminal_turn_discounts)
    if len(transition_rewards) != len(turns):
        raise ProjectionError(
            f"finalised transition_rewards length {len(transition_rewards)} != "
            f"turns recorded {len(turns)}"
        )
    if len(turn_discounts) != len(turns):
        raise ProjectionError(
            f"finalised turn_discounts length {len(turn_discounts)} != turns recorded {len(turns)}"
        )

    terminal_stop_reason = StopReason(finalised.terminal_stop_reason)

    episode = Episode(
        episode_id=started.episode_id,
        turns=turns,
        steps=steps,
        observations=observations,
        belief_snapshots=belief_snapshots,
        belief_updates=belief_updates,
        tool_calls=tool_calls,
        metric_breakdowns=metric_breakdowns,
        transition_rewards=transition_rewards,
        turn_discounts=turn_discounts,
        terminal_stop_reason=terminal_stop_reason,
    )

    trial = _build_trial(trial_submitted, trial_scored)

    return Trajectory(
        trajectory_id=started.trajectory_id,
        task_spec=started.task_spec,
        episode=episode,
        trial=trial,
        seed=started.seed,
        created_at_iso=finalised.created_at_iso,
        dependency_digests=dict(finalised.dependency_digests),
        run_metadata=dict(finalised.run_metadata),
    )


def _build_trial(
    submitted: TrialSubmittedEvent | None,
    scored: TrialScoredEvent | None,
) -> Trial:
    if submitted is None and scored is None:
        raise ProjectionError(
            "episode finalised without any trial events; Trajectory needs a Trial. "
            "For error-terminated episodes, emit an explicit TrialSubmittedEvent with "
            "submitted_program=None followed by a TrialScoredEvent with a reward_source "
            "registered as heuristic (e.g. 'heuristic.no_submission')."
        )
    if scored is None:
        # Submitted but not scored yet: can happen if finalise races the scorer.
        # Treat as an error-terminated trial with an explicit 'unknown' scoring
        # would be a silent default; instead refuse.
        raise ProjectionError(
            f"trial {submitted.trial_id if submitted else '?'} submitted but not scored"
        )
    if submitted is None:
        # Scored without a preceding submission: defensive check. The outer
        # `events_to_trajectory` already rejects this stream shape, but the
        # projector accepts any source of events; make the contract explicit
        # here too so `_build_trial` cannot silently synthesise a Trial with
        # an empty `language` field.
        raise ProjectionError(
            f"trial {scored.trial_id} scored without a preceding trial_submitted event"
        )
    trial_id = scored.trial_id
    if submitted.trial_id != trial_id:
        raise ProjectionError(
            f"trial_id mismatch between submitted ({submitted.trial_id}) and scored ({trial_id})"
        )
    return Trial(
        trial_id=trial_id,
        submitted_program=submitted.submitted_program,
        language=submitted.language,
        sandbox_backend=scored.sandbox_backend,
        reward=scored.reward,
        verifier_stdout=scored.verifier_stdout,
        verifier_stderr=scored.verifier_stderr,
        exit_code=scored.exit_code,
        wall_seconds=scored.wall_seconds,
        artefacts=dict(scored.artefacts),
    )


def trajectory_to_trace_record(trajectory: Trajectory) -> TraceRecord:
    """Schemas-shaped view: synthstats_schemas.trace.TraceRecord.

    A TraceStep represents ONE message, not one turn. We flatten every turn's
    messages_input + assistant_message into step entries. Observations, tool
    calls, and metrics land in `artifacts`. synthstats_schemas.trace.TraceStep
    has no first-class `tool_call_id` field, so it is folded into `metadata`.
    """
    from synthstats_schemas.trace import TraceRecord, TraceStep

    steps: list[TraceStep] = []
    step_counter = 0
    for turn in trajectory.episode.turns:
        for msg in (*turn.messages_input, turn.assistant_message):
            step_metadata: dict[str, Any] = {
                "turn_index": turn.turn_index,
                "parsed_action_type": turn.parsed_action_type.value,
                "stop_reason": turn.stop_reason.value,
            }
            if msg.tool_call_id is not None:
                step_metadata["tool_call_id"] = msg.tool_call_id
            steps.append(
                TraceStep(
                    step_id=f"{trajectory.episode.episode_id}:t{turn.turn_index}:s{step_counter}",
                    role=msg.role,
                    content=msg.content,
                    reasoning_content=msg.reasoning_content,
                    metadata=step_metadata,
                )
            )
            step_counter += 1

    artifacts = {
        "observations": [o.model_dump(mode="json") for o in trajectory.episode.observations],
        "tool_calls": [tc.model_dump(mode="json") for tc in trajectory.episode.tool_calls],
        "metric_breakdowns": [
            m.model_dump(mode="json") for m in trajectory.episode.metric_breakdowns
        ],
    }

    return TraceRecord(
        trace_id=trajectory.episode.episode_id,
        task_id=trajectory.task_spec.task_id,
        trial_id=trajectory.trial.trial_id,
        steps=steps,
        artifacts=artifacts,
        metadata={
            "trajectory_id": trajectory.trajectory_id,
            "seed": trajectory.seed,
            "schema_version": trajectory.schema_version,
        },
    )


def trajectory_to_trial_record(trajectory: Trajectory) -> TrialRecord:
    """Schemas-shaped view: synthstats_schemas.trace.TrialRecord."""
    from synthstats_schemas.trace import RuntimeDescriptor, TrialRecord

    runtime = RuntimeDescriptor(
        name=trajectory.trial.language,
        execution_backend=trajectory.trial.sandbox_backend,
        metadata={
            "exit_code": trajectory.trial.exit_code,
            "wall_seconds": trajectory.trial.wall_seconds,
        },
    )

    return TrialRecord(
        trial_id=trajectory.trial.trial_id,
        task_id=trajectory.task_spec.task_id,
        language=trajectory.trial.language,
        program=trajectory.trial.submitted_program or "",
        reward=trajectory.trial.reward.total,
        runtime=runtime,
        trace_id=trajectory.episode.episode_id,
        artifacts=dict(trajectory.trial.artefacts),
        metadata={
            "trajectory_id": trajectory.trajectory_id,
            "reward_source": trajectory.trial.reward.reward_source,
            "evidence_kind": trajectory.trial.reward.evidence_kind.value,
            "truth_owner": trajectory.trial.reward.truth_owner.value,
            "producer": trajectory.trial.reward.producer,
        },
    )


__all__ = [
    "ProjectionError",
    "events_to_trajectory",
    "trajectory_to_trace_record",
    "trajectory_to_trial_record",
]

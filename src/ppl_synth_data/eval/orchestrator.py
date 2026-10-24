"""Run deterministic scalar episodes and persist their events.

The loop emits an episode-start event, requests agent turns, executes environment queries or
sandbox trials, and emits an episode-finalised event on submission or a stop condition. The
journal projects these events into a trajectory. Exceptions from turn generation or sandbox
execution produce an error trial, a terminal metric and a finalised event.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ppl_synth_data.eval.agents.base import EpisodeAgent
from ppl_synth_data.eval.persistence import (
    EpisodeJournal,
    events_to_trajectory,
)
from ppl_synth_data.eval.sandbox.trivial import TrivialSandbox, TrivialSandboxResult
from ppl_synth_data.eval.seed_ledger import SeedLedger
from ppl_synth_data.eval.toy_env import ToyScalarEnv
from ppl_synth_data.schemas.trajectory import (
    ActionType,
    EvalTaskSpec,
    Message,
    MetricBreakdown,
    MetricSchedule,
    Observation,
    RewardRecord,
    Step,
    StopReason,
    Trajectory,
    Turn,
)

# Type alias: the minimal surface an env must offer the orchestrator.
EnvLike = ToyScalarEnv  # Scalar environment interface.


@dataclass
class OrchestratorConfig:
    """Stop-condition bundle."""

    max_turns: int = 10
    trajectory_id: str = "traj-toy-001"
    episode_id: str = "ep-toy-001"
    clock: Callable[[], str] | None = None  # injectable; None -> journal default
    # Optional event emitter.
    otel_emitter: Callable[[str, dict[str, Any]], None] | None = None


@dataclass
class OrchestratorResult:
    trajectory: Trajectory
    journal: EpisodeJournal
    seed_ledger_snapshot: dict[str, int] = field(default_factory=dict)


class ToyOrchestrator:
    """Episode loop for ToyScalarEnv, scripted actions and a fixed-reward test sandbox."""

    def __init__(
        self,
        *,
        env: EnvLike,
        agent: EpisodeAgent,
        task_spec: EvalTaskSpec,
        journal: EpisodeJournal,
        seed_ledger: SeedLedger,
        sandbox: TrivialSandbox | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._env = env
        self._agent = agent
        self._task_spec = task_spec
        self._journal = journal
        self._seed_ledger = seed_ledger
        self._sandbox = sandbox if sandbox is not None else TrivialSandbox()
        self._config = config if config is not None else OrchestratorConfig()

    def run_episode(self) -> OrchestratorResult:
        seed_env = self._seed_ledger.for_subsystem("env")
        self._env.reset(seed_env)

        self._journal.append(
            {
                "event_type": "episode_started",
                "episode_id": self._config.episode_id,
                "trajectory_id": self._config.trajectory_id,
                "task_spec": self._task_spec.model_dump(),
                "seed": self._seed_ledger.root_seed,
            }
        )
        self._emit_otel("episode.started", {"episode_id": self._config.episode_id})

        messages = list(self._agent.initial_messages(self._task_spec))
        terminal_stop_reason = StopReason.length
        trial_phase: str = "none"  # "none" | "submitted" | "scored"
        error_info: dict[str, Any] | None = None

        try:
            for turn_index in range(self._config.max_turns):
                agent_seed = self._seed_ledger.for_subsystem(f"agent.turn_{turn_index}")
                try:
                    turn_result = self._agent.generate_turn(
                        task_spec=self._task_spec,
                        messages=messages,
                        turn_index=turn_index,
                        seed=agent_seed,
                    )
                except IndexError as exc:
                    # Scripted-style agents signal exhaustion by IndexError.
                    # Honour it as a stop_token terminus, not as a crash.
                    terminal_stop_reason = StopReason.stop_token
                    error_info = {"kind": "agent_exhausted", "detail": str(exc)}
                    break

                messages.append(turn_result.assistant_message)

                if turn_result.parsed_action_type is ActionType.submit_program:
                    trial_id = f"trial-{self._config.episode_id}"
                    action = turn_result.parsed_action or {}
                    program = str(action.get("program", ""))
                    language = str(action.get("language", self._task_spec.goal.goal_type))
                    turn = Turn(
                        turn_index=turn_index,
                        messages_input=messages[:-1],
                        assistant_message=turn_result.assistant_message,
                        token_ids=turn_result.token_ids,
                        token_logprobs=turn_result.token_logprobs,
                        loss_mask=turn_result.loss_mask,
                        eos_logprobs=turn_result.eos_logprobs,
                        parsed_action_type=turn_result.parsed_action_type,
                        parsed_action=turn_result.parsed_action,
                        stop_reason=turn_result.stop_reason,
                    )
                    self._journal.append({"event_type": "turn_recorded", "turn": turn.model_dump()})
                    self._emit_otel("turn.recorded", {"turn_index": turn_index})

                    self._journal.append(
                        {
                            "event_type": "trial_submitted",
                            "trial_id": trial_id,
                            "submitted_program": program,
                            "language": language,
                        }
                    )
                    trial_phase = "submitted"

                    try:
                        result = self._sandbox.execute(
                            language=language,
                            program=program,
                            task_package={},
                            timeout_seconds=60.0,
                        )
                    except Exception as exc:
                        # Sandbox failed after trial_submitted was emitted; the
                        # journal would be left with a submitted-but-not-scored
                        # trial, which `events_to_trajectory` refuses. Close the
                        # trial with an explicit error-reward scoring so the
                        # projection succeeds.
                        error_reward = self._emit_trial_scored_error_event(trial_id, exc)
                        # Advance trial_phase the MOMENT the trial_scored event
                        # is durably appended. The terminal-metric emission
                        # below may still raise (journal IO, Pydantic, etc.);
                        # if it does, the outer except catches, but the post-
                        # loop fallback must NOT re-emit a second trial_scored.
                        trial_phase = "scored"
                        terminal_stop_reason = StopReason.error
                        error_info = {"kind": "sandbox_error", "detail": str(exc)}
                        self._emit_terminal_metric(error_reward)
                    else:
                        self._emit_trial_scored_event(trial_id, result)
                        trial_phase = "scored"  # durable before metric, see above
                        terminal_stop_reason = StopReason.completed
                        self._emit_terminal_metric(result.reward_record)
                    break

                if turn_result.parsed_action_type is ActionType.query:
                    action = turn_result.parsed_action or {}
                    x_value = float(action.get("x", 0.0))
                    observation_id = f"obs-{self._config.episode_id}-{turn_index}"
                    state_before = self._env.state_digest()
                    y = self._env.run_experiment(x_value)
                    state_after = self._env.state_digest()

                    observation = Observation(
                        observation_id=observation_id,
                        step_index=turn_index,
                        triggering_turn_index=turn_index,
                        payload={"x": x_value, "y": y},
                        evidence_kind=self._env.observation_evidence_kind(),
                        truth_owner=self._env.observation_truth_owner(),
                        source="toy_scalar_env",
                    )
                    step = Step(
                        step_index=turn_index,
                        triggering_turn_index=turn_index,
                        observation_id=observation_id,
                        env_state_digest_before=state_before,
                        env_state_digest_after=state_after,
                        wall_seconds=0.0,
                    )
                    turn = Turn(
                        turn_index=turn_index,
                        messages_input=messages[:-1],
                        assistant_message=turn_result.assistant_message,
                        token_ids=turn_result.token_ids,
                        token_logprobs=turn_result.token_logprobs,
                        loss_mask=turn_result.loss_mask,
                        eos_logprobs=turn_result.eos_logprobs,
                        parsed_action_type=turn_result.parsed_action_type,
                        parsed_action=turn_result.parsed_action,
                        tool_call_ids=[],
                        triggered_step_indices=[step.step_index],
                        stop_reason=turn_result.stop_reason,
                    )
                    self._journal.append({"event_type": "turn_recorded", "turn": turn.model_dump()})
                    self._journal.append(
                        {
                            "event_type": "observation_recorded",
                            "observation": observation.model_dump(),
                        }
                    )
                    self._journal.append({"event_type": "step_recorded", "step": step.model_dump()})
                    messages.append(
                        Message(
                            role="tool",
                            content=f"observation: x={x_value:.4f} y={y:.4f}",
                        )
                    )
                    continue

                # No actionable turn: think / unknown. Record and continue.
                turn = Turn(
                    turn_index=turn_index,
                    messages_input=messages[:-1],
                    assistant_message=turn_result.assistant_message,
                    token_ids=turn_result.token_ids,
                    token_logprobs=turn_result.token_logprobs,
                    loss_mask=turn_result.loss_mask,
                    eos_logprobs=turn_result.eos_logprobs,
                    parsed_action_type=turn_result.parsed_action_type,
                    parsed_action=turn_result.parsed_action,
                    stop_reason=turn_result.stop_reason,
                )
                self._journal.append({"event_type": "turn_recorded", "turn": turn.model_dump()})
        except Exception as exc:
            # Unexpected failure in agent/env/parsing. Fall through to the
            # finalise block below, which ensures the episode always closes
            # with a valid Trial + EpisodeFinalised pair.
            terminal_stop_reason = StopReason.error
            error_info = {"kind": type(exc).__name__, "detail": str(exc)}

        # Bring the journal to a finalisable state regardless of which path
        # exited the loop.
        trial_id = f"trial-{self._config.episode_id}"
        if trial_phase == "none":
            # No submission happened (length timeout, agent exhaustion, or
            # mid-turn error before reaching submit).
            self._journal.append(
                {
                    "event_type": "trial_submitted",
                    "trial_id": trial_id,
                    "submitted_program": None,
                    "language": "",
                }
            )
            unknown_reward = self._emit_unknown_trial_scored_event(
                trial_id, terminal_stop_reason, error_info
            )
            self._emit_terminal_metric(unknown_reward)
        elif trial_phase == "submitted":
            # Trial was submitted but the trial_scored write itself failed
            # mid-episode. This IS reachable: if a journal IO error fires
            # during `_emit_trial_scored_event`, the outer except catches
            # without advancing trial_phase, and we must close the trial
            # here with an error reward.
            error_reward = self._emit_trial_scored_error_event(
                trial_id, RuntimeError("trial submitted but not scored")
            )
            self._emit_terminal_metric(error_reward)

        turn_count = self._turn_count_from_journal()
        self._journal.append(
            {
                "event_type": "episode_finalised",
                "episode_id": self._config.episode_id,
                "terminal_transition_rewards": [0.0] * turn_count,
                "terminal_turn_discounts": [1.0] * turn_count,
                "terminal_stop_reason": terminal_stop_reason.value,
                "created_at_iso": self._resolve_clock(),
                "dependency_digests": {},
                "run_metadata": {
                    "seed_ledger": self._seed_ledger.snapshot(),
                    "seed_ledger_order": self._seed_ledger.order(),
                    "error_info": error_info,
                },
            }
        )

        trajectory = events_to_trajectory(self._journal.read_events())
        self._journal.finalise(trajectory)
        self._emit_otel("episode.finalised", {"episode_id": self._config.episode_id})

        return OrchestratorResult(
            trajectory=trajectory,
            journal=self._journal,
            seed_ledger_snapshot=self._seed_ledger.snapshot(),
        )

    # --- helpers ---------------------------------------------------------------
    #
    # The trial-scored helpers come in an "event-only" flavour: each appends
    # the trial_scored event to the journal and returns the RewardRecord.
    # The caller MUST advance `trial_phase = "scored"` between the event
    # append and any subsequent work (e.g. _emit_terminal_metric), so that a
    # failure in the metric emission does not leave the journal in a state
    # where post-loop fallback re-emits a second trial_scored event.

    def _emit_trial_scored_event(self, trial_id: str, result: TrivialSandboxResult) -> None:
        self._journal.append(
            {
                "event_type": "trial_scored",
                "trial_id": trial_id,
                "reward": result.reward_record.model_dump(),
                "sandbox_backend": self._sandbox.backend_name,
                "verifier_stdout": result.stdout,
                "verifier_stderr": result.stderr,
                "exit_code": result.exit_code,
                "wall_seconds": result.wall_seconds,
                "artefacts": dict(result.artefacts),
            }
        )

    def _emit_terminal_metric(self, reward: RewardRecord) -> None:
        """Emit the terminal scalar reward and its provenance as a MetricBreakdown."""
        breakdown = MetricBreakdown(
            metric_id="trivial.trial_reward",
            metric_name="trial_reward",
            turn_index=None,
            value=reward.total,
            components=dict(reward.components),
            schedule=MetricSchedule.terminal,
            reward_source=reward.reward_source,
            evidence_kind=reward.evidence_kind,
            truth_owner=reward.truth_owner,
            producer="ppl_synth_data.eval.orchestrator",
        )
        self._journal.append({"event_type": "metric_recorded", "metric": breakdown.model_dump()})

    def _emit_trial_scored_error_event(self, trial_id: str, exc: Exception) -> RewardRecord:
        """Append a trial_scored event with an error reward.

        Event-only variant: returns the RewardRecord so the caller can pass
        it to `_emit_terminal_metric` AFTER advancing `trial_phase`.

        Accepts `Exception` (not `BaseException`) to mirror the outer
        `except Exception` clause: control-flow interrupts like
        `KeyboardInterrupt` / `SystemExit` must never be silently wrapped
        into a "scored" outcome.
        """
        from synthstats_schemas import get_reward_source

        spec = get_reward_source("heuristic.judge_exception")
        reward = RewardRecord(
            total=0.0,
            reward_source=spec.name,
            evidence_kind=spec.evidence_kind,
            truth_owner=spec.truth_owner,
            producer="ppl_synth_data.eval.orchestrator",
            info={"error_kind": type(exc).__name__, "error_detail": str(exc)},
        )
        self._journal.append(
            {
                "event_type": "trial_scored",
                "trial_id": trial_id,
                "reward": reward.model_dump(),
                "sandbox_backend": None,
                "verifier_stdout": "",
                "verifier_stderr": f"{type(exc).__name__}: {exc}",
                "exit_code": -1,
                "wall_seconds": 0.0,
                "artefacts": {},
            }
        )
        return reward

    def _emit_unknown_trial_scored_event(
        self,
        trial_id: str,
        stop_reason: StopReason,
        error_info: dict[str, Any] | None,
    ) -> RewardRecord:
        """Append a trial_scored event for a trial that never had a submission.

        Event-only variant: returns the RewardRecord; caller pairs with
        `_emit_terminal_metric` after advancing phase.
        """
        from synthstats_schemas import get_reward_source

        spec = get_reward_source("heuristic.no_submission")
        info: dict[str, Any] = {"stop_reason_without_submission": stop_reason.value}
        if error_info is not None:
            info["error_info"] = error_info
        reward = RewardRecord(
            total=0.0,
            reward_source=spec.name,
            evidence_kind=spec.evidence_kind,
            truth_owner=spec.truth_owner,
            producer="ppl_synth_data.eval.orchestrator",
            info=info,
        )
        self._journal.append(
            {
                "event_type": "trial_scored",
                "trial_id": trial_id,
                "reward": reward.model_dump(),
                "sandbox_backend": None,
            }
        )
        return reward

    def _resolve_clock(self) -> str:
        if self._config.clock is not None:
            return self._config.clock()
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def _emit_otel(self, span_name: str, attributes: dict[str, Any]) -> None:
        if self._config.otel_emitter is not None:
            self._config.otel_emitter(span_name, attributes)

    def _turn_count_from_journal(self) -> int:
        return sum(1 for e in self._journal.read_events() if e.event_type == "turn_recorded")


__all__ = [
    "EnvLike",
    "OrchestratorConfig",
    "OrchestratorResult",
    "ToyOrchestrator",
]

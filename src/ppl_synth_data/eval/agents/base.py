"""EpisodeAgent protocol and TurnResult dataclass.

An agent consumes the current message history and the seed for its turn;
it returns the assistant message, token-level data, and a parsed action.
The orchestrator is responsible for everything else (env stepping, metric
scheduling, WAL persistence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ppl_synth_data.schemas.trajectory import (
    ActionType,
    EvalTaskSpec,
    Message,
    StopReason,
)


@dataclass(frozen=True)
class TurnResult:
    """What an agent returns for one turn.

    token_ids/token_logprobs/loss_mask are per-token for the assistant's
    generation in this turn. eos_logprobs is optional (empty = vanilla TB).
    """

    assistant_message: Message
    token_ids: list[int]
    token_logprobs: list[float]
    loss_mask: list[bool]
    parsed_action_type: ActionType
    parsed_action: dict[str, Any] | None = None
    eos_logprobs: list[float] = field(default_factory=list)
    stop_reason: StopReason = StopReason.completed


@runtime_checkable
class EpisodeAgent(Protocol):
    """Minimal surface an orchestrator needs from an agent."""

    def initial_messages(self, task_spec: EvalTaskSpec) -> list[Message]:
        """Seed the conversation; typically just a system message."""
        ...

    def generate_turn(
        self,
        *,
        task_spec: EvalTaskSpec,
        messages: list[Message],
        turn_index: int,
        seed: int,
    ) -> TurnResult:
        """Produce one turn's worth of assistant output."""
        ...


__all__ = ["EpisodeAgent", "TurnResult"]

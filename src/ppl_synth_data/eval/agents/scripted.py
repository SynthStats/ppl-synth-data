"""Deterministic scripted agent for the toy kernel tests.

No LLM, no randomness (beyond the seed-derived rollout of scripted turns).
Useful for pinning the orchestrator loop against a fixed sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ppl_synth_data.eval.agents.base import EpisodeAgent, TurnResult
from ppl_synth_data.schemas.trajectory import (
    ActionType,
    EvalTaskSpec,
    Message,
    StopReason,
)


@dataclass(frozen=True)
class ScriptedTurn:
    """One scripted turn the agent will emit in order."""

    content: str
    action_type: ActionType
    action: dict[str, Any] | None = None
    token_ids: tuple[int, ...] = (1, 2, 3)
    token_logprobs: tuple[float, ...] = (-0.1, -0.1, -0.1)
    loss_mask: tuple[bool, ...] = (True, True, True)
    stop_reason: StopReason = StopReason.completed


class ScriptedAgent(EpisodeAgent):
    """Plays a fixed sequence of turns, verbatim.

    The orchestrator still drives the loop and emits events; the agent just
    provides deterministic turn content. Passing `system_message` replaces
    the default system seed message.
    """

    def __init__(
        self,
        turns: list[ScriptedTurn],
        *,
        system_message: str = "You are a test agent.",
    ) -> None:
        if not turns:
            raise ValueError("ScriptedAgent needs at least one turn")
        self._turns = list(turns)
        self._system_message = system_message

    def initial_messages(self, task_spec: EvalTaskSpec) -> list[Message]:
        return [Message(role="system", content=self._system_message)]

    def generate_turn(
        self,
        *,
        task_spec: EvalTaskSpec,
        messages: list[Message],
        turn_index: int,
        seed: int,
    ) -> TurnResult:
        if turn_index >= len(self._turns):
            raise IndexError(
                f"ScriptedAgent ran out of turns at index {turn_index}; "
                f"only {len(self._turns)} were provided."
            )
        scripted = self._turns[turn_index]
        return TurnResult(
            assistant_message=Message(role="assistant", content=scripted.content),
            token_ids=list(scripted.token_ids),
            token_logprobs=list(scripted.token_logprobs),
            loss_mask=list(scripted.loss_mask),
            parsed_action_type=scripted.action_type,
            parsed_action=dict(scripted.action) if scripted.action is not None else None,
            stop_reason=scripted.stop_reason,
        )


__all__ = ["ScriptedAgent", "ScriptedTurn"]

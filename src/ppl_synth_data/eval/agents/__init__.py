"""Episode agents: orchestrator-facing abstraction for one LLM seat.

The orchestrator is agent-agnostic. A minimal EpisodeAgent returns a
TurnResult per turn; concrete agents wrap LLM clients, scripted actions,
or multi-agent coordinators.
"""

from ppl_synth_data.eval.agents.base import EpisodeAgent, TurnResult
from ppl_synth_data.eval.agents.scripted import ScriptedAgent

__all__ = ["EpisodeAgent", "ScriptedAgent", "TurnResult"]

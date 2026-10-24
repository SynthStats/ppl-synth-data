"""Sandbox implementations for terminal-trial scoring.

The Sandbox interface is defined in eval.protocols. TrivialSandbox supplies a fixed reward for
deterministic orchestrator tests.
"""

from ppl_synth_data.eval.sandbox.trivial import TrivialSandbox, TrivialSandboxResult

__all__ = ["TrivialSandbox", "TrivialSandboxResult"]

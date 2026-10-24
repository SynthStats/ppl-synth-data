"""Episode orchestration, event persistence and deterministic replay.

ToyOrchestrator connects ToyScalarEnv, ScriptedAgent and TrivialSandbox to exercise the
trajectory schema, write-ahead journal and seed ledger. The protocols module defines typing
interfaces for beliefs, metrics, sandboxes and active evaluation environments. Language
execution backends live separately in ppl_synth_data.runtimes.
"""

__all__: list[str] = []

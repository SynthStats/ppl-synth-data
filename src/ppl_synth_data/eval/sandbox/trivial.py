"""Fixed-reward sandbox for deterministic tests; it does not execute submitted code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from synthstats_schemas import get_reward_source

from ppl_synth_data.schemas.trajectory import RewardRecord


@dataclass(frozen=True)
class TrivialSandboxResult:
    """Typed return of TrivialSandbox.execute, matches SandboxResult protocol."""

    reward_record: RewardRecord
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    wall_seconds: float = 0.0
    artefacts: Mapping[str, Any] = field(default_factory=dict)


class TrivialSandbox:
    """Return a fixed reward with registered provenance for deterministic tests."""

    backend_name = "trivial"

    def __init__(
        self,
        *,
        reward_value: float = 1.0,
        reward_source: str = "validation.compilation",
        producer: str = "ppl_synth_data.eval.sandbox.trivial",
    ) -> None:
        self._reward_value = reward_value
        self._reward_source = reward_source
        self._producer = producer

    def execute(
        self,
        *,
        language: str,
        program: str,
        task_package: Mapping[str, Any],
        timeout_seconds: float,
    ) -> TrivialSandboxResult:
        spec = get_reward_source(self._reward_source)
        reward_record = RewardRecord(
            total=self._reward_value,
            components={"trivial": self._reward_value},
            reward_source=self._reward_source,
            evidence_kind=spec.evidence_kind,
            truth_owner=spec.truth_owner,
            producer=self._producer,
        )
        return TrivialSandboxResult(
            reward_record=reward_record,
            stdout="",
            stderr="",
            exit_code=0,
            wall_seconds=0.0,
            artefacts={},
        )


__all__ = ["TrivialSandbox", "TrivialSandboxResult"]

"""Deterministic scalar regression environment for orchestrator tests.

Queries specify x and return y = alpha + beta * x + noise. Observations accumulate. The seed
determines the hidden parameters and observation noise.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from synthstats_schemas import EvidenceKind, TruthOwner


class ToyScalarEnv:
    """Scalar linear environment exposing the query interface used by orchestrator tests."""

    def __init__(
        self,
        *,
        true_alpha: float = 1.5,
        true_beta: float = 0.7,
        noise_scale: float = 0.1,
    ) -> None:
        self._true_alpha = true_alpha
        self._true_beta = true_beta
        self._noise_scale = noise_scale
        self._rng: random.Random | None = None
        self._history: list[tuple[float, float]] = []

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._history = []

    def run_experiment(self, x_value: float) -> float:
        """Execute one experiment. Returns a noisy y observation.

        Not using numpy so the test determinism test does not depend on
        numpy's RNG being left alone by other subsystems. Python's
        random.Random is per-instance and deterministic.
        """
        if self._rng is None:
            raise RuntimeError("ToyScalarEnv.reset must be called before run_experiment")
        y = self._true_alpha + self._true_beta * x_value
        noise = self._rng.gauss(0.0, self._noise_scale)
        observed = y + noise
        self._history.append((x_value, observed))
        return observed

    @property
    def history(self) -> Sequence[tuple[float, float]]:
        return tuple(self._history)

    # Factories used by the scalar-environment tests.
    @staticmethod
    def available_goals() -> tuple[str, ...]:
        return ("identify_linear_params",)

    @staticmethod
    def design_space() -> Mapping[str, Any]:
        return {"x_min": -5.0, "x_max": 5.0}

    @staticmethod
    def observation_evidence_kind() -> EvidenceKind:
        return EvidenceKind.verifier

    @staticmethod
    def observation_truth_owner() -> TruthOwner:
        return TruthOwner.verifier

    def ground_truth(self) -> dict[str, float]:
        return {
            "alpha": self._true_alpha,
            "beta": self._true_beta,
            "noise_scale": self._noise_scale,
        }

    def state_digest(self) -> str:
        """Content digest of the env's externally-visible state."""
        import hashlib

        digest = hashlib.sha256()
        digest.update(f"alpha={self._true_alpha}|beta={self._true_beta}".encode())
        digest.update(f"|noise={self._noise_scale}|n={len(self._history)}".encode())
        for x, y in self._history:
            digest.update(f"|{x:.12f}:{y:.12f}".encode())
        return "sha256:" + digest.hexdigest()[:16]


__all__ = ["ToyScalarEnv"]

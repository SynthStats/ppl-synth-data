"""Per-subsystem seed derivation for deterministic replay.

Each stochastic subsystem derives its seed from the root seed and subsystem
name, independently of ambient process state.

Rule: same root_seed + same subsystem name -> same derived seed. Forever.

Derivation is a stable SHA-256 hash of "<subsystem>|<root_seed>" reduced to
a 32-bit positive integer. Reduction to 32 bits matches NumPy's seed range
and is wide enough that collisions between distinct (root, name) pairs are
negligible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from threading import RLock


def derive_seed(root_seed: int, subsystem: str) -> int:
    """Deterministic 32-bit positive seed for a subsystem.

    Content-addressable: any reader of the ledger can independently
    reconstruct the same per-subsystem seed from (root_seed, subsystem).
    """
    if not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError(f"root_seed must be a non-negative int, got {root_seed!r}")
    if not subsystem:
        raise ValueError("subsystem name must be non-empty")
    digest = hashlib.sha256(f"{subsystem}|{root_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


@dataclass
class SeedLedger:
    """Record of which (subsystem, seed) pairs this episode used.

    Not a source of randomness; a bookkeeper. The orchestrator calls `for_subsystem(name)`; the
    ledger derives the seed, records that the subsystem was consulted, and returns it. The final
    snapshot is embedded in the trajectory's `run_metadata` so replay can confirm the same
    subsystems draw in the same order.

    Repeated calls for the same subsystem return the same seed; that is the invariant. If a
    caller needs a stream of distinct seeds, it should request named subsystems (e.g.
    "pymc.sample.turn_4") rather than stepping a counter.
    """

    root_seed: int
    _seeds: dict[str, int] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock)

    def for_subsystem(self, subsystem: str) -> int:
        with self._lock:
            cached = self._seeds.get(subsystem)
            if cached is not None:
                return cached
            seed = derive_seed(self.root_seed, subsystem)
            self._seeds[subsystem] = seed
            self._order.append(subsystem)
            return seed

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._seeds)

    def order(self) -> list[str]:
        with self._lock:
            return list(self._order)


__all__ = ["SeedLedger", "derive_seed"]

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceMetrics:
    total: int
    accepted: int

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.accepted / self.total

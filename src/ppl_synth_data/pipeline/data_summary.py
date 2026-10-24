from __future__ import annotations

from typing import Any

import numpy as np


def summarize_numeric(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
    }

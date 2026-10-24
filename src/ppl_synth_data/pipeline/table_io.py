from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["write_parquet"]


def write_parquet(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path

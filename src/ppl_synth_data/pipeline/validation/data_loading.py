from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_sample_data(parquet_path: Path) -> dict[str, Any] | None:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return None
    sample: dict[str, Any] = {"n": len(df)}
    for col in df.columns:
        sample[col] = np.array(df[col].values)
    return sample


def try_load_sample_data(
    env_id: str | None,
    data_dir: Path,
    *,
    hidden: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    if not env_id:
        return None, "missing env_id; cannot resolve Stage C data parquet"
    parquet_dir = data_dir / "hidden" if hidden else data_dir
    parquet_path = parquet_dir / f"{env_id}.parquet"
    if not parquet_path.exists():
        kind = "hidden" if hidden else "visible"
        return None, f"missing Stage C {kind} data parquet for env_id={env_id}"
    try:
        sample_data = load_sample_data(parquet_path)
        if sample_data is None:
            kind = "hidden" if hidden else "visible"
            return None, f"empty Stage C {kind} data parquet for env_id={env_id}"
        return sample_data, None
    except Exception as exc:
        kind = "hidden" if hidden else "visible"
        error = (
            f"failed to load Stage C {kind} data parquet for env_id={env_id}: {type(exc).__name__}"
        )
        log.warning("failed to load sample data for %s: %s", env_id, error, exc_info=True)
        return None, error

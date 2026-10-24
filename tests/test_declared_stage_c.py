"""Real Stage B/C integration for the closed four-family input contract."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev

import pandas as pd
import pytest

from ppl_synth_data.pipeline.artifacts import resolve_run_paths
from ppl_synth_data.pipeline.io import write_jsonl
from ppl_synth_data.pipeline.split_policy import SplitPolicyIneligibleError
from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b
from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c
from ppl_synth_data.specs.models import FourFamilySpec


def _prepared_paths(tmp_path: Path):
    paths = resolve_run_paths(tmp_path, "declared-stage-c")
    spec = FourFamilySpec.model_validate(
        {
            "env_id": "synthetic-count-allocation",
            "env_name": "synthetic_count_allocation",
            "domain": "synthetic",
            "scenario": {
                "intent": "Measure count response under a fixed three-level allocation",
                "source_row_sha256": "a" * 64,
            },
            "generative_family": "regression_count",
            "features": [
                {
                    "name": "month",
                    "law": {"dist": "fixed_counts", "levels": [1, 2, 3], "counts": [18, 30, 42]},
                },
                {"name": "signal", "law": {"dist": "normal", "mean": 50.0, "sd": 2.0}},
            ],
            "response": {
                "family": "regression_count",
                "intercept_log_rate": -0.8,
                "coefficients": {"month": 0.3, "signal": 0.02},
                "count_distribution": "poisson",
            },
            "data_plan": {"n_observations": 90},
        }
    )
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec.model_dump(mode="json")])
    result = run_stage_b(paths)
    assert result.status == "success"
    return paths, spec


def _read_split(paths, env_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = paths.stage_dirs["C"] / "data"
    visible = pd.read_parquet(data_dir / f"{env_id}.parquet")
    hidden = pd.read_parquet(data_dir / "hidden" / f"{env_id}.parquet")
    return visible, hidden


def test_declared_stage_c_materializes_exact_allocation_and_replays(tmp_path: Path) -> None:
    paths, spec = _prepared_paths(tmp_path)

    result = run_stage_c(paths, seed=20260930, split_policy="random_row", visible_ratio=0.6)

    assert result.status == "success"
    visible, hidden = _read_split(paths, spec.env_id)
    assert (len(visible), len(hidden)) == (54, 36)
    assert set(visible.columns) == {"month", "signal", "y"}
    assert set(hidden.columns) == {"month", "signal", "y"}
    combined = pd.concat([visible, hidden], ignore_index=True)
    assert Counter(combined["month"].tolist()) == {1: 18, 2: 30, 3: 42}
    assert pd.api.types.is_integer_dtype(combined["y"].dtype)
    assert (combined["y"] >= 0).all()
    signal = combined["signal"].tolist()
    assert abs(fmean(signal) - 50.0) < 0.7
    assert abs(pstdev(signal) - 2.0) < 0.7
    assert min(signal) > 9.0

    run_stage_c(paths, seed=20260930, split_policy="random_row", visible_ratio=0.6)
    replay_visible, replay_hidden = _read_split(paths, spec.env_id)
    pd.testing.assert_frame_equal(visible, replay_visible)
    pd.testing.assert_frame_equal(hidden, replay_hidden)


def test_declared_stage_c_refuses_group_holdout_without_group_semantics(tmp_path: Path) -> None:
    paths, spec = _prepared_paths(tmp_path)

    with pytest.raises(SplitPolicyIneligibleError, match="group_holdout"):
        run_stage_c(paths, seed=20260930, split_policy="group_holdout", visible_ratio=0.6)

    assert not (paths.stage_dirs["C"] / "data" / f"{spec.env_id}.parquet").exists()

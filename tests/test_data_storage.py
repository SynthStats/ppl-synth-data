from __future__ import annotations

from pathlib import Path

import pandas as pd

from ppl_synth_data.pipeline.table_io import write_parquet


def test_write_parquet_creates_file(tmp_path: Path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    out = tmp_path / "sub" / "out.parquet"
    result = write_parquet(rows, out)
    assert result == out
    assert out.exists()


def test_write_parquet_readable(tmp_path: Path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    out = tmp_path / "data.parquet"
    write_parquet(rows, out)
    df = pd.read_parquet(out)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
    assert df["a"].tolist() == [1, 2]


def test_write_parquet_empty_rows(tmp_path: Path):
    out = tmp_path / "empty.parquet"
    result = write_parquet([], out)
    assert result == out
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 0

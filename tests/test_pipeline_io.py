from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppl_synth_data.pipeline.io import (
    atomic_write_text,
    read_jsonl,
    write_json,
    write_jsonl,
    write_split_files,
)


def test_write_json(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    write_json(out, {"key": "value", "num": 42})
    loaded = json.loads(out.read_text())
    assert loaded == {"key": "value", "num": 42}


def test_write_json_creates_parents(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "dir" / "out.json"
    write_json(out, {"a": 1})
    assert out.exists()


def test_write_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "data.jsonl"
    rows = [{"id": 1}, {"id": 2}]
    result = write_jsonl(out, rows)
    assert result == out
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"id": 1}


def test_read_jsonl(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text('{"a":1}\n{"a":2}\n')
    rows = read_jsonl(data)
    assert len(rows) == 2
    assert rows[0] == {"a": 1}


def test_read_jsonl_skips_blank(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text('{"a":1}\n\n{"a":2}\n')
    rows = read_jsonl(data)
    assert len(rows) == 2


def test_read_jsonl_missing() -> None:
    with pytest.raises(FileNotFoundError):
        read_jsonl(Path("/nonexistent/data.jsonl"))


def test_write_split_files(tmp_path: Path) -> None:
    rows = {
        "train": [{"x": 1}],
        "validation": [{"x": 2}],
        "test": [{"x": 3}],
    }
    paths = write_split_files(tmp_path, "sft", rows)
    assert paths["train"].name == "sft.train.jsonl"
    assert json.loads(paths["train"].read_text().strip()) == {"x": 1}
    assert paths["validation"].exists()
    assert paths["test"].exists()


def test_write_split_files_empty_split(tmp_path: Path) -> None:
    paths = write_split_files(tmp_path, "data", {})
    for split_path in paths.values():
        assert split_path.exists()
        assert split_path.read_text().strip() == ""


def test_atomic_write_text_final_mode_is_container_readable(tmp_path: Path) -> None:
    """mkstemp creates temp files at 0600; exported task files are read by an
    unprivileged container user over a bind mount, so the final mode must be
    0644 regardless of umask. A silent chmod removal would only fail on a
    real Linux Docker host, so pin it here."""
    import os
    import stat

    target = tmp_path / "model.py"
    atomic_write_text(target, "def model(data):\n    pass\n")
    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode == 0o644, f"expected 0o644, got {oct(mode)}"

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via mkstemp + os.replace.

    ``mkstemp`` creates the temp file with mode 0600 (owner-only) regardless
    of umask. Many callers (Harbor task exporters in particular) write files
    that a non-root user inside a Docker container must later read via a
    bind mount, such as ``solution/model.py`` / ``solution/Submission.hs`` read
    by the container's unprivileged ``solver`` user. Left at 0600, those
    reads fail closed with ``Permission denied`` (or, for Haskell modules,
    surface as a confusing "not in scope" compile error instead of the real
    permission problem). Explicitly widen to 0644 (world-readable,
    owner-writable) after the atomic rename so every caller gets a sane,
    container-safe default; none of the current callers write secrets.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, content)
    return path


def write_split_files(
    stage_dir: Path,
    prefix: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Path]:
    paths = {
        "train": stage_dir / f"{prefix}.train.jsonl",
        "validation": stage_dir / f"{prefix}.validation.jsonl",
        "test": stage_dir / f"{prefix}.test.jsonl",
    }
    for split, out_path in paths.items():
        write_jsonl(out_path, rows_by_split.get(split, []))
    return paths

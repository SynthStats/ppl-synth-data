"""Shared manifest/hash/template utilities for the Harbor exporters.

These primitives are runtime-neutral: they are used by more than one of
`export_pymc`, `export_lazyppl`, and `export_skyrl_bundle` (or, in the case of
the hashing helpers, are generic file/tree-digest infrastructure rather than
domain logic for a specific runtime).
"""

from __future__ import annotations

import hashlib
import re
from importlib import resources
from pathlib import Path
from typing import Any

import jinja2

from ppl_synth_data.pipeline.contracts import STAGE_DIR_NAMES
from ppl_synth_data.pipeline.io import read_jsonl
from ppl_synth_data.runtime_registry import default_runtime_name, normalize_runtime_name
from ppl_synth_data.specs.models import CompletionRecord

_FENCE_RE = re.compile(r"^```(?:python)?\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
_SEQ_RE = re.compile(r"\d+")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _strip_markdown_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing to bundle symlinked Harbor artifact: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _toml_escape(value: str) -> str:
    """Escape a string for safe embedding inside a TOML quoted string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _load_jinja_env() -> jinja2.Environment:
    pkg = resources.files("ppl_synth_data").joinpath("harbor/templates")

    def _loader(name: str) -> str | None:
        try:
            return pkg.joinpath(name).read_text(encoding="utf-8")
        except (FileNotFoundError, TypeError):
            return None

    env = jinja2.Environment(
        loader=jinja2.FunctionLoader(_loader),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    env.filters["toml_escape"] = _toml_escape
    return env


def _read_data_refs(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Read Stage C's data_refs.jsonl (env_id -> data_ref/hidden_data_ref/split_policy)."""
    refs_path = run_dir / STAGE_DIR_NAMES["C"] / "data_refs.jsonl"
    refs: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(refs_path):
        env_id = row.get("env_id")
        if isinstance(env_id, str) and env_id:
            refs[env_id] = row
    return refs


def _resolve_run_relative_path(run_dir: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else run_dir / path


def _completion_runtime_name(rec: CompletionRecord) -> str:
    provenance_runtime = (
        rec.provenance.runtime_name
        if rec.provenance is not None and rec.provenance.runtime_name
        else None
    )
    return (
        normalize_runtime_name(provenance_runtime, default=default_runtime_name())
        or default_runtime_name()
    )

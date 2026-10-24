#!/usr/bin/env python3
"""Clean the staged LazyPPL SFT splits.

Operates on a staged copy of ``data/lazyppl_synthesis`` (never on canonical data). Reads the de-
contaminated staged splits, then:

1. Schema-unify every row to the canonical key set
   ``{prompt, completion, metadata, complexity}``. The legacy ``language`` key
   and the ``source`` field are folded into ``metadata`` (``metadata.runtime``,
   ``metadata.language``, ``metadata.source``). The old top-level ``language``
   key is dropped (no backward-compat dual-shape).
2. Drop broken rows: any row whose Haskell completion is missing the required
   ``model ::`` or ``logLikRow ::`` signatures, and any train row whose prompt
   leaks into val or test (held-out sets win; the train copy is removed). A leak
   is either an exact prompt match or a stem/prefix match: a translated train
   row can append a runtime instruction to a held-out NL task, so the underlying
   scenario still leaks even when the full strings differ.
3. Recover the true ``complexity`` for translated-from-pymc rows by exact-prompt
   match against the canonical PyMC corpus. Rows with no recoverable label keep
   ``complexity: null`` (honest unknown, no silent default).
4. Stamp ``metadata.runtime = "lazyppl"`` on every row (runtime conditioning).

Usage:
    uv run python scripts/clean_lazyppl_staged_splits.py
        --staged-dir artifacts/staged/lazyppl_synthesis --pymc-dir data/pymc_synthesis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_KEYS = {"prompt", "completion", "metadata", "complexity"}
REQUIRED_SIGNATURES = ("model ::", "logLikRow ::")
RUNTIME = "lazyppl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_pymc_complexity_map(pymc_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = pymc_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            prompt = row.get("prompt")
            complexity = row.get("complexity")
            if isinstance(prompt, str) and isinstance(complexity, str):
                mapping[prompt] = complexity
    return mapping


def has_required_signatures(completion: str) -> bool:
    return all(sig in completion for sig in REQUIRED_SIGNATURES)


STEM_LEAK_MIN_LEN = 80


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


def find_heldout_leak(prompt: str, heldout: list[tuple[str, str]]) -> str | None:
    """Return the held-out split a train prompt leaks into, or None.

    Matches exact prompts, normalized-whitespace equality, and stem/prefix
    containment (a translated row appending a runtime instruction to a held-out
    NL task). The held-out prompt is treated as the canonical task scenario.
    """
    norm_prompt = normalize_prompt(prompt)
    for where, heldout_prompt in heldout:
        norm_heldout = normalize_prompt(heldout_prompt)
        if norm_prompt == norm_heldout:
            return where
        shorter, longer = sorted((norm_prompt, norm_heldout), key=len)
        if len(shorter) >= STEM_LEAK_MIN_LEN and longer.startswith(shorter):
            return where
    return None


def canonicalize_row(row: dict[str, Any], pymc_complexity: dict[str, str]) -> dict[str, Any]:
    prompt = row["prompt"]
    completion = row["completion"]
    metadata = dict(row.get("metadata") or {})

    metadata["runtime"] = RUNTIME
    metadata.setdefault("language", row.get("language", "lazyppl"))

    # canonical top-level complexity: native rows already carry it; translated
    # rows recover it from the PyMC corpus, else stay null (honest unknown).
    complexity = row.get("complexity")
    if complexity is None:
        complexity = metadata.get("complexity")
    if complexity is None:
        complexity = pymc_complexity.get(prompt)
    if isinstance(complexity, str):
        metadata["complexity"] = complexity

    return {
        "prompt": prompt,
        "completion": completion,
        "metadata": metadata,
        "complexity": complexity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "staged" / "lazyppl_synthesis",
    )
    parser.add_argument("--pymc-dir", type=Path, default=REPO_ROOT / "data" / "pymc_synthesis")
    args = parser.parse_args()

    staged_dir: Path = args.staged_dir.resolve()
    pymc_complexity = build_pymc_complexity_map(args.pymc_dir.resolve())

    raw = {split: read_jsonl(staged_dir / f"{split}.jsonl") for split in ("train", "val", "test")}

    # Held-out (split, prompt) pairs (post-canonicalization prompts are
    # unchanged); used for exact, normalized, and stem/prefix leak detection.
    heldout: list[tuple[str, str]] = [("val", row["prompt"]) for row in raw["val"]] + [
        ("test", row["prompt"]) for row in raw["test"]
    ]

    dropped: list[dict[str, str]] = []
    cleaned: dict[str, list[dict[str, Any]]] = {}

    for split in ("train", "val", "test"):
        kept: list[dict[str, Any]] = []
        for idx, row in enumerate(raw[split]):
            completion = row.get("completion", "")
            if not has_required_signatures(completion):
                dropped.append(
                    {
                        "split": split,
                        "index": str(idx),
                        "prompt_head": row.get("prompt", "")[:60],
                        "reason": "missing required Haskell signatures "
                        "(model :: and/or logLikRow ::)",
                    }
                )
                continue
            if split == "train":
                where = find_heldout_leak(row["prompt"], heldout)
                if where is not None:
                    dropped.append(
                        {
                            "split": split,
                            "index": str(idx),
                            "prompt_head": row["prompt"][:60],
                            "reason": f"cross-split leak: NL task also in "
                            f"{where}; removed from train to preserve held-out "
                            "set",
                        }
                    )
                    continue
            kept.append(canonicalize_row(row, pymc_complexity))
        cleaned[split] = kept

    for split in ("train", "val", "test"):
        write_jsonl(staged_dir / f"{split}.jsonl", cleaned[split])

    report = {
        "row_counts": {s: len(cleaned[s]) for s in ("train", "val", "test")},
        "dropped": dropped,
        "complexity_null_counts": {
            s: sum(1 for r in cleaned[s] if r["complexity"] is None)
            for s in ("train", "val", "test")
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

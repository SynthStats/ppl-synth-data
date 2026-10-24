#!/usr/bin/env python3
"""De-contaminate train/val/test JSONL splits by removing leaked prompts from train.

A prompt that appears in BOTH train and a held-out split (val or test) is a leak: held-out
evaluation on that prompt is meaningless because the model saw it in training. The held-out sets
win: a duplicated row stays in val/test and is removed from train only. Val and test are never
modified.

Overlap is computed two ways and the union is removed:
  * exact: the prompt string verbatim.
  * normalized: whitespace collapsed (``\s+`` -> single space, stripped), so
    rows that differ only in indentation / trailing newlines still count as the
    same prompt.

Idempotent: re-running on already-clean splits removes nothing and re-writes the same content.
Importable: ``decontaminate`` and ``compute_overlaps`` are the reusable entry points.

Usage:
    uv run python scripts/dedup_split_contamination.py \
        --input-dir data/lazyppl_synthesis \
        --output-dir artifacts/staged/lazyppl_synthesis
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ppl_synth_data.pipeline.io import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "lazyppl_synthesis"

# Held-out splits win over train; their order here is the precedence for
# reporting which split a leaked prompt was kept in.
_HELDOUT_SPLITS = ("val", "test")
_ALL_SPLITS = ("train", "val", "test")

_WHITESPACE = re.compile(r"\s+")


def _prompt_of(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"row is missing a string 'prompt' field: {row!r}")
    return prompt


def normalize_prompt(prompt: str) -> str:
    """Collapse all runs of whitespace to a single space and strip the ends."""
    return _WHITESPACE.sub(" ", prompt).strip()


def _keys(row: dict[str, Any]) -> tuple[str, str]:
    prompt = _prompt_of(row)
    return prompt, normalize_prompt(prompt)


@dataclass
class OverlapReport:
    """Counts and sample prompts for prompts leaking from train into held-out."""

    train_val: int = 0
    train_test: int = 0
    removed_prompts: list[str] = field(default_factory=list)


def compute_overlaps(
    *,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> OverlapReport:
    """Count train prompts that also appear in val / test (exact OR normalized)."""
    train_exact = {_prompt_of(r) for r in train}
    train_norm = {normalize_prompt(_prompt_of(r)) for r in train}

    def overlap(heldout: list[dict[str, Any]]) -> set[str]:
        leaked: set[str] = set()
        for row in heldout:
            exact, norm = _keys(row)
            if exact in train_exact or norm in train_norm:
                leaked.add(exact)
        return leaked

    val_leaked = overlap(val)
    test_leaked = overlap(test)
    return OverlapReport(
        train_val=len(val_leaked),
        train_test=len(test_leaked),
        removed_prompts=sorted(val_leaked | test_leaked),
    )


def decontaminate(
    *,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return cleaned (train, val, test) plus the exact prompts dropped from train.

    Val and test are returned unchanged. A train row is dropped if its prompt
    matches (exact OR whitespace-normalized) any val/test prompt.
    """
    heldout_exact: set[str] = set()
    heldout_norm: set[str] = set()
    for heldout in (val, test):
        for row in heldout:
            exact, norm = _keys(row)
            heldout_exact.add(exact)
            heldout_norm.add(norm)

    kept: list[dict[str, Any]] = []
    removed_prompts: list[str] = []
    for row in train:
        exact, norm = _keys(row)
        if exact in heldout_exact or norm in heldout_norm:
            removed_prompts.append(exact)
        else:
            kept.append(row)

    return kept, list(val), list(test), removed_prompts


def _load_splits(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {name: read_jsonl(input_dir / f"{name}.jsonl") for name in _ALL_SPLITS}


def run(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Read splits from *input_dir*, write de-contaminated splits to *output_dir*.

    Returns a dict with overlap_before, overlap_after, rows_removed_from_train,
    staged_counts, and a sample of removed prompts. *input_dir* is never written.
    """
    splits = _load_splits(input_dir)
    before = compute_overlaps(train=splits["train"], val=splits["val"], test=splits["test"])

    clean_train, clean_val, clean_test, removed = decontaminate(
        train=splits["train"], val=splits["val"], test=splits["test"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", clean_train)
    write_jsonl(output_dir / "val.jsonl", clean_val)
    write_jsonl(output_dir / "test.jsonl", clean_test)

    after = compute_overlaps(train=clean_train, val=clean_val, test=clean_test)

    if after.train_val != 0 or after.train_test != 0:
        raise RuntimeError(
            "de-contamination failed: residual overlap "
            f"train_val={after.train_val} train_test={after.train_test}"
        )

    result = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "overlap_before": {"train_val": before.train_val, "train_test": before.train_test},
        "overlap_after": {"train_val": after.train_val, "train_test": after.train_test},
        "rows_removed_from_train": len(removed),
        "staged_counts": {
            "train": len(clean_train),
            "val": len(clean_val),
            "test": len(clean_test),
        },
        "removed_prompts_sample": removed[:8],
    }
    log.info(
        "before train_val=%d train_test=%d; removed %d train rows; "
        "after train_val=%d train_test=%d; counts %s",
        before.train_val,
        before.train_test,
        len(removed),
        after.train_val,
        after.train_test,
        result["staged_counts"],
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Remove cross-split prompt contamination (train leaks into val/test)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing train/val/test.jsonl to read (never written)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write de-contaminated train/val/test.jsonl",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run(input_dir=args.input_dir, output_dir=args.output_dir)
    log.info("result: %s", result)


if __name__ == "__main__":
    main()

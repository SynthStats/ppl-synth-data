#!/usr/bin/env python3
"""Create a held-out synthesis evaluation set disjoint from SFT and ATIF prompts.

Candidate rows use {prompt, completion, metadata, complexity} and include visible and hidden
sample data under eval_data.visible and eval_data.hidden. Exact and whitespace-normalised prompt
keys are shared with dedup_split_contamination.py. Each selected prompt is excluded from every
train, validation and test split and from the caller-supplied ATIF prompt corpus.

--candidates selects the input pool. --dry-run validates inputs without writing the set.
--target-size selects the requested number of tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Use the shared exact and normalized prompt keys for contamination checks.
from dedup_split_contamination import normalize_prompt  # noqa: E402

from ppl_synth_data.pipeline.io import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYMC_SPLIT_DIR = REPO_ROOT / "data" / "pymc_synthesis"
DEFAULT_OUT = REPO_ROOT / "data" / "eval" / "pymc_heldout_v0.jsonl"

# The 13 PyMC families the eval set should balance across.
# Used only for a balance report, not as a hard gate; a small candidate pool
# may not cover all 13, and that is surfaced rather than silently accepted.
_REQUIRED_EVAL_KEYS = ("prompt", "completion")


@dataclass(frozen=True)
class ExclusionSet:
    """Exact + normalized prompt keys to exclude an eval row against."""

    exact: frozenset[str]
    norm: frozenset[str]
    source: str

    def contains(self, prompt: str) -> bool:
        return prompt in self.exact or normalize_prompt(prompt) in self.norm


def _prompt_keys(prompts: list[str]) -> tuple[set[str], set[str]]:
    exact = {p for p in prompts}
    norm = {normalize_prompt(p) for p in prompts}
    return exact, norm


def load_split_prompts(split_dir: Path) -> list[str]:
    """All prompts across train/val/test in a pymc_synthesis-style split dir."""
    prompts: list[str] = []
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"expected split file not found: {path}")
        for row in read_jsonl(path):
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"row in {path} missing string 'prompt': {row!r}")
            prompts.append(prompt)
    return prompts


def _atif_prompt_from_json(doc: dict[str, Any]) -> str | None:
    """Extract the user prompt from one ATIF trajectory document.

    ATIF trajectory JSONs in this repo store ``steps: [{source, message}, ...]``
    where ``message`` is a plain string and the prompt is the first
    ``source == 'user'`` step. Returns None if no user step is present.
    """
    steps = doc.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and step.get("source") == "user":
                message = step.get("message")
                if isinstance(message, str) and message.strip():
                    return message
    # Fallback: an ATIF row exported as a flat {prompt, ...} record.
    prompt = doc.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return None


def load_atif_prompts(atif_path: Path) -> list[str]:
    """Load ATIF rescore prompts from a JSONL file or a dir of ATIF JSONs."""
    if not atif_path.exists():
        raise FileNotFoundError(f"--atif-prompts path not found: {atif_path}")

    prompts: list[str] = []
    if atif_path.is_dir():
        for json_path in sorted(atif_path.glob("*.json")):
            if json_path.name == "manifest.json":
                continue
            try:
                doc = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("skipping unparseable ATIF json: %s", json_path)
                continue
            prompt = _atif_prompt_from_json(doc)
            if prompt is not None:
                prompts.append(prompt)
    else:
        for row in read_jsonl(atif_path):
            prompt = _atif_prompt_from_json(row)
            if prompt is not None:
                prompts.append(prompt)

    if not prompts:
        raise ValueError(
            f"no ATIF prompts extracted from {atif_path}; refusing to freeze "
            "an eval set without an ATIF exclusion set (zero-overlap can't be "
            "proven against an empty set)"
        )
    return prompts


def build_exclusions(*, split_dir: Path, atif_path: Path) -> list[ExclusionSet]:
    split_prompts = load_split_prompts(split_dir)
    atif_prompts = load_atif_prompts(atif_path)
    split_exact, split_norm = _prompt_keys(split_prompts)
    atif_exact, atif_norm = _prompt_keys(atif_prompts)
    return [
        ExclusionSet(frozenset(split_exact), frozenset(split_norm), "pymc_synthesis"),
        ExclusionSet(frozenset(atif_exact), frozenset(atif_norm), "atif_rescore"),
    ]


def _validate_eval_row(row: dict[str, Any], *, index: int) -> None:
    for key in _REQUIRED_EVAL_KEYS:
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"candidate row {index} missing string '{key}'")
    eval_data = row.get("eval_data")
    if not isinstance(eval_data, dict):
        raise ValueError(
            f"candidate row {index} missing 'eval_data' object with "
            "visible/hidden sample data (required by the holdout-logp scorer)"
        )
    if "visible" not in eval_data or "hidden" not in eval_data:
        raise ValueError(
            f"candidate row {index} 'eval_data' must carry both 'visible' and "
            f"'hidden' sample data; got keys {sorted(eval_data)}"
        )


def _family_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in ("model_type", "family", "model_family", "domain"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
    return "unknown"


def _complexity_of(row: dict[str, Any]) -> str:
    value = row.get("complexity")
    if isinstance(value, str) and value:
        return value
    meta = row.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("complexity"), str):
        return meta["complexity"]
    return "unknown"


def filter_candidates(
    *, candidates: list[dict[str, Any]], exclusions: list[ExclusionSet]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop any candidate whose prompt overlaps an exclusion set."""
    kept: list[dict[str, Any]] = []
    dropped_by_source: Counter[str] = Counter()
    seen_prompts: set[str] = set()
    intra_dupes = 0
    for index, row in enumerate(candidates):
        _validate_eval_row(row, index=index)
        prompt = row["prompt"]
        norm = normalize_prompt(prompt)
        if norm in seen_prompts:
            intra_dupes += 1
            continue
        excluded = False
        for exclusion in exclusions:
            if exclusion.contains(prompt):
                dropped_by_source[exclusion.source] += 1
                excluded = True
                break
        if excluded:
            continue
        seen_prompts.add(norm)
        kept.append(row)
    report = dict(dropped_by_source)
    report["intra_candidate_duplicates"] = intra_dupes
    return kept, report


def assert_zero_overlap(*, eval_rows: list[dict[str, Any]], exclusions: list[ExclusionSet]) -> None:
    """Fail loudly if any frozen eval prompt overlaps an exclusion set."""
    for row in eval_rows:
        for exclusion in exclusions:
            if exclusion.contains(row["prompt"]):
                raise RuntimeError(
                    "contamination: a frozen eval prompt overlaps "
                    f"{exclusion.source}; freezing aborted"
                )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,
    candidates_path: Path,
    split_dir: Path,
    atif_path: Path,
    out_path: Path,
    target_size: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    exclusions = build_exclusions(split_dir=split_dir, atif_path=atif_path)
    candidates = read_jsonl(candidates_path)
    kept, drop_report = filter_candidates(candidates=candidates, exclusions=exclusions)
    assert_zero_overlap(eval_rows=kept, exclusions=exclusions)

    if target_size is not None and len(kept) > target_size:
        kept = kept[:target_size]

    family_balance = dict(Counter(_family_of(r) for r in kept))
    complexity_balance = dict(Counter(_complexity_of(r) for r in kept))

    result: dict[str, Any] = {
        "candidates_path": str(candidates_path),
        "out_path": str(out_path),
        "candidate_count": len(candidates),
        "kept_count": len(kept),
        "dropped": drop_report,
        "exclusion_sizes": {e.source: len(e.exact) for e in exclusions},
        "family_balance": family_balance,
        "complexity_balance": complexity_balance,
        "dry_run": dry_run,
        "frozen": False,
        "sha256": None,
    }

    if dry_run:
        log.info("dry-run OK: would freeze %d eval rows -> %s", len(kept), out_path)
        return result

    if not kept:
        raise RuntimeError(
            "no clean eval rows remained after exclusion; refusing to write an "
            "empty eval set. The candidate pool must come from validated "
            "generation."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, kept)
    result["frozen"] = True
    result["sha256"] = _sha256(out_path)
    log.info(
        "froze %d eval rows -> %s (sha256=%s)",
        len(kept),
        out_path,
        result["sha256"],
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a held-out PyMC eval set with zero train/ATIF overlap"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="JSONL of validated candidate eval rows (must carry eval_data)",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_PYMC_SPLIT_DIR,
        help="pymc_synthesis split dir to exclude (train/val/test.jsonl)",
    )
    parser.add_argument(
        "--atif-prompts",
        type=Path,
        required=True,
        help="ATIF rescore prompts: a JSONL file or a dir of ATIF trajectory JSONs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to write the frozen eval set",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Optional cap on frozen eval set size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate wiring + exclusion only; do not write the eval set",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run(
        candidates_path=args.candidates,
        split_dir=args.split_dir,
        atif_path=args.atif_prompts,
        out_path=args.out,
        target_size=args.target_size,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

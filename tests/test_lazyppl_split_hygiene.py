"""LazyPPL train, validation and test sets keep scenario groups disjoint.

Rows with the same scenario can contain different sampled observations. Group-level checks
detect their overlap even when the full prompts differ. Default tests cover the detector and
shipped data; heavy_lane also checks an externally supplied overlap-positive dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ppl_synth_data.dataset.split import group_split_overlaps

REPO_ROOT = Path(__file__).resolve().parents[1]

# The shipped, canonical LazyPPL SFT splits. Override with
# LAZYPPL_SFT_RELEASE_DIR for a different release directory.
DEFAULT_SHIPPED_DIR = REPO_ROOT / "data" / "lazyppl_synthesis"
SHIPPED_DIR = Path(os.environ.get("LAZYPPL_SFT_RELEASE_DIR", str(DEFAULT_SHIPPED_DIR)))

# Optional overlap-positive dataset for the contamination detector.
OVERLAP_FIXTURE_DIR = (
    Path(os.environ.get(
        "LAZYPPL_OVERLAP_FIXTURE_DIR",
        str(REPO_ROOT / "artifacts" / "overlap-fixtures" / "lazyppl"),
    ))
)

SPLITS = ("train", "val", "test")
# The published release carries no reasoning traces: the field is a build
# artefact, not part of the task. The test split ships as tasks only, without
# completions, behind its canary line.
CANONICAL_KEYS = {
    "completion",
    "complexity",
    "metadata",
    "prompt",
    "split_group_id",
}
TASK_ONLY_KEYS = CANONICAL_KEYS - {"completion"}
STEM_LEAK_MIN_LEN = 80


def _read_jsonl(path: Path) -> list[dict]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    # The published test split opens with a canary line; it is not a task row.
    return [r for r in records if not ("canary" in r and "prompt" not in r)]


def _load(directory: Path) -> dict[str, list[dict]]:
    if not directory.exists():
        raise RuntimeError(
            f"this test requires LazyPPL SFT splits at {directory}; build them with "
            "scripts/rebuild_lazyppl_splits.py (or set LAZYPPL_SFT_RELEASE_DIR) before "
            "running"
        )
    rows: dict[str, list[dict]] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.exists():
            raise RuntimeError(f"this test requires the split file {path}")
        rows[split] = _read_jsonl(path)
    return rows


def _normalize(prompt: str) -> str:
    return " ".join(prompt.split())


def _scenario(text: str, columns: str) -> str:
    return (
        f"Write a LazyPPL Haskell Bayesian model for this data-modelling task. "
        f"Scenario: {text}\nObserved row fields JSON: {columns}\n"
        "Return only one fenced haskell code block."
    )


def test_group_split_overlaps_flags_a_scenario_shared_across_splits() -> None:
    """Sibling rows of one scenario in two splits must be reported."""
    shared = "A shelter logs how many days each rescued animal stays before adoption."
    splits = {
        "train": [{"prompt": _scenario(shared, '{"days":[3,4,10]}')}],
        "val": [{"prompt": _scenario("A vineyard logs grape brix at harvest.", '{"brix":[21]}')}],
        "test": [{"prompt": _scenario(shared, '{"days":[9,11,2]}')}],
    }

    overlaps = group_split_overlaps(splits)

    assert overlaps["train__test"] == [shared]
    assert overlaps["train__val"] == []
    assert overlaps["val__test"] == []


def test_group_split_overlaps_is_empty_when_scenarios_are_disjoint() -> None:
    splits = {
        "train": [{"prompt": _scenario("A shelter logs animal stays.", '{"days":[3]}')}],
        "val": [{"prompt": _scenario("A vineyard logs grape brix.", '{"brix":[21]}')}],
        "test": [{"prompt": _scenario("A quarry logs blast vibration.", '{"pv":[7]}')}],
    }

    assert all(shared == [] for shared in group_split_overlaps(splits).values())


def test_shipped_lazyppl_splits_are_scenario_group_disjoint() -> None:
    """No scenario group may appear in more than one shipped LazyPPL split."""
    overlaps = group_split_overlaps(_load(SHIPPED_DIR))

    shared = {pair: keys for pair, keys in overlaps.items() if keys}
    assert not shared, f"scenario groups cross split boundaries in {SHIPPED_DIR}: " + "; ".join(
        f"{pair} shares {len(keys)} groups" for pair, keys in shared.items()
    )


def test_shipped_lazyppl_splits_have_zero_cross_split_prompt_overlap() -> None:
    """Whole-prompt equality is implied by scenario disjointness; check it directly too.

    A bug in scenario_group_key itself (wrong marker, wrong terminator) would
    not be caught by the scenario-level test above, since both sides of that
    comparison go through the same function. This check uses a different,
    independent method (whole-prompt and stem/prefix comparison) so a single
    bug in the shared extractor cannot pass both.
    """
    rows = _load(SHIPPED_DIR)
    prompts = {split: [r["prompt"] for r in rows[split]] for split in SPLITS}

    exact = {split: set(prompts[split]) for split in SPLITS}
    normalized = {split: {_normalize(p) for p in prompts[split]} for split in SPLITS}

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (exact[left] & exact[right]), (
            f"exact prompt overlap between {left} and {right}: {exact[left] & exact[right]}"
        )
        assert not (normalized[left] & normalized[right]), (
            f"normalized prompt overlap between {left} and {right}"
        )

    # Stem/prefix leaks: a train row appending a runtime instruction to a
    # held-out NL task still leaks the underlying scenario.
    train_prompts = [_normalize(p) for p in prompts["train"]]
    heldout_prompts = [_normalize(p) for p in prompts["val"] + prompts["test"]]
    for train_prompt in train_prompts:
        for heldout_prompt in heldout_prompts:
            if train_prompt == heldout_prompt:
                continue
            shorter, longer = sorted((train_prompt, heldout_prompt), key=len)
            assert not (len(shorter) >= STEM_LEAK_MIN_LEN and longer.startswith(shorter)), (
                f"stem/prefix leak from held-out task into train: {shorter[:60]!r}"
            )


def test_shipped_lazyppl_splits_have_uniform_canonical_keys() -> None:
    rows = _load(SHIPPED_DIR)

    for split in SPLITS:
        assert rows[split], f"shipped {split} split is empty"
        expected_keys = TASK_ONLY_KEYS if split == "test" else CANONICAL_KEYS
        for record in rows[split]:
            assert set(record) == expected_keys, (
                f"non-canonical keys in shipped {split} split: {sorted(record)}"
            )


def test_shipped_lazyppl_rows_carry_runtime_conditioning() -> None:
    rows = _load(SHIPPED_DIR)
    for split in SPLITS:
        for record in rows[split]:
            metadata = record.get("metadata") or {}
            assert metadata.get("runtime") == "lazyppl", (
                f"{split} row missing metadata.runtime=lazyppl: {record['prompt'][:60]!r}"
            )


def test_shipped_lazyppl_completions_have_required_signatures() -> None:
    rows = _load(SHIPPED_DIR)
    # The published test split is tasks only; completions ship with train/val.
    for split in ("train", "val"):
        for record in rows[split]:
            completion = record["completion"]
            assert "model ::" in completion and "logLikRow ::" in completion, (
                f"{split} row missing model::/logLikRow:: signatures: {record['prompt'][:60]!r}"
            )


@pytest.mark.heavy_lane
def test_detector_finds_cross_split_overlap() -> None:
    """The overlap-positive dataset must produce train/validation and train/test group matches."""
    overlaps = group_split_overlaps(_load(OVERLAP_FIXTURE_DIR))

    assert overlaps["train__val"], "detector found no train/val scenario overlap to reproduce"
    assert overlaps["train__test"], "detector found no train/test scenario overlap to reproduce"

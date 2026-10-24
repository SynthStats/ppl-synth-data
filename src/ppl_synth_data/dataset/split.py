"""Split assignment for generated datasets.

Two units of splitting live here and they are not interchangeable. A row-level
key (`deterministic_split`) is only safe when the key is already the unit that
must not cross a split boundary, such as an env id. SFT corpora sample several
rows of observed data from one natural-language scenario, so there the unit is
the scenario and the row-level key leaks: sibling rows of one scenario scatter
across train, validation and test, and a model that memorised the training
programme reproduces it at test time. `group_disjoint_split` is the splitter for
that shape.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCENARIO_MARKER = "Scenario:"
SCENARIO_TERMINATORS = (
    "Assume ",
    "Observed data JSON:",
    "Observed row fields JSON:",
    "Return only one fenced",
)
SPLIT_NAMES = ("train", "val", "test")


def scenario_group_key(prompt: str) -> str:
    """Return the scenario sentence that identifies a prompt's group.

    Everything after the scenario is per-row: the sampled observed data, the
    column listing, the response instructions. Keying on the whole prompt is
    what scatters one scenario across splits, so the key stops at the first
    structural marker that follows the scenario text.

    Raises on a prompt that carries no scenario marker or no terminator rather
    than falling back to the whole prompt, because a silent fallback is
    indistinguishable from a correct key and reintroduces the leak.
    """
    start = prompt.find(SCENARIO_MARKER)
    if start < 0:
        raise RuntimeError(
            f"prompt carries no {SCENARIO_MARKER!r} marker, so its scenario group cannot be "
            f"derived: {prompt[:200]!r}"
        )
    rest = prompt[start + len(SCENARIO_MARKER) :]
    cuts = [pos for pos in (rest.find(t) for t in SCENARIO_TERMINATORS) if pos >= 0]
    if not cuts:
        raise RuntimeError(
            "prompt carries no scenario terminator "
            f"(one of {SCENARIO_TERMINATORS}), so its scenario group cannot be "
            f"derived: {prompt[:200]!r}"
        )
    key = rest[: min(cuts)].strip()
    if not key:
        raise RuntimeError(f"prompt has an empty scenario: {prompt[:200]!r}")
    return key


def group_split_overlaps(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[str]]:
    """Report scenario groups shared between any two splits.

    This measures the group, not the prompt. Whole-prompt equality is the check
    that let scenario-level leakage through: sibling rows of one scenario carry
    different sampled data, so their prompts differ and a prompt-set
    intersection is empty while the scenario is on both sides.
    """
    grouped = {
        split: {scenario_group_key(row["prompt"]) for row in rows} for split, rows in splits.items()
    }
    names = list(grouped)
    overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}__{right}"] = sorted(grouped[left] & grouped[right])
    return overlaps


def group_disjoint_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict[str, list[Mapping[str, Any]]]:
    """Partition rows so every scenario group lands wholly in one split.

    Every input row appears in exactly one output split: this is a
    re-partition, never a filter. Groups are assigned as units, stratified by
    structural class so no split loses a class, and greedily balanced towards
    the requested row ratios. Whole groups make the ratios approximate; the
    caller reports the achieved counts.

    `seed` is required. The assignment is a pure function of the row contents
    and the seed, independent of input order.
    """
    if not 0.0 < val_ratio < 1.0 or not 0.0 < test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must each be in (0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[scenario_group_key(row["prompt"])].append(row)
    if not groups:
        raise RuntimeError("group_disjoint_split received no rows")

    strata: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        classes = {row["metadata"]["structural_class"] for row in members}
        if len(classes) != 1:
            raise RuntimeError(
                f"scenario group {key[:80]!r} spans several structural classes {sorted(classes)}; "
                "the stratum is ambiguous"
            )
        strata[classes.pop()].append(key)

    total_rows = sum(len(members) for members in groups.values())
    fractions = {
        "train": 1.0 - val_ratio - test_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    def rank(key: str) -> str:
        return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()

    # Each stratum releases only its own share of the row budget, so no class can
    # fill a split on its own. A single global budget lets the largest classes do
    # exactly that, which drains the small classes into validation and test and
    # leaves the three splits measuring different class mixtures. The budget is
    # cumulative rather than reset per stratum, so whole-group rounding in one
    # class is corrected by the next: largest classes are placed while the
    # residual is still zero, and the single-row classes absorb the correction.
    budget = dict.fromkeys(SPLIT_NAMES, 0.0)
    rows_now = dict.fromkeys(SPLIT_NAMES, 0)
    stratum_order = sorted(
        strata, key=lambda name: (-sum(len(groups[k]) for k in strata[name]), name)
    )
    for stratum in stratum_order:
        stratum_rows = sum(len(groups[k]) for k in strata[stratum])
        for name, fraction in fractions.items():
            budget[name] += stratum_rows * fraction
        # Largest groups first for balance; the seeded rank breaks ties reproducibly.
        keys = sorted(strata[stratum], key=lambda k: (-len(groups[k]), rank(k)))
        uncovered = list(SPLIT_NAMES)
        for position, key in enumerate(keys):
            remaining = len(keys) - position
            if uncovered and remaining <= len(uncovered):
                # Only enough groups left to cover the splits that still lack this class.
                choices = uncovered
            else:
                choices = list(SPLIT_NAMES)
            target = max(choices, key=lambda name: (budget[name] - rows_now[name], name))
            assigned[target].append(key)
            rows_now[target] += len(groups[key])
            if target in uncovered:
                uncovered.remove(target)

    splits = {
        name: [row for key in sorted(keys, key=rank) for row in groups[key]]
        for name, keys in assigned.items()
    }
    placed = sum(len(rows_out) for rows_out in splits.values())
    if placed != total_rows:
        raise RuntimeError(f"split dropped rows: {placed} placed of {total_rows} input")
    return splits


def deterministic_split(key: str, train_ratio: float = 0.9, val_ratio: float = 0.05) -> str:
    if not (0.0 <= train_ratio <= 1.0):
        raise ValueError("train_ratio must be in [0, 1]")
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError("val_ratio must be in [0, 1]")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1")

    raw = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    h = int.from_bytes(raw, "big") / float(2**64)
    if h < train_ratio:
        return "train"
    if h < train_ratio + val_ratio:
        return "validation"
    return "test"

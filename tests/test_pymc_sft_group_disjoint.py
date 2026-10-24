"""Scenario groups must not cross a split boundary in the PyMC SFT release.

Several rows share one natural-language scenario and differ only in their
sampled observed data. When such siblings sit on both sides of a split, a test
score measures memorisation of the training programme rather than
generalisation, so the scenario is the unit that must stay whole.

Two lanes here, deliberately. The detector tests run in the default lane so the
detection logic cannot rot unnoticed. The release tests read the out-of-repo
bundle and carry `heavy_lane` per this repo's marker convention; run them with
`-m heavy_lane` on the machine holding the artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ppl_synth_data.dataset.split import group_split_overlaps, scenario_group_key

REPO_ROOT = Path(__file__).resolve().parents[1]

# Set PYMC_SFT_RELEASE_DIR to the group-disjoint release directory.
DEFAULT_RELEASE_DIR = REPO_ROOT / "artifacts" / "releases" / "pymc"
RELEASE_DIR = Path(os.environ.get("PYMC_SFT_RELEASE_DIR", str(DEFAULT_RELEASE_DIR)))

# Optional overlap-positive dataset for the contamination detector.
OVERLAP_FIXTURE_DIR = Path(os.environ.get(
    "PYMC_OVERLAP_FIXTURE_DIR",
    str(REPO_ROOT / "artifacts" / "overlap-fixtures" / "pymc"),
))

SPLITS = ("train", "val", "test")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _load_release(directory: Path) -> dict[str, list[dict]]:
    if not directory.exists():
        raise RuntimeError(
            f"this lane requires the PyMC SFT release at {directory}; build it with "
            "scripts/build_pymc_sft_group_disjoint_split.py (or set PYMC_SFT_RELEASE_DIR) "
            "before running"
        )
    rows: dict[str, list[dict]] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.exists():
            raise RuntimeError(f"this lane requires the split file {path}")
        rows[split] = _read_jsonl(path)
    return rows


def _scenario(text: str, columns: str) -> str:
    return (
        f"Write a PyMC v5 model.\nScenario: {text}\n"
        f"Observed data JSON: {columns}\nReturn only one fenced block."
    )


def test_group_split_overlaps_flags_a_scenario_shared_across_splits() -> None:
    """Sibling rows of one scenario in two splits must be reported."""
    shared = "A depot records parcel handling delays across three shifts."
    splits = {
        "train": [{"prompt": _scenario(shared, '{"delay": [3, 4]}')}],
        "val": [{"prompt": _scenario("An orchard records fruit set per tree.", '{"set": [1]}')}],
        "test": [{"prompt": _scenario(shared, '{"delay": [9, 11, 2]}')}],
    }

    overlaps = group_split_overlaps(splits)

    assert overlaps["train__test"] == [shared]
    assert overlaps["train__val"] == []
    assert overlaps["val__test"] == []


def test_group_split_overlaps_is_empty_when_scenarios_are_disjoint() -> None:
    splits = {
        "train": [{"prompt": _scenario("A depot records parcel delays.", '{"delay": [3]}')}],
        "val": [{"prompt": _scenario("An orchard records fruit set.", '{"set": [1]}')}],
        "test": [{"prompt": _scenario("A reservoir records inflow.", '{"inflow": [7]}')}],
    }

    assert all(shared == [] for shared in group_split_overlaps(splits).values())


def test_scenario_group_key_raises_rather_than_falling_back() -> None:
    """A prompt without the structural markers has no derivable group."""
    with pytest.raises(RuntimeError, match="no 'Scenario:' marker"):
        scenario_group_key("Write a PyMC v5 model for the attached data.")
    with pytest.raises(RuntimeError, match="no scenario terminator"):
        scenario_group_key("Scenario: a depot records parcel delays and nothing follows")


@pytest.mark.heavy_lane
def test_released_pymc_splits_are_group_disjoint() -> None:
    """No scenario group may appear in more than one shipped split."""
    overlaps = group_split_overlaps(_load_release(RELEASE_DIR))

    shared = {pair: keys for pair, keys in overlaps.items() if keys}
    assert not shared, f"scenario groups cross split boundaries in {RELEASE_DIR}: " + "; ".join(
        f"{pair} shares {len(keys)} groups" for pair, keys in shared.items()
    )


@pytest.mark.heavy_lane
def test_detector_finds_cross_split_overlap() -> None:
    """The overlap-positive dataset must produce train/test and train/validation group matches."""
    overlaps = group_split_overlaps(_load_release(OVERLAP_FIXTURE_DIR))

    assert overlaps["train__test"], "detector found no train/test scenario overlap to reproduce"
    assert overlaps["train__val"], "detector found no train/val scenario overlap to reproduce"

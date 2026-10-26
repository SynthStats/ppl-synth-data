"""Re-partition the PyMC SFT corpus so no scenario group crosses a split.

This is a re-partition and never a filter: every input row appears in exactly
one output split, and the script raises if the totals disagree. Splitting logic
lives in `ppl_synth_data.dataset.split`; this entrypoint reads a source release,
writes the group-disjoint splits, and records their provenance.

    uv run python scripts/build_pymc_sft_group_disjoint_split.py \
        --source <source-pymc-dir> --out <destination-root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ppl_synth_data.dataset.split import (
    group_disjoint_split,
    group_split_overlaps,
    scenario_group_key,
)

SPLITS = ("train", "val", "test")
METADATA_DIMENSIONS = ("domain", "model_type", "structural_class")
DEFAULT_SEED = 20260724
GROUP_ID_PREFIX = "scenario-sha256-"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"required source split is missing: {path}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def group_id(scenario: str) -> str:
    return GROUP_ID_PREFIX + hashlib.sha256(scenario.encode("utf-8")).hexdigest()


def distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {
        dimension: Counter(row["metadata"][dimension] for row in rows)
        for dimension in METADATA_DIMENSIONS
    }
    counts["complexity"] = Counter(row["complexity"] for row in rows)
    return {
        dimension: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
        for dimension, counter in counts.items()
    }


def coverage_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Which metadata values reach every split, and which cannot by construction.

    A value carried by fewer scenario groups than there are splits can only
    reach that many splits, whatever the assignment. Those are named rather
    than left to look like an imbalance the splitter chose.
    """
    all_rows = [row for rows in splits.values() for row in rows]
    groups_per_value: dict[str, dict[str, set[str]]] = {
        dimension: defaultdict(set) for dimension in (*METADATA_DIMENSIONS, "complexity")
    }
    for row in all_rows:
        scenario = scenario_group_key(row["prompt"])
        for dimension in METADATA_DIMENSIONS:
            groups_per_value[dimension][row["metadata"][dimension]].add(scenario)
        groups_per_value["complexity"][row["complexity"]].add(scenario)

    report: dict[str, Any] = {}
    for dimension, per_value in groups_per_value.items():
        present = {
            split: {
                row["metadata"][dimension] if dimension != "complexity" else row["complexity"]
                for row in rows
            }
            for split, rows in splits.items()
        }
        spanning = sorted(value for value in per_value if all(value in s for s in present.values()))
        limited = {
            value: {
                "scenario_groups": len(groups),
                "splits_reached": sum(1 for s in present.values() if value in s),
            }
            for value, groups in sorted(per_value.items())
            if len(groups) < len(SPLITS)
        }
        short_but_spread = sorted(
            value
            for value, groups in per_value.items()
            if len(groups) >= len(SPLITS) and value not in spanning
        )
        report[dimension] = {
            "values_total": len(per_value),
            "values_in_every_split": len(spanning),
            "values_too_few_groups_to_span": limited,
            "values_with_enough_groups_but_not_spanning": short_but_spread,
        }
    return report


def build(source: Path, destination: Path, seed: int) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    source_files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = source / f"{split}.jsonl"
        rows = read_jsonl(path)
        source_counts[split] = len(rows)
        source_files[f"{split}.jsonl"] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(rows),
        }
        source_rows.extend(rows)
    if not source_rows:
        raise RuntimeError(f"no rows found under {source}")

    splits = group_disjoint_split(source_rows, seed=seed, val_ratio=0.2, test_ratio=0.2)

    written: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        stamped = []
        for row in rows:
            enriched = dict(row)
            enriched["split_group_id"] = group_id(scenario_group_key(row["prompt"]))
            stamped.append(enriched)
        written[split] = stamped

    total_out = sum(len(rows) for rows in written.values())
    if total_out != len(source_rows):
        raise RuntimeError(
            f"re-partition changed the row count: {len(source_rows)} in, {total_out} out"
        )

    overlaps = group_split_overlaps(written)
    if any(overlaps.values()):
        raise RuntimeError(f"scenario groups cross splits after re-partition: {overlaps}")

    out_dir = destination / "pymc"
    files: dict[str, dict[str, Any]] = {}
    for split, rows in written.items():
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        files[f"pymc/{split}.jsonl"] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(rows),
        }

    group_ids = {split: {row["split_group_id"] for row in rows} for split, rows in written.items()}
    manifest = {
        "runtime": "pymc",
        "unit_of_split": "scenario",
        "grouping_key": {
            "description": (
                "the scenario sentence: prompt text after 'Scenario:' up to the first of "
                "'Assume ', 'Observed data JSON:', 'Observed row fields JSON:', "
                "'Return only one fenced'; stripped and compared exactly"
            ),
            "row_field": "split_group_id",
            "row_field_format": GROUP_ID_PREFIX + "<sha256 of the scenario sentence>",
            "raises_when": "a prompt carries no scenario marker or no terminator",
        },
        "seed": seed,
        "ratios_requested": {"train": 0.6, "val": 0.2, "test": 0.2},
        "source": {
            "dataset": "SynthStats/ppl-synthesis-sft-bootstrap",
            "revision": "4286dfa29947599ca056202d04c2a1397e55513c",
            "local_directory": str(source),
            "files": source_files,
            "rows": dict(source_counts),
            "rows_total": len(source_rows),
        },
        "counts": {
            "rows_total": total_out,
            "rows_per_split": {split: len(rows) for split, rows in written.items()},
            "row_fractions": {
                split: round(len(rows) / total_out, 4) for split, rows in written.items()
            },
            "scenario_groups_total": len(set().union(*group_ids.values())),
            "scenario_groups_per_split": {split: len(ids) for split, ids in group_ids.items()},
        },
        "cross_split_scenario_group_overlaps": {pair: len(keys) for pair, keys in overlaps.items()},
        "metadata_distribution": {split: distribution(rows) for split, rows in written.items()},
        "stratification": {
            "primary": "metadata.structural_class",
            "coverage": coverage_report(written),
        },
        "files": files,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help=(
                                                                       'source pymc split directory'
                                                                   ))
    parser.add_argument("--out", required=True, type=Path, help="destination release root")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    manifest = build(args.source.resolve(), args.out.resolve(), args.seed)
    print(
        json.dumps(
            {
                "rows_per_split": manifest["counts"]["rows_per_split"],
                "scenario_groups_per_split": manifest["counts"]["scenario_groups_per_split"],
                "cross_split_scenario_group_overlaps": manifest[
                    "cross_split_scenario_group_overlaps"
                ],
                "rows_total": manifest["counts"]["rows_total"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge translated + native LazyPPL records into train/val/test splits.

Reads translated splits (train/val/test) and native records, filters to
typecheck_ok=True and wrapped translated records with wrap_typecheck_ok=True,
assigns native records to splits via deterministic hash
of the prompt prefix (first 200 chars), and writes final
data/lazyppl_synthesis/{train,val,test}.jsonl.

Usage:
    uv run python scripts/merge_lazyppl_splits.py \
        --input-dir generated/lazyppl_synthesis/inputs \
        --output-dir data/lazyppl_synthesis
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections import Counter
from pathlib import Path

from ppl_synth_data.dataset.split import deterministic_split

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "generated" / "lazyppl_synthesis" / "inputs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "lazyppl_synthesis"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"required input file not found: {path}")
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"expected JSON object in {path}:{line_number}")
                records.append(record)
    return records


def _write_jsonl_atomic(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(tmp_handle.name)
    try:
        with tmp_handle as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def merge_splits(*, input_dir: Path, output_dir: Path) -> None:
    train_translated = load_jsonl(input_dir / "train_translated.jsonl")
    try:
        val_translated = load_jsonl(input_dir / "val_translated.jsonl")
    except FileNotFoundError:
        val_translated = []
        log.warning("val_translated.jsonl not found, proceeding without")
    try:
        test_translated = load_jsonl(input_dir / "test_translated.jsonl")
    except FileNotFoundError:
        test_translated = []
        log.warning("test_translated.jsonl not found, proceeding without")

    native = load_jsonl(input_dir / "native_train.jsonl")

    log.info(
        "Loaded: %d train-translated, %d val-translated, %d test-translated, %d native",
        len(train_translated),
        len(val_translated),
        len(test_translated),
        len(native),
    )

    def is_ok(r: dict) -> bool:
        metadata = r.get("metadata")
        return (
            isinstance(metadata, dict)
            and metadata.get("typecheck_ok") is True
            and metadata.get("wrap_typecheck_ok", True) is True
        )

    train_translated = [r for r in train_translated if is_ok(r)]
    val_translated = [r for r in val_translated if is_ok(r)]
    test_translated = [r for r in test_translated if is_ok(r)]
    native = [r for r in native if is_ok(r)]

    log.info(
        "After typecheck filter: %d train, %d val, %d test, %d native",
        len(train_translated),
        len(val_translated),
        len(test_translated),
        len(native),
    )

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for r in native:
        # full prompt, not prefix; similar-prefix families would cluster otherwise
        prompt = r.get("prompt", "")
        family = (r.get("metadata") or {}).get("family", "")
        key = f"{family}:{prompt}"
        split_name = deterministic_split(key, train_ratio=0.8, val_ratio=0.1)
        splits[split_name].append(r)

    log.info(
        "Native split: %d train, %d val, %d test",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )

    # merge
    final_train = train_translated + splits["train"]
    final_val = val_translated + splits["validation"]
    final_test = test_translated + splits["test"]
    total = len(final_train) + len(final_val) + len(final_test)

    if total == 0:
        raise ValueError(
                             'all records were filtered out; refusing to ov'
                             'erwrite existing outputs'
                         )

    # report family coverage
    for split_name, records in [("train", final_train), ("val", final_val), ("test", final_test)]:
        families: Counter[str] = Counter()
        for r in records:
            meta = r.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            fam = meta.get("family", meta.get("generative_family", "translated"))
            families[fam] += 1
        log.info("%s: %d records, families: %s", split_name, len(records), dict(families))

    outputs = {
        output_dir / "train.jsonl": final_train,
        output_dir / "val.jsonl": final_val,
        output_dir / "test.jsonl": final_test,
    }
    staged_paths: dict[Path, Path] = {}
    try:
        for output_path, records in outputs.items():
            staged_paths[output_path] = _write_jsonl_atomic(output_path, records)
        for output_path, staged_path in staged_paths.items():
            staged_path.replace(output_path)
    except Exception:
        for staged_path in staged_paths.values():
            staged_path.unlink(missing_ok=True)
        raise

    log.info(
        "Wrote %d total records (%d/%d/%d)",
        total,
        len(final_train),
        len(final_val),
        len(final_test),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge translated and native LazyPPL records")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
                 'Directory containing train/val/test translate'
                 'd JSONL files and native_train.jsonl'
             ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write canonical train/val/test JSONL splits",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    merge_splits(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

from ppl_synth_data.audit.batch import (
    BatchSpec,
    build_batch_text,
    build_entry_list,
    make_batches,
)


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_build_entry_list_reads_splits(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": "p1", "completion": "c1"}])
    _write_jsonl(tmp_path / "val.jsonl", [{"prompt": "p2", "completion": "c2"}])

    entries = build_entry_list(tmp_path)

    assert len(entries) == 2
    assert entries[0] == (1, {"prompt": "p1", "completion": "c1"}, "train")
    assert entries[1] == (2, {"prompt": "p2", "completion": "c2"}, "val")


def test_build_entry_list_skips_missing_splits(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": "p1", "completion": "c1"}])
    entries = build_entry_list(tmp_path)
    assert len(entries) == 1
    assert entries[0][2] == "train"


def test_build_entry_list_global_idx_is_1_based(tmp_path):
    _write_jsonl(
        tmp_path / "train.jsonl",
        [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(3)],
    )
    entries = build_entry_list(tmp_path)
    assert [e[0] for e in entries] == [1, 2, 3]


def test_make_batches_splits_correctly():
    entries = [(i, {}, "train") for i in range(1, 6)]
    batches = make_batches(entries, batch_size=2)

    assert len(batches) == 3
    assert batches[0].batch_num == 1
    assert batches[0].start == 1
    assert batches[0].end == 2
    assert len(batches[0].entries) == 2
    assert batches[2].batch_num == 3
    assert len(batches[2].entries) == 1


def test_make_batches_single_batch():
    entries = [(1, {}, "val"), (2, {}, "val")]
    batches = make_batches(entries, batch_size=10)
    assert len(batches) == 1
    assert batches[0].start == 1
    assert batches[0].end == 2


def test_build_batch_text_contains_entries():
    spec = BatchSpec(
        batch_num=1,
        start=1,
        end=1,
        entries=[(1, {"prompt": "test prompt", "completion": "test code"}, "train")],
    )
    text = build_batch_text(spec)

    assert "Batch 1" in text
    assert "test prompt" in text
    assert "test code" in text
    assert "Entry 1" in text
    assert "split: train" in text


def test_build_batch_text_custom_system_prompt():
    spec = BatchSpec(
        batch_num=2,
        start=3,
        end=4,
        entries=[
            (3, {"prompt": "p", "completion": "c"}, "val"),
        ],
    )
    text = build_batch_text(spec, system_prompt="Custom {batch_num} {start} {end}\n")
    assert text.startswith("Custom 2 3 4")

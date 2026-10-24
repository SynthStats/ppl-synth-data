from __future__ import annotations

import json

from ppl_synth_data.audit.parser import EntryResult
from ppl_synth_data.audit.report import _split_map, render_report
from ppl_synth_data.evidence import AuditVerdict


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_split_map_assigns_indices(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [{"a": 1}, {"a": 2}])
    _write_jsonl(tmp_path / "val.jsonl", [{"a": 3}])

    sm = _split_map(tmp_path)

    assert sm == {1: "train", 2: "train", 3: "val"}


def test_split_map_skips_missing_files(tmp_path):
    _write_jsonl(tmp_path / "test.jsonl", [{"a": 1}])
    sm = _split_map(tmp_path)
    assert sm == {1: "test"}


def test_render_report_structure(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [{"a": 1}, {"a": 2}])
    _write_jsonl(tmp_path / "val.jsonl", [{"a": 3}])

    results = [
        EntryResult(1, AuditVerdict.PASS, "5/5", "5/5", "OK", "None", batch_num=1),
        EntryResult(2, AuditVerdict.FLAG, "3/5", "4/5", "OK", "minor issue", batch_num=1),
        EntryResult(3, AuditVerdict.CRITICAL, "2/5", "1/5", "wrong type", "bad dist", batch_num=1),
    ]
    report = render_report(results, tmp_path)

    assert "# Dataset Audit Report" in report
    assert "| # | Split | Verdict" in report
    assert "## Summary Statistics" in report
    assert "PASS" in report
    assert "FLAG" in report
    assert "CRITICAL" in report


def test_render_report_assigns_splits(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [{"a": 1}])
    _write_jsonl(tmp_path / "val.jsonl", [{"a": 2}])

    results = [
        EntryResult(1, AuditVerdict.PASS, "5/5", "5/5", "OK", "None", batch_num=1),
        EntryResult(2, AuditVerdict.PASS, "5/5", "5/5", "OK", "None", batch_num=1),
    ]
    render_report(results, tmp_path)

    assert results[0].split == "train"
    assert results[1].split == "val"


def test_render_report_empty(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [])
    report = render_report([], tmp_path)
    assert "Coverage:** 0 entries" in report

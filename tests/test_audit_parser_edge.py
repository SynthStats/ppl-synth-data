"""Edge case tests for audit.parser: multiline issues, missing fields, split field."""

from __future__ import annotations

from ppl_synth_data.audit.parser import EntryResult, parse_log


def test_multiline_issues_captured() -> None:
    log = (
        "Entry 10: FLAG\n"
        "Prompt score: 6/10\n"
        "Completion score: 5/10\n"
        "Metadata: ok\n"
        "Issues: first line of issue\n"
        "second line of issue\n"
        "third line\n"
    )
    results = parse_log(log, batch_num=0)
    assert len(results) == 1
    assert "first line" in results[0].issues
    assert "second line" in results[0].issues


def test_missing_scores_default_to_question_mark() -> None:
    log = "Entry 0: PASS\nIssues: none\n"
    results = parse_log(log, batch_num=0)
    assert len(results) == 1
    assert results[0].prompt_score == "?"
    assert results[0].completion_score == "?"
    assert results[0].metadata_note == "?"


def test_entry_result_default_split_is_empty() -> None:
    r = EntryResult(
        global_idx=0,
        verdict="PASS",
        prompt_score="10",
        completion_score="10",
        metadata_note="ok",
        issues="none",
        batch_num=0,
    )
    assert r.split == ""


def test_entry_result_split_can_be_set() -> None:
    r = EntryResult(
        global_idx=0,
        verdict="PASS",
        prompt_score="10",
        completion_score="10",
        metadata_note="ok",
        issues="none",
        batch_num=0,
        split="train",
    )
    assert r.split == "train"


def test_garbage_between_entries_ignored() -> None:
    log = (
        "some preamble garbage\n"
        "## batch header\n"
        "\n"
        "Entry 1: PASS\n"
        "Prompt score: 8\n"
        "Completion score: 7\n"
        "Metadata: fine\n"
        "Issues: none\n"
        "\n"
        "some mid-section noise\n"
        "\n"
        "Entry 2: CRITICAL\n"
        "Prompt score: 1\n"
        "Completion score: 1\n"
        "Metadata: broken\n"
        "Issues: everything wrong\n"
    )
    results = parse_log(log, batch_num=5)
    assert len(results) == 2
    assert results[0].global_idx == 1
    assert results[0].verdict == "PASS"
    assert results[1].global_idx == 2
    assert results[1].verdict == "CRITICAL"
    assert results[0].batch_num == 5


def test_large_entry_index() -> None:
    log = "Entry 999: FLAG\nPrompt score: 5\nCompletion score: 4\nMetadata: ok\nIssues: big index\n"
    results = parse_log(log, batch_num=0)
    assert len(results) == 1
    assert results[0].global_idx == 999


def test_issues_field_missing_entirely() -> None:
    log = "Entry 0: PASS\nPrompt score: 9\nCompletion score: 8\nMetadata: good\n"
    results = parse_log(log, batch_num=0)
    assert len(results) == 1
    assert results[0].issues == "?"

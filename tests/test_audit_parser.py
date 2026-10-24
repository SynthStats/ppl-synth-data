from __future__ import annotations

from ppl_synth_data.audit.parser import VERDICT_EMOJI, parse_log
from ppl_synth_data.evidence import AuditVerdict, SemanticStatus


def test_empty_text():
    assert parse_log("", batch_num=0) == []


def test_no_entries():
    assert parse_log("Some random text\nwith no entries", batch_num=1) == []


def test_single_pass_entry():
    log = "Entry 0: PASS\nPrompt score: 9/10\nCompletion score: 8/10\nMetadata: ok\nIssues: none\n"
    results = parse_log(log, batch_num=3)
    assert len(results) == 1
    r = results[0]
    assert r.global_idx == 0
    assert r.verdict == "PASS"
    assert r.prompt_score == "9/10"
    assert r.completion_score == "8/10"
    assert r.metadata_note == "ok"
    assert "none" in r.issues
    assert r.batch_num == 3


def test_critical_entry_with_issues():
    log = (
        "Entry 5: CRITICAL\n"
        "Prompt score: 2/10\n"
        "Completion score: 1/10\n"
        "Metadata: missing fields\n"
        "Issues: leaked distribution name\n"
    )
    results = parse_log(log, batch_num=1)
    assert len(results) == 1
    assert results[0].verdict == "CRITICAL"
    assert "leaked" in results[0].issues


def test_multiple_entries():
    log = (
        "Entry 1: PASS\n"
        "Prompt score: 10/10\n"
        "Completion score: 9/10\n"
        "Metadata: fine\n"
        "Issues: none\n"
        "\n"
        "Entry 2: FLAG\n"
        "Prompt score: 6/10\n"
        "Completion score: 5/10\n"
        "Metadata: weak\n"
        "Issues: vague prompt\n"
    )
    results = parse_log(log, batch_num=0)
    assert len(results) == 2
    assert results[0].global_idx == 1
    assert results[0].verdict == "PASS"
    assert results[1].global_idx == 2
    assert results[1].verdict == "FLAG"


def test_case_insensitive_verdict():
    log = "Entry 3: pass\nPrompt score: 7\nCompletion score: 7\nMetadata: ok\nIssues: none\n"
    results = parse_log(log, batch_num=0)
    assert len(results) == 1
    assert results[0].verdict == "PASS"


def test_verdict_emoji_mapping():
    assert set(VERDICT_EMOJI.keys()) == {"PASS", "FLAG", "CRITICAL"}


def test_audit_verdict_carries_shared_semantic_taxonomy():
    result = parse_log(
        "Entry 1: critical\nPrompt score: 1\nCompletion score: 1\nMetadata: bad\nIssues: bad\n",
        batch_num=0,
    )[0]

    assert result.verdict == AuditVerdict.CRITICAL
    assert result.evidence_kind == "audit"
    assert result.truth_owner == "auditor"
    assert result.semantic_status == SemanticStatus.critical

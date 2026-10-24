"""Render a Markdown audit report from EntryResult objects."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ppl_synth_data.audit.parser import VERDICT_EMOJI, EntryResult
from ppl_synth_data.evidence import AuditVerdict


def _split_map(data_dir: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    idx = 1
    for split in ("train", "val", "test"):
        p = data_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result[idx] = split
                idx += 1
    return result


def render_report(results: Sequence[EntryResult], data_dir: Path) -> str:
    split_map = _split_map(data_dir)
    for r in results:
        r.split = split_map.get(r.global_idx, "?")

    sorted_r = sorted(results, key=lambda r: r.global_idx)
    total = len(sorted_r)
    pass_n = sum(1 for r in sorted_r if r.verdict == AuditVerdict.PASS)
    flag_n = sum(1 for r in sorted_r if r.verdict == AuditVerdict.FLAG)
    crit_n = sum(1 for r in sorted_r if r.verdict == AuditVerdict.CRITICAL)

    lines = [
        "# Dataset Audit Report",
        "",
        f"**Coverage:** {total} entries parsed",
        "",
        "---",
        "",
        "## Full Results Table",
        "",
        "| # | Split | Verdict | Prompt | Completion | Metadata | Batch |",
        "|---|-------|---------|--------|------------|----------|-------|",
    ]

    for r in sorted_r:
        icon = VERDICT_EMOJI.get(r.verdict, "?")
        meta = r.metadata_note[:40] + ("..." if len(r.metadata_note) > 40 else "")
        lines.append(
            f"| {r.global_idx} | {r.split} | {icon} {r.verdict} | "
            f"{r.prompt_score} | {r.completion_score} | {meta} | {r.batch_num} |"
        )

    def _entry_block(r: EntryResult) -> list[str]:
        return [
            f"### Entry {r.global_idx} (split: {r.split}, batch: {r.batch_num})",
            f"- Prompt score: {r.prompt_score}",
            f"- Completion score: {r.completion_score}",
            f"- Metadata: {r.metadata_note}",
            f"- Issues: {r.issues}",
            "---",
            "",
        ]

    critical = [r for r in sorted_r if r.verdict == AuditVerdict.CRITICAL]
    lines += ["", "---", "", f"## Critical Issues ({crit_n} entries)", ""]
    for r in critical:
        lines += _entry_block(r)

    flagged = [r for r in sorted_r if r.verdict == AuditVerdict.FLAG]
    lines += ["", "---", "", f"## Flagged Issues ({flag_n} entries)", ""]
    for r in flagged:
        lines += _entry_block(r)

    lines += [
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Verdict | Count | % |",
        "|---------|-------|---|",
        f"| ✅ PASS | {pass_n} | {100 * pass_n / max(total, 1):.1f}% |",
        f"| ⚠️ FLAG | {flag_n} | {100 * flag_n / max(total, 1):.1f}% |",
        f"| 🚨 CRITICAL | {crit_n} | {100 * crit_n / max(total, 1):.1f}% |",
        "",
    ]

    for split_name in ("train", "val", "test"):
        sr = [r for r in sorted_r if r.split == split_name]
        if not sr:
            continue
        sp = sum(1 for r in sr if r.verdict == AuditVerdict.PASS)
        sf = sum(1 for r in sr if r.verdict == AuditVerdict.FLAG)
        sc = sum(1 for r in sr if r.verdict == AuditVerdict.CRITICAL)
        lines.append(f"**{split_name}** ({len(sr)} entries): {sp} PASS, {sf} FLAG, {sc} CRITICAL")

    return "\n".join(lines) + "\n"

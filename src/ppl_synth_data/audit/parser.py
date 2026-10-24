"""Parse GPT audit output into structured EntryResult objects."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ppl_synth_data.evidence import AuditVerdict, EvidenceKind, SemanticStatus, TruthOwner

VERDICT_EMOJI = {verdict.value: verdict.emoji for verdict in AuditVerdict}


@dataclass
class EntryResult:
    global_idx: int
    verdict: AuditVerdict
    prompt_score: str
    completion_score: str
    metadata_note: str
    issues: str
    batch_num: int
    split: str = ""

    @property
    def evidence_kind(self) -> EvidenceKind:
        return self.verdict.evidence_kind

    @property
    def truth_owner(self) -> TruthOwner:
        return self.verdict.truth_owner

    @property
    def semantic_status(self) -> SemanticStatus:
        return self.verdict.semantic_status


def parse_log(log_text: str, batch_num: int) -> list[EntryResult]:
    # Split on "Entry N:" anchors; each chunk is one entry's verdict block.
    chunks = re.split(r"(?=^Entry\s+\d+\s*:)", log_text, flags=re.MULTILINE)

    results = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        m = re.match(r"Entry\s+(\d+)\s*:\s*(PASS|FLAG|CRITICAL)", chunk, re.IGNORECASE)
        if not m:
            continue

        def extract(pattern: str, default: str = "?", _text: str = chunk) -> str:
            hit = re.search(pattern, _text, re.IGNORECASE)
            return hit.group(1).strip() if hit else default

        issues_m = re.search(
            r"Issues\s*:\s*(.+?)(?=\nEntry\s+\d+|^##|\Z)",
            chunk,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )

        results.append(
            EntryResult(
                global_idx=int(m.group(1)),
                verdict=AuditVerdict.parse(m.group(2)),
                prompt_score=extract(r"Prompt\s+score\s*:\s*(.+?)(?:\n|$)"),
                completion_score=extract(r"Completion\s+score\s*:\s*(.+?)(?:\n|$)"),
                metadata_note=extract(r"Metadata\s*:\s*(.+?)(?:\n|$)"),
                issues=issues_m.group(1).strip() if issues_m else "?",
                batch_num=batch_num,
            )
        )

    return results

from __future__ import annotations

from enum import StrEnum


class EvidenceKind(StrEnum):
    verifier = "verifier"
    heuristic = "heuristic"
    audit = "audit"


class TruthOwner(StrEnum):
    verifier = "verifier"
    heuristic = "heuristic"
    auditor = "auditor"


class SemanticStatus(StrEnum):
    not_scored = "not_scored"
    pass_ = "pass"
    flag = "flag"
    critical = "critical"


class AuditVerdict(StrEnum):
    PASS = "PASS"
    FLAG = "FLAG"
    CRITICAL = "CRITICAL"

    @property
    def emoji(self) -> str:
        return {
            AuditVerdict.PASS: "✅",
            AuditVerdict.FLAG: "⚠️",
            AuditVerdict.CRITICAL: "🚨",
        }[self]

    @property
    def semantic_status(self) -> SemanticStatus:
        return {
            AuditVerdict.PASS: SemanticStatus.pass_,
            AuditVerdict.FLAG: SemanticStatus.flag,
            AuditVerdict.CRITICAL: SemanticStatus.critical,
        }[self]

    @property
    def evidence_kind(self) -> EvidenceKind:
        return EvidenceKind.audit

    @property
    def truth_owner(self) -> TruthOwner:
        return TruthOwner.auditor

    @classmethod
    def parse(cls, value: str) -> AuditVerdict:
        return cls(value.strip().upper())

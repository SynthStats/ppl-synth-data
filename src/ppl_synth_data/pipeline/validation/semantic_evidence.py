from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from synthstats_schemas import RewardArtifact

from ppl_synth_data.evidence import EvidenceKind, SemanticStatus, TruthOwner
from ppl_synth_data.specs.models import CompletionRecord

LAZYPPL_SEMANTIC_EVIDENCE_VERSION = "lazyppl_harbor_semantic_scorer.v1"
LAZYPPL_REWARD_SOURCE = "validation.lazyppl_elpd"
_VERIFIER_EVIDENCE = str(EvidenceKind.verifier)
_VERIFIER_TRUTH_OWNER = str(TruthOwner.verifier)


def lazyppl_semantic_evidence_failure(rec: CompletionRecord) -> str | None:
    """Return the fail-closed reason for a LazyPPL row, or None when usable."""
    validation = rec.validation
    if validation is None:
        return "missing LazyPPL semantic validation"
    if not validation.ok:
        return "LazyPPL validation is not ok"
    if validation.semantic_status != SemanticStatus.pass_:
        return f"LazyPPL semantic status is {validation.semantic_status!r}"
    if validation.evidence_kind != EvidenceKind.verifier:
        return "LazyPPL semantic evidence is not verifier-owned"
    if validation.truth_owner != TruthOwner.verifier:
        return "LazyPPL semantic truth owner is not verifier"

    details = validation.semantic_details
    if details.get("semantic_evidence_version") != LAZYPPL_SEMANTIC_EVIDENCE_VERSION:
        return "missing current LazyPPL Harbor semantic evidence version"

    reward_artifact = details.get("reward_artifact")
    if not isinstance(reward_artifact, dict):
        return "missing LazyPPL semantic reward artifact"
    return _lazyppl_reward_artifact_failure(reward_artifact)


def _lazyppl_reward_artifact_failure(reward_artifact: dict[str, Any]) -> str | None:
    try:
        artifact = RewardArtifact.model_validate(reward_artifact)
    except ValidationError as exc:
        return f"LazyPPL semantic reward artifact fails RewardArtifact contract: {exc}"
    if artifact.reward_source != LAZYPPL_REWARD_SOURCE:
        return "LazyPPL semantic reward source is not validation.lazyppl_elpd"
    if str(artifact.evidence_kind) != _VERIFIER_EVIDENCE:
        return "LazyPPL semantic reward artifact is not verifier evidence"
    if str(artifact.truth_owner) != _VERIFIER_TRUTH_OWNER:
        return "LazyPPL semantic reward artifact truth owner is not verifier"
    return None

"""Producer-side reward scalarization for rich SynthStats reward artifacts."""

from __future__ import annotations

import math
from typing import Any

from synthstats_schemas import (
    EvidenceKind,
    RewardArtifact,
    RewardComponent,
    RewardComponentKind,
    RewardRubric,
    RewardScalarization,
    RewardScalarizationMethod,
    RubricCriterion,
    TruthOwner,
)
from synthstats_schemas.reward import TRAINING_LOG_REWARD_MIN

_COMPOSITE_PRODUCER = "ppl_synth_data.reward.scalarization"
_RUBRIC_PRODUCER = "ppl_synth_data.reward.scalarization"


def _as_reward_artifact(artifact: RewardArtifact | dict[str, Any]) -> RewardArtifact:
    if isinstance(artifact, RewardArtifact):
        return artifact
    return RewardArtifact.model_validate(artifact)


def _finite_score(value: float, *, name: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite")
    return score


def _bounded_log(value: float) -> float:
    return max(math.log(max(value, math.exp(TRAINING_LOG_REWARD_MIN))), TRAINING_LOG_REWARD_MIN)


def _component_log_contribution(component: RewardComponent) -> float:
    if component.log_contribution is not None:
        return float(component.log_contribution)
    return _bounded_log(float(component.value))


def build_rubric_composite_reward(
    *,
    verifier_artifact: RewardArtifact | dict[str, Any],
    rubric_score: float,
    rubric_id: str,
    rubric_version: str,
    rubric_title: str,
    rubric_criteria: tuple[RubricCriterion, ...] | list[RubricCriterion | dict[str, Any]],
    rubric_weight: float = 0.1,
    verifier_weight: float = 1.0,
    rubric_component_name: str = "llm_rubric",
    rubric_notes: str = "",
    parent_rubric_version: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> RewardArtifact:
    """Build a strict mixed verifier/rubric reward artifact.

    The top-level reward is heuristic-owned because the scalarized aggregate
    includes a heuristic rubric component. Verifier components remain preserved
    with their original source/evidence/truth provenance.
    """

    base = _as_reward_artifact(verifier_artifact)
    score = _finite_score(rubric_score, name="rubric_score")
    if score < 0.0 or score > 1.0:
        raise ValueError("rubric_score must be in [0, 1]")
    rubric_weight = _finite_score(rubric_weight, name="rubric_weight")
    verifier_weight = _finite_score(verifier_weight, name="verifier_weight")
    if rubric_weight < 0.0:
        raise ValueError("rubric_weight must be non-negative")
    if verifier_weight < 0.0:
        raise ValueError("verifier_weight must be non-negative")

    criteria = tuple(
        criterion
        if isinstance(criterion, RubricCriterion)
        else RubricCriterion.model_validate(criterion)
        for criterion in rubric_criteria
    )
    provenance_payload = dict(provenance or {})
    rubric = RewardRubric(
        rubric_id=rubric_id,
        version=rubric_version,
        title=rubric_title,
        criteria=criteria,
        parent_version=parent_rubric_version,
        evolution_note=rubric_notes,
        provenance={
            "source_reward_source": base.reward_source,
            **provenance_payload,
        },
    )
    rubric_component = RewardComponent(
        name=rubric_component_name,
        kind=RewardComponentKind.process_rubric,
        value=score,
        source="heuristic.llm_rubric",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        producer=_RUBRIC_PRODUCER,
        weight=rubric_weight,
        log_contribution=_bounded_log(score),
        confidence=score,
        metrics={"rubric_score": score},
        diagnostics={},
        provenance={
            "rubric_id": rubric_id,
            "rubric_version": rubric_version,
            **provenance_payload,
        },
    )

    verifier_components = tuple(base.reward_components)
    if not verifier_components:
        verifier_components = (
            RewardComponent(
                name="verifier_log_reward",
                kind=RewardComponentKind.terminal_verifier,
                value=float(base.reward if base.reward is not None else base.training_log_reward),
                source=base.reward_source,
                evidence_kind=base.evidence_kind,
                truth_owner=base.truth_owner,
                producer=base.producer,
                weight=verifier_weight,
                log_contribution=base.training_log_reward,
                metrics=dict(base.metrics),
                diagnostics=dict(base.diagnostics),
                provenance=dict(base.provenance),
            ),
        )

    weights = {component.name: verifier_weight for component in verifier_components}
    weights[rubric_component.name] = rubric_weight
    components = (*verifier_components, rubric_component)
    scalarized_log_reward = sum(
        weights[component.name] * _component_log_contribution(component) for component in components
    )
    training_log_reward = max(float(scalarized_log_reward), TRAINING_LOG_REWARD_MIN)
    scalarization = RewardScalarization(
        method=RewardScalarizationMethod.weighted_sum,
        component_names=tuple(component.name for component in components),
        weights=weights,
        lower_log_reward_bound=TRAINING_LOG_REWARD_MIN,
        diagnostics={
            "aggregate": "training_log_reward",
            "source_reward_source": base.reward_source,
            "contains_heuristic_component": True,
        },
    )

    return RewardArtifact(
        reward_source="heuristic.composite_reward",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        producer=_COMPOSITE_PRODUCER,
        training_log_reward=training_log_reward,
        reward=max(math.exp(training_log_reward), 0.0),
        reward_transform={
            "kind": "weighted_verifier_rubric_composite",
            "lower_log_reward_bound": TRAINING_LOG_REWARD_MIN,
            "source_reward_source": base.reward_source,
        },
        metrics={
            "source_training_log_reward": base.training_log_reward,
            "rubric_score": score,
            "scalarized_log_reward": training_log_reward,
        },
        diagnostics={
            "source_truth_owner": base.truth_owner.value,
            "source_evidence_kind": base.evidence_kind.value,
        },
        provenance={
            "source_reward_source": base.reward_source,
            "source_producer": base.producer,
            **provenance_payload,
        },
        components={component.name: float(component.value) for component in components},
        reward_components=components,
        scalarization=scalarization,
        rubric=rubric,
    )

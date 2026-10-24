from __future__ import annotations

import math

import pytest
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

from ppl_synth_data.reward.scalarization import build_rubric_composite_reward


def _verifier_artifact(*, training_log_reward: float = -1.5) -> RewardArtifact:
    component = RewardComponent(
        name="holdout_elpd",
        kind=RewardComponentKind.terminal_verifier,
        value=-12.0,
        source="validation.lazyppl_elpd",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        producer="ppl_synth_data.docker.lazyppl_harbor.score",
        weight=1.0,
        log_contribution=training_log_reward,
        metrics={"elpd": -12.0},
        provenance={"record_id": "row-0"},
    )
    return RewardArtifact(
        reward_source="validation.lazyppl_elpd",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        producer="ppl_synth_data.docker.lazyppl_harbor.score",
        training_log_reward=training_log_reward,
        reward_transform={"kind": "test"},
        metrics={"elpd": -12.0},
        reward=math.exp(training_log_reward),
        components={"holdout_elpd": -12.0},
        reward_components=(component,),
        scalarization=RewardScalarization(
            method=RewardScalarizationMethod.weighted_sum,
            component_names=("holdout_elpd",),
            weights={"holdout_elpd": 1.0},
        ),
    )


def test_build_rubric_composite_reward_preserves_verifier_and_heuristic_provenance() -> None:
    artifact = build_rubric_composite_reward(
        verifier_artifact=_verifier_artifact(),
        rubric_score=0.8,
        rubric_id="ppl-critique",
        rubric_version="2026-05-16",
        rubric_title="PPL critique rubric",
        rubric_criteria=(
            RubricCriterion(
                criterion_id="structure",
                description="Program has a coherent likelihood and prior structure.",
                weight=1.0,
            ),
        ),
        rubric_weight=0.1,
        provenance={"atif_id": "row-0"},
    )

    assert artifact.reward_source == "heuristic.composite_reward"
    assert artifact.evidence_kind is EvidenceKind.heuristic
    assert artifact.truth_owner is TruthOwner.heuristic
    assert artifact.producer == "ppl_synth_data.reward.scalarization"
    assert artifact.training_log_reward == pytest.approx(-1.5 + 0.1 * math.log(0.8))
    assert artifact.training_log_reward >= TRAINING_LOG_REWARD_MIN
    assert artifact.reward_components[0].source == "validation.lazyppl_elpd"
    assert artifact.reward_components[0].truth_owner is TruthOwner.verifier
    assert artifact.reward_components[1].source == "heuristic.llm_rubric"
    assert artifact.reward_components[1].truth_owner is TruthOwner.heuristic
    assert artifact.scalarization is not None
    assert artifact.scalarization.component_names == ("holdout_elpd", "llm_rubric")
    assert artifact.scalarization.weights == {"holdout_elpd": 1.0, "llm_rubric": 0.1}
    assert isinstance(artifact.rubric, RewardRubric)
    assert artifact.rubric.rubric_id == "ppl-critique"
    assert RewardArtifact.model_validate(artifact.model_dump(mode="json")) == artifact


def test_build_rubric_composite_reward_clamps_to_contract_floor() -> None:
    artifact = build_rubric_composite_reward(
        verifier_artifact=_verifier_artifact(training_log_reward=-19.9),
        rubric_score=0.0,
        rubric_id="ppl-critique",
        rubric_version="2026-05-16",
        rubric_title="PPL critique rubric",
        rubric_criteria=[{"criterion_id": "structure", "description": "Coherent model structure."}],
    )

    assert artifact.training_log_reward == TRAINING_LOG_REWARD_MIN


def test_build_rubric_composite_reward_rejects_invalid_rubric_score() -> None:
    with pytest.raises(ValueError, match="rubric_score must be in"):
        build_rubric_composite_reward(
            verifier_artifact=_verifier_artifact(),
            rubric_score=1.1,
            rubric_id="ppl-critique",
            rubric_version="2026-05-16",
            rubric_title="PPL critique rubric",
            rubric_criteria=[
                {"criterion_id": "structure", "description": "Coherent model structure."}
            ],
        )

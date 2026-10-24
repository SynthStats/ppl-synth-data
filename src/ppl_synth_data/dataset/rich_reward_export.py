"""Export rich reward ATIF directories from existing strict ATIF artifacts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from synthstats_schemas import (
    TRAINING_LOG_REWARD_MIN,
    ATIFRecord,
    LiveRichRewardManifest,
    RewardArtifact,
    RewardComponent,
    RewardComponentKind,
    RewardRubric,
    RewardScalarization,
    RewardScalarizationMethod,
    RichRewardATIFExportManifest,
    RubricCriterion,
)

from ppl_synth_data.llm.backend import LLMRequest
from ppl_synth_data.llm.openai_backend import OpenAIBackend
from ppl_synth_data.pipeline.io import write_json

_REWARD_SCHEMA_VERSION = "synthstats.harbor.reward.v1"
_COMPOSITE_REWARD_SOURCE = "heuristic.composite_reward"
_COMPOSITE_PRODUCER = "ppl_synth_data.dataset.rich_reward_export"
_RUBRIC_REWARD_SOURCE = "heuristic.llm_rubric"
_RUBRIC_PRODUCER = "ppl_synth_data.dataset.rich_reward_export"
_RUBRIC_COMPONENT_NAME = "llm_trace_rubric"
_PROCESS_INFO_REWARD_SOURCE = "heuristic.expected_information_gain_proxy"
_PROCESS_INFO_PRODUCER = "ppl_synth_data.dataset.rich_reward_export"
_PROCESS_INFO_COMPONENT_NAME = "process_information_gain_proxy"
_RUBRIC_ID = "synthstats_trace_quality"
_RUBRIC_VERSION = "2026-05-16"
_RUBRIC_SCORE_SCHEMA_VERSION = "ppl_synth_data.rich_reward_rubric_scores.v1"
_BASE_VERIFIER_REWARD_SOURCES = frozenset(
    {
        "validation.lazyppl_elpd",
        "validation.pymc_elpd",
    }
)


_RUBRIC_CRITERIA: tuple[RubricCriterion, ...] = (
    RubricCriterion(
        criterion_id="data_contract_use",
        description=(
            "The trace respects the visible task/data contract and avoids "
            "inventing unavailable fields or APIs."
        ),
        weight=0.30,
    ),
    RubricCriterion(
        criterion_id="likelihood_structure",
        description=(
            "The submitted probabilistic program uses a plausible likelihood "
            "and latent structure for the observed data."
        ),
        weight=0.35,
    ),
    RubricCriterion(
        criterion_id="uncertainty_reasoning",
        description=(
            "The trace handles uncertainty, posterior diagnostics, censoring, "
            "hierarchy, or other statistical complications when relevant."
        ),
        weight=0.25,
    ),
    RubricCriterion(
        criterion_id="submission_hygiene",
        description=(
            "The final answer is a clean, executable submission rather than "
            "unstructured prose or an incomplete sketch."
        ),
        weight=0.10,
    ),
)


_RUBRIC = RewardRubric(
    rubric_id=_RUBRIC_ID,
    version=_RUBRIC_VERSION,
    title="SynthStats Trace Quality Rubric",
    criteria=_RUBRIC_CRITERIA,
    evolution_note=(
        "Initial rich-reward proof rubric. It is deliberately heuristic and "
        "must stay separated from verifier-owned ELPD rewards."
    ),
    provenance={
        "grounding": [
            "BoxingGym expected-information-gain framing",
            "LLM-as-a-judge process/rubric reward literature",
            "experience-replay LLM post-training literature",
        ],
    },
)


@dataclass(frozen=True)
class RubricScore:
    """Structured LLM-rubric score for one ATIF trace."""

    criterion_scores: dict[str, float]
    rationale: str
    confidence: float

    @property
    def weighted_score(self) -> float:
        total_weight = sum(float(criterion.weight) for criterion in _RUBRIC_CRITERIA)
        if total_weight <= 0.0:
            raise RuntimeError("rubric criteria weights must sum to a positive value")
        weighted = 0.0
        for criterion in _RUBRIC_CRITERIA:
            score = self.criterion_scores.get(criterion.criterion_id)
            if score is None:
                raise ValueError(f"rubric score is missing criterion {criterion.criterion_id!r}")
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError(f"rubric criterion {criterion.criterion_id!r} must be in [0, 1]")
            weighted += float(criterion.weight) * float(score)
        if not math.isfinite(self.confidence) or self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("rubric confidence must be in [0, 1]")
        return float(weighted / total_weight)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RubricScore:
        raw_scores = payload.get("criterion_scores")
        if not isinstance(raw_scores, Mapping):
            raise ValueError("rubric payload must contain criterion_scores object")
        scores: dict[str, float] = {}
        for criterion in _RUBRIC_CRITERIA:
            value = raw_scores.get(criterion.criterion_id)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"rubric criterion {criterion.criterion_id!r} must be numeric")
            scores[criterion.criterion_id] = float(value)
        rationale = payload.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rubric payload must contain non-empty rationale")
        confidence = payload.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ValueError("rubric payload must contain numeric confidence")
        return cls(
            criterion_scores=scores,
            rationale=rationale.strip(),
            confidence=float(confidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_scores": dict(self.criterion_scores),
            "weighted_score": self.weighted_score,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


def _assert_fresh_output_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    entries = [entry for entry in out_dir.iterdir() if entry.name != ".DS_Store"]
    if entries:
        raise FileExistsError(f"rich reward output directory already contains files: {out_dir}")


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _trial_dirs(atif_dir: Path) -> list[Path]:
    if not atif_dir.is_dir():
        raise FileNotFoundError(f"ATIF directory not found: {atif_dir}")
    trials = sorted(
        path
        for path in atif_dir.iterdir()
        if path.is_dir()
        and (path / "trajectory.json").is_file()
        and (path / "reward.json").is_file()
    )
    if not trials:
        raise FileNotFoundError(f"no ATIF trial directories found in {atif_dir}")
    return trials


def _load_scores(path: Path) -> dict[str, RubricScore]:
    payload = _json_object(path)
    if payload.get("schema_version") != _RUBRIC_SCORE_SCHEMA_VERSION:
        raise ValueError(
            f"rubric scores file must use schema_version {_RUBRIC_SCORE_SCHEMA_VERSION!r}"
        )
    raw_rubric = payload.get("rubric")
    if not isinstance(raw_rubric, Mapping):
        raise ValueError("rubric scores file must contain rubric metadata")
    rubric = RewardRubric.model_validate(raw_rubric)
    if rubric != _RUBRIC:
        raise ValueError("rubric scores file does not match the active rubric version")
    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("rubric scores file must contain a scores object")
    parsed: dict[str, RubricScore] = {}
    for atif_id, score_payload in scores.items():
        if not isinstance(atif_id, str) or not atif_id:
            raise ValueError("rubric score keys must be non-empty ATIF ids")
        if not isinstance(score_payload, Mapping):
            raise ValueError(f"rubric score for {atif_id!r} must be an object")
        parsed[atif_id] = RubricScore.from_mapping(score_payload)
    return parsed


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and item.get("type", "text") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _trace_process_information_score(atif: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """Estimate trace-level information value without claiming verifier-owned EIG.

    This is a deterministic process proxy for rich-reward experiments. True
    Bayesian expected information gain remains the verifier-owned
    ``validation.expected_information_gain`` source.
    """

    agent_question_marks = 0
    agent_reasoning_chars = 0
    observation_count = 0
    tool_call_count = 0
    tool_names: set[str] = set()
    for step in atif.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        source = step.get("source") or step.get("role")
        text = _message_text(step.get("message"))
        reasoning = step.get("reasoning_content")
        if isinstance(reasoning, str):
            text = "\n".join(part for part in (text, reasoning) if part)
        if source == "agent":
            agent_question_marks += text.count("?")
            think_start = text.find("<think>")
            think_end = text.find("</think>")
            if think_start >= 0 and think_end > think_start:
                agent_reasoning_chars += think_end - think_start
            else:
                agent_reasoning_chars += len(str(reasoning or ""))

        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, str | bytes):
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                tool_call_count += 1
                raw_name = (
                    tool_call.get("name") or tool_call.get("tool_name") or tool_call.get("function")
                )
                if isinstance(raw_name, Mapping):
                    raw_name = raw_name.get("name")
                if isinstance(raw_name, str) and raw_name.strip():
                    tool_names.add(raw_name.strip())

        observation = step.get("observation")
        if isinstance(observation, Mapping) and observation:
            observation_count += 1

    question_score = min(agent_question_marks / 3.0, 1.0)
    tool_score = min(tool_call_count / 3.0, 1.0)
    observation_score = min(observation_count / 3.0, 1.0)
    reasoning_score = min(agent_reasoning_chars / 800.0, 1.0)
    score = (
        0.30 * question_score
        + 0.30 * tool_score
        + 0.20 * observation_score
        + 0.20 * reasoning_score
    )
    metrics: dict[str, Any] = {
        "agent_question_marks": agent_question_marks,
        "agent_reasoning_chars": agent_reasoning_chars,
        "tool_call_count": tool_call_count,
        "observation_count": observation_count,
        "distinct_tool_count": len(tool_names),
        "question_score": question_score,
        "tool_score": tool_score,
        "observation_score": observation_score,
        "reasoning_score": reasoning_score,
        "score": score,
    }
    return float(max(0.0, min(score, 1.0))), metrics


def _trim_text(value: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _rubric_user_prompt(
    atif: Mapping[str, Any],
    reward: RewardArtifact,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    prompt_parts: list[str] = []
    completion_parts: list[str] = []
    for step in atif.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        text = _message_text(step.get("message"))
        if step.get("source") == "user" and text:
            prompt_parts.append(text)
        elif step.get("source") == "agent" and text:
            completion_parts.append(text)

    prompt_text, prompt_truncated = _trim_text(
        "\n\n".join(prompt_parts),
        max_chars=max_chars,
    )
    completion_text, completion_truncated = _trim_text(
        "\n\n".join(completion_parts),
        max_chars=max_chars,
    )
    verifier_payload = json.dumps(
        {
            "reward_source": reward.reward_source,
            "training_log_reward": reward.training_log_reward,
            "metrics": reward.metrics,
            "diagnostics": reward.diagnostics,
            "components": reward.components,
        },
        indent=2,
        sort_keys=True,
    )
    rubric_payload = json.dumps(
        {
            criterion.criterion_id: {
                "description": criterion.description,
                "weight": criterion.weight,
            }
            for criterion in _RUBRIC_CRITERIA
        },
        indent=2,
        sort_keys=True,
    )
    user = (
        "Score this probabilistic-program synthesis trace with the fixed rubric.\n"
        "This score is heuristic: do not restate verifier metrics as ground truth, "
        "and do not reward confident prose over executable modeling quality.\n\n"
        f"ATIF session_id: {atif.get('session_id')}\n\n"
        f"Rubric:\n{rubric_payload}\n\n"
        f"Task prompt:\n{prompt_text}\n\n"
        f"Agent completion:\n{completion_text}\n\n"
        f"Verifier reward artifact summary:\n{verifier_payload}\n\n"
        "Return only JSON matching the requested schema."
    )
    return user, bool(prompt_truncated or completion_truncated)


def _rubric_response_schema() -> dict[str, Any]:
    criterion_properties = {
        criterion.criterion_id: {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        }
        for criterion in _RUBRIC_CRITERIA
    }
    return {
        "type": "json_schema",
        "name": "synthstats_trace_rubric_score",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "criterion_scores": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": criterion_properties,
                    "required": [criterion.criterion_id for criterion in _RUBRIC_CRITERIA],
                },
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["criterion_scores", "rationale", "confidence"],
        },
    }


def score_atif_rubrics_with_openai(
    atif_dir: Path,
    *,
    model: str,
    out_path: Path,
    limit: int | None = None,
    max_chars: int = 12000,
    reasoning_effort: str | None = "low",
) -> dict[str, Any]:
    """Score ATIF traces with an LLM rubric and write strict score JSON."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    trials = _trial_dirs(atif_dir)
    if limit is not None:
        trials = trials[:limit]
    artifact_dir = out_path.parent / "openai_artifacts"
    backend = OpenAIBackend(artifact_dir=artifact_dir)
    requests: list[tuple[str, LLMRequest]] = []
    request_to_atif: dict[str, str] = {}
    truncation: dict[str, bool] = {}
    for index, trial_dir in enumerate(trials):
        atif = ATIFRecord.model_validate_json(
            (trial_dir / "trajectory.json").read_text(encoding="utf-8")
        ).model_dump(mode="json")
        reward = RewardArtifact.model_validate_json(
            (trial_dir / "reward.json").read_text(encoding="utf-8")
        )
        atif_id = str(atif["session_id"])
        user, was_truncated = _rubric_user_prompt(
            atif,
            reward,
            max_chars=max_chars,
        )
        request_id = f"rubric-{index:06d}"
        request_to_atif[request_id] = atif_id
        truncation[atif_id] = was_truncated
        requests.append(
            (
                request_id,
                LLMRequest(
                    model=model,
                    system=(
                        "You are a strict statistical modeling trace judge. "
                        "Return calibrated numeric scores and a concise rationale."
                    ),
                    user=user,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=900,
                    text_format=_rubric_response_schema(),
                ),
            )
        )

    handle = backend.submit_batch(requests)
    results = backend.collect_batch(handle, allow_partial_failures=False)
    missing = [request_id for request_id, _ in requests if request_id not in results]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"rubric batch response missing {len(missing)} request_ids; first missing: {preview}"
        )
    scores: dict[str, dict[str, Any]] = {}
    raw_responses: dict[str, Any] = {}
    for request_id, _ in requests:
        atif_id = request_to_atif[request_id]
        result = results[request_id]
        parsed = json.loads(result.content)
        score = RubricScore.from_mapping(parsed)
        scores[atif_id] = score.to_dict()
        raw_responses[atif_id] = result.raw

    payload = {
        "schema_version": "ppl_synth_data.rich_reward_rubric_scores.v1",
        "source_atif_dir": str(atif_dir),
        "model": model,
        "scoring_backend": "openai_batch",
        "batch": {
            "backend": handle.backend,
            "job_id": handle.job_id,
            "request_count": len(requests),
        },
        "rubric": _RUBRIC.model_dump(mode="json"),
        "scores": scores,
        "prompt_truncated": truncation,
        "raw_responses": raw_responses,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    return payload


def _compose_rich_reward(
    *,
    base: RewardArtifact,
    atif: Mapping[str, Any],
    atif_id: str,
    score: RubricScore,
    heuristic_weight: float,
    process_weight: float,
    rubric_model: str | None = None,
    rubric_backend: str | None = None,
) -> RewardArtifact:
    if not math.isfinite(heuristic_weight) or heuristic_weight <= 0.0:
        raise ValueError("heuristic_weight must be finite and positive")
    if not math.isfinite(process_weight) or process_weight < 0.0:
        raise ValueError("process_weight must be finite and non-negative")
    if base.reward_source not in _BASE_VERIFIER_REWARD_SOURCES:
        raise ValueError(
            f"base reward artifact must be a verifier-owned ELPD reward; got {base.reward_source!r}"
        )
    if base.evidence_kind.value != "verifier" or base.truth_owner.value != "verifier":
        raise ValueError("base reward artifact must be verifier-owned")
    if not base.reward_components:
        raise ValueError("base reward artifact must expose reward_components")
    base_components = list(base.reward_components)
    if not any(
        component.kind is RewardComponentKind.terminal_verifier
        and component.truth_owner.value == "verifier"
        for component in base_components
    ):
        raise ValueError("base reward artifact must expose a terminal verifier component")
    base_names = [component.name for component in base_components]
    if len(set(base_names)) != len(base_names):
        raise ValueError("base reward component names must be unique")
    if _RUBRIC_COMPONENT_NAME in base_names:
        raise ValueError(f"base reward already has component {_RUBRIC_COMPONENT_NAME!r}")
    if process_weight > 0.0 and _PROCESS_INFO_COMPONENT_NAME in base_names:
        raise ValueError(f"base reward already has component {_PROCESS_INFO_COMPONENT_NAME!r}")

    rubric_score = score.weighted_score
    rubric_centered = rubric_score - 0.5
    rubric_log_delta = heuristic_weight * rubric_centered
    base_is_at_floor = float(base.training_log_reward) <= TRAINING_LOG_REWARD_MIN
    training_log_reward = (
        TRAINING_LOG_REWARD_MIN
        if base_is_at_floor
        else max(
            float(base.training_log_reward) + float(rubric_log_delta),
            TRAINING_LOG_REWARD_MIN,
        )
    )
    rubric_component = RewardComponent.model_validate(dict(
        name=_RUBRIC_COMPONENT_NAME,
        kind=RewardComponentKind.process_rubric,
        value=rubric_score,
        source=_RUBRIC_REWARD_SOURCE,
        evidence_kind="heuristic",
        truth_owner="heuristic",
        producer=_RUBRIC_PRODUCER,
        weight=heuristic_weight,
        log_contribution=rubric_log_delta,
        confidence=score.confidence,
        metrics=score.criterion_scores,
        diagnostics={"rationale": score.rationale},
        provenance={
            "rubric_id": _RUBRIC_ID,
            "rubric_version": _RUBRIC_VERSION,
            "atif_id": atif_id,
            "base_reward_source": base.reward_source,
        },
    ))
    reward_components = (*base_components, rubric_component)
    process_component: RewardComponent | None = None
    process_score = 0.0
    process_metrics: dict[str, Any] = {}
    process_log_delta = 0.0
    if process_weight > 0.0:
        process_score, process_metrics = _trace_process_information_score(atif)
        process_log_delta = max(
            float(process_weight) * (process_score - 0.5),
            TRAINING_LOG_REWARD_MIN,
        )
        process_component = RewardComponent.model_validate(dict(
            name=_PROCESS_INFO_COMPONENT_NAME,
            kind=RewardComponentKind.process_information_gain,
            value=process_score,
            source=_PROCESS_INFO_REWARD_SOURCE,
            evidence_kind="heuristic",
            truth_owner="heuristic",
            producer=_PROCESS_INFO_PRODUCER,
            weight=process_weight,
            log_contribution=process_log_delta,
            confidence=0.65,
            metrics=process_metrics,
            diagnostics={
                "calibration": "deterministic process proxy, not verifier-owned Bayesian EIG",
            },
            provenance={
                "atif_id": atif_id,
                "base_reward_source": base.reward_source,
                "source": "ppl_synth_data.dataset.rich_reward_export",
            },
        ))
        reward_components = (*reward_components, process_component)

    training_log_reward = (
        TRAINING_LOG_REWARD_MIN
        if base_is_at_floor
        else max(
            float(training_log_reward) + float(process_log_delta),
            TRAINING_LOG_REWARD_MIN,
        )
    )
    original_weights: dict[str, float] = {}
    if base.scalarization is not None:
        original_weights.update(base.scalarization.weights)
    weights = {
        component.name: float(original_weights.get(component.name, component.weight))
        for component in base_components
    }
    weights[_RUBRIC_COMPONENT_NAME] = float(heuristic_weight)
    if process_component is not None:
        weights[_PROCESS_INFO_COMPONENT_NAME] = float(process_weight)
    component_names = tuple(component.name for component in reward_components)
    components = dict(base.components)
    for component in reward_components:
        components.setdefault(component.name, float(component.value))
    metrics = dict(base.metrics)
    metrics.update(
        {
            "base_training_log_reward": float(base.training_log_reward),
            "rubric_score": rubric_score,
            "rubric_log_delta": rubric_log_delta,
            "heuristic_weight": float(heuristic_weight),
            "process_information_score": process_score,
            "process_information_log_delta": process_log_delta,
            "process_weight": float(process_weight),
        }
    )
    diagnostics = dict(base.diagnostics)
    diagnostics["rich_reward"] = {
        "base_reward_source": base.reward_source,
        "base_truth_owner": base.truth_owner.value,
        "heuristic_component": _RUBRIC_COMPONENT_NAME,
        "process_information_component": (
            _PROCESS_INFO_COMPONENT_NAME if process_component is not None else None
        ),
        "process_information_score": process_score,
        "rubric_rationale": score.rationale,
    }
    provenance = dict(base.provenance)
    provenance["rich_reward"] = {
        "source": "ppl_synth_data.dataset.rich_reward_export",
        "base_reward_source": base.reward_source,
        "rubric_id": _RUBRIC_ID,
        "rubric_version": _RUBRIC_VERSION,
        "process_information_proxy": process_metrics,
    }
    if rubric_model is not None:
        provenance["rich_reward"]["rubric_model"] = rubric_model
    if rubric_backend is not None:
        provenance["rich_reward"]["rubric_backend"] = rubric_backend
    return RewardArtifact.model_validate(dict(
        schema_version=_REWARD_SCHEMA_VERSION,
        reward=math.exp(min(training_log_reward, 20.0)),
        training_log_reward=training_log_reward,
        reward_transform={
            "kind": "verifier_log_reward_plus_process_components",
            "base_training_log_reward": float(base.training_log_reward),
            "rubric_centered": rubric_centered,
            "heuristic_weight": float(heuristic_weight),
            "rubric_log_delta": rubric_log_delta,
            "process_information_centered": process_score - 0.5,
            "process_weight": float(process_weight),
            "process_information_log_delta": process_log_delta,
            "lower_log_reward_bound": TRAINING_LOG_REWARD_MIN,
        },
        reward_source=_COMPOSITE_REWARD_SOURCE,
        evidence_kind="heuristic",
        truth_owner="heuristic",
        producer=_COMPOSITE_PRODUCER,
        metrics=metrics,
        diagnostics=diagnostics,
        provenance=provenance,
        components=components,
        reward_components=reward_components,
        scalarization=RewardScalarization(
            method=RewardScalarizationMethod.verifier_gated_weighted_sum,
            component_names=component_names,
            weights=weights,
            lower_log_reward_bound=TRAINING_LOG_REWARD_MIN,
            diagnostics={
                "aggregate": "training_log_reward",
                "base_reward_source": base.reward_source,
                "base_scalarization": (
                    base.scalarization.model_dump(mode="json")
                    if base.scalarization is not None
                    else None
                ),
                "heuristic_component": _RUBRIC_COMPONENT_NAME,
                "heuristic_weight": float(heuristic_weight),
                "process_information_component": (
                    _PROCESS_INFO_COMPONENT_NAME if process_component is not None else None
                ),
                "process_weight": float(process_weight),
            },
        ),
        rubric=_RUBRIC,
    ))


def export_rich_reward_atif_dir(
    atif_dir: Path,
    out_dir: Path,
    *,
    rubric_scores_path: Path,
    heuristic_weight: float = 0.1,
    process_weight: float = 0.05,
    min_exported: int | None = None,
) -> dict[str, Any]:
    """Write a fresh ATIF dir with verifier, rubric, and process components."""
    scores = _load_scores(rubric_scores_path)
    trials = _trial_dirs(atif_dir)
    _assert_fresh_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    dropped: list[dict[str, str]] = []
    exported_component_sources: set[str] = set()
    for trial_dir in trials:
        atif = ATIFRecord.model_validate_json(
            (trial_dir / "trajectory.json").read_text(encoding="utf-8")
        ).model_dump(mode="json")
        base_reward = RewardArtifact.model_validate_json(
            (trial_dir / "reward.json").read_text(encoding="utf-8")
        )
        atif_id = str(atif["session_id"])
        score = scores.get(atif_id)
        if score is None:
            dropped.append({"atif_id": atif_id, "reason": "missing rubric score"})
            continue
        rich_reward = _compose_rich_reward(
            base=base_reward,
            atif=atif,
            atif_id=atif_id,
            score=score,
            heuristic_weight=heuristic_weight,
            process_weight=process_weight,
        )
        exported_component_sources.update(
            component.source for component in rich_reward.reward_components
        )
        atif["reward_artifact"] = rich_reward.model_dump(mode="json")
        validated_atif = ATIFRecord.model_validate(atif).model_dump(mode="json")

        dst = out_dir / trial_dir.name
        dst.mkdir(parents=True, exist_ok=False)
        write_json(dst / "trajectory.json", validated_atif)
        write_json(dst / "reward.json", rich_reward.model_dump(mode="json"))
        exported.append(atif_id)

    if min_exported is not None:
        if min_exported <= 0:
            raise ValueError("min_exported must be positive")
        if len(exported) < min_exported:
            raise ValueError(
                f"rich reward export produced {len(exported)} rows, "
                f"below required minimum {min_exported}"
            )

    manifest = {
        "status": "success",
        "schema_version": "ppl_synth_data.rich_reward_atif_export.v1",
        "source_atif_dir": str(atif_dir),
        "out_dir": str(out_dir),
        "rubric_scores_path": str(rubric_scores_path),
        "reward_source": _COMPOSITE_REWARD_SOURCE,
        "base_reward_sources": sorted(
            {
                RewardArtifact.model_validate_json(
                    (trial / "reward.json").read_text(encoding="utf-8")
                ).reward_source
                for trial in trials
            }
        ),
        "component_reward_sources": sorted(exported_component_sources),
        "rubric_id": _RUBRIC_ID,
        "rubric_version": _RUBRIC_VERSION,
        "heuristic_weight": float(heuristic_weight),
        "process_weight": float(process_weight),
        "count_input": len(trials),
        "count_exported": len(exported),
        "count_dropped": len(dropped),
        "atif_ids": exported,
        "dropped": dropped,
        "training_input_mode": "message_only_requires_policy_rescore",
    }
    manifest = RichRewardATIFExportManifest.model_validate(manifest).model_dump(mode="json")
    write_json(out_dir / "manifest.json", manifest)
    tmp_path = out_dir / ".dropped.jsonl.tmp"
    tmp_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dropped),
        encoding="utf-8",
    )
    os.replace(tmp_path, out_dir / "dropped.jsonl")
    write_json(
        out_dir / "metadata.json",
        {
            "schema_version": "ppl_synth_data.rich_reward_rubric_scores.copy.v1",
            "source": str(rubric_scores_path),
            "scores": {key: score.to_dict() for key, score in scores.items()},
        },
    )
    return manifest


def _first_existing_path(candidates: Sequence[Path], *, label: str) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    preview = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find {label}; checked {preview}")


def _load_first_valid_atif(
    candidates: Sequence[Path],
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    invalid: list[str] = []
    missing: list[str] = []
    for path in candidates:
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            atif = ATIFRecord.model_validate_json(path.read_text(encoding="utf-8")).model_dump(
                mode="json"
            )
        except ValidationError as exc:
            errors = exc.errors()
            message = str(errors[0].get("msg", str(exc))) if errors else str(exc)
            invalid.append(f"{path}: {message}")
            continue
        return path, atif
    checked = ", ".join(str(path) for path in candidates)
    if invalid:
        raise ValueError(
            f"could not find valid {label}; invalid candidates: "
            f"{'; '.join(invalid)}; checked: {checked}"
        )
    raise FileNotFoundError(f"could not find {label}; missing candidates: {', '.join(missing)}")


def _reward_candidates(trial_dir: Path, reward_path: Path | None) -> tuple[Path, ...]:
    candidates = [
        trial_dir / "reward.json",
        trial_dir / "artifacts" / "reward.json",
        trial_dir / "logs" / "artifacts" / "reward.json",
    ]
    if reward_path is not None:
        candidates.insert(0, reward_path)
    return tuple(dict.fromkeys(candidates))


def _trajectory_candidates(
    trial_dir: Path,
    trajectory_path: Path | None,
) -> tuple[Path, ...]:
    candidates = [
        trial_dir / "trajectory.json",
        trial_dir / "artifacts" / "trajectory.json",
        trial_dir / "agent" / "trajectory.json",
        trial_dir / "logs" / "artifacts" / "trajectory.json",
        trial_dir / "logs" / "agent" / "trajectory.json",
    ]
    if trajectory_path is not None:
        candidates.insert(0, trajectory_path)
    return tuple(dict.fromkeys(candidates))


def _score_single_trace_rubric_with_openai(
    *,
    atif: Mapping[str, Any],
    reward: RewardArtifact,
    model: str,
    artifact_dir: Path,
    max_chars: int,
    reasoning_effort: str | None,
) -> tuple[RubricScore, dict[str, Any]]:
    backend = OpenAIBackend(artifact_dir=artifact_dir)
    user, was_truncated = _rubric_user_prompt(atif, reward, max_chars=max_chars)
    request = LLMRequest(
        model=model,
        system=(
            "You are a strict statistical modeling trace judge. "
            "Return calibrated numeric scores and a concise rationale."
        ),
        user=user,
        reasoning_effort=reasoning_effort,
        max_output_tokens=900,
        text_format=_rubric_response_schema(),
    )
    result = backend.complete(request)
    parsed = json.loads(result.content)
    return (
        RubricScore.from_mapping(parsed),
        {
            "model": model,
            "scoring_backend": "openai_responses",
            "prompt_truncated": was_truncated,
            "raw_response": result.raw,
        },
    )


def compose_live_rich_reward(
    *,
    trial_dir: Path,
    trajectory_path: Path | None = None,
    reward_path: Path | None = None,
    rubric_model: str = "gpt-5.5",
    heuristic_weight: float = 0.1,
    process_weight: float = 0.05,
    max_judge_chars: int = 12000,
    reasoning_effort: str | None = "low",
) -> dict[str, Any]:
    """Synchronously enrich one live Harbor trial with a rich reward artifact."""
    trial_dir = Path(trial_dir)
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"trial_dir not found: {trial_dir}")
    reward_candidates = _reward_candidates(trial_dir, reward_path)
    trajectory_candidates = _trajectory_candidates(trial_dir, trajectory_path)
    base_reward_path = _first_existing_path(reward_candidates, label="reward.json")
    base_reward = RewardArtifact.model_validate_json(base_reward_path.read_text(encoding="utf-8"))
    base_trajectory_path, atif = _load_first_valid_atif(
        trajectory_candidates,
        label="trajectory.json",
    )
    atif_id = str(atif["session_id"])
    score, rubric_metadata = _score_single_trace_rubric_with_openai(
        atif=atif,
        reward=base_reward,
        model=rubric_model,
        artifact_dir=trial_dir / "rich_reward_openai_artifacts",
        max_chars=max_judge_chars,
        reasoning_effort=reasoning_effort,
    )
    rich_reward = _compose_rich_reward(
        base=base_reward,
        atif=atif,
        atif_id=atif_id,
        score=score,
        heuristic_weight=heuristic_weight,
        process_weight=process_weight,
        rubric_model=rubric_model,
        rubric_backend="openai_responses",
    )
    atif["reward_artifact"] = rich_reward.model_dump(mode="json")
    validated_atif = ATIFRecord.model_validate(atif).model_dump(mode="json")

    written_rewards: list[str] = []
    for path in reward_candidates:
        if path.is_file():
            write_json(path, rich_reward.model_dump(mode="json"))
            written_rewards.append(str(path))
    written_trajectories: list[str] = []
    for path in trajectory_candidates:
        if path.is_file():
            write_json(path, validated_atif)
            written_trajectories.append(str(path))
    manifest = {
        "status": "success",
        "schema_version": "ppl_synth_data.live_rich_reward.v1",
        "trial_dir": str(trial_dir),
        "atif_id": atif_id,
        "base_trajectory_path": str(base_trajectory_path),
        "reward_source": rich_reward.reward_source,
        "component_reward_sources": sorted(
            {component.source for component in rich_reward.reward_components}
        ),
        "component_names": [component.name for component in rich_reward.reward_components],
        "scalarization": rich_reward.scalarization.model_dump(mode="json")
        if rich_reward.scalarization is not None
        else None,
        "rubric": rubric_metadata,
        "written_reward_paths": written_rewards,
        "written_trajectory_paths": written_trajectories,
    }
    manifest = LiveRichRewardManifest.model_validate(manifest).model_dump(mode="json")
    write_json(trial_dir / "live_rich_reward_manifest.json", manifest)
    return manifest


__all__ = [
    "RubricScore",
    "compose_live_rich_reward",
    "export_rich_reward_atif_dir",
    "score_atif_rubrics_with_openai",
]

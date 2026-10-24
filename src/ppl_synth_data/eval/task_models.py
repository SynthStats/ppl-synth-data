"""Producer-internal task/evaluation-spec models.

These are NOT part of the cross-repo ABI (``synthstats_schemas``). They exist
solely to support ``schemas.adapters.task_spec_from_env_spec``, which projects
an ``EnvSpec`` into a task description. Nothing outside this producer consumes
them, so they stay strict Pydantic models local to ``ppl_synth_data`` rather
than living in the curated cross-repo ``schemas`` surface.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from synthstats_schemas.base import TASK_SCHEMA_VERSION
from synthstats_schemas.trace import RuntimeDescriptor


class ObservationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_type: str | None = None
    input_description: str | None = None
    output_type: str | None = None
    target_name: str | None = None
    feature_names: list[str] = Field(default_factory=list)
    grouping: list[str] = Field(default_factory=list)
    covariates: list[str] = Field(default_factory=list)
    n_observations: int | None = Field(default=None, ge=1)


class EvaluationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_metric: str
    metrics: list[str] = Field(default_factory=list)
    split_policy: str | None = None
    verifier: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = TASK_SCHEMA_VERSION
    task_id: str
    title: str
    domain: str | None = None
    complexity: str | None = None
    model_type: str | None = None
    generative_family: str | None = None
    key_concepts: list[str] = Field(default_factory=list)
    prompt_guidance: dict[str, Any] = Field(default_factory=dict)
    observation_schema: ObservationSchema
    evaluation_spec: EvaluationSpec
    runtime: RuntimeDescriptor | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "EvaluationSpec",
    "ObservationSchema",
    "TaskSpec",
]

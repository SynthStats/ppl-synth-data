from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from ppl_synth_data.evidence import EvidenceKind, SemanticStatus, TruthOwner

ENV_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

FiniteCountParameter = Annotated[float, Field(strict=True, allow_inf_nan=False)]
CountFeatureName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")]


class _CountResponseBase(BaseModel):
    """The EnvSpec-4F count response on raw feature values."""

    model_config = ConfigDict(extra="forbid", strict=True)

    family: Literal["regression_count"]
    intercept_log_rate: FiniteCountParameter
    coefficients: dict[CountFeatureName, FiniteCountParameter] = Field(min_length=1)


class PoissonCountResponse(_CountResponseBase):
    count_distribution: Literal["poisson"]


class NB2CountResponse(_CountResponseBase):
    count_distribution: Literal["nb2"]
    nb2_alpha: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]


CountResponse = Annotated[
    PoissonCountResponse | NB2CountResponse, Field(discriminator="count_distribution")
]


class _ClosedDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_serializer(mode="wrap")
    def _serialize_supplied(self, handler):
        return {key: value for key, value in handler(self).items() if key in self.model_fields_set}


class NormalInput(_ClosedDeclaration):
    dist: Literal["normal"]
    mean: FiniteCountParameter
    sd: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class TruncatedNormalInput(_ClosedDeclaration):
    """Parent Normal location and scale, conditioned on the declared interval."""

    dist: Literal["truncated_normal"]
    mean: FiniteCountParameter
    sd: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    low: FiniteCountParameter | None = None
    high: FiniteCountParameter | None = None

    @model_validator(mode="after")
    def _bounds(self):
        field = None
        if self.low is None and self.high is None:
            field, message = "low", "at least one numeric bound is required"
        elif self.low is not None and self.high is not None and self.low >= self.high:
            field, message = "high", "high must exceed low"
        elif (self.low is not None and self.mean < self.low) or (
            self.high is not None and self.mean > self.high
        ):
            field, message = "mean", "parent mean must lie inside the bounds"
        if field is not None:
            # Field-local errors retain their full JSON pointer when nested.
            raise ValidationError.from_exception_data(
                type(self).__name__,
                [
                    {
                        "type": "value_error",
                        "loc": (field,),
                        "input": getattr(self, field),
                        "ctx": {"error": ValueError(message)},
                    }
                ],
            )
        return self


class DiscreteUniformInput(_ClosedDeclaration):
    dist: Literal["discrete_uniform"]
    low: int
    high: int

    @model_validator(mode="after")
    def _bounds(self):
        if self.low >= self.high:
            raise ValueError("/law/high: must exceed /law/low")
        return self


class BernoulliInput(_ClosedDeclaration):
    dist: Literal["bernoulli"]
    p: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]


class CategoricalInput(_ClosedDeclaration):
    dist: Literal["categorical"]
    levels: list[Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_]{0,30}$")]] = Field(
        min_length=2, max_length=8
    )
    probs: list[Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]] = Field(
        min_length=2, max_length=8
    )

    @model_validator(mode="after")
    def _probabilities(self):
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("/law/levels: duplicate levels")
        if len(self.probs) != len(self.levels) or not math.isclose(
            math.fsum(self.probs), 1.0, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("/law/probs: require one probability per level summing to 1")
        return self


class FixedCountsInput(_ClosedDeclaration):
    dist: Literal["fixed_counts"]
    levels: list[int | FiniteCountParameter | str] = Field(min_length=2, max_length=24)
    counts: list[Annotated[int, Field(ge=1)]] = Field(min_length=2, max_length=24)

    @model_validator(mode="after")
    def _allocation(self):
        if len(set(self.levels)) != len(self.levels) or len(set(map(str, self.levels))) != len(
            self.levels
        ):
            raise ValueError("/law/levels: duplicate or ambiguous reference-coding levels")
        if len(self.levels) != len(self.counts):
            raise ValueError("/law/counts: require one count per level")
        return self


InputLaw = Annotated[
    NormalInput
    | TruncatedNormalInput
    | DiscreteUniformInput
    | BernoulliInput
    | CategoricalInput
    | FixedCountsInput,
    Field(discriminator="dist"),
]


class DeclaredFeature(_ClosedDeclaration):
    name: CountFeatureName
    law: InputLaw
    units: str = ""

    @property
    def uses_reference_coding(self) -> bool:
        return isinstance(self.law, CategoricalInput) or (
            isinstance(self.law, FixedCountsInput) and isinstance(self.law.levels[0], str)
        )


class _ResponseDeclaration(_ClosedDeclaration):
    coefficients: dict[str, FiniteCountParameter] = Field(min_length=1)


class ContinuousResponse(_ResponseDeclaration):
    family: Literal["regression_continuous"]
    intercept: FiniteCountParameter
    noise_sigma: Annotated[float, Field(gt=0, allow_inf_nan=False)]


CountFamily = TypeVar("CountFamily", bound=str)


class _CountDeclarationBase(_ResponseDeclaration, Generic[CountFamily]):
    family: CountFamily
    intercept_log_rate: FiniteCountParameter
    count_distribution: Literal["poisson", "nb2"]
    nb2_alpha: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def _dispersion(self):
        if (self.count_distribution == "nb2") != ("nb2_alpha" in self.model_fields_set):
            raise ValueError("/response/nb2_alpha: required exactly when count_distribution is nb2")
        if self.count_distribution == "nb2" and self.nb2_alpha is None:
            raise ValueError("/response/nb2_alpha: requires a positive shape")
        return self


class CountDeclaration(_CountDeclarationBase[Literal["regression_count"]]):
    pass


class ZeroInflatedResponse(_CountDeclarationBase[Literal["zero_inflated_count"]]):
    zero_inflation: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]


class BinaryResponse(_ResponseDeclaration):
    family: Literal["binary_choice"]
    intercept: FiniteCountParameter
    link: Literal["logit"]


class ScenarioDeclaration(_ClosedDeclaration):
    intent: str = Field(min_length=1)
    source_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ObservationPlan(_ClosedDeclaration):
    n_observations: int = Field(ge=20, le=5000)


class FourFamilySpec(_ClosedDeclaration):
    """Closed EnvSpec-4F v0.2 declarations, separate from historical EnvSpecs."""

    env_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,80}$")
    env_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    scenario: ScenarioDeclaration
    generative_family: Literal[
        "regression_continuous", "regression_count", "zero_inflated_count", "binary_choice"
    ]
    features: list[DeclaredFeature] = Field(min_length=1, max_length=8)
    response: Annotated[
        ContinuousResponse | CountDeclaration | ZeroInflatedResponse | BinaryResponse,
        Field(discriminator="family"),
    ]
    data_plan: ObservationPlan
    units: dict[str, str] = Field(default_factory=dict)
    allowed_hints: list[str] = Field(default_factory=list)
    forbidden_hints: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _fixed_count_level_types(cls, value):
        if isinstance(value, dict) and isinstance(value.get("features"), list):
            for index, feature in enumerate(value["features"]):
                law = feature.get("law") if isinstance(feature, dict) else None
                if not isinstance(law, dict) or law.get("dist") != "fixed_counts":
                    continue
                levels = law.get("levels")
                if (
                    isinstance(levels, list)
                    and any(type(v) in (int, float) for v in levels)
                    and any(isinstance(v, str) for v in levels)
                ):
                    raise ValueError(
                        f"/features/{index}/law/levels: mixed numeric and string levels "
                        "have no declared coding"
                    )
        return value

    @model_validator(mode="after")
    def _cross_fields(self):
        if self.response.family != self.generative_family:
            raise ValueError("/response/family: must equal /generative_family")
        names = [feature.name for feature in self.features]
        if len(set(names)) != len(names):
            raise ValueError("/features: duplicate names")
        expected: set[str] = set()
        allocations = 0
        for index, feature in enumerate(self.features):
            law = feature.law
            if feature.uses_reference_coding and isinstance(
                law, (CategoricalInput, FixedCountsInput)
            ):
                expected.update(f"{feature.name}={level}" for level in law.levels[1:])
            else:
                expected.add(feature.name)
            if isinstance(law, FixedCountsInput):
                allocations += 1
                if sum(law.counts) != self.data_plan.n_observations:
                    raise ValueError(
                        f"/features/{index}/law/counts: sum must equal /data_plan/n_observations"
                    )
        if allocations > 1:
            raise ValueError(
                
                    "/features: multiple fixed_counts laws require unsupport"
                    "ed joint allocation semantics"
                
            )
        if set(self.response.coefficients) != expected:
            raise ValueError(f"/response/coefficients: expected exactly {sorted(expected)!r}")
        return self


def _non_empty_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


class Complexity(StrEnum):
    simple = "simple"
    medium = "medium"
    complex = "complex"


class InputType(StrEnum):
    float = "float"
    int = "int"
    json = "json"
    string = "string"
    none = "none"

    @classmethod
    def _missing_(cls, value: object) -> InputType | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        aliases = {"number": cls.float, "integer": cls.int}
        return aliases.get(normalized)


class IOSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_type: InputType
    input_description: str
    output_type: str
    target_name: str
    feature_names: list[str] = Field(default_factory=list)
    feature_types: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Declared type per feature. Stage C supports only independent continuous "
            "features with explicit bounds; missing or other types fail at use."
        ),
    )
    feature_domains: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_feature_domains(self) -> IOSchema:
        for name, (low, high) in self.feature_domains.items():
            singleton_integer = (
                self.feature_types.get(name) == "integer" and low == high and low.is_integer()
            )
            if not (low < high or singleton_integer):
                raise ValueError(
                    f"IOSchema.feature_domains[{name!r}] must have low < high "
                    f"(or a singleton integer domain), got ({low}, {high})"
                )
        return self


class DataPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_observations: int = Field(ge=1)
    grouping: list[str] = Field(default_factory=list)
    covariates: list[str] = Field(default_factory=list)


class EnvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str
    env_name: str

    @field_validator("env_id")
    @classmethod
    def _validate_env_id(cls, v: str) -> str:
        if not ENV_ID_RE.match(v):
            raise ValueError(f"env_id must match {ENV_ID_RE.pattern!r}, got {v!r}")
        return v

    domain: str
    complexity: Complexity
    model_type: str
    key_concepts: list[str]

    io_schema: IOSchema
    data_plan: DataPlan

    generative_family: str
    generative_params: dict[str, Any] = Field(default_factory=dict)
    prompt_guidance: dict[str, Any] = Field(default_factory=dict)


class PromptProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_name: str | None = None
    domain: str | None = None
    complexity: str | None = None
    model_type: str | None = None
    generative_family: str | None = None
    key_concepts: list[str] = Field(default_factory=list)
    has_observed_data: bool | None = None
    runtime_name: str | None = None
    runtime_version: str | None = None
    source_category: str | None = None
    source_url_or_repo: str | None = None
    license_signal: str | None = None
    adaptation_mode: str | None = None
    source_retrieval_date: str | None = None


class ValidationTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=0)
    phase: str
    ok: bool
    message: str
    evidence_kind: EvidenceKind | None = None
    truth_owner: TruthOwner | None = None
    semantic_status: SemanticStatus | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[ValidationTraceStep] = Field(default_factory=list)


class SemanticValidationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pymc_draws: int = Field(gt=0)
    pymc_tune: int = Field(gt=0)
    pymc_timeout_sec: float = Field(gt=0)


class ValidationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = False
    message: str = ""
    repaired: bool = False
    leakage_rating: str | None = None
    sample_data_error: str | None = None
    evidence_kind: EvidenceKind | None = None
    truth_owner: TruthOwner | None = None
    semantic_status: SemanticStatus | None = None
    semantic_metric: str | None = None
    semantic_score: float | None = None
    semantic_settings: SemanticValidationSettings | None = None
    semantic_details: dict[str, Any] = Field(default_factory=dict)
    trace: ValidationTrace | None = None


class PromptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    split: str
    env_id: str
    prompt_text: str
    provenance: PromptProvenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt_id: str
    env_id: str = ""
    split: str = ""
    prompt_text: str
    raw_completion: str
    candidate_outputs: list[str] = Field(default_factory=list)
    provider_response: dict[str, Any] = Field(default_factory=dict)
    parsed_program: str | None = None
    provenance: PromptProvenance | None = None
    validation: ValidationInfo | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CritiqueRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    completion_id: str
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    env_spec: EnvSpec
    prompt: PromptRecord
    completion: CompletionRecord | None = None
    critique: CritiqueRecord | None = None

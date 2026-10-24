from __future__ import annotations

import json
from typing import Any

from ppl_synth_data.llm.backend import BackendCapabilities, LLMRequest, LLMResult
from ppl_synth_data.specs.models import Complexity, DataPlan, EnvSpec, IOSchema


class CaptureBackend:
    capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=True,
    )

    def __init__(self, responses: list[str | tuple[str, dict[str, Any]]]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if not self.responses:
            raise IndexError(
                f"CaptureBackend exhausted: {len(self.requests)} calls "
                f"but only {len(self.requests) - 1} responses were scripted"
            )
        scripted = self.responses.pop(0)
        if isinstance(scripted, tuple):
            text, raw = scripted
        else:
            text, raw = scripted, {}
        return LLMResult(content=text, raw=raw)


def make_numeric_spec(feature_names: list[str], *, input_type: str = "float") -> EnvSpec:
    weights = (1.0, 1.15, 1.3)
    return EnvSpec(
        env_id="orchestrator_query_spec_v1",
        env_name="orchestrator_query_spec_v1",
        domain="healthcare",
        complexity=Complexity.medium,
        model_type="predictive",
        key_concepts=["regression"],
        io_schema=IOSchema(
            input_type=input_type,
            input_description="numeric features",
            output_type="float",
            target_name="y",
            feature_names=feature_names,
            feature_types={name: "continuous" for name in feature_names},
            feature_domains={name: (-5.0, 5.0) for name in feature_names},
        ),
        data_plan=DataPlan(n_observations=10, grouping=[], covariates=[]),
        generative_family="regression_continuous",
        generative_params={
            "intercept": 0.0,
            "noise_sigma": 1.0,
            "coefficients": dict(zip(feature_names, weights[: len(feature_names)], strict=True)),
        },
        prompt_guidance={},
    )


def minimal_envspec_record(*, env_id: str, env_name: str) -> dict[str, object]:
    return {
        "env_id": env_id,
        "env_name": env_name,
        "domain": "healthcare",
        "complexity": "medium",
        "model_type": "predictive",
        "key_concepts": ["hierarchical_model"],
        "io_schema": {
            "input_type": "json",
            "input_description": "JSON",
            "output_type": "float",
            "target_name": "y",
            "feature_names": ["group"],
            "feature_types": {"group": "continuous"},
            "feature_domains": {"group": [0.0, 10.0]},
        },
        "data_plan": {
            "n_observations": 3,
            "grouping": ["group"],
            "covariates": [],
        },
        "generative_family": "hierarchical_grouped",
        "generative_params": {"base_mean": 0.0, "noise_sigma": 1.0},
        "prompt_guidance": {},
    }


def write_jsonl(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def ensure_minimal_configs(root) -> None:
    """Create a minimal configs/ directory so load_pipeline_defaults() succeeds."""
    from pathlib import Path

    cfg = Path(root) / "configs"
    (cfg / "stages").mkdir(parents=True, exist_ok=True)
    if not (cfg / "default.yaml").exists():
        (cfg / "default.yaml").write_text("run:\n  root: artifacts/runs\n", encoding="utf-8")

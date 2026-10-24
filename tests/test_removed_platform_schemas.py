"""Schema ownership and import boundaries.

Trace records belong to synthstats_schemas.trace. TaskSpec, ObservationSchema
and EvaluationSpec belong to ppl_synth_data.eval.task_models. The tests reject
a schemas.platform module or schemas package re-exports, and check the types
returned by schemas.adapters.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
from synthstats_schemas.trace import (
    GenerationRecord as ContractsGenerationRecord,
)
from synthstats_schemas.trace import (
    PromptView as ContractsPromptView,
)
from synthstats_schemas.trace import (
    RuntimeDescriptor as ContractsRuntimeDescriptor,
)
from synthstats_schemas.trace import (
    TraceRecord as ContractsTraceRecord,
)
from synthstats_schemas.trace import (
    TraceStep as ContractsTraceStep,
)
from synthstats_schemas.trace import (
    TrialRecord as ContractsTrialRecord,
)

import ppl_synth_data.schemas.adapters as schema_adapters
from ppl_synth_data.eval.task_models import TaskSpec
from ppl_synth_data.runtime_registry import (
    build_harbor_task_id,
    default_runtime_name,
    get_runtime_spec,
    harbor_task_id_prefix,
    normalize_runtime_name,
    runtime_name_from_metadata,
    runtime_name_from_payload,
    runtime_spec_from_payload,
    score_artifact_key_from_payload,
)
from ppl_synth_data.schemas.adapters import (
    generation_record_from_completion_record,
    task_spec_from_env_spec,
    trace_record_from_dataset_row,
    trace_record_from_messages,
    trial_record_from_agent_result,
)
from ppl_synth_data.specs.models import (
    CompletionRecord,
    Complexity,
    DataPlan,
    EnvSpec,
    InputType,
    IOSchema,
    PromptProvenance,
    ValidationInfo,
    ValidationTrace,
)


def test_platform_module_no_longer_exists() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ppl_synth_data.schemas.platform")


def test_schemas_package_has_no_curated_reexport_surface() -> None:
    import ppl_synth_data.schemas as schemas_package

    # No curated public API left: nothing re-exported from the package root.
    assert not hasattr(schemas_package, "TraceRecord")
    assert not hasattr(schemas_package, "TaskSpec")
    assert not hasattr(schemas_package, "generation_record_from_completion_record")


def test_adapters_produce_contracts_models() -> None:
    prompt = schema_adapters.prompt_view_from_prompt_record(
        {"id": "prompt-001", "env_id": "dugongs", "prompt_text": "hi"}
    )
    assert isinstance(prompt, ContractsPromptView)

    trial = trial_record_from_agent_result(
        {
            "program": "model code",
            "language": "pymc",
            "reward": 1.0,
            "task_config": {"env": {"name": "dugongs"}},
        },
        trial_id="trial-001",
    )
    assert isinstance(trial, ContractsTrialRecord)
    assert trial.runtime is not None
    assert isinstance(trial.runtime, ContractsRuntimeDescriptor)

    trace = trace_record_from_messages(
        [{"role": "user", "content": "hi"}],
        trace_id="trace-001",
    )
    assert isinstance(trace, ContractsTraceRecord)
    assert len(trace.steps) == 1
    assert isinstance(trace.steps[0], ContractsTraceStep)

    generation = generation_record_from_completion_record(
        {
            "id": "completion-001",
            "prompt_id": "prompt-001",
            "env_id": "dugongs",
            "raw_completion": "model code",
        }
    )
    assert isinstance(generation, ContractsGenerationRecord)

    task_spec = task_spec_from_env_spec({"env_id": "dugongs", "env_name": "Dugongs"})
    assert isinstance(task_spec, TaskSpec)


def test_env_spec_maps_to_producer_internal_task_spec() -> None:
    env_spec = EnvSpec(
        env_id="dugongs",
        env_name="Dugongs",
        domain="ecology",
        complexity=Complexity.medium,
        model_type="hierarchical",
        key_concepts=["pooled estimates", "counts"],
        io_schema=IOSchema(
            input_type=InputType.float,
            input_description="observed counts by region",
            output_type="posterior_samples",
            target_name="count",
            feature_names=["region", "year"],
        ),
        data_plan=DataPlan(
            n_observations=128,
            grouping=["region"],
            covariates=["year"],
        ),
        generative_family="poisson",
        generative_params={"mu": 3.0},
        prompt_guidance={"style": "observational"},
    )

    task_spec = task_spec_from_env_spec(
        env_spec.model_dump(mode="python"),
        runtime_name="pymc",
        runtime_version="5.20.0",
    )

    assert isinstance(task_spec, TaskSpec)
    assert task_spec.task_id == "dugongs"
    assert task_spec.title == "Dugongs"
    assert task_spec.runtime is not None
    assert isinstance(task_spec.runtime, ContractsRuntimeDescriptor)
    assert task_spec.runtime.name == "pymc"
    assert task_spec.runtime.execution_backend == "python"
    assert task_spec.observation_schema.target_name == "count"
    assert task_spec.observation_schema.grouping == ["region"]
    assert task_spec.evaluation_spec.primary_metric == "semantic_validation"


def test_runtime_registry_normalizes_aliases_and_metadata() -> None:
    assert default_runtime_name() == "pymc"
    assert normalize_runtime_name("haskell") == "lazyppl"
    assert runtime_name_from_metadata({"lazyppl_version": "lazyppl-1.0"}) == "lazyppl"
    assert harbor_task_id_prefix("haskell") == "lazyppl"
    assert (
        build_harbor_task_id("haskell", "gp_regression", sequence=1)
        == "lazyppl-gp-regression-00001"
    )

    runtime_spec = get_runtime_spec("lazy-ppl")

    assert runtime_spec.name == "lazyppl"
    assert runtime_spec.metadata_version_key == "lazyppl_version"
    assert runtime_spec.execution_backend == "haskell"
    assert runtime_spec.harbor_docker_image is None
    assert runtime_spec.harbor_verifier_timeout_sec == 180.0


def test_runtime_registry_resolves_runtime_from_payload() -> None:
    payload = {
        "shared_trial_record": {
            "trial_id": "completion-001",
            "runtime": {"name": "haskell", "version": "1.0.0"},
        },
        "lazyppl_version": "1.0.0",
    }

    runtime_name = runtime_name_from_payload(payload)
    runtime_spec = runtime_spec_from_payload(payload)
    score_artifact_key = score_artifact_key_from_payload(payload)

    assert runtime_name == "lazyppl"
    assert runtime_spec is not None
    assert runtime_spec.name == "lazyppl"
    assert runtime_spec.execution_backend == "haskell"
    assert score_artifact_key == "log_lik_matrix"


def test_agent_result_trial_record_normalizes_runtime_aliases() -> None:
    agent_result = {
        "program": "model code",
        "language": "haskell",
        "reward": 0.5,
        "task_config": {"env": {"name": "dugongs"}},
    }

    trial = trial_record_from_agent_result(
        agent_result,
        trial_id="trial-001",
        trace_id="trace-001",
    )

    assert isinstance(trial, ContractsTrialRecord)
    assert trial.language == "lazyppl"
    assert trial.runtime is not None
    assert trial.runtime.name == "lazyppl"
    assert trial.runtime.execution_backend == "haskell"


def test_completion_record_maps_to_shared_generation_record() -> None:
    record = CompletionRecord(
        id="completion-001",
        prompt_id="prompt-001",
        env_id="dugongs",
        split="train",
        prompt_text="Write a model for grouped counts.",
        raw_completion="<submit_program>model code</submit_program>",
        candidate_outputs=["<submit_program>alternate model</submit_program>"],
        provider_response={"choices": [{"message": {"content": "model code"}}]},
        parsed_program="model code",
        provenance=PromptProvenance(runtime_name="pymc", runtime_version="5.20.0"),
        validation=ValidationInfo(
            ok=True,
            message="passed",
            repaired=False,
            trace=ValidationTrace.model_validate(
                {
                    "steps": [
                        {
                            "attempt": 0,
                            "phase": "semantic_validation",
                            "ok": True,
                            "message": "pass",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "pass",
                            "metadata": {
                                "metric_name": "psis_loo",
                                "metric_value": 0.81,
                            },
                        }
                    ]
                }
            ),
        ),
        metadata={"source": "stage-e"},
    )

    generation = generation_record_from_completion_record(record.model_dump(mode="python"))

    assert isinstance(generation, ContractsGenerationRecord)
    assert generation.record_id == "completion-001"
    assert generation.task_id == "dugongs"
    assert generation.prompt_id == "prompt-001"
    assert generation.runtime is not None
    assert generation.runtime.name == "pymc"
    assert generation.runtime.execution_backend == "python"
    assert generation.prompt is not None
    assert generation.prompt.content == "Write a model for grouped counts."
    assert generation.candidate_outputs == ["<submit_program>alternate model</submit_program>"]
    assert generation.provider_response == {"choices": [{"message": {"content": "model code"}}]}
    assert generation.validation["ok"] is True
    assert generation.validation["trace"]["steps"][0]["evidence_kind"] == "verifier"
    assert generation.validation["trace"]["steps"][0]["truth_owner"] == "verifier"
    assert generation.validation["trace"]["steps"][0]["semantic_status"] == "pass"
    assert generation.metadata == {"source": "stage-e"}


def test_stage_f_trace_artifact_validates_as_contracts_trace_record() -> None:
    """The trace_json shape schemas.adapters emits/consumes: taxonomy lives in
    each step's `metadata`, never as a top-level TraceStep field (contracts
    TraceStep has no evidence_kind/truth_owner/semantic_status fields at all,
    and is extra="forbid")."""
    trace_record = {
        "trace_id": "completion-001:validation",
        "task_id": "dugongs",
        "trial_id": "completion-001",
        "steps": [
            {
                "step_id": "step-0000",
                "role": "validator",
                "content": "CLEAN",
                "metadata": {
                    "attempt": 0,
                    "phase": "leakage_judge",
                    "ok": True,
                    "evidence_kind": "heuristic",
                    "truth_owner": "heuristic",
                    "semantic_status": "pass",
                },
            },
            {
                "step_id": "step-0001",
                "role": "validator",
                "content": "ok",
                "metadata": {
                    "attempt": 0,
                    "phase": "runtime",
                    "ok": True,
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "semantic_status": "not_scored",
                },
            },
        ],
        "artifacts": {
            "outcome": "validated",
            "prompt_id": "prompt-001",
            "split": "train",
        },
        "metadata": {
            "stage": "F",
            "runtime": "pymc",
            "env_id": "dugongs",
            "split": "train",
        },
    }

    shared_trace = ContractsTraceRecord.model_validate(trace_record)

    assert shared_trace.trace_id == "completion-001:validation"
    assert shared_trace.task_id == "dugongs"
    assert len(shared_trace.steps) == 2
    assert shared_trace.steps[0].metadata["evidence_kind"] == "heuristic"
    assert shared_trace.steps[0].metadata["truth_owner"] == "heuristic"
    assert shared_trace.steps[0].metadata["semantic_status"] == "pass"
    assert shared_trace.steps[0].metadata["phase"] == "leakage_judge"


def test_contracts_trace_step_rejects_legacy_top_level_taxonomy_fields() -> None:
    """The pre-migration platform.TraceStep shape (evidence_kind/truth_owner/
    semantic_status as top-level fields) is no longer valid: contracts
    TraceStep is extra="forbid" and declares no such fields, so it must fail
    loud rather than silently drop or accept the legacy shape."""
    legacy_shaped_record = {
        "trace_id": "completion-legacy-001:validation",
        "steps": [
            {
                "step_id": "step-0000",
                "role": "validator",
                "content": "ok",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
                "semantic_status": "not_scored",
            }
        ],
    }

    with pytest.raises(Exception, match="evidence_kind|Extra inputs"):
        ContractsTraceRecord.model_validate(legacy_shaped_record)


def test_trace_record_from_messages_preserves_taxonomy_and_metadata() -> None:
    trace = trace_record_from_messages(
        [
            {
                "role": "assistant",
                "content": "judge says pass",
                "evidence_kind": "heuristic",
                "truth_owner": "heuristic",
                "semantic_status": "pass",
                "metadata": {"phase": "leakage_judge", "score": 0.2},
            },
            {
                "role": "assistant",
                "content": "verifier accepted",
                "tool_call_id": "call-1",
                "metadata": {
                    "phase": "semantic_validation",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "semantic_status": "pass",
                    "metric_name": "psis_loo",
                },
            },
        ],
        trace_id="trace-taxonomy",
        task_id="dugongs",
        trial_id="trial-001",
    )

    assert isinstance(trace, ContractsTraceRecord)
    assert trace.trace_id == "trace-taxonomy"
    assert trace.steps[0].metadata["evidence_kind"] == "heuristic"
    assert trace.steps[0].metadata["truth_owner"] == "heuristic"
    assert trace.steps[0].metadata["semantic_status"] == "pass"
    assert trace.steps[0].metadata["phase"] == "leakage_judge"
    assert trace.steps[0].metadata["score"] == 0.2
    assert trace.steps[1].metadata["evidence_kind"] == "verifier"
    assert trace.steps[1].metadata["truth_owner"] == "verifier"
    assert trace.steps[1].metadata["semantic_status"] == "pass"
    assert trace.steps[1].metadata["metric_name"] == "psis_loo"
    assert trace.steps[1].metadata["tool_call_id"] == "call-1"


def test_stage_g_with_data_trace_json_validates_as_contracts_trace_record(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pydantic_settings")
    from ppl_synth_data.pipeline.artifacts import resolve_run_paths
    from ppl_synth_data.pipeline.stages import run_stage_g
    from ppl_synth_data.runtimes import get_runtime

    paths = resolve_run_paths(tmp_path, "run_stage_g_trace_contract")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    env_spec = EnvSpec(
        env_id="dugongs",
        env_name="Dugongs",
        domain="ecology",
        complexity=Complexity.medium,
        model_type="hierarchical",
        key_concepts=["counts"],
        io_schema=IOSchema(
            input_type=InputType.float,
            input_description="observed counts by region",
            output_type="posterior_samples",
            target_name="count",
            feature_names=["region"],
        ),
        data_plan=DataPlan(n_observations=32, grouping=["region"], covariates=[]),
        generative_family="poisson",
        generative_params={"mu": 3.0},
        prompt_guidance={},
    )
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(env_spec.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps(
            {
                "env_id": "dugongs",
                "data_ref": "stage_C_data/data/dugongs.parquet",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completion = CompletionRecord(
        id="completion-001",
        prompt_id="prompt-001",
        env_id="dugongs",
        split="train",
        prompt_text="Write a model for grouped counts.",
        raw_completion="<submit_program>model code</submit_program>",
        candidate_outputs=["<submit_program>alternate model</submit_program>"],
        parsed_program="model code",
        provenance=PromptProvenance(runtime_name="pymc", runtime_version="5.20.0"),
        validation=ValidationInfo(
            ok=True,
            message="passed",
            repaired=False,
            trace=ValidationTrace.model_validate(
                {
                    "steps": [
                        {
                            "attempt": 0,
                            "phase": "runtime",
                            "ok": True,
                            "message": "ok",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "not_scored",
                        },
                        {
                            "attempt": 0,
                            "phase": "semantic_validation",
                            "ok": True,
                            "message": "pass",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "pass",
                            "metadata": {
                                "metric_name": "psis_loo",
                                "metric_value": 0.81,
                            },
                        },
                    ]
                }
            ),
        ),
        metadata={"seed": 7},
    )
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(completion.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")
    trace_record = {
        "trace_id": "completion-001:validation",
        "task_id": "dugongs",
        "trial_id": "completion-001",
        "steps": [
            {
                "step_id": "step-0000",
                "role": "validator",
                "content": "ok",
                "metadata": {
                    "attempt": 0,
                    "phase": "runtime",
                    "ok": True,
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "semantic_status": "not_scored",
                },
            }
        ],
        "artifacts": {
            "outcome": "validated",
            "prompt_id": "prompt-001",
            "split": "train",
        },
        "metadata": {
            "stage": "F",
            "runtime": "pymc",
            "env_id": "dugongs",
            "split": "train",
        },
    }
    (paths.stage_dirs["F"] / "traces.jsonl").write_text(
        json.dumps(trace_record) + "\n",
        encoding="utf-8",
    )

    result = cast(Any, run_stage_g)(
        paths,
        repo_id=None,
        hf_token=None,
        skip_upload=True,
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    row = json.loads(
        (paths.stage_dirs["G"] / "pymc_with_data.train.jsonl").read_text(encoding="utf-8").strip()
    )
    shared_trace = ContractsTraceRecord.model_validate(json.loads(row["trace_json"]))
    validation_payload = json.loads(row["validation_json"])

    assert row["trace_id"] == "completion-001:validation"
    assert json.loads(row["candidate_outputs_json"]) == [
        "<submit_program>alternate model</submit_program>"
    ]
    assert shared_trace.trace_id == "completion-001:validation"
    assert shared_trace.trial_id == "completion-001"
    assert shared_trace.steps[0].metadata["evidence_kind"] == "verifier"
    assert shared_trace.steps[0].metadata["truth_owner"] == "verifier"
    assert shared_trace.steps[0].metadata["semantic_status"] == "not_scored"
    assert validation_payload["trace"]["steps"][0]["evidence_kind"] == "verifier"
    assert validation_payload["trace"]["steps"][0]["truth_owner"] == "verifier"
    assert validation_payload["trace"]["steps"][0]["semantic_status"] == "not_scored"
    assert "evidence_kind" not in validation_payload["trace"]["steps"][0]["metadata"]
    assert "truth_owner" not in validation_payload["trace"]["steps"][0]["metadata"]
    assert "semantic_status" not in validation_payload["trace"]["steps"][0]["metadata"]
    assert validation_payload["trace"]["steps"][-1]["semantic_status"] == "pass"


def test_dataset_row_trace_falls_back_to_validation_trace_when_trace_json_missing() -> None:
    dataset_row = {
        "id": "completion-compat-001",
        "prompt_id": "prompt-compat-001",
        "env_id": "dugongs",
        "split": "train",
        "prompt": "Write a model for grouped counts.",
        "completion": "model code",
        "pymc_version": "5.20.0",
        "trace_id": "completion-compat-001:validation",
        "provenance_json": json.dumps(
            {
                "id": "completion-compat-001",
                "prompt_id": "prompt-compat-001",
                "provenance": {"runtime_name": "pymc", "runtime_version": "5.20.0"},
                "metadata": {},
            }
        ),
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "trace": {
                    "steps": [
                        {
                            "attempt": 0,
                            "phase": "runtime",
                            "ok": True,
                            "message": "ok",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "not_scored",
                        },
                        {
                            "attempt": 0,
                            "phase": "semantic_validation",
                            "ok": True,
                            "message": "pass",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "pass",
                            "metadata": {
                                "metric_name": "psis_loo",
                                "metric_value": 0.81,
                            },
                        },
                    ]
                },
            }
        ),
    }

    trace = trace_record_from_dataset_row(dataset_row)

    assert trace is not None
    assert isinstance(trace, ContractsTraceRecord)
    assert trace.trace_id == "completion-compat-001:validation"
    assert trace.metadata["trace_origin"] == "validation_json.trace"
    assert [step.content for step in trace.steps] == ["ok", "pass"]
    assert trace.steps[0].metadata["phase"] == "runtime"
    assert trace.steps[0].metadata["evidence_kind"] == "verifier"
    assert trace.steps[0].metadata["truth_owner"] == "verifier"
    assert trace.steps[0].metadata["semantic_status"] == "not_scored"
    assert trace.steps[1].metadata["metric_name"] == "psis_loo"
    assert trace.steps[1].metadata["evidence_kind"] == "verifier"
    assert trace.steps[1].metadata["truth_owner"] == "verifier"
    assert trace.steps[1].metadata["semantic_status"] == "pass"


def test_dataset_row_trace_promotes_validation_trace_taxonomy_from_metadata() -> None:
    dataset_row = {
        "id": "completion-legacy-validation-001",
        "env_id": "dugongs",
        "trace_id": "completion-legacy-validation-001:validation",
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "trace": {
                    "steps": [
                        {
                            "attempt": 0,
                            "message": "ok",
                            "metadata": {
                                "phase": "runtime",
                                "evidence_kind": "verifier",
                                "truth_owner": "verifier",
                                "semantic_status": "pass",
                            },
                        }
                    ]
                },
            }
        ),
    }

    trace = trace_record_from_dataset_row(dataset_row)

    assert trace is not None
    assert trace.metadata["trace_origin"] == "validation_json.trace"
    assert trace.steps[0].metadata["evidence_kind"] == "verifier"
    assert trace.steps[0].metadata["truth_owner"] == "verifier"
    assert trace.steps[0].metadata["semantic_status"] == "pass"
    assert trace.steps[0].metadata["phase"] == "runtime"


def test_dataset_row_trace_enriches_empty_trace_json_from_validation_trace() -> None:
    dataset_row = {
        "id": "completion-empty-trace-001",
        "prompt_id": "prompt-empty-trace-001",
        "env_id": "dugongs",
        "split": "train",
        "prompt": "Write a model for grouped counts.",
        "completion": "model code",
        "pymc_version": "5.20.0",
        "trace_id": "completion-empty-trace-001:validation",
        "trace_json": json.dumps(
            {
                "trace_id": "completion-empty-trace-001:validation",
                "task_id": "dugongs",
                "trial_id": "completion-empty-trace-001",
                "steps": [],
                "artifacts": {"outcome": "validated"},
                "metadata": {"source": "stale_trace_json"},
            }
        ),
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "trace": {
                    "steps": [
                        {
                            "attempt": 0,
                            "phase": "runtime",
                            "ok": True,
                            "message": "ok",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "not_scored",
                        }
                    ]
                },
            }
        ),
    }

    trace = trace_record_from_dataset_row(dataset_row)

    assert trace is not None
    assert trace.metadata["source"] == "stale_trace_json"
    assert trace.metadata["trace_origin"] == "trace_json.empty+validation_json.trace"
    assert trace.artifacts["outcome"] == "validated"
    assert len(trace.steps) == 1
    assert trace.steps[0].metadata["phase"] == "runtime"
    assert trace.steps[0].metadata["evidence_kind"] == "verifier"
    assert trace.steps[0].metadata["truth_owner"] == "verifier"
    assert trace.steps[0].metadata["semantic_status"] == "not_scored"


def test_dataset_row_trace_rejects_mismatched_task_id_in_trace_json() -> None:
    dataset_row = {
        "id": "completion-mismatch-task-001",
        "env_id": "dugongs",
        "trace_id": "completion-mismatch-task-001:validation",
        "trace_json": json.dumps(
            {
                "trace_id": "completion-mismatch-task-001:validation",
                "task_id": "wrong-task",
                "trial_id": "completion-mismatch-task-001",
                "steps": [],
                "metadata": {"source": "trace_json"},
            }
        ),
    }

    with pytest.raises(ValueError, match="dataset row task_id does not match"):
        trace_record_from_dataset_row(dataset_row)


def test_dataset_row_trace_rejects_mismatched_trace_id_in_trace_json() -> None:
    dataset_row = {
        "id": "completion-mismatch-trace-001",
        "env_id": "dugongs",
        "trace_id": "completion-mismatch-trace-001:validation",
        "trace_json": json.dumps(
            {
                "trace_id": "stale-trace-id",
                "task_id": "dugongs",
                "trial_id": "completion-mismatch-trace-001",
                "steps": [],
                "metadata": {"source": "trace_json"},
            }
        ),
    }

    with pytest.raises(ValueError, match="dataset row trace_id does not match"):
        trace_record_from_dataset_row(dataset_row)


def test_dataset_row_trace_rejects_mismatched_trial_id_when_enriching_empty_trace() -> None:
    dataset_row = {
        "id": "completion-mismatch-trial-001",
        "env_id": "dugongs",
        "trace_id": "completion-mismatch-trial-001:validation",
        "trace_json": json.dumps(
            {
                "trace_id": "completion-mismatch-trial-001:validation",
                "task_id": "dugongs",
                "trial_id": "stale-trial-id",
                "steps": [],
                "metadata": {"source": "stale_trace_json"},
            }
        ),
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "trace": {
                    "steps": [
                        {
                            "attempt": 0,
                            "phase": "runtime",
                            "ok": True,
                            "message": "ok",
                        }
                    ]
                },
            }
        ),
    }

    with pytest.raises(ValueError, match="dataset row trial_id does not match"):
        trace_record_from_dataset_row(dataset_row)


def test_dataset_row_trace_keeps_legacy_trace_json_taxonomy_in_metadata() -> None:
    """A trace_json blob that already nests taxonomy inside step metadata
    (the only shape contracts.TraceStep accepts) round-trips unchanged: there
    is no promotion to dedicated fields because contracts.TraceStep declares
    none."""
    dataset_row = {
        "id": "completion-legacy-trace-001",
        "prompt_id": "prompt-legacy-trace-001",
        "env_id": "dugongs",
        "split": "train",
        "prompt": "Write a model for grouped counts.",
        "completion": "model code",
        "pymc_version": "5.20.0",
        "trace_id": "completion-legacy-trace-001:validation",
        "trace_json": json.dumps(
            {
                "trace_id": "completion-legacy-trace-001:validation",
                "task_id": "dugongs",
                "trial_id": "completion-legacy-trace-001",
                "steps": [
                    {
                        "step_id": "step-0000",
                        "role": "validator",
                        "content": "ok",
                        "metadata": {
                            "phase": "runtime",
                            "ok": True,
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "not_scored",
                        },
                    }
                ],
                "artifacts": {"outcome": "validated"},
                "metadata": {"source": "legacy_trace_json"},
            }
        ),
    }

    trace = trace_record_from_dataset_row(dataset_row)

    assert trace is not None
    assert trace.metadata["source"] == "legacy_trace_json"
    assert len(trace.steps) == 1
    assert trace.steps[0].metadata["phase"] == "runtime"
    assert trace.steps[0].metadata["evidence_kind"] == "verifier"
    assert trace.steps[0].metadata["truth_owner"] == "verifier"
    assert trace.steps[0].metadata["semantic_status"] == "not_scored"

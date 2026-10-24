from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ppl_synth_data.evidence import SemanticStatus
from ppl_synth_data.llm.backend import (
    BackendCapabilities,
    BatchHandle,
    LLMRequest,
    LLMResult,
)
from ppl_synth_data.pipeline.artifacts import resolve_run_paths
from ppl_synth_data.pipeline.stages import (
    run_stage_a,
    run_stage_b,
    run_stage_d,
    run_stage_e,
    run_stage_f,
    run_stage_g,
)
from ppl_synth_data.pipeline.stages.stage_a_envspec import _generation_plan
from ppl_synth_data.pipeline.validation.semantic_evidence import (
    LAZYPPL_SEMANTIC_EVIDENCE_VERSION,
)
from ppl_synth_data.runtimes import get_runtime
from ppl_synth_data.validation import lazyppl_semantics
from ppl_synth_data.validation.pipeline import (
    ValidationRunResult,
    ValidationTraceEntry,
)
from ppl_synth_data.validation.pymc_semantics import SemanticValidationResult

from .helpers import CaptureBackend, minimal_envspec_record, write_jsonl


def _write_stage_c_visible_data(paths, env_id: str) -> None:
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]}).to_parquet(
        data_dir / f"{env_id}.parquet",
        index=False,
    )


def _semantic_not_scored(*args, **kwargs) -> SemanticValidationResult:
    _ = (args, kwargs)
    return SemanticValidationResult(
        ok=True,
        status=SemanticStatus.not_scored,
        message="semantic validation not exercised by this test",
        metric_name=None,
        metric_value=None,
        details={},
    )


def _lazyppl_semantic_validation() -> dict:
    return {
        "ok": True,
        "message": "LazyPPL semantic validation passed",
        "repaired": False,
        "leakage_rating": "CLEAN",
        "evidence_kind": "verifier",
        "truth_owner": "verifier",
        "semantic_status": "pass",
        "semantic_metric": "avg_holdout_logp",
        "semantic_score": -1.25,
        "semantic_settings": {
            "pymc_draws": 100,
            "pymc_tune": 100,
            "pymc_timeout_sec": 180,
        },
        "semantic_details": {
            "semantic_evidence_version": LAZYPPL_SEMANTIC_EVIDENCE_VERSION,
            "reward_artifact": {
                "schema_version": "synthstats.harbor.reward.v1",
                "reward_source": "validation.lazyppl_elpd",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
                "producer": "ppl_synth_data.docker.lazyppl_harbor.score",
                "training_log_reward": -1.25,
                "reward_transform": {"kind": "baseline_normalized_log_reward"},
                "metrics": {"avg_holdout_logp": -1.25},
                "diagnostics": {},
                "provenance": {},
                "reward": None,
            },
        },
    }


def test_stage_b_canonicalizes_duplicate_env_names(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_005")
    a_dir = paths.stage_dirs["A"]
    a_dir.mkdir(parents=True, exist_ok=True)

    records = [
        minimal_envspec_record(env_id="e1", env_name="duplicate_env"),
        minimal_envspec_record(env_id="e2", env_name="duplicate_env"),
    ]
    write_jsonl(a_dir / "envspecs.jsonl", records)

    result = run_stage_b(paths)

    assert result.status == "success"
    manifest = json.loads((paths.stage_dirs["B"] / "manifest.json").read_text())
    assert manifest["renamed_env_name_count"] == 1
    assert manifest["renamed_env_names"] == [
        {
            "env_id": "e2",
            "original_env_name": "duplicate_env",
            "env_name": "duplicate_env_e2",
        }
    ]
    canonical = [
        json.loads(line)
        for line in (paths.stage_dirs["B"] / "envspecs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["env_name"] for row in canonical] == [
        "duplicate_env",
        "duplicate_env_e2",
    ]
    assert (paths.stage_dirs["B"] / "envs" / "duplicate_env.py").exists()
    assert (paths.stage_dirs["B"] / "envs" / "duplicate_env_e2.py").exists()


def test_stage_b_rejects_duplicate_env_ids(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_005b")
    a_dir = paths.stage_dirs["A"]
    a_dir.mkdir(parents=True, exist_ok=True)

    records = [
        minimal_envspec_record(env_id="dup-id", env_name="env_one"),
        minimal_envspec_record(env_id="dup-id", env_name="env_two"),
    ]
    write_jsonl(a_dir / "envspecs.jsonl", records)

    try:
        run_stage_b(paths)
    except ValueError as exc:
        assert "unique env_id" in str(exc)
    else:
        raise AssertionError("expected duplicate env_id rejection")


def test_stage_b_skips_uncompilable_generated_envspecs(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_b_skip_bad_env")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    good = minimal_envspec_record(env_id="good-env", env_name="good_env")
    bad = minimal_envspec_record(env_id="bad-env", env_name="bad_env")
    bad["io_schema"]["target_name"] = "bad-name"
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [good, bad])

    result = run_stage_b(paths)

    assert result.status == "completed_with_refusals"
    manifest = json.loads((paths.stage_dirs["B"] / "manifest.json").read_text())
    assert manifest["count"] == 2
    assert manifest["compiled_count"] == 1
    assert manifest["skipped_env_count"] == 1
    skipped = json.loads((paths.stage_dirs["B"] / "skipped_envs.jsonl").read_text(encoding="utf-8"))
    assert skipped["env_id"] == "bad-env"


def test_stage_b_reports_mixed_batch_compiler_refusals(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_b_refused_law")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    good = minimal_envspec_record(env_id="good-law", env_name="good_law")
    good["generative_family"] = "regression_continuous"
    good["generative_params"] = {
        "intercept": 120.0,
        "noise_sigma": 8.5,
        "coefficients": {"group": 12.0},
    }
    bad = {**good, "env_id": "refused-law", "env_name": "refused_law"}
    bad["generative_params"] = {"intercept": 120.0, "noise_sigma": 8.5}
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [good, bad])

    result = run_stage_b(paths)

    assert result.status == "completed_with_refusals"
    manifest = json.loads((paths.stage_dirs["B"] / "manifest.json").read_text())
    assert manifest == result.outputs
    assert manifest["status"] == "completed_with_refusals"
    assert manifest["count"] == 2
    assert manifest["compiled_count"] == 1
    assert manifest["skipped_env_count"] == 1
    assert manifest["resumable"] is False
    refusals = manifest["skipped_envs"]
    assert len(refusals) == 1
    assert refusals[0]["env_id"] == "refused-law"
    assert refusals[0]["exception_type"] == "ValueError"
    assert "coefficients" in refusals[0]["error"]
    recorded = json.loads((paths.stage_dirs["B"] / "skipped_envs.jsonl").read_text())
    assert recorded == refusals[0]


def test_stage_b_clears_stale_compiled_modules_before_recompile(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_b_clears_stale")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    write_jsonl(
        paths.stage_dirs["A"] / "envspecs.jsonl",
        [minimal_envspec_record(env_id="fresh-env", env_name="fresh_env")],
    )
    env_dir = paths.stage_dirs["B"] / "envs"
    env_dir.mkdir(parents=True, exist_ok=True)
    stale = env_dir / "stale_env.py"
    stale.write_text("class Stale: pass\n", encoding="utf-8")

    result = run_stage_b(paths)

    assert result.status == "success"
    assert not stale.exists()
    manifest = json.loads((paths.stage_dirs["B"] / "manifest.json").read_text())
    assert manifest["compiled_count"] == 1


def test_stage_b_removes_target_names_and_refuses_predictorless_spec(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_b_target_feature")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    spec = minimal_envspec_record(env_id="target-feature", env_name="target_feature")
    spec["generative_family"] = "mixture_continuous"
    spec["io_schema"]["target_name"] = "vibration_rms"
    spec["io_schema"]["feature_names"] = ["vibration_rms", "vibration_rms"]
    good = minimal_envspec_record(env_id="good-env", env_name="good_env")
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec, good])

    result = run_stage_b(paths)

    assert result.status == "completed_with_refusals"
    manifest = json.loads((paths.stage_dirs["B"] / "manifest.json").read_text())
    assert manifest == result.outputs
    assert manifest["status"] == "completed_with_refusals"
    assert manifest["count"] == 2
    assert manifest["compiled_count"] == 1
    assert manifest["skipped_env_count"] == 1
    assert manifest["resumable"] is False
    refusal = manifest["skipped_envs"][0]
    assert refusal["env_id"] == "target-feature"
    assert refusal["exception_type"] == "ValueError"
    assert "feature_names" in refusal["error"]
    assert manifest["canonicalized_io_schema_count"] == 1
    assert manifest["canonicalized_io_schemas"][0]["removed_feature_names"] == [
        "vibration_rms",
        "vibration_rms",
    ]
    env_dir = paths.stage_dirs["B"] / "envs"
    assert not (env_dir / "target_feature.py").exists()
    assert (env_dir / "good_env.py").exists()
    canonical = [
        json.loads(line)
        for line in (paths.stage_dirs["B"] / "envspecs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["env_id"] for row in canonical] == ["good-env"]
    canonicalized = manifest["canonicalized_io_schemas"][0]
    assert canonicalized["env_id"] == "target-feature"
    assert canonicalized["target_name"] == "vibration_rms"
    assert canonicalized["feature_names"] == []


def test_stage_g_writes_dropped_records(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_006")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="known-env", env_name="known_env")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(spec) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps({"env_id": "known-env", "data_ref": "stage_C_data/data/known-env.parquet"})
        + "\n",
        encoding="utf-8",
    )
    bad_validated = {
        "id": "c-1",
        "prompt_id": "p-1",
        "prompt_text": "Prompt",
        "raw_completion": "<think>x</think>\n```python\ndef model(data):\n  return data\n```",
        "parsed_program": None,
        "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
        "metadata": {
            "env_id": "missing-env",
            "split": "train",
            "pymc_version": "v5",
        },
    }
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(bad_validated) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dropped every validated completion"):
        run_stage_g(
            paths,
            repo_id=None,
            hf_token=None,
            skip_upload=True,
            runtime=get_runtime("pymc"),
        )

    dropped = (paths.stage_dirs["G"] / "dropped.jsonl").read_text(encoding="utf-8").strip()
    assert dropped


def test_stage_g_fails_when_invalid_seed_drops_every_record(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_006_seed")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="known-env", env_name="known_env")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(spec) + "\n",
        encoding="utf-8",
    )
    data_ref_row = {
        "env_id": "known-env",
        "data_ref": "stage_C_data/data/known-env.parquet",
    }
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps(data_ref_row) + "\n",
        encoding="utf-8",
    )
    validated = {
        "id": "c-2",
        "prompt_id": "p-2",
        "prompt_text": "Prompt",
        "raw_completion": "<think>x</think>\n```python\ndef model(data):\n  return data\n```",
        "parsed_program": None,
        "metadata": {
            "env_id": "known-env",
            "split": "train",
            "seed": "not-an-int",
            "pymc_version": "v5",
        },
        "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
    }
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(validated) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dropped every validated completion"):
        run_stage_g(
            paths,
            repo_id=None,
            hf_token=None,
            skip_upload=True,
            runtime=get_runtime("pymc"),
        )

    dropped_rows = (paths.stage_dirs["G"] / "dropped.jsonl").read_text(encoding="utf-8")
    assert "invalid_seed" in dropped_rows


def test_stage_g_drops_validated_records_without_materialized_data_ref(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_006_missing_data_ref")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="known-env", env_name="known_env")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(spec) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text("", encoding="utf-8")
    validated = {
        "id": "c-3",
        "prompt_id": "p-3",
        "prompt_text": "Prompt",
        "raw_completion": "<think>x</think>\n```python\ndef model(data):\n  return data\n```",
        "parsed_program": None,
        "metadata": {
            "env_id": "known-env",
            "split": "train",
            "seed": 0,
            "pymc_version": "v5",
        },
        "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
    }
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(validated) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dropped every validated completion"):
        run_stage_g(
            paths,
            repo_id=None,
            hf_token=None,
            skip_upload=True,
            runtime=get_runtime("pymc"),
        )

    dropped_rows = (paths.stage_dirs["G"] / "dropped.jsonl").read_text(encoding="utf-8")
    assert "missing_data_ref" in dropped_rows


def test_stage_g_fails_when_stage_f_processed_but_validated_nothing(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_006_no_validated")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="known-env", env_name="known_env")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(spec) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text("", encoding="utf-8")
    (paths.stage_dirs["F"] / "validated.jsonl").write_text("", encoding="utf-8")
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text(
        json.dumps({"id": "completion-1"}) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "manifest.json").write_text(
        json.dumps({"status": "success", "count_total": 1, "runtime": "pymc"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no validated completions"):
        run_stage_g(
            paths,
            repo_id=None,
            hf_token=None,
            skip_upload=True,
            runtime=get_runtime("pymc"),
        )


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_uses_stage_yaml_defaults_when_model_unset(monkeypatch, tmp_path: Path) -> None:
    configs = tmp_path / "configs" / "stages"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "prompts.yaml").write_text(
        "stage: prompts\nmodel: yaml-prompt-model\nreasoning_effort: low\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    paths = resolve_run_paths(tmp_path / "runs", "run_yaml_stage_d")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(minimal_envspec_record(env_id="env-1", env_name="env_1")) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "summaries.jsonl").write_text(
        json.dumps({"env_id": "env-1", "n": 3}) + "\n",
        encoding="utf-8",
    )

    backend = CaptureBackend(["prompt-text"])
    run_stage_d(
        paths,
        limit=1,
        use_batch=False,
        model=None,  # type: ignore[arg-type]
        backend=backend,
        runtime=get_runtime("pymc"),
    )
    assert backend.requests[0].model == "yaml-prompt-model"
    assert backend.requests[0].reasoning_effort == "low"
    assert backend.requests[0].max_output_tokens == 4096
    manifest = json.loads((paths.stage_dirs["D"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["configured_reasoning_effort"] == "low"
    assert manifest["reasoning_effort"] == "low"
    assert manifest["max_output_tokens"] == 4096


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_omits_reasoning_effort_when_configured(monkeypatch, tmp_path: Path) -> None:
    configs = tmp_path / "configs" / "stages"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "prompts.yaml").write_text(
        ("stage: prompts\nmodel: gpt-5.5-pro\nreasoning_effort: omit\nmax_output_tokens: 4096\n"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_omit_reasoning")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(minimal_envspec_record(env_id="env-1", env_name="env_1")) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "summaries.jsonl").write_text(
        json.dumps({"env_id": "env-1", "n": 3}) + "\n",
        encoding="utf-8",
    )

    backend = CaptureBackend(["prompt-text"])
    run_stage_d(
        paths,
        limit=1,
        use_batch=False,
        backend=backend,
        runtime=get_runtime("pymc"),
    )

    assert backend.requests[0].model == "gpt-5.5-pro"
    assert backend.requests[0].reasoning_effort is None
    assert backend.requests[0].max_output_tokens == 4096
    manifest = json.loads((paths.stage_dirs["D"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["configured_reasoning_effort"] == "omit"
    assert manifest["reasoning_effort"] is None
    assert manifest["max_output_tokens"] == 4096


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_skips_specs_without_materialized_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_missing_summary")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(minimal_envspec_record(env_id="env-2", env_name="env_2")) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "summaries.jsonl").write_text("", encoding="utf-8")

    backend = CaptureBackend(["prompt-text"])
    with pytest.raises(RuntimeError, match="no materialized environment summaries"):
        run_stage_d(
            paths,
            limit=1,
            use_batch=False,
            backend=backend,
            runtime=get_runtime("pymc"),
        )

    assert backend.requests == []


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_only_prompts_materialized_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_record_local")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    specs = [
        minimal_envspec_record(env_id="env-missing", env_name="env_missing"),
        minimal_envspec_record(env_id="env-present", env_name="env_present"),
    ]
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", specs)
    write_jsonl(paths.stage_dirs["C"] / "summaries.jsonl", [{"env_id": "env-present", "n": 5}])

    backend = CaptureBackend(["prompt-a"])
    run_stage_d(paths, limit=2, use_batch=False, backend=backend, runtime=get_runtime("pymc"))

    rows = []
    with (paths.stage_dirs["D"] / "prompts.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    flags = {row["env_id"]: row["provenance"]["has_observed_data"] for row in rows}
    assert flags == {"env-present": True}
    provenance = rows[0]["provenance"]
    assert provenance["source_category"] == "synthetic_invented"
    assert provenance["source_url_or_repo"] == "not_applicable_invented"
    assert provenance["license_signal"] == "not_applicable_invented"
    assert provenance["adaptation_mode"] == "invented"
    assert provenance["source_retrieval_date"] == "not_applicable_invented"


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_batch_partial_failure_writes_skipped_prompts(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_partial_batch")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    specs = [minimal_envspec_record(env_id=f"env-{idx}", env_name=f"env_{idx}") for idx in range(3)]
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", specs)
    write_jsonl(
        paths.stage_dirs["C"] / "summaries.jsonl",
        [{"env_id": f"env-{idx}", "n": 5} for idx in range(3)],
    )

    class PartialPromptBatchBackend:
        capabilities = BackendCapabilities(
            supports_batch=True,
            supports_json_schema=True,
            supports_reasoning_effort=True,
        )

        def __init__(self) -> None:
            self.requests: list[tuple[str, LLMRequest]] = []
            self.allow_partial_failures: bool | None = None

        def complete(self, request: LLMRequest) -> LLMResult:
            raise AssertionError("stage D should use batch path")

        def submit_batch(
            self,
            requests: list[tuple[str, LLMRequest]],
        ) -> BatchHandle:
            self.requests = list(requests)
            return BatchHandle(
                backend="test",
                job_id="batch-stage-d-partial",
                request_ids=tuple(request_id for request_id, _ in requests),
            )

        def collect_batch(
            self,
            handle: BatchHandle,
            *,
            allow_partial_failures: bool = False,
        ) -> dict[str, LLMResult]:
            assert handle.job_id == "batch-stage-d-partial"
            self.allow_partial_failures = allow_partial_failures
            return {
                "prompt-000000": LLMResult(content="Prompt zero", raw={}),
                "prompt-000002": LLMResult(content="Prompt two", raw={}),
            }

    backend = PartialPromptBatchBackend()
    result = run_stage_d(
        paths,
        limit=3,
        use_batch=True,
        backend=backend,
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    assert backend.allow_partial_failures is True
    assert not (paths.stage_dirs["D"] / "pending_batch.json").exists()
    rows = [
        json.loads(line)
        for line in (paths.stage_dirs["D"] / "prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    skipped = [
        json.loads(line)
        for line in (paths.stage_dirs["D"] / "skipped_prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    manifest = json.loads((paths.stage_dirs["D"] / "manifest.json").read_text())

    assert [row["id"] for row in rows] == ["prompt-000000", "prompt-000002"]
    assert [row["env_id"] for row in rows] == ["env-0", "env-2"]
    assert skipped == [
        {
            "id": "prompt-000001",
            "env_id": "env-1",
            "reason": "missing_or_failed_batch_output",
        }
    ]
    assert manifest["count_requested"] == 3
    assert manifest["count"] == 2
    assert manifest["skipped_count"] == 1
    assert manifest["skipped_output"].endswith("skipped_prompts.jsonl")


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_carries_source_provenance_from_envspec(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_source_provenance")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    spec = minimal_envspec_record(env_id="env-present", env_name="env_present")
    spec["prompt_guidance"] = {
        "source_provenance": {
            "source_category": "pymc_examples",
            "source_url_or_repo": "https://github.com/pymc-devs/pymc-examples",
            "license_signal": "permissive",
            "adaptation_mode": "source-inspired",
            "source_retrieval_date": "2026-05-16",
        }
    }
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])
    write_jsonl(paths.stage_dirs["C"] / "summaries.jsonl", [{"env_id": "env-present", "n": 5}])

    backend = CaptureBackend(["prompt-a"])
    run_stage_d(paths, limit=1, use_batch=False, backend=backend, runtime=get_runtime("pymc"))

    row = json.loads((paths.stage_dirs["D"] / "prompts.jsonl").read_text(encoding="utf-8"))
    provenance = row["provenance"]
    assert provenance["source_category"] == "pymc_examples"
    assert provenance["source_url_or_repo"] == "https://github.com/pymc-devs/pymc-examples"
    assert provenance["license_signal"] == "permissive"
    assert provenance["adaptation_mode"] == "source-inspired"
    assert provenance["source_retrieval_date"] == "2026-05-16"


_PRIVATE_SOURCE_PATH = "/private/non_public/source-plan/synthetic-supervision-transcript.md"


def test_stage_a_strips_non_public_source_url_from_slot() -> None:
    """A private local path in a source-plan slot must never reach the EnvSpec
    prompt."""
    from ppl_synth_data.pipeline.stages.stage_a_envspec import (
        _source_plan_prompt_suffix,
        _source_provenance_from_slot,
    )

    slot = {
        "family": "pymc",
        "domain": "clinical",
        "complexity": "medium",
        "diversity_seed": 1,
        "source_provenance": {
            "source_category": "real_adapted",
            "source_url_or_repo": _PRIVATE_SOURCE_PATH,
            "adaptation_mode": "source-inspired",
            "source_retrieval_date": "2026-05-16",
        },
    }
    provenance = _source_provenance_from_slot(slot)
    assert provenance["source_url_or_repo"] == "not_applicable_invented"
    assert _PRIVATE_SOURCE_PATH not in json.dumps(provenance)

    suffix = _source_plan_prompt_suffix(provenance)
    assert _PRIVATE_SOURCE_PATH not in suffix
    assert "/Users/" not in suffix


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_strips_non_public_source_url(monkeypatch, tmp_path: Path) -> None:
    """Stage D also strips private paths that reach an EnvSpec."""
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_strip_private_source")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    spec = minimal_envspec_record(env_id="env-present", env_name="env_present")
    spec["prompt_guidance"] = {
        "source_provenance": {
            "source_category": "real_adapted",
            "source_url_or_repo": _PRIVATE_SOURCE_PATH,
            "license_signal": "unknown",
            "adaptation_mode": "source-inspired",
            "source_retrieval_date": "2026-05-16",
        }
    }
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])
    write_jsonl(paths.stage_dirs["C"] / "summaries.jsonl", [{"env_id": "env-present", "n": 5}])

    backend = CaptureBackend(["prompt-a"])
    run_stage_d(paths, limit=1, use_batch=False, backend=backend, runtime=get_runtime("pymc"))

    raw = (paths.stage_dirs["D"] / "prompts.jsonl").read_text(encoding="utf-8")
    assert _PRIVATE_SOURCE_PATH not in raw
    assert "/Users/" not in raw
    provenance = json.loads(raw)["provenance"]
    assert provenance["source_url_or_repo"] == "not_applicable_invented"


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_appends_lazyppl_taskdata_schema(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_lazyppl_schema")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="env-present", env_name="env_present")
    spec["io_schema"] = {
        "input_type": "json",
        "input_description": "JSON",
        "output_type": "float",
        "target_name": "y",
        "feature_names": ["smoker", "count"],
    }
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])
    write_jsonl(paths.stage_dirs["C"] / "summaries.jsonl", [{"env_id": "env-present", "n": 5}])
    parquet_path = data_dir / "env-present.parquet"
    pd.DataFrame(
        {
            "smoker": [0.0, 1.0],
            "count": pd.Series([1, 2], dtype="int64"),
            "y": [0.5, 0.75],
        }
    ).to_parquet(parquet_path)
    write_jsonl(
        paths.stage_dirs["C"] / "data_refs.jsonl",
        [
            {
                "env_id": "env-present",
                "env_name": "env_present",
                "data_ref": "stage_C_data/data/env-present.parquet",
            }
        ],
    )

    backend = CaptureBackend(["Prompt body."])
    run_stage_d(paths, limit=1, use_batch=False, backend=backend, runtime=get_runtime("lazyppl"))

    row = json.loads((paths.stage_dirs["D"] / "prompts.jsonl").read_text(encoding="utf-8"))
    prompt_text = row["prompt_text"]
    assert "## LazyPPL TaskData schema" in prompt_text
    assert "smoker :: Double" in prompt_text
    assert "count :: Integer" in prompt_text
    assert "Target accessor: `y`" in prompt_text
    assert "Feature accessors: `smoker`, `count`" in prompt_text
    assert "Observed TaskData summaries from the exact rows used by the verifier" in prompt_text
    assert "`smoker`: float64, min=0" in prompt_text
    assert "`count`: int64, min=1" in prompt_text
    assert "use row-derived quantities" in prompt_text


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_lazyppl_schema_uses_materialized_columns_over_envspec(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_lazyppl_schema_drift")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="env-present", env_name="env_present")
    spec["io_schema"] = {
        "input_type": "json",
        "input_description": "JSON",
        "output_type": "float",
        "target_name": "measured_growth_rate_per_hour",
        "feature_names": ["temperature_c", "nutrient_level"],
    }
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])
    write_jsonl(paths.stage_dirs["C"] / "summaries.jsonl", [{"env_id": "env-present", "n": 2}])
    parquet_path = data_dir / "env-present.parquet"
    pd.DataFrame(
        {
            "strain_id": ["0.0", "0.9"],
            "measured_growth_rate_per_hour": [1.1, 1.4],
        }
    ).to_parquet(parquet_path)
    write_jsonl(
        paths.stage_dirs["C"] / "data_refs.jsonl",
        [
            {
                "env_id": "env-present",
                "env_name": "env_present",
                "data_ref": "stage_C_data/data/env-present.parquet",
            }
        ],
    )

    backend = CaptureBackend(["Prompt body."])
    run_stage_d(paths, limit=1, use_batch=False, backend=backend, runtime=get_runtime("lazyppl"))

    row = json.loads((paths.stage_dirs["D"] / "prompts.jsonl").read_text(encoding="utf-8"))
    prompt_text = row["prompt_text"]
    assert "strain_id :: Double" in prompt_text
    assert "Target accessor: `measured_growth_rate_per_hour`" in prompt_text
    assert "Feature accessors: `strain_id`." in prompt_text
    assert "temperature_c" not in prompt_text
    assert "nutrient_level" not in prompt_text
    assert "Only the accessors in the Row type above exist in TaskData" in prompt_text


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_d_pymc_schema_uses_materialized_columns_over_envspec(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_d_pymc_schema_drift")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="env-present", env_name="env_present")
    spec["io_schema"] = {
        "input_type": "json",
        "input_description": "JSON",
        "output_type": "float",
        "target_name": "rumination_coping_dynamics",
        "feature_names": ["time_day", "rumination_score", "coping_effort_score"],
    }
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])
    write_jsonl(paths.stage_dirs["C"] / "summaries.jsonl", [{"env_id": "env-present", "n": 24}])
    pd.DataFrame(
        {
            "t": [0.0, 1.0],
            "prey_count": [9, 10],
            "predator_count": [11, 10],
        }
    ).to_parquet(data_dir / "env-present.parquet")
    write_jsonl(
        paths.stage_dirs["C"] / "data_refs.jsonl",
        [
            {
                "env_id": "env-present",
                "env_name": "env_present",
                "data_ref": "stage_C_data/data/env-present.parquet",
            }
        ],
    )

    backend = CaptureBackend(["Prompt body."])
    run_stage_d(paths, limit=1, use_batch=False, backend=backend, runtime=get_runtime("pymc"))

    row = json.loads((paths.stage_dirs["D"] / "prompts.jsonl").read_text(encoding="utf-8"))
    prompt_text = row["prompt_text"]
    assert "## PyMC data dictionary" in prompt_text
    assert "Observed target key: `predator_count`." in prompt_text
    assert "- `t`" in prompt_text
    assert "- `prey_count`" in prompt_text
    assert "- `predator_count`" in prompt_text
    assert "`t`: float64, min=0" in prompt_text
    assert "time_day" not in prompt_text
    assert "rumination_score" not in prompt_text


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_uses_stage_yaml_defaults_when_model_unset(monkeypatch, tmp_path: Path) -> None:
    configs = tmp_path / "configs" / "stages"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "completions.yaml").write_text(
        ("stage: completions\nmodel: yaml-completion-model\nreasoning_effort: medium\n"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    paths = resolve_run_paths(tmp_path / "runs", "run_yaml_stage_e")
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "prompt-000001",
                "split": "train",
                "env_id": "env-1",
                "prompt_text": "Prompt body",
                "metadata": {"domain": "healthcare"},
                "provenance": {
                    "env_name": "env_1",
                    "domain": "healthcare",
                    "complexity": "medium",
                    "model_type": "predictive",
                    "generative_family": "hierarchical_grouped",
                    "key_concepts": ["hierarchical_model"],
                    "has_observed_data": True,
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    backend = CaptureBackend(["<think>x</think>\n```python\nprint('ok')\n```"])
    run_stage_e(
        paths,
        limit=1,
        use_batch=False,
        model=None,  # type: ignore[arg-type]
        backend=backend,
        runtime=get_runtime("pymc"),
    )
    assert backend.requests[0].model == "yaml-completion-model"
    assert backend.requests[0].reasoning_effort == "medium"
    assert backend.requests[0].max_output_tokens == 20000

    row = json.loads((paths.stage_dirs["E"] / "completions.jsonl").read_text(encoding="utf-8"))
    assert row["env_id"] == "env-1"
    assert row["split"] == "train"
    assert row["provenance"]["generative_family"] == "hierarchical_grouped"
    assert row["provenance"]["runtime_name"] == "pymc"


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_omits_reasoning_effort_when_configured(
    monkeypatch,
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs" / "stages"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "completions.yaml").write_text(
        (
            "stage: completions\n"
            "model: gpt-5.5-pro\n"
            "reasoning_effort: omit\n"
            "max_output_tokens: 16384\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_omit_reasoning")
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "prompt-000001",
                "split": "train",
                "env_id": "env-1",
                "prompt_text": "Prompt body",
                "metadata": {},
                "provenance": {
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    backend = CaptureBackend(["<think>x</think>\n```python\nprint('ok')\n```"])
    run_stage_e(
        paths,
        limit=1,
        use_batch=False,
        backend=backend,
        runtime=get_runtime("pymc"),
    )

    assert backend.requests[0].model == "gpt-5.5-pro"
    assert backend.requests[0].reasoning_effort is None
    assert backend.requests[0].max_output_tokens == 16384
    manifest = json.loads((paths.stage_dirs["E"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["configured_reasoning_effort"] == "omit"
    assert manifest["reasoning_effort"] is None
    assert manifest["max_output_tokens"] == 16384


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_appends_exact_data_contract_when_envspec_exists(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_data_contract")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    write_jsonl(
        paths.stage_dirs["A"] / "envspecs.jsonl",
        [minimal_envspec_record(env_id="env-1", env_name="env_1")],
    )
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "prompt-000001",
                "split": "train",
                "env_id": "env-1",
                "prompt_text": "Prompt body",
                "provenance": {
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    backend = CaptureBackend(["<think>x</think>\n```python\nprint('ok')\n```"])
    run_stage_e(
        paths,
        limit=1,
        use_batch=False,
        model="m",
        backend=backend,
        runtime=get_runtime("pymc"),
    )

    assert "Data dictionary contract for the submitted program:" in backend.requests[0].user
    assert 'observed target key: "y"' in backend.requests[0].user
    assert '- "group"' in backend.requests[0].user
    assert "Final answer contract for this request:" in backend.requests[0].user
    assert "Never return a blank" in backend.requests[0].user
    row = json.loads((paths.stage_dirs["E"] / "completions.jsonl").read_text(encoding="utf-8"))
    assert row["prompt_text"] == backend.requests[0].user
    assert row["metadata"]["stage_d_prompt_text"] == "Prompt body"
    assert row["metadata"]["stage_e_prompt_augmented"] is True


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_appends_lazyppl_row_accessor_contract(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_lazyppl_row_contract")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    write_jsonl(
        paths.stage_dirs["A"] / "envspecs.jsonl",
        [minimal_envspec_record(env_id="env-1", env_name="env_1")],
    )
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "prompt-000001",
                "split": "train",
                "env_id": "env-1",
                "prompt_text": "Prompt body",
                "provenance": {
                    "runtime_name": "lazyppl",
                    "runtime_version": "lazyppl-1.0",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    backend = CaptureBackend(
        [
            "<think>x</think>\n"
            '<submit_program language="haskell">\n'
            "model :: [Row] -> Meas State\n"
            "model = undefined\n"
            "logLikRow :: State -> Row -> Double\n"
            "logLikRow = undefined\n"
            "</submit_program>",
        ]
    )
    run_stage_e(
        paths,
        limit=1,
        use_batch=False,
        model="m",
        backend=backend,
        runtime=get_runtime("lazyppl"),
    )

    user_prompt = backend.requests[0].user
    assert "Data dictionary contract for the submitted LazyPPL program:" in user_prompt
    assert "observed target Row accessor: `y row`" in user_prompt
    assert "- `group row`" in user_prompt
    assert "Final answer contract for this request:" in user_prompt
    assert '<submit_program language="haskell">' in user_prompt
    assert "def model(data)" not in user_prompt


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_uses_materialized_lazyppl_row_schema_as_authority(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_lazyppl_materialized")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    spec = minimal_envspec_record(env_id="env-1", env_name="env_1")
    spec["io_schema"]["target_name"] = "growth_rate"
    spec["io_schema"]["feature_names"] = ["temperature_c", "nutrient_level"]
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])
    stage_d_prompt = (
        "Prompt body\n"
        "## LazyPPL TaskData schema\n\n"
        "```haskell\n"
        "data Row = Row\n"
        "  { strain_id :: Int\n"
        "  , measured_growth_rate_per_hour :: Double\n"
        "  } deriving (Show)\n"
        "```\n"
        "Target accessor: `measured_growth_rate_per_hour`.\n"
        "Feature accessors: `strain_id`.\n"
    )
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "prompt-000001",
                "split": "train",
                "env_id": "env-1",
                "prompt_text": stage_d_prompt,
                "provenance": {
                    "runtime_name": "lazyppl",
                    "runtime_version": "lazyppl-1.0",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    backend = CaptureBackend(
        [
            "<think>x</think>\n"
            '<submit_program language="haskell">\n'
            "model :: [Row] -> Meas State\n"
            "model = undefined\n"
            "logLikRow :: State -> Row -> Double\n"
            "logLikRow = undefined\n"
            "</submit_program>",
        ]
    )
    run_stage_e(
        paths,
        limit=1,
        use_batch=False,
        model="m",
        backend=backend,
        runtime=get_runtime("lazyppl"),
    )

    user_prompt = backend.requests[0].user
    assert "## LazyPPL TaskData schema" in user_prompt
    assert "only Row accessor contract" in user_prompt
    assert "strain_id :: Int" in user_prompt
    assert "measured_growth_rate_per_hour :: Double" in user_prompt
    assert "Final answer contract for this request:" in user_prompt
    assert "`temperature_c row`" not in user_prompt
    assert "`nutrient_level row`" not in user_prompt


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_uses_materialized_pymc_data_schema_as_authority(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_pymc_materialized")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    spec = minimal_envspec_record(env_id="env-1", env_name="env_1")
    spec["io_schema"]["target_name"] = "rumination_coping_dynamics"
    spec["io_schema"]["feature_names"] = [
        "time_day",
        "rumination_score",
        "coping_effort_score",
    ]
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])
    stage_d_prompt = (
        "Prompt body\n"
        "## PyMC data dictionary\n\n"
        "Observed target key: `predator_count`.\n"
        "Observed predictor keys:\n"
        "- `t`\n"
        "- `prey_count`\n\n"
        "Allowed `data` keys:\n"
        "- `t`\n"
        "- `prey_count`\n"
        "- `predator_count`\n"
    )
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "prompt-000001",
                "split": "train",
                "env_id": "env-1",
                "prompt_text": stage_d_prompt,
                "provenance": {
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    backend = CaptureBackend(["<think>x</think>\n<submit_program>\nprint('ok')\n</submit_program>"])
    run_stage_e(
        paths,
        limit=1,
        use_batch=False,
        model="m",
        backend=backend,
        runtime=get_runtime("pymc"),
    )

    user_prompt = backend.requests[0].user
    assert "## PyMC data dictionary" in user_prompt
    assert "the only `data` key contract" in user_prompt
    assert "`predator_count`" in user_prompt
    assert "`prey_count`" in user_prompt
    assert '"time_day"' not in user_prompt
    assert '"rumination_score"' not in user_prompt
    assert '"coping_effort_score"' not in user_prompt


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_rejects_prompt_runtime_mismatch(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_runtime_mismatch")
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "prompt-000001",
                "split": "train",
                "env_id": "env-1",
                "prompt_text": "Prompt body",
                "metadata": {},
                "provenance": {
                    "runtime_name": "lazyppl",
                    "runtime_version": "lazyppl-1.0",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        run_stage_e(
            paths,
            limit=1,
            use_batch=False,
            backend=CaptureBackend(["ignored"]),
            runtime=get_runtime("pymc"),
        )
    except ValueError as exc:
        assert "runtime" in str(exc)
        assert "lazyppl" in str(exc)
    else:
        raise AssertionError("expected prompt runtime mismatch rejection")


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_persists_candidate_outputs_and_provider_response(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_candidates")
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "prompt-000001",
                "split": "train",
                "env_id": "env-1",
                "prompt_text": "Prompt body",
                "metadata": {"domain": "healthcare"},
                "provenance": {
                    "env_name": "env_1",
                    "domain": "healthcare",
                    "complexity": "medium",
                    "model_type": "predictive",
                    "generative_family": "hierarchical_grouped",
                    "key_concepts": ["hierarchical_model"],
                    "has_observed_data": True,
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    primary = "<think>x</think>\n```python\nprint('ok')\n```"
    alternate = "```python\nprint('alt')\n```"
    backend = CaptureBackend(
        [
            (
                primary,
                {
                    "id": "resp_stage_e_test",
                    "usage": {"input_tokens": 7, "output_tokens": 11},
                    "choices": [
                        {"message": {"content": primary}},
                        {"message": {"content": alternate}},
                    ],
                },
            )
        ]
    )

    run_stage_e(
        paths,
        limit=1,
        use_batch=False,
        backend=backend,
        runtime=get_runtime("pymc"),
    )

    row = json.loads((paths.stage_dirs["E"] / "completions.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((paths.stage_dirs["E"] / "manifest.json").read_text(encoding="utf-8"))

    assert row["candidate_outputs"] == [alternate]
    assert row["provider_response"]["choices"][1]["message"]["content"] == alternate
    assert row["metadata"]["provider_request_id"] == "completion-000000"
    assert row["metadata"]["provider_response_id"] == "resp_stage_e_test"
    assert len(row["metadata"]["provider_response_sha256"]) == 64
    assert len(row["metadata"]["raw_completion_sha256"]) == 64
    assert row["metadata"]["provider_usage"] == {"input_tokens": 7, "output_tokens": 11}
    assert manifest["count_with_candidates"] == 1
    assert manifest["request_ids"] == ["completion-000000"]
    assert manifest["success_request_ids"] == ["completion-000000"]
    assert manifest["skipped_request_ids"] == []


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_skips_empty_completion_outputs(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_empty_output")
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    prompts = [
        {
            "id": "prompt-000001",
            "split": "train",
            "env_id": "env-1",
            "prompt_text": "Prompt one",
            "metadata": {},
            "provenance": {
                "runtime_name": "pymc",
                "runtime_version": "v5",
            },
        },
        {
            "id": "prompt-000002",
            "split": "train",
            "env_id": "env-2",
            "prompt_text": "Prompt two",
            "metadata": {},
            "provenance": {
                "runtime_name": "pymc",
                "runtime_version": "v5",
            },
        },
    ]
    write_jsonl(paths.stage_dirs["D"] / "prompts.jsonl", prompts)

    run_stage_e(
        paths,
        limit=2,
        use_batch=False,
        backend=CaptureBackend(["", "<think>x</think>\n```python\nprint('ok')\n```"]),
        runtime=get_runtime("pymc"),
    )

    rows = [
        json.loads(line)
        for line in (paths.stage_dirs["E"] / "completions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    skipped = [
        json.loads(line)
        for line in (paths.stage_dirs["E"] / "skipped_completions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    manifest = json.loads((paths.stage_dirs["E"] / "manifest.json").read_text())

    assert len(rows) == 1
    assert rows[0]["id"] == "completion-000001"
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "empty_output"
    assert manifest["count_requested"] == 2
    assert manifest["count"] == 1
    assert manifest["skipped_count"] == 1


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_batch_partial_failure_writes_skipped_completions(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_partial_batch")
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    prompts = [
        {
            "id": "prompt-000000",
            "split": "train",
            "env_id": "env-0",
            "prompt_text": "Prompt zero",
            "metadata": {},
            "provenance": {
                "runtime_name": "pymc",
                "runtime_version": "v5",
            },
        },
        {
            "id": "prompt-000001",
            "split": "validation",
            "env_id": "env-1",
            "prompt_text": "Prompt one",
            "metadata": {},
            "provenance": {
                "runtime_name": "pymc",
                "runtime_version": "v5",
            },
        },
        {
            "id": "prompt-000002",
            "split": "test",
            "env_id": "env-2",
            "prompt_text": "Prompt two",
            "metadata": {},
            "provenance": {
                "runtime_name": "pymc",
                "runtime_version": "v5",
            },
        },
    ]
    write_jsonl(paths.stage_dirs["D"] / "prompts.jsonl", prompts)

    class PartialBatchBackend:
        capabilities = BackendCapabilities(
            supports_batch=True,
            supports_json_schema=False,
            supports_reasoning_effort=False,
        )

        def __init__(self) -> None:
            self.allow_partial_failures: bool | None = None
            self.last_batch_failures = {"completion-000001": "empty output_text"}

        def complete(self, request: LLMRequest) -> LLMResult:
            raise AssertionError("batch path should not call complete")

        def submit_batch(
            self,
            requests: list[tuple[str, LLMRequest]],
        ) -> BatchHandle:
            assert [request_id for request_id, _ in requests] == [
                "completion-000000",
                "completion-000001",
                "completion-000002",
            ]
            return BatchHandle(
                backend="fake",
                job_id="batch-stage-e-partial",
                request_ids=tuple(request_id for request_id, _ in requests),
            )

        def collect_batch(
            self,
            handle: BatchHandle,
            *,
            allow_partial_failures: bool = False,
        ) -> dict[str, LLMResult]:
            assert handle.job_id == "batch-stage-e-partial"
            self.allow_partial_failures = allow_partial_failures
            return {
                "completion-000000": LLMResult(
                    content="<think>x</think>\n```python\nprint('zero')\n```",
                    raw={"id": "zero"},
                ),
                "completion-000002": LLMResult(
                    content="<think>x</think>\n```python\nprint('two')\n```",
                    raw={"id": "two"},
                ),
            }

    backend = PartialBatchBackend()

    result = run_stage_e(
        paths,
        limit=3,
        use_batch=True,
        backend=backend,
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    assert backend.allow_partial_failures is True
    rows = [
        json.loads(line)
        for line in (paths.stage_dirs["E"] / "completions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    skipped = [
        json.loads(line)
        for line in (paths.stage_dirs["E"] / "skipped_completions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    manifest = json.loads((paths.stage_dirs["E"] / "manifest.json").read_text())

    assert [row["id"] for row in rows] == ["completion-000000", "completion-000002"]
    assert [row["env_id"] for row in rows] == ["env-0", "env-2"]
    assert rows[0]["provider_response"] == {"id": "zero"}
    assert rows[0]["metadata"]["provider_request_id"] == "completion-000000"
    assert rows[0]["metadata"]["provider_response_id"] == "zero"
    assert len(skipped) == 1
    assert skipped[0] == {
        "id": "completion-000001",
        "prompt_id": "prompt-000001",
        "env_id": "env-1",
        "reason": "missing_or_empty_batch_output",
        "provider_failure": "empty output_text",
    }
    assert manifest["count_requested"] == 3
    assert manifest["count"] == 2
    assert manifest["skipped_count"] == 1
    assert manifest["request_ids"] == [
        "completion-000000",
        "completion-000001",
        "completion-000002",
    ]
    assert manifest["success_request_ids"] == [
        "completion-000000",
        "completion-000002",
    ]
    assert manifest["skipped_request_ids"] == ["completion-000001"]
    assert manifest["batch_receipt"].endswith("batch_receipt.json")
    assert manifest["use_batch"] is True


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_e_sync_resumes_existing_partial_outputs(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_e_resume_partial")
    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    prompts = [
        {
            "id": "prompt-000001",
            "split": "train",
            "env_id": "env-1",
            "prompt_text": "Prompt one",
            "metadata": {},
            "provenance": {
                "runtime_name": "pymc",
                "runtime_version": "v5",
            },
        },
        {
            "id": "prompt-000002",
            "split": "train",
            "env_id": "env-2",
            "prompt_text": "Prompt two",
            "metadata": {},
            "provenance": {
                "runtime_name": "pymc",
                "runtime_version": "v5",
            },
        },
    ]
    write_jsonl(paths.stage_dirs["D"] / "prompts.jsonl", prompts)
    write_jsonl(
        paths.stage_dirs["E"] / "completions.jsonl",
        [
            {
                "id": "completion-000000",
                "prompt_id": "prompt-000001",
                "env_id": "env-1",
                "split": "train",
                "prompt_text": "Prompt one",
                "raw_completion": "<think>old</think>\n```python\nprint('old')\n```",
                "candidate_outputs": [],
                "provider_response": {},
                "provenance": {
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
                "metadata": {
                    "stage_d_prompt_text": "Prompt one",
                    "stage_e_prompt_augmented": False,
                },
            }
        ],
    )
    write_jsonl(paths.stage_dirs["E"] / "skipped_completions.jsonl", [])

    backend = CaptureBackend(["<think>new</think>\n```python\nprint('new')\n```"])
    run_stage_e(
        paths,
        limit=2,
        use_batch=False,
        backend=backend,
        runtime=get_runtime("pymc"),
    )

    rows = [
        json.loads(line)
        for line in (paths.stage_dirs["E"] / "completions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    progress = json.loads((paths.stage_dirs["E"] / "progress.json").read_text())
    manifest = json.loads((paths.stage_dirs["E"] / "manifest.json").read_text())

    assert [row["id"] for row in rows] == ["completion-000000", "completion-000001"]
    assert len(backend.requests) == 1
    assert backend.requests[0].user.startswith("Prompt two")
    assert progress["status"] == "success"
    assert progress["count_completed"] == 2
    assert manifest["count"] == 2


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_a_uses_generation_plan(monkeypatch, tmp_path: Path) -> None:
    """Stage A builds prompts from build_generation_plan and includes family in each request."""
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_a_gen_plan")

    spec = minimal_envspec_record(env_id="gen-1", env_name="gen_1")
    backend = CaptureBackend([json.dumps(spec)] * 3)
    backend.capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=True,
        supports_reasoning_effort=True,
    )
    result = run_stage_a(paths, limit=3, seed=42, use_batch=False, backend=backend)

    assert result.status == "success"
    assert result.outputs["count"] == 3
    assert result.outputs["seed"] == 42

    from ppl_synth_data.specs.sampling import build_generation_plan

    plan = build_generation_plan(3, seed=42)
    for i, req in enumerate(backend.requests):
        assert f"Family: {plan[i]['family']}" in req.user
        assert f"Domain: {plan[i]['domain']}" in req.user
        assert f"Complexity: {plan[i]['complexity']}" in req.user
        assert req.text_format is not None
        assert req.text_format["type"] == "json_schema"
        assert req.text_format["strict"] is False
        assert req.text_format["name"] == "EnvSpec"
        assert req.reasoning_effort == "none"
        assert req.max_output_tokens == 8192
    manifest = json.loads((paths.stage_dirs["A"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["configured_reasoning_effort"] == "none"
    assert manifest["reasoning_effort"] == "none"
    assert manifest["max_output_tokens"] == 8192


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_a_source_plan_injects_source_provenance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_a_source_plan")
    source_plan = tmp_path / "source_plan.jsonl"
    source_plan.write_text(
        json.dumps(
            {
                "family": "hierarchical_grouped",
                "domain": "psychology",
                "complexity": "medium",
                "source_category": "probmods_webppl",
                "source_url_or_repo": "https://github.com/probmods/webppl",
                "license_signal": "MIT",
                "adaptation_mode": "source-inspired",
                "source_retrieval_date": "2026-05-16",
                "runtime_target": "lazyppl",
                "problem_hint": "Recursive social-reasoning choice model",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = minimal_envspec_record(env_id="source-env", env_name="source_env")
    spec["generative_family"] = "hierarchical_grouped"
    spec["domain"] = "psychology"
    backend = CaptureBackend([json.dumps(spec)])

    result = run_stage_a(
        paths,
        limit=3,
        seed=42,
        source_plan_path=source_plan,
        use_batch=False,
        backend=backend,
    )

    assert result.status == "success"
    assert result.outputs["count"] == 1
    assert result.outputs["source_plan_count"] == 1
    assert result.outputs["source_plan_used_count"] == 1
    assert "Source provenance:" in backend.requests[0].user
    assert "https://github.com/probmods/webppl" in backend.requests[0].user
    assert "Runtime target: lazyppl" in backend.requests[0].user
    assert "Recursive social-reasoning choice model" in backend.requests[0].user

    rows = [
        json.loads(line)
        for line in (paths.stage_dirs["A"] / "envspecs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    source_provenance = rows[0]["prompt_guidance"]["source_provenance"]
    assert source_provenance == {
        "source_category": "probmods_webppl",
        "source_url_or_repo": "https://github.com/probmods/webppl",
        "license_signal": "MIT",
        "adaptation_mode": "source-inspired",
        "source_retrieval_date": "2026-05-16",
    }


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_a_source_plan_normalizes_invented_source_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_a_invented_source")
    source_plan = tmp_path / "source_plan.jsonl"
    source_plan.write_text(
        json.dumps(
            {
                "family": "simple_regression",
                "domain": "manufacturing",
                "complexity": "simple",
                "source_category": "synthetic_invented",
                "source_url_or_repo": None,
                "license_signal": "not_applicable_invented",
                "adaptation_mode": "invented",
                "source_retrieval_date": "2026-05-16",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = minimal_envspec_record(env_id="invented-env", env_name="invented_env")
    spec["generative_family"] = "simple_regression"
    spec["domain"] = "manufacturing"
    backend = CaptureBackend([json.dumps(spec)])

    run_stage_a(
        paths,
        limit=1,
        seed=42,
        source_plan_path=source_plan,
        use_batch=False,
        backend=backend,
    )

    row = json.loads((paths.stage_dirs["A"] / "envspecs.jsonl").read_text())
    source_provenance = row["prompt_guidance"]["source_provenance"]
    assert source_provenance == {
        "source_category": "synthetic_invented",
        "source_url_or_repo": "not_applicable_invented",
        "license_signal": "not_applicable_invented",
        "adaptation_mode": "invented",
        "source_retrieval_date": "2026-05-16",
    }


def test_generation_plan_filters_source_plan_by_runtime_target(tmp_path: Path) -> None:
    source_plan = tmp_path / "source_plan.jsonl"
    rows = [
        {
            "id": "pymc-row",
            "family": "regression",
            "domain": "manufacturing",
            "complexity": "medium",
            "runtime_target": "pymc",
        },
        {
            "id": "lazyppl-row",
            "family": "memoized_regime",
            "domain": "operations",
            "complexity": "complex",
            "runtime_target": "lazy-ppl",
        },
        {
            "id": "runtime-neutral-row",
            "family": "hierarchical_count",
            "domain": "ecology",
            "complexity": "medium",
        },
    ]
    write_jsonl(source_plan, rows)

    plan, manifest = _generation_plan(
        limit=10,
        seed=0,
        source_plan_path=source_plan,
        runtime_target="lazyppl",
    )

    assert [row["id"] for row in plan] == [
        "lazyppl-row",
        "runtime-neutral-row",
    ]
    assert manifest["source_plan_count"] == 3
    assert manifest["source_plan_used_count"] == 2
    assert manifest["source_plan_runtime_target"] == "lazyppl"
    assert manifest["source_plan_runtime_eligible_count"] == 2
    assert manifest["source_plan_runtime_skipped_count"] == 1


def test_generation_plan_rejects_unknown_source_plan_runtime(tmp_path: Path) -> None:
    source_plan = tmp_path / "source_plan.jsonl"
    write_jsonl(
        source_plan,
        [
            {
                "family": "regression",
                "domain": "manufacturing",
                "complexity": "medium",
                "runtime_target": "webppl",
            }
        ],
    )

    with pytest.raises(ValueError, match="unknown runtime_target"):
        _generation_plan(
            limit=1,
            seed=0,
            source_plan_path=source_plan,
            runtime_target="pymc",
        )


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_a_batch_partial_failure_writes_skipped_envspecs(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_a_partial_batch")

    class PartialBatchBackend:
        capabilities = BackendCapabilities(
            supports_batch=True,
            supports_json_schema=True,
            supports_reasoning_effort=True,
        )

        def __init__(self) -> None:
            self.requests: list[tuple[str, LLMRequest]] = []
            self.allow_partial_failures: bool | None = None

        def complete(self, request: LLMRequest) -> LLMResult:
            raise AssertionError("stage A should use batch path")

        def submit_batch(
            self,
            requests: list[tuple[str, LLMRequest]],
        ) -> BatchHandle:
            self.requests = list(requests)
            return BatchHandle(
                backend="test",
                job_id="batch-partial",
                request_ids=tuple(request_id for request_id, _ in requests),
            )

        def collect_batch(
            self,
            handle: BatchHandle,
            *,
            allow_partial_failures: bool = False,
        ) -> dict[str, LLMResult]:
            assert handle.job_id == "batch-partial"
            self.allow_partial_failures = allow_partial_failures
            specs = [
                minimal_envspec_record(env_id="env-0", env_name="env_0"),
                minimal_envspec_record(env_id="env-2", env_name="env_2"),
            ]
            return {
                "spec-000000": LLMResult(content=json.dumps(specs[0]), raw={}),
                "spec-000002": LLMResult(content=json.dumps(specs[1]), raw={}),
            }

    backend = PartialBatchBackend()
    result = run_stage_a(paths, limit=3, seed=42, use_batch=True, backend=backend)

    assert result.status == "success"
    assert result.outputs["count"] == 2
    assert result.outputs["count_requested"] == 3
    assert result.outputs["skipped_count"] == 1
    assert backend.allow_partial_failures is True
    rows = [
        json.loads(line)
        for line in (paths.stage_dirs["A"] / "envspecs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["env_id"] for row in rows] == ["env-0", "env-2"]
    skipped = json.loads(
        (paths.stage_dirs["A"] / "skipped_envspecs.jsonl").read_text(encoding="utf-8")
    )
    assert skipped == {
        "id": "spec-000001",
        "reason": "missing_or_failed_batch_output",
    }


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_a_falls_back_to_json_object_without_schema_support(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = resolve_run_paths(tmp_path / "runs", "run_stage_a_json_object")

    spec = minimal_envspec_record(env_id="gen-1", env_name="gen_1")
    backend = CaptureBackend([json.dumps(spec)])

    result = run_stage_a(paths, limit=1, seed=7, use_batch=False, backend=backend)

    assert result.status == "success"
    assert backend.requests[0].text_format == {"type": "json_object"}


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_f_with_no_records_does_not_build_backend(monkeypatch, tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_empty")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["E"] / "completions.jsonl").write_text("", encoding="utf-8")

    def boom(_name):  # pragma: no cover - asserted via behavior
        raise RuntimeError("backend should not be built")

    monkeypatch.setattr("ppl_synth_data.pipeline.stages.stage_f_validate.build_backend", boom)
    result = run_stage_f(paths, backend_name="openai", runtime=get_runtime("pymc"))

    assert result.status == "success"
    assert result.outputs["count_total"] == 0
    assert result.outputs["runtime"] == "pymc"


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_deterministic_judge_does_not_build_judge_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_deterministic_judge")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    _write_stage_c_visible_data(paths, "env-1")
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-local-judge",
                "prompt_id": "p-local-judge",
                "prompt_text": "Observed daily counts with seasonal patterns.",
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-1",
                "split": "train",
                "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def boom(*args, **kwargs):  # pragma: no cover - asserted via behavior
        _ = (args, kwargs)
        raise RuntimeError("backend should not be built")

    monkeypatch.setattr("ppl_synth_data.pipeline.stages.stage_f_validate.build_backend", boom)
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: ValidationRunResult(
            ok=True,
            final_code="def model(data):\n    return 1\n",
            message="ok",
            trace=(),
        ),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.score_pymc_semantics",
        _semantic_not_scored,
    )

    class DummyBackend:
        pass

    result = run_stage_f(
        paths,
        backend=DummyBackend(),
        judge_backend_name="deterministic",
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    assert result.outputs["leakage_judge"] == "deterministic_banned_terms"
    row = json.loads((paths.stage_dirs["F"] / "validated.jsonl").read_text(encoding="utf-8"))
    first_step = row["validation"]["trace"]["steps"][0]
    assert first_step["phase"] == "leakage_judge"
    assert first_step["metadata"]["method"] == "deterministic_banned_terms"


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_deterministic_judge_masks_runtime_prompt_markers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_runtime_contract_sections")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    _write_stage_c_visible_data(paths, "env-1")
    prompt_text = (
        "Observed continuous readings with one covariate.\n\n"
        "## PyMC data dictionary\n\n"
        "The verifier calls `def model(data)` with exact materialized keys.\n"
        "Observed target key: `y`.\n"
        "Observed predictor keys:\n"
        "- `x`\n\n"
        "PyMC submission constraint: the data dictionary above is the only "
        "`data` key contract.\n\n"
        "Final answer contract for this request:\n"
        "- The submitted Python must define `def model(data):` and return a "
        "`pm.Model`."
    )
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-runtime-contract",
                "prompt_id": "p-runtime-contract",
                "prompt_text": prompt_text,
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-1",
                "split": "train",
                "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: ValidationRunResult(
            ok=True,
            final_code="def model(data):\n    return 1\n",
            message="ok",
            trace=(),
        ),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.score_pymc_semantics",
        _semantic_not_scored,
    )

    class DummyBackend:
        pass

    result = run_stage_f(
        paths,
        backend=DummyBackend(),
        judge_backend_name="deterministic",
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    assert result.outputs["count_validated"] == 1
    row = json.loads((paths.stage_dirs["F"] / "validated.jsonl").read_text(encoding="utf-8"))
    first_step = row["validation"]["trace"]["steps"][0]
    assert first_step["phase"] == "leakage_judge"
    assert first_step["message"] == "CLEAN"
    assert "## PyMC data dictionary" in first_step["metadata"]["masked_runtime_markers"]
    assert "PyMC submission constraint:" in first_step["metadata"]["masked_runtime_markers"]
    assert (
        "Final answer contract for this request:"
        in (first_step["metadata"]["masked_runtime_markers"])
    )
    assert "PyMC" not in first_step["metadata"]["evidence"]


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_deterministic_judge_rejects_family_leakage_after_runtime_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_runtime_marker_then_leak")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    prompt_text = (
        "Observed daily counts.\n\n"
        "## PyMC data dictionary\n"
        "Allowed `data` keys: `count`, `day`.\n"
        "Use a Poisson likelihood for the submitted model."
    )
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-runtime-marker-leak",
                "prompt_id": "p-runtime-marker-leak",
                "prompt_text": prompt_text,
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-1",
                "split": "train",
                "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    called = {"repair": 0}

    def _unexpected_repair(*args, **kwargs):
        called["repair"] += 1
        raise AssertionError("repair should not run after leakage rejection")

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        _unexpected_repair,
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.stages.stage_f_validate.build_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("backend should not be built after leakage rejection")
        ),
    )

    result = run_stage_f(
        paths,
        backend=None,
        judge_backend_name="deterministic",
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    assert called["repair"] == 0
    row = json.loads((paths.stage_dirs["F"] / "rejected.jsonl").read_text(encoding="utf-8"))
    first_step = row["validation_trace"]["steps"][0]
    assert row["reason"] == "prompt_leakage"
    assert "poisson" in first_step["metadata"]["evidence"]
    assert "## PyMC data dictionary" in first_step["metadata"]["masked_runtime_markers"]


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_uses_metadata_family_when_partial_provenance_is_missing_it(
    monkeypatch, tmp_path: Path
) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_partial_provenance")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    _write_stage_c_visible_data(paths, "env-1")
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-1",
                "prompt_id": "p-1",
                "prompt_text": "Prompt",
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-1",
                "split": "train",
                "provenance": {"runtime_name": "pymc"},
                "metadata": {
                    "env_id": "env-1",
                    "generative_family": "legacy-family",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen_families: list[str | None] = []

    def fake_judge(prompt_text, backend, model, actual_family=None):
        _ = (prompt_text, backend, model)
        seen_families.append(actual_family)
        return {"rating": "CLEAN", "evidence": "", "recovered_family": None, "leaked": False}

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.judge_leakage",
        fake_judge,
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.stages.stage_f_validate.build_backend",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: (True, "def model(data):\n    return 1\n", "ok"),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.score_pymc_semantics",
        _semantic_not_scored,
    )

    class DummyBackend:
        pass

    result = run_stage_f(
        paths,
        backend=DummyBackend(),
        judge_backend_name="openai",
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    assert seen_families == ["legacy-family"]


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_happy_path_persists_validation_on_typed_field(monkeypatch, tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_happy_path")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    _write_stage_c_visible_data(paths, "env-typed")
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-typed",
                "prompt_id": "p-typed",
                "prompt_text": "Prompt",
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-typed",
                "split": "train",
                "provenance": {
                    "env_name": "env_typed",
                    "domain": "healthcare",
                    "complexity": "medium",
                    "model_type": "predictive",
                    "generative_family": "hierarchical_grouped",
                    "key_concepts": ["hierarchical_model"],
                    "has_observed_data": True,
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
                "metadata": {"seed": 7},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.judge_leakage",
        lambda *args, **kwargs: {
            "rating": "CLEAN",
            "evidence": "",
            "recovered_family": None,
            "leaked": False,
        },
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.stages.stage_f_validate.build_backend",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: ValidationRunResult(
            ok=True,
            final_code="def model(data):\n    return 2\n",
            message="repaired",
            trace=(
                ValidationTraceEntry(
                    attempt=0,
                    phase="syntax",
                    ok=False,
                    message="missing return",
                ),
                ValidationTraceEntry(
                    attempt=0,
                    phase="repair",
                    ok=True,
                    message="repair requested after syntax failure",
                    metadata={"trigger": "syntax", "code_changed": True},
                ),
                ValidationTraceEntry(
                    attempt=1,
                    phase="syntax",
                    ok=True,
                    message="ok",
                ),
                ValidationTraceEntry(
                    attempt=1,
                    phase="runtime",
                    ok=True,
                    message="ok",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.score_pymc_semantics",
        _semantic_not_scored,
    )

    class DummyBackend:
        pass

    result = run_stage_f(
        paths,
        backend=DummyBackend(),
        judge_backend_name="openai",
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    manifest = json.loads((paths.stage_dirs["F"] / "manifest.json").read_text(encoding="utf-8"))
    assert result.outputs["validation_pass_rate"] == 1.0
    assert manifest["validation_pass_rate"] == 1.0
    assert "compile_pass_rate" not in manifest
    row = json.loads((paths.stage_dirs["F"] / "validated.jsonl").read_text(encoding="utf-8"))
    trace_row = json.loads((paths.stage_dirs["F"] / "traces.jsonl").read_text(encoding="utf-8"))
    assert row["parsed_program"] == "def model(data):\n    return 2\n"
    assert row["validation"]["ok"] is True
    assert row["validation"]["repaired"] is True
    assert row["validation"]["message"] == "repaired"
    assert row["validation"]["evidence_kind"] == "verifier"
    assert row["validation"]["truth_owner"] == "verifier"
    assert row["validation"]["semantic_status"] == "not_scored"
    assert row["validation"]["trace"]["steps"][0]["phase"] == "leakage_judge"
    assert row["validation"]["trace"]["steps"][0]["evidence_kind"] == "heuristic"
    assert row["validation"]["trace"]["steps"][0]["truth_owner"] == "heuristic"
    assert row["validation"]["trace"]["steps"][0]["semantic_status"] == "pass"
    assert "evidence_kind" not in row["validation"]["trace"]["steps"][0]["metadata"]
    assert "truth_owner" not in row["validation"]["trace"]["steps"][0]["metadata"]
    assert "semantic_status" not in row["validation"]["trace"]["steps"][0]["metadata"]
    assert row["validation"]["trace"]["steps"][-1]["phase"] == "semantic_validation"
    assert row["validation"]["trace"]["steps"][-1]["evidence_kind"] == "verifier"
    assert row["validation"]["trace"]["steps"][-1]["truth_owner"] == "verifier"
    assert row["validation"]["trace"]["steps"][-1]["semantic_status"] == "not_scored"
    assert "evidence_kind" not in row["validation"]["trace"]["steps"][-1]["metadata"]
    assert "truth_owner" not in row["validation"]["trace"]["steps"][-1]["metadata"]
    assert "semantic_status" not in row["validation"]["trace"]["steps"][-1]["metadata"]
    assert trace_row["trace_id"] == "c-typed:validation"
    assert trace_row["artifacts"]["outcome"] == "validated"
    assert trace_row["steps"][0]["metadata"]["evidence_kind"] == "heuristic"
    assert trace_row["steps"][0]["metadata"]["truth_owner"] == "heuristic"
    assert trace_row["steps"][0]["metadata"]["semantic_status"] == "pass"
    assert trace_row["steps"][0]["metadata"]["phase"] == "leakage_judge"
    assert row["metadata"] == {"seed": 7}
    progress = json.loads((paths.stage_dirs["F"] / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "success"
    assert progress["count_processed"] == 1
    assert progress["count_validated"] == 1
    assert progress["count_rejected"] == 0


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_rejects_missing_stage_c_visible_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_missing_stage_c_data")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-missing-data",
                "prompt_id": "p-missing-data",
                "prompt_text": "Prompt",
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-missing-data",
                "split": "train",
                "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    called = {"repair": 0}

    def _unexpected_repair(*args, **kwargs):
        called["repair"] += 1
        raise AssertionError("repair must not run without real Stage C data")

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        _unexpected_repair,
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.stages.stage_f_validate.build_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("backend should not be built before Stage C data exists")
        ),
    )

    result = run_stage_f(
        paths,
        backend=None,
        judge_backend_name="deterministic",
        runtime=get_runtime("pymc"),
    )

    rejected_row = json.loads(
        (paths.stage_dirs["F"] / "rejected.jsonl").read_text(encoding="utf-8")
    )
    trace_row = json.loads((paths.stage_dirs["F"] / "traces.jsonl").read_text(encoding="utf-8"))

    assert result.status == "success"
    assert called["repair"] == 0
    assert rejected_row["reason"] == "missing_stage_c_data"
    assert rejected_row["message"] == (
        "missing Stage C visible data parquet for env_id=env-missing-data"
    )
    assert str(tmp_path) not in json.dumps(rejected_row)
    assert rejected_row["sample_data_error"] == rejected_row["message"]
    assert rejected_row["validation_trace"]["steps"][-1]["phase"] == "stage_c_data"
    assert rejected_row["validation_trace"]["steps"][-1]["semantic_status"] == "critical"
    assert trace_row["artifacts"]["reason"] == "missing_stage_c_data"
    assert trace_row["steps"][-1]["metadata"]["phase"] == "stage_c_data"


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_resumes_from_partial_checkpoint(monkeypatch, tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_resume")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    _write_stage_c_visible_data(paths, "env-resume-2")
    first_record = {
        "id": "c-resume-1",
        "prompt_id": "p-resume-1",
        "prompt_text": "Prompt one",
        "raw_completion": "<think>x</think>\n```python\ndef model(data):\n    return 1\n```",
        "env_id": "env-resume-1",
        "split": "train",
        "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
        "metadata": {},
    }
    second_record = {
        **first_record,
        "id": "c-resume-2",
        "prompt_id": "p-resume-2",
        "prompt_text": "Prompt two",
        "env_id": "env-resume-2",
    }
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(first_record) + "\n" + json.dumps(second_record) + "\n",
        encoding="utf-8",
    )

    stage_f_dir = paths.stage_dirs["F"]
    stage_f_dir.mkdir(parents=True, exist_ok=True)
    (stage_f_dir / "validated.jsonl").write_text(json.dumps(first_record) + "\n")
    (stage_f_dir / "rejected.jsonl").write_text("")
    (stage_f_dir / "traces.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "c-resume-1:validation",
                "task_id": "env-resume-1",
                "trial_id": "c-resume-1",
                "steps": [],
                "artifacts": {"outcome": "validated"},
                "metadata": {"stage": "F", "runtime": "pymc"},
                "schema_version": "ppl.platform.record.v0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen_records: list[str] = []

    def fake_judge(prompt_text, backend, model, actual_family=None):
        _ = (backend, model, actual_family)
        seen_records.append(prompt_text)
        return {"rating": "CLEAN", "evidence": "", "recovered_family": None, "leaked": False}

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.stages.stage_f_validate.build_backend",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.judge_leakage",
        fake_judge,
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: ValidationRunResult(
            ok=True,
            final_code="def model(data):\n    return 2\n",
            message="ok",
            trace=(
                ValidationTraceEntry(
                    attempt=0,
                    phase="syntax",
                    ok=True,
                    message="ok",
                ),
                ValidationTraceEntry(
                    attempt=0,
                    phase="runtime",
                    ok=True,
                    message="ok",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.score_pymc_semantics",
        _semantic_not_scored,
    )

    result = run_stage_f(paths, runtime=get_runtime("pymc"))

    assert result.status == "success"
    assert seen_records == ["Prompt two"]
    validated = [
        json.loads(line)
        for line in (stage_f_dir / "validated.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = [
        json.loads(line)
        for line in (stage_f_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    progress = json.loads((stage_f_dir / "progress.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in validated] == ["c-resume-1", "c-resume-2"]
    assert [row["trace_id"] for row in traces] == [
        "c-resume-1:validation",
        "c-resume-2:validation",
    ]
    assert progress["status"] == "success"
    assert progress["count_processed"] == 2


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_discards_checkpoint_rows_not_in_current_stage_e(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_discards_stale_checkpoint")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    _write_stage_c_visible_data(paths, "env-current")
    stale_record = {
        "id": "c-stale",
        "prompt_id": "p-stale",
        "prompt_text": "Stale prompt",
        "raw_completion": "<think>x</think>\n```python\ndef model(data):\n    return 1\n```",
        "env_id": "env-stale",
        "split": "train",
        "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
        "metadata": {},
    }
    current_record = {
        **stale_record,
        "id": "c-current",
        "prompt_id": "p-current",
        "prompt_text": "Current prompt",
        "env_id": "env-current",
    }
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(current_record) + "\n",
        encoding="utf-8",
    )

    stage_f_dir = paths.stage_dirs["F"]
    stage_f_dir.mkdir(parents=True, exist_ok=True)
    (stage_f_dir / "validated.jsonl").write_text(json.dumps(stale_record) + "\n")
    (stage_f_dir / "rejected.jsonl").write_text("")
    (stage_f_dir / "traces.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "c-stale:validation",
                "task_id": "env-stale",
                "trial_id": "c-stale",
                "steps": [],
                "artifacts": {"outcome": "validated"},
                "metadata": {"stage": "F", "runtime": "pymc"},
                "schema_version": "ppl.platform.record.v0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: ValidationRunResult(
            ok=True,
            final_code="def model(data):\n    return 2\n",
            message="ok",
            trace=(
                ValidationTraceEntry(
                    attempt=0,
                    phase="syntax",
                    ok=True,
                    message="ok",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.score_pymc_semantics",
        _semantic_not_scored,
    )

    class DummyBackend:
        pass

    result = run_stage_f(
        paths,
        backend=DummyBackend(),
        judge_backend_name="deterministic",
        runtime=get_runtime("pymc"),
    )

    validated = [
        json.loads(line)
        for line in (stage_f_dir / "validated.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = [
        json.loads(line)
        for line in (stage_f_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    progress = json.loads((stage_f_dir / "progress.json").read_text(encoding="utf-8"))

    assert result.status == "success"
    assert [row["id"] for row in validated] == ["c-current"]
    assert [row["trace_id"] for row in traces] == ["c-current:validation"]
    assert progress["count_total"] == 1
    assert progress["count_processed"] == 1


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_rejects_prompt_leakage_without_running_repair(monkeypatch, tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_leakage")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-leak",
                "prompt_id": "p-leak",
                "prompt_text": "Use a Poisson likelihood for observed counts.",
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-leak",
                "split": "train",
                "provenance": {
                    "generative_family": "hierarchical_grouped",
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    called = {"repair": 0}

    def _unexpected_repair(*args, **kwargs):
        called["repair"] += 1
        raise AssertionError("repair should not run after leakage rejection")

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        _unexpected_repair,
    )

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.stages.stage_f_validate.build_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("backend should not be built after leakage rejection")
        ),
    )

    result = run_stage_f(
        paths,
        backend=None,
        judge_backend_name="deterministic",
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    assert called["repair"] == 0
    rejected_row = json.loads(
        (paths.stage_dirs["F"] / "rejected.jsonl").read_text(encoding="utf-8")
    )
    trace_row = json.loads((paths.stage_dirs["F"] / "traces.jsonl").read_text(encoding="utf-8"))
    assert rejected_row["reason"] == "prompt_leakage"
    assert rejected_row["evidence_kind"] == "heuristic"
    assert rejected_row["truth_owner"] == "heuristic"
    assert rejected_row["semantic_status"] == "critical"
    assert rejected_row["validation_trace"]["steps"][0]["phase"] == "leakage_judge"
    assert rejected_row["validation_trace"]["steps"][0]["evidence_kind"] == "heuristic"
    assert rejected_row["validation_trace"]["steps"][0]["truth_owner"] == "heuristic"
    assert rejected_row["validation_trace"]["steps"][0]["semantic_status"] == "critical"
    assert "evidence_kind" not in rejected_row["validation_trace"]["steps"][0]["metadata"]
    assert "truth_owner" not in rejected_row["validation_trace"]["steps"][0]["metadata"]
    assert "semantic_status" not in rejected_row["validation_trace"]["steps"][0]["metadata"]
    assert trace_row["artifacts"]["outcome"] == "rejected"
    assert trace_row["artifacts"]["reason"] == "prompt_leakage"
    assert trace_row["steps"][0]["metadata"]["evidence_kind"] == "heuristic"
    assert trace_row["steps"][0]["metadata"]["truth_owner"] == "heuristic"
    assert trace_row["steps"][0]["metadata"]["semantic_status"] == "critical"


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_records_semantic_signal_for_pymc(monkeypatch, tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_semantic_signal")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = data_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]}).to_parquet(
        data_dir / "env-semantic.parquet",
        index=False,
    )
    pd.DataFrame({"y": [3.0], "x": [2.0]}).to_parquet(
        hidden_dir / "env-semantic.parquet",
        index=False,
    )
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-semantic",
                "prompt_id": "p-semantic",
                "prompt_text": "Prompt",
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-semantic",
                "split": "train",
                "provenance": {
                    "generative_family": "linear_regression",
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.judge_leakage",
        lambda *args, **kwargs: {
            "rating": "CLEAN",
            "evidence": "",
            "recovered_family": None,
            "leaked": False,
        },
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: ValidationRunResult(
            ok=True,
            final_code="def model(data):\n    return 2\n",
            message="repaired",
            trace=(
                ValidationTraceEntry(
                    attempt=0,
                    phase="syntax",
                    ok=True,
                    message="ok",
                ),
                ValidationTraceEntry(
                    attempt=0,
                    phase="runtime",
                    ok=True,
                    message="ok",
                ),
            ),
        ),
    )
    semantic_call: dict[str, object] = {}

    def fake_score_pymc_semantics(*args, **kwargs):
        semantic_call["kwargs"] = kwargs
        return SemanticValidationResult(
            ok=True,
            status=SemanticStatus.pass_,
            message="semantic validation passed",
            metric_name="avg_holdout_logp",
            metric_value=-0.123,
            details={"pareto_k_max": 0.2, "hidden_n_obs": 1},
        )

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.score_pymc_semantics",
        fake_score_pymc_semantics,
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.stages.stage_f_validate.build_backend",
        lambda *args, **kwargs: object(),
    )

    class DummyBackend:
        pass

    result = run_stage_f(
        paths,
        backend=DummyBackend(),
        judge_backend_name="openai",
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    row = json.loads((paths.stage_dirs["F"] / "validated.jsonl").read_text(encoding="utf-8"))
    trace_row = json.loads((paths.stage_dirs["F"] / "traces.jsonl").read_text(encoding="utf-8"))
    assert row["validation"]["semantic_status"] == "pass"
    assert row["validation"]["semantic_metric"] == "avg_holdout_logp"
    assert row["validation"]["semantic_score"] == -0.123
    assert row["validation"]["semantic_settings"] == {
        "pymc_draws": 100,
        "pymc_tune": 100,
        "pymc_timeout_sec": 180,
    }
    assert row["validation"]["semantic_details"] == {
        "pareto_k_max": 0.2,
        "hidden_n_obs": 1,
    }
    assert semantic_call["kwargs"]["draws"] == 100
    assert semantic_call["kwargs"]["tune"] == 100
    assert semantic_call["kwargs"]["timeout_sec"] == 180
    assert row["validation"]["trace"]["steps"][-1]["phase"] == "semantic_validation"
    assert row["validation"]["trace"]["steps"][-1]["evidence_kind"] == "verifier"
    assert row["validation"]["trace"]["steps"][-1]["truth_owner"] == "verifier"
    assert row["validation"]["trace"]["steps"][-1]["semantic_status"] == "pass"
    assert "evidence_kind" not in row["validation"]["trace"]["steps"][-1]["metadata"]
    assert "truth_owner" not in row["validation"]["trace"]["steps"][-1]["metadata"]
    assert "semantic_status" not in row["validation"]["trace"]["steps"][-1]["metadata"]
    assert trace_row["steps"][-1]["metadata"]["evidence_kind"] == "verifier"
    assert trace_row["steps"][-1]["metadata"]["truth_owner"] == "verifier"
    assert trace_row["steps"][-1]["metadata"]["semantic_status"] == "pass"
    assert trace_row["steps"][-1]["metadata"]["phase"] == "semantic_validation"
    assert trace_row["steps"][-1]["metadata"]["semantic_settings"] == {
        "pymc_draws": 100,
        "pymc_tune": 100,
        "pymc_timeout_sec": 180,
    }
    manifest = json.loads((paths.stage_dirs["F"] / "manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((paths.stage_dirs["F"] / "progress.json").read_text(encoding="utf-8"))
    assert manifest["semantic_validation"]["settings"] == {
        "pymc_draws": 100,
        "pymc_tune": 100,
        "pymc_timeout_sec": 180,
    }
    assert progress["semantic_validation"] == manifest["semantic_validation"]


def test_stage_f_acceptance_semantic_budget_fails_closed(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_acceptance_budget")

    with pytest.raises(ValueError, match="requires at least 200 semantic draws"):
        run_stage_f(
            paths,
            runtime=get_runtime("pymc"),
            pymc_semantic_draws=100,
            acceptance_min_pymc_semantic_draws=200,
        )


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_real_pymc_semantic_uses_hidden_holdout_data(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_real_pymc_holdout")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = data_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}).to_parquet(
        data_dir / "env-real-semantic.parquet",
        index=False,
    )
    pd.DataFrame({"y": [4.0, 5.0], "x": [3.0, 4.0]}).to_parquet(
        hidden_dir / "env-real-semantic.parquet",
        index=False,
    )
    code = """
def model(data):
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    with pm.Model() as m:
        intercept = pm.Normal("intercept", 0, 5)
        slope = pm.Normal("slope", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        mu = intercept + slope * x
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
""".strip()
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-real-semantic",
                "prompt_id": "p-real-semantic",
                "prompt_text": "Fit observed readings y from covariate x.",
                "raw_completion": f"<think>ok</think>\n```python\n{code}\n```",
                "env_id": "env-real-semantic",
                "split": "train",
                "provenance": {
                    "generative_family": "continuous_relationship",
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class NoRepairBackend:
        def complete(self, request: LLMRequest) -> LLMResult:
            raise AssertionError("valid PyMC completion should not request repair")

    result = run_stage_f(
        paths,
        backend=NoRepairBackend(),
        judge_backend_name="deterministic",
        runtime=get_runtime("pymc"),
    )

    row = json.loads((paths.stage_dirs["F"] / "validated.jsonl").read_text(encoding="utf-8"))
    trace_row = json.loads((paths.stage_dirs["F"] / "traces.jsonl").read_text(encoding="utf-8"))
    semantic_step = row["validation"]["trace"]["steps"][-1]

    assert result.status == "success"
    assert row["validation"]["ok"] is True
    assert row["validation"]["semantic_metric"] == "avg_holdout_logp"
    assert isinstance(row["validation"]["semantic_score"], float)
    assert row["validation"]["semantic_details"]["hidden_data_present"] is True
    assert row["validation"]["semantic_details"]["hidden_n_obs"] == 2
    assert "hidden_data_error" not in row["validation"]["semantic_details"]
    assert semantic_step["phase"] == "semantic_validation"
    assert semantic_step["ok"] is True
    assert semantic_step["evidence_kind"] == "verifier"
    assert semantic_step["truth_owner"] == "verifier"
    assert semantic_step["metadata"]["metric_name"] == "avg_holdout_logp"
    assert trace_row["steps"][-1]["metadata"]["evidence_kind"] == "verifier"
    assert trace_row["steps"][-1]["metadata"]["truth_owner"] == "verifier"


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_pymc_without_hidden_data_is_not_holdout_evidence(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_pymc_without_holdout")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}).to_parquet(
        data_dir / "env-no-holdout.parquet",
        index=False,
    )
    code = """
def model(data):
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    with pm.Model() as m:
        intercept = pm.Normal("intercept", 0, 5)
        slope = pm.Normal("slope", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        mu = intercept + slope * x
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
""".strip()
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-no-holdout",
                "prompt_id": "p-no-holdout",
                "prompt_text": "Fit observed readings y from covariate x.",
                "raw_completion": f"<think>ok</think>\n```python\n{code}\n```",
                "env_id": "env-no-holdout",
                "split": "train",
                "provenance": {
                    "generative_family": "continuous_relationship",
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class NoRepairBackend:
        def complete(self, request: LLMRequest) -> LLMResult:
            raise AssertionError("valid PyMC completion should not request repair")

    result = run_stage_f(
        paths,
        backend=NoRepairBackend(),
        judge_backend_name="deterministic",
        runtime=get_runtime("pymc"),
    )

    row = json.loads((paths.stage_dirs["F"] / "validated.jsonl").read_text(encoding="utf-8"))
    semantic_details = row["validation"]["semantic_details"]

    assert result.status == "success"
    assert row["validation"]["semantic_status"] == "flag"
    assert row["validation"]["semantic_metric"] == "train_elpd_loo"
    assert row["validation"]["semantic_metric"] != "avg_holdout_logp"
    assert "avg_holdout_logp" not in semantic_details
    assert semantic_details["hidden_data_present"] is False
    assert semantic_details["hidden_n_obs"] == 0
    assert semantic_details["hidden_data_error"] == (
        "missing Stage C hidden data parquet for env_id=env-no-holdout"
    )


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_lazyppl_semantic_routes_through_real_scorer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Narrow unit test: exercises OUR TaskData mounting/parsing contract
    against a hand-built RewardArtifact payload shape, by mocking the
    containerized Harbor scorer subprocess boundary
    (``run_with_process_group_timeout``). The real Docker-backed boundary is
    covered for real by ``test_stage_f_lazyppl_semantic_real_scorer``
    (heavy_lane, real_boundary), which runs this same routing path with no
    mocks against the actual ``lazyppl-harbor`` image."""
    monkeypatch.setenv("LAZYPPL_HARBOR_IMAGE", "lazyppl-harbor:test")
    paths = resolve_run_paths(tmp_path, "run_stage_f_lazyppl_semantic_route")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = data_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"xVal": [1.25, 2.5]}).to_parquet(
        data_dir / "env-lazyppl-semantic.parquet",
        index=False,
    )
    pd.DataFrame({"xVal": [3.75]}).to_parquet(
        hidden_dir / "env-lazyppl-semantic.parquet",
        index=False,
    )
    code = """
module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State Double deriving (Show)

model :: [Row] -> Meas State
model rows = do
  mu <- sample (normal 0.0 1.0)
  let state = State mu
  mapM_ (\\row -> score (exp (logLikRow state row))) rows
  return state

logLikRow :: State -> Row -> Double
logLikRow (State mu) row = normalLogPdf mu 1.0 (xVal row)
""".strip()
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-lazyppl-semantic",
                "prompt_id": "p-lazyppl-semantic",
                "prompt_text": "Fit values between zero and one.",
                "raw_completion": f"<think>ok</think>\n```haskell\n{code}\n```",
                "env_id": "env-lazyppl-semantic",
                "split": "train",
                "provenance": {
                    "generative_family": "continuous_relationship",
                    "runtime_name": "lazyppl",
                    "runtime_version": "lazyppl-1.0",
                },
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: ValidationRunResult(
            ok=True,
            final_code=code,
            message="ok",
            trace=(
                ValidationTraceEntry(
                    attempt=0,
                    phase="runtime",
                    ok=True,
                    message="ok",
                ),
            ),
        ),
    )

    semantic_commands: list[list[str]] = []

    def fake_harbor_run(command, **kwargs):
        semantic_commands.append(command)
        mounts = [item for item in command if item.endswith(":ro")]
        mounted_paths = [Path(item.split(":", 1)[0]) for item in mounts]
        taskdata_path = next(path for path in mounted_paths if path.name == "TaskData.hs")
        hidden_taskdata_path = next(
            path for path in mounted_paths if path.name == "HiddenTaskData.hs"
        )
        taskdata = taskdata_path.read_text(encoding="utf-8")
        hidden_taskdata = hidden_taskdata_path.read_text(encoding="utf-8")
        assert kwargs["timeout"] == lazyppl_semantics._DEFAULT_TIMEOUT_SEC
        assert "xVal = 1.25" in taskdata
        assert "xVal = 2.5" in taskdata
        assert "xVal = 3.75" in hidden_taskdata
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "schema_version": "synthstats.harbor.reward.v1",
                    "reward_source": "validation.lazyppl_elpd",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "producer": "ppl_synth_data.docker.lazyppl_harbor.score",
                    "training_log_reward": 0.5,
                    "reward_transform": {"kind": "baseline_normalized_log_reward"},
                    "metrics": {
                        "avg_holdout_logp": -1.25,
                        "elpd_loo": -12.3,
                        "pareto_k_max": 0.4,
                        "compile_ok": 1.0,
                        "n_draws": 2200,
                        "n_obs": 2,
                    },
                    "diagnostics": {"pareto_k_warning": False},
                    "provenance": {"runtime": "lazyppl", "scorer": "arviz_loo"},
                    "reward": 1.648721,
                }
            ),
        )

    monkeypatch.setattr(
        lazyppl_semantics,
        "run_with_process_group_timeout",
        fake_harbor_run,
    )

    class NoRepairBackend:
        def complete(self, request: LLMRequest) -> LLMResult:
            raise AssertionError("valid LazyPPL completion should not request repair")

    result = run_stage_f(
        paths,
        backend=NoRepairBackend(),
        judge_backend_name="deterministic",
        runtime=get_runtime("lazyppl"),
    )

    row = json.loads((paths.stage_dirs["F"] / "validated.jsonl").read_text(encoding="utf-8"))
    semantic_step = row["validation"]["trace"]["steps"][-1]

    assert result.status == "success"
    assert len(semantic_commands) == 1
    assert row["validation"]["semantic_metric"] == "avg_holdout_logp"
    assert row["validation"]["semantic_score"] == -1.25
    assert row["validation"]["semantic_details"]["reward_source"] == "validation.lazyppl_elpd"
    assert semantic_step["phase"] == "semantic_validation"
    assert semantic_step["evidence_kind"] == "verifier"
    assert semantic_step["truth_owner"] == "verifier"
    assert semantic_step["metadata"]["metric_name"] == "avg_holdout_logp"


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_stage_f_lazyppl_semantic_real_scorer(tmp_path: Path) -> None:
    """Real-Docker twin of ``test_stage_f_lazyppl_semantic_routes_through_real_scorer`` in the
    Docker runtime.

    No monkeypatch of ``run_with_process_group_timeout`` (nor of ``validate_and_repair``): the
    whole Stage F validation pipeline runs for real against the real ``lazyppl-harbor`` Docker
    image, producing a real contract-valid ``RewardArtifact`` with a real computed
    ``avg_holdout_logp``. Hard-asserts Docker is present and raises (never skips) if it is not.
    """
    import shutil

    if not shutil.which("docker"):
        raise RuntimeError(
            "heavy_lane test requires a real docker binary on PATH and the built "
            "lazyppl-harbor image; run this suite on the AWS heavy-lane instance"
        )

    paths = resolve_run_paths(tmp_path, "run_stage_f_lazyppl_semantic_real")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = data_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"xVal": [1.25, 2.5, -0.5, 0.75]}).to_parquet(
        data_dir / "env-lazyppl-semantic-real.parquet",
        index=False,
    )
    pd.DataFrame({"xVal": [3.75, -1.0]}).to_parquet(
        hidden_dir / "env-lazyppl-semantic-real.parquet",
        index=False,
    )
    code = """
module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State Double deriving (Show)

model :: [Row] -> Meas State
model rows = do
  mu <- sample (normal 0.0 1.0)
  let state = State mu
  mapM_ (\\row -> score (exp (logLikRow state row))) rows
  return state

logLikRow :: State -> Row -> Double
logLikRow (State mu) row = normalLogPdf mu 1.0 (xVal row)
""".strip()
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-lazyppl-semantic-real",
                "prompt_id": "p-lazyppl-semantic-real",
                "prompt_text": "Fit values between zero and one.",
                "raw_completion": f"<think>ok</think>\n```haskell\n{code}\n```",
                "env_id": "env-lazyppl-semantic-real",
                "split": "train",
                "provenance": {
                    "generative_family": "continuous_relationship",
                    "runtime_name": "lazyppl",
                    "runtime_version": "lazyppl-1.0",
                },
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class NoRepairBackend:
        def complete(self, request: LLMRequest) -> LLMResult:
            raise AssertionError("valid LazyPPL completion should not request repair")

    result = run_stage_f(
        paths,
        backend=NoRepairBackend(),
        judge_backend_name="deterministic",
        runtime=get_runtime("lazyppl"),
    )

    row = json.loads((paths.stage_dirs["F"] / "validated.jsonl").read_text(encoding="utf-8"))
    semantic_step = row["validation"]["trace"]["steps"][-1]

    assert result.status == "success", row["validation"]
    assert row["validation"]["semantic_metric"] == "avg_holdout_logp"
    assert isinstance(row["validation"]["semantic_score"], float)
    assert row["validation"]["semantic_details"]["reward_source"] == "validation.lazyppl_elpd"
    assert semantic_step["phase"] == "semantic_validation"
    assert semantic_step["evidence_kind"] == "verifier"
    assert semantic_step["truth_owner"] == "verifier"


@pytest.mark.narrow_unit_not_acceptance
def test_stage_f_rejects_critical_semantic_validation(monkeypatch, tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_semantic_reject")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    data_dir = paths.stage_dirs["C"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]}).to_parquet(
        data_dir / "env-semantic-reject.parquet",
        index=False,
    )
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-semantic-reject",
                "prompt_id": "p-semantic-reject",
                "prompt_text": "Prompt",
                "raw_completion": (
                    "<think>x</think>\n```python\ndef model(data):\n    return 1\n```"
                ),
                "env_id": "env-semantic-reject",
                "split": "train",
                "provenance": {
                    "generative_family": "linear_regression",
                    "runtime_name": "pymc",
                    "runtime_version": "v5",
                },
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.judge_leakage",
        lambda *args, **kwargs: {
            "rating": "CLEAN",
            "evidence": "",
            "recovered_family": None,
            "leaked": False,
        },
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.stages.stage_f_validate.build_backend",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.validate_and_repair",
        lambda *args, **kwargs: ValidationRunResult(
            ok=True,
            final_code="def model(data):\n    return 2\n",
            message="repaired",
            trace=(
                ValidationTraceEntry(
                    attempt=0,
                    phase="syntax",
                    ok=True,
                    message="ok",
                ),
                ValidationTraceEntry(
                    attempt=0,
                    phase="runtime",
                    ok=True,
                    message="ok",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "ppl_synth_data.pipeline.validation.record_validation.score_pymc_semantics",
        lambda *args, **kwargs: SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message="posterior sampling failed",
            metric_name="train_elpd_loo",
            metric_value=-3.0,
            details={"pareto_k_max": 1.1},
        ),
    )

    class DummyBackend:
        pass

    result = run_stage_f(
        paths,
        backend=DummyBackend(),
        judge_backend_name="openai",
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    rejected_row = json.loads(
        (paths.stage_dirs["F"] / "rejected.jsonl").read_text(encoding="utf-8")
    )
    trace_row = json.loads((paths.stage_dirs["F"] / "traces.jsonl").read_text(encoding="utf-8"))
    assert rejected_row["reason"] == "semantic_validation"
    assert rejected_row["evidence_kind"] == "verifier"
    assert rejected_row["truth_owner"] == "verifier"
    assert rejected_row["semantic_status"] == "critical"
    assert rejected_row["semantic_metric"] == "train_elpd_loo"
    assert rejected_row["semantic_score"] == -3.0
    assert rejected_row["semantic_details"] == {"pareto_k_max": 1.1}
    assert rejected_row["parsed_program"] == "def model(data):\n    return 2\n"
    assert rejected_row["validation_trace"]["steps"][-1]["phase"] == "semantic_validation"
    assert rejected_row["validation_trace"]["steps"][-1]["evidence_kind"] == "verifier"
    assert rejected_row["validation_trace"]["steps"][-1]["truth_owner"] == "verifier"
    assert rejected_row["validation_trace"]["steps"][-1]["semantic_status"] == "critical"
    assert "evidence_kind" not in rejected_row["validation_trace"]["steps"][-1]["metadata"]
    assert "truth_owner" not in rejected_row["validation_trace"]["steps"][-1]["metadata"]
    assert "semantic_status" not in rejected_row["validation_trace"]["steps"][-1]["metadata"]
    assert trace_row["artifacts"]["reason"] == "semantic_validation"
    assert trace_row["steps"][-1]["metadata"]["evidence_kind"] == "verifier"
    assert trace_row["steps"][-1]["metadata"]["truth_owner"] == "verifier"
    assert trace_row["steps"][-1]["metadata"]["semantic_status"] == "critical"


def test_stage_f_rejects_completion_runtime_mismatch(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_f_runtime_mismatch")
    paths.stage_dirs["E"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["E"] / "completions.jsonl").write_text(
        json.dumps(
            {
                "id": "c-mismatch",
                "prompt_id": "p-mismatch",
                "prompt_text": "Prompt",
                "raw_completion": "x",
                "env_id": "env-mismatch",
                "split": "train",
                "provenance": {"runtime_name": "lazyppl", "runtime_version": "lazyppl-1.0"},
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        run_stage_f(paths, backend_name="openai", runtime=get_runtime("pymc"))
    except ValueError as exc:
        assert "runtime 'lazyppl'" in str(exc) or 'runtime "lazyppl"' in str(exc)
    else:
        raise AssertionError("expected runtime mismatch rejection")


@pytest.mark.narrow_unit_not_acceptance
def test_stage_g_lazyppl_produces_lazyppl_prefixed_files(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_lazyppl_g")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="env-lz", env_name="env_lz")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(spec) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps({"env_id": "env-lz", "data_ref": "ref"}) + "\n",
        encoding="utf-8",
    )
    validated = {
        "id": "c-lz",
        "prompt_id": "p-lz",
        "prompt_text": "Prompt",
        "raw_completion": "<think>ok</think>\n```haskell\nmodel = undefined\n```",
        "parsed_program": "model = undefined",
        "env_id": "env-lz",
        "split": "train",
        "provenance": {
            "env_name": "env_lz",
            "domain": "healthcare",
            "complexity": "medium",
            "model_type": "predictive",
            "generative_family": "hierarchical_grouped",
            "key_concepts": ["hierarchical_model"],
            "has_observed_data": True,
            "runtime_name": "lazyppl",
            "runtime_version": "lazyppl-1.0",
        },
        "validation": _lazyppl_semantic_validation(),
        "metadata": {
            "env_id": "env-lz",
            "split": "train",
            "lazyppl_version": "lazyppl-1.0",
            "seed": 42,
        },
    }
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(validated) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")
    (paths.stage_dirs["F"] / "traces.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "ppl.platform.record.v0",
                "trace_id": "c-lz:validation",
                "task_id": "env-lz",
                "trial_id": "c-lz",
                "steps": [
                    {
                        "step_id": "step-0000",
                        "role": "validator",
                        "content": "ok",
                        "metadata": {"attempt": 0, "phase": "runtime", "ok": True},
                    }
                ],
                "artifacts": {
                    "outcome": "validated",
                    "prompt_id": "p-lz",
                    "split": "train",
                },
                "metadata": {
                    "stage": "F",
                    "runtime": "lazyppl",
                    "env_id": "env-lz",
                    "split": "train",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runtime = get_runtime("lazyppl")
    result = run_stage_g(
        paths,
        repo_id=None,
        hf_token=None,
        skip_upload=True,
        runtime=runtime,
    )

    assert result.status == "success"
    g_dir = paths.stage_dirs["G"]
    assert (g_dir / "lazyppl_sft.train.jsonl").exists()
    assert (g_dir / "lazyppl_with_data.train.jsonl").exists()
    assert not (g_dir / "pymc_sft.train.jsonl").exists()
    assert not (g_dir / "pymc_with_data.train.jsonl").exists()

    manifest = json.loads((g_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count_written"] == 1
    assert manifest["count_rows_with_trace"] == 1
    assert manifest["runtime"] == "lazyppl"
    assert "sft_counts" in manifest
    assert "with_data_counts" in manifest

    row = json.loads((g_dir / "lazyppl_sft.train.jsonl").read_text(encoding="utf-8").strip())
    assert "lazyppl_version" in row
    assert "pymc_version" not in row
    assert row["env_id"] == "env-lz"
    assert row["split"] == "train"
    assert row["prompt"] == "Prompt"
    assert row["completion"] == "<think>ok</think>\n```haskell\nmodel = undefined\n```"
    assert row["model_type"] == "predictive"
    assert row["domain"] == "healthcare"
    assert row["complexity"] == "medium"
    assert row["key_concepts"] == ["hierarchical_model"]
    assert row["has_observed_data"] is True
    assert row["trace_id"] == "c-lz:validation"
    assert json.loads(row["provenance_json"])["provenance"]["runtime_name"] == "lazyppl"
    with_data_row = json.loads(
        (g_dir / "lazyppl_with_data.train.jsonl").read_text(encoding="utf-8").strip()
    )
    assert with_data_row["data_ref"] == "ref"
    assert with_data_row["parsed_program"] == "model = undefined"
    assert with_data_row["seed"] == 42
    assert with_data_row["env_name"] == "env_lz"
    assert with_data_row["trace_id"] == "c-lz:validation"
    assert json.loads(with_data_row["trace_json"])["trial_id"] == "c-lz"
    assert json.loads(with_data_row["validation_json"])["ok"] is True


@pytest.mark.narrow_unit_not_acceptance
def test_stage_g_rejects_stale_lazyppl_semantic_evidence(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_lazyppl_g_stale")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="env-lz", env_name="env_lz")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(spec) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps({"env_id": "env-lz", "data_ref": "ref"}) + "\n",
        encoding="utf-8",
    )
    stale = {
        "id": "c-lz",
        "prompt_id": "p-lz",
        "prompt_text": "Prompt",
        "raw_completion": "<think>ok</think>\n```haskell\nmodel = undefined\n```",
        "parsed_program": "model = undefined",
        "env_id": "env-lz",
        "split": "train",
        "provenance": {
            "runtime_name": "lazyppl",
            "runtime_version": "lazyppl-1.0",
        },
        "validation": {
            "ok": True,
            "message": "old typecheck-only validation",
            "repaired": False,
            "leakage_rating": "CLEAN",
        },
        "metadata": {"seed": 42},
    }
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(stale) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")
    (paths.stage_dirs["F"] / "traces.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dropped every validated completion"):
        run_stage_g(
            paths,
            repo_id=None,
            hf_token=None,
            skip_upload=True,
            runtime=get_runtime("lazyppl"),
        )

    dropped = [
        json.loads(line)
        for line in (paths.stage_dirs["G"] / "dropped.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert dropped == [
        {
            "id": "c-lz",
            "prompt_id": "p-lz",
            "reason": "missing_lazyppl_semantic_evidence",
            "env_id": "env-lz",
            "detail": "LazyPPL semantic status is None",
        }
    ]


@pytest.mark.narrow_unit_not_acceptance
def test_stage_g_partial_provenance_falls_back_to_metadata(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_g_partial_provenance")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="env-partial", env_name="env_partial")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(spec) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps({"env_id": "env-partial", "data_ref": "ref"}) + "\n",
        encoding="utf-8",
    )
    validated = {
        "id": "c-partial",
        "prompt_id": "p-partial",
        "prompt_text": "Prompt",
        "raw_completion": "<think>ok</think>\n```python\ndef model(data):\n    return 1\n```",
        "parsed_program": "def model(data):\n    return 1\n",
        "env_id": "env-partial",
        "split": "train",
        "provenance": {"runtime_name": "pymc"},
        "metadata": {
            "env_id": "env-partial",
            "split": "train",
            "key_concepts": ["legacy-kc"],
            "has_observed_data": False,
            "source_category": "pymc_examples",
            "source_url_or_repo": "https://github.com/pymc-devs/pymc-examples",
            "license_signal": "permissive",
            "adaptation_mode": "source-inspired",
            "source_retrieval_date": "2026-05-16",
            "pymc_version": "v5",
            "seed": 3,
        },
    }
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(validated) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")

    result = run_stage_g(
        paths,
        repo_id=None,
        hf_token=None,
        skip_upload=True,
        runtime=get_runtime("pymc"),
    )

    assert result.status == "success"
    row = json.loads(
        (paths.stage_dirs["G"] / "pymc_sft.train.jsonl").read_text(encoding="utf-8").strip()
    )
    with_data_row = json.loads(
        (paths.stage_dirs["G"] / "pymc_with_data.train.jsonl").read_text(encoding="utf-8").strip()
    )
    assert row["key_concepts"] == ["legacy-kc"]
    assert row["has_observed_data"] is False
    assert with_data_row["parsed_program"] == "def model(data):\n    return 1\n"
    provenance = json.loads(row["provenance_json"])["provenance"]
    assert provenance["source_category"] == "pymc_examples"
    assert provenance["source_url_or_repo"] == "https://github.com/pymc-devs/pymc-examples"
    assert provenance["license_signal"] == "permissive"
    assert provenance["adaptation_mode"] == "source-inspired"
    assert provenance["source_retrieval_date"] == "2026-05-16"


def test_stage_g_drops_extra_run_records_with_runtime_mismatch(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_g_primary")
    extra_paths = resolve_run_paths(tmp_path, "run_stage_g_extra")
    for run_paths in (paths, extra_paths):
        run_paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
        run_paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
        run_paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    primary_spec = minimal_envspec_record(env_id="env-primary", env_name="env_primary")
    extra_spec = minimal_envspec_record(env_id="env-extra", env_name="env_extra")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(primary_spec) + "\n",
        encoding="utf-8",
    )
    (extra_paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(extra_spec) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps({"env_id": "env-primary", "data_ref": "primary-ref"}) + "\n",
        encoding="utf-8",
    )
    (extra_paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps({"env_id": "env-extra", "data_ref": "extra-ref"}) + "\n",
        encoding="utf-8",
    )
    primary_validated = {
        "id": "c-primary",
        "prompt_id": "p-primary",
        "prompt_text": "Prompt",
        "raw_completion": "<think>ok</think>\n```python\ndef model(data):\n    return 1\n```",
        "parsed_program": "def model(data):\n    return 1\n",
        "env_id": "env-primary",
        "split": "train",
        "provenance": {"runtime_name": "pymc", "runtime_version": "v5"},
        "metadata": {"env_id": "env-primary", "split": "train", "pymc_version": "v5"},
    }
    extra_validated = {
        "id": "c-extra",
        "prompt_id": "p-extra",
        "prompt_text": "Prompt",
        "raw_completion": "<think>ok</think>\n```haskell\nmodel = undefined\n```",
        "parsed_program": "model = undefined",
        "env_id": "env-extra",
        "split": "train",
        "provenance": {"runtime_name": "lazyppl", "runtime_version": "lazyppl-1.0"},
        "metadata": {"env_id": "env-extra", "split": "train", "lazyppl_version": "lazyppl-1.0"},
    }
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(primary_validated) + "\n",
        encoding="utf-8",
    )
    (extra_paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(extra_validated) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")
    (extra_paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")

    result = run_stage_g(
        paths,
        repo_id=None,
        hf_token=None,
        skip_upload=True,
        runtime=get_runtime("pymc"),
        extra_run_dirs=[extra_paths],
    )

    assert result.status == "success"
    manifest = json.loads((paths.stage_dirs["G"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count_written"] == 1
    assert manifest["count_dropped"] == 1
    dropped_rows = (paths.stage_dirs["G"] / "dropped.jsonl").read_text(encoding="utf-8")
    assert "runtime_mismatch" in dropped_rows


def test_stage_g_rejects_records_with_unknown_runtime(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_stage_g_unknown_runtime")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    paths.stage_dirs["F"].mkdir(parents=True, exist_ok=True)

    spec = minimal_envspec_record(env_id="env-unknown", env_name="env_unknown")
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text(
        json.dumps(spec) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text(
        json.dumps({"env_id": "env-unknown", "data_ref": "unknown-ref"}) + "\n",
        encoding="utf-8",
    )
    validated = {
        "id": "c-unknown",
        "prompt_id": "p-unknown",
        "prompt_text": "Prompt",
        "raw_completion": "haskell-ish",
        "parsed_program": None,
        "env_id": "env-unknown",
        "split": "train",
        "metadata": {"env_id": "env-unknown", "split": "train"},
    }
    (paths.stage_dirs["F"] / "validated.jsonl").write_text(
        json.dumps(validated) + "\n",
        encoding="utf-8",
    )
    (paths.stage_dirs["F"] / "rejected.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dropped every validated completion"):
        run_stage_g(
            paths,
            repo_id=None,
            hf_token=None,
            skip_upload=True,
            runtime=get_runtime("pymc"),
        )

    dropped_rows = (paths.stage_dirs["G"] / "dropped.jsonl").read_text(encoding="utf-8")
    assert "runtime_unknown" in dropped_rows

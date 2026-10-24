from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from synthstats_schemas import ATIFRecord, RewardArtifact

from ppl_synth_data.config import Settings
from ppl_synth_data.dataset import rich_reward_export as rich_reward_module
from ppl_synth_data.dataset.rich_reward_export import (
    compose_live_rich_reward,
    export_rich_reward_atif_dir,
    score_atif_rubrics_with_openai,
)
from ppl_synth_data.llm.backend import BatchHandle, LLMResult


def _write_strict_atif_trial(
    root: Path,
    *,
    atif_id: str = "trial-1",
    reward_source: str = "validation.lazyppl_elpd",
    evidence_kind: str = "verifier",
    truth_owner: str = "verifier",
    producer: str = "ppl_synth_data.docker.lazyppl_harbor.score",
    training_log_reward: float = -0.25,
) -> Path:
    trial_dir = root / atif_id
    trial_dir.mkdir(parents=True)
    reward = {
        "schema_version": "synthstats.harbor.reward.v1",
        "reward_source": reward_source,
        "evidence_kind": evidence_kind,
        "truth_owner": truth_owner,
        "producer": producer,
        "training_log_reward": training_log_reward,
        "reward_transform": {
            "kind": "baseline_normalized_log_reward",
            "lower_log_reward_bound": -20.0,
        },
        "metrics": {"holdout_elpd": -12.5, "compile_ok": 1.0},
        "diagnostics": {},
        "provenance": {"runtime": "lazyppl"},
        "reward": math.exp(min(training_log_reward, 20.0)),
        "components": {"holdout_elpd": -12.5},
        "reward_components": [
            {
                "name": "holdout_elpd",
                "kind": "terminal_verifier",
                "value": -12.5,
                "source": reward_source,
                "evidence_kind": evidence_kind,
                "truth_owner": truth_owner,
                "producer": producer,
                "weight": 1.0,
                "log_contribution": training_log_reward,
            }
        ],
        "scalarization": {
            "method": "weighted_sum",
            "component_names": ["holdout_elpd"],
            "weights": {"holdout_elpd": 1.0},
        },
    }
    trajectory = {
        "schema_version": "ATIF-v1.6",
        "session_id": atif_id,
        "metadata": {
            "dataset_row": {"id": atif_id, "env_id": "env-1"},
            "shared_generation_record": {"record_id": atif_id},
            "shared_trial_record": {"trial_id": atif_id},
            "shared_trace_record": {"trace_id": f"{atif_id}:validation"},
        },
        "steps": [
            {"step_id": "system", "source": "system", "message": "system"},
            {"step_id": "user", "source": "user", "message": "fit a model"},
            {
                "step_id": "agent",
                "source": "agent",
                "message": (
                    "<think>Which fields are visible? The schema gives only "
                    "the observed covariates, so I should not invent groups.</think>"
                    '<submit_program language="haskell">model = undefined</submit_program>'
                ),
                "tool_calls": [{"name": "inspect_schema"}],
                "observation": {"schema_checked": True},
            },
        ],
        "reward_artifact": reward,
    }
    (trial_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (trial_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    return trial_dir


def _write_rubric_scores(path: Path, *, atif_id: str = "trial-1") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ppl_synth_data.rich_reward_rubric_scores.v1",
                "rubric": rich_reward_module._RUBRIC.model_dump(mode="json"),
                "scores": {
                    atif_id: {
                        "criterion_scores": {
                            "data_contract_use": 0.8,
                            "likelihood_structure": 0.7,
                            "uncertainty_reasoning": 0.6,
                            "submission_hygiene": 0.9,
                        },
                        "rationale": "Uses the visible contract and submits structured code.",
                        "confidence": 0.75,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.narrow_unit_not_acceptance
def test_score_atif_rubrics_with_openai_uses_batch_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "atif"
    _write_strict_atif_trial(source, atif_id="trial-1")
    _write_strict_atif_trial(source, atif_id="trial-2")
    captured: dict[str, object] = {}

    class _BatchOnlyBackend:
        def __init__(self, **kwargs: object) -> None:
            captured["artifact_dir"] = kwargs["artifact_dir"]

        def complete(self, *_args: object, **_kwargs: object) -> LLMResult:
            raise AssertionError("rich reward scoring must use provider-native batch")

        def submit_batch(self, requests: list[tuple[str, object]]) -> BatchHandle:
            captured["requests"] = requests
            return BatchHandle(
                backend="openai",
                job_id="batch-rich-rubric",
                request_ids=tuple(request_id for request_id, _ in requests),
            )

        def collect_batch(
            self,
            handle: BatchHandle,
            *,
            allow_partial_failures: bool = False,
        ) -> dict[str, LLMResult]:
            assert handle.job_id == "batch-rich-rubric"
            assert allow_partial_failures is False
            return {
                request_id: LLMResult(
                    content=json.dumps(
                        {
                            "criterion_scores": {
                                "data_contract_use": 0.8,
                                "likelihood_structure": 0.7,
                                "uncertainty_reasoning": 0.6,
                                "submission_hygiene": 0.9,
                            },
                            "rationale": f"strict score for {request_id}",
                            "confidence": 0.75,
                        }
                    ),
                    raw={"request_id": request_id},
                )
                for request_id in handle.request_ids
            }

    monkeypatch.setattr(rich_reward_module, "OpenAIBackend", _BatchOnlyBackend)

    out_path = tmp_path / "scores" / "rubric_scores.json"
    payload = score_atif_rubrics_with_openai(
        source,
        model="gpt-5.5",
        out_path=out_path,
        limit=2,
    )

    requests = captured["requests"]
    assert isinstance(requests, list)
    assert [request_id for request_id, _ in requests] == [
        "rubric-000000",
        "rubric-000001",
    ]
    assert all(request.text_format["type"] == "json_schema" for _, request in requests)
    assert payload["scoring_backend"] == "openai_batch"
    assert payload["batch"] == {
        "backend": "openai",
        "job_id": "batch-rich-rubric",
        "request_count": 2,
    }
    assert set(payload["scores"]) == {"trial-1", "trial-2"}
    assert out_path.is_file()


@pytest.mark.paid_lane
def test_real_openai_rich_reward_batch_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-API twin of test_score_atif_rubrics_with_openai_uses_batch_backend.

    That test fakes OpenAIBackend entirely (submit_batch/collect_batch return canned
    LLMResults); no real HTTP call, batch-job polling, or JSON-schema acceptance is ever
    exercised there. This test drives the real score_atif_rubrics_with_openai entrypoint against
    the real OpenAI Batch API on one tiny strict-ATIF trial fixture (the ATIF payload is schema-
    real-but-synthetic via the existing _write_strict_atif_trial helper; the boundary under test
    is the PROVIDER contract, not the trial data).

    The only deviation from calling score_atif_rubrics_with_openai untouched is a bounded poll
    timeout: the Batch API's only completion_window is 24h
    (ppl_synth_data.llm.openai_batch.create_batch), so the production default poll would block
    up to 24h. The wrapper below still performs the real HTTP poll against the real batch; it
    only shortens how long this one test waits before treating the batch as still-pending,
    mirroring the same real-boundary choice documented in test_real_openai_batch_acceptance.
    """
    settings = Settings()
    settings.require_openai()  # hard assert: no silent skip on missing credentials

    source = tmp_path / "atif"
    _write_strict_atif_trial(source, atif_id="trial-real-1")

    import ppl_synth_data.llm.openai_backend as openai_backend_module

    real_poll = openai_backend_module.poll_batch_until_terminal
    real_backend_cls = rich_reward_module.OpenAIBackend
    captured_backends: list[object] = []

    def _bounded_poll(client, batch_id, poll_seconds=10.0, timeout_seconds=86400.0, on_poll=None):
        return real_poll(
            client, batch_id, poll_seconds=poll_seconds, timeout_seconds=300.0, on_poll=on_poll
        )

    def _capturing_backend(*args: object, **kwargs: object) -> object:
        backend = real_backend_cls(*args, **kwargs)
        captured_backends.append(backend)
        return backend

    monkeypatch.setattr(openai_backend_module, "poll_batch_until_terminal", _bounded_poll)
    monkeypatch.setattr(rich_reward_module, "OpenAIBackend", _capturing_backend)

    out_path = tmp_path / "scores" / "rubric_scores.json"
    try:
        payload = score_atif_rubrics_with_openai(
            source,
            model="gpt-5.4-mini",
            out_path=out_path,
            limit=1,
        )
    except TimeoutError:
        backend = captured_backends[0]
        snapshot = getattr(backend, "_last_batch_status_snapshot", None)
        assert snapshot is not None and snapshot.get("id"), (
            "expected a real batch status snapshot even on poll timeout"
        )
        assert snapshot.get("status") in {"validating", "in_progress", "finalizing"}
        receipt = {"outcome": "pending_real_boundary_confirmed", "status_snapshot": snapshot}
        (tmp_path / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
        print(json.dumps(receipt, indent=2, default=str))
        return

    assert payload["scoring_backend"] == "openai_batch"
    assert payload["batch"]["backend"] == "openai"
    assert payload["batch"]["job_id"]
    assert set(payload["scores"]) == {"trial-real-1"}
    assert out_path.is_file()
    receipt = {
        "outcome": "completed",
        "job_id": payload["batch"]["job_id"],
        "scores": payload["scores"],
    }
    (tmp_path / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps(receipt, indent=2, default=str))


@pytest.mark.narrow_unit_not_acceptance
def test_export_rich_reward_atif_dir_writes_strict_composite_rewards(
    tmp_path: Path,
) -> None:
    source = tmp_path / "atif"
    _write_strict_atif_trial(source)
    scores_path = tmp_path / "scores.json"
    _write_rubric_scores(scores_path)
    out = tmp_path / "rich"

    manifest = export_rich_reward_atif_dir(
        source,
        out,
        rubric_scores_path=scores_path,
        heuristic_weight=0.2,
        process_weight=0.05,
        min_exported=1,
    )

    assert manifest["count_exported"] == 1
    assert manifest["count_dropped"] == 0
    assert manifest["reward_source"] == "heuristic.composite_reward"
    reward = RewardArtifact.model_validate_json(
        (out / "trial-1" / "reward.json").read_text(encoding="utf-8")
    )
    atif = ATIFRecord.model_validate_json(
        (out / "trial-1" / "trajectory.json").read_text(encoding="utf-8")
    )
    assert reward.reward_source == "heuristic.composite_reward"
    assert reward.truth_owner == "heuristic"
    assert reward.training_log_reward > -0.25
    assert reward.rubric is not None
    assert reward.rubric.rubric_id == "synthstats_trace_quality"
    assert [component.name for component in reward.reward_components] == [
        "holdout_elpd",
        "llm_trace_rubric",
        "process_information_gain_proxy",
    ]
    assert reward.reward_components[0].truth_owner == "verifier"
    assert reward.reward_components[1].truth_owner == "heuristic"
    assert reward.reward_components[1].source == "heuristic.llm_rubric"
    assert reward.reward_components[2].truth_owner == "heuristic"
    assert reward.reward_components[2].kind == "process_information_gain"
    assert reward.reward_components[2].source == "heuristic.expected_information_gain_proxy"
    assert reward.scalarization is not None
    assert reward.scalarization.method == "verifier_gated_weighted_sum"
    assert reward.scalarization.component_names == (
        "holdout_elpd",
        "llm_trace_rubric",
        "process_information_gain_proxy",
    )
    assert set(manifest["component_reward_sources"]) == {
        "validation.lazyppl_elpd",
        "heuristic.llm_rubric",
        "heuristic.expected_information_gain_proxy",
    }
    assert atif.reward_artifact is not None
    assert atif.reward_artifact.reward_source == "heuristic.composite_reward"


@pytest.mark.narrow_unit_not_acceptance
def test_export_rich_reward_atif_dir_requires_rubric_scores(tmp_path: Path) -> None:
    source = tmp_path / "atif"
    _write_strict_atif_trial(source)
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "schema_version": "ppl_synth_data.rich_reward_rubric_scores.v1",
                "rubric": rich_reward_module._RUBRIC.model_dump(mode="json"),
                "scores": {},
            }
        ),
        encoding="utf-8",
    )

    try:
        export_rich_reward_atif_dir(
            source,
            tmp_path / "rich",
            rubric_scores_path=scores_path,
            min_exported=1,
        )
    except ValueError as exc:
        assert "below required minimum 1" in str(exc)
    else:
        raise AssertionError("expected missing rubric score to fail min_exported")


@pytest.mark.narrow_unit_not_acceptance
def test_export_rich_reward_atif_dir_rejects_stale_rubric_metadata(tmp_path: Path) -> None:
    source = tmp_path / "atif"
    _write_strict_atif_trial(source)
    scores_path = tmp_path / "scores.json"
    _write_rubric_scores(scores_path)
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    payload["rubric"]["version"] = "stale"
    scores_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="active rubric version"):
        export_rich_reward_atif_dir(
            source,
            tmp_path / "rich",
            rubric_scores_path=scores_path,
            min_exported=1,
        )


@pytest.mark.narrow_unit_not_acceptance
def test_export_rich_reward_atif_dir_rejects_heuristic_base_reward(tmp_path: Path) -> None:
    source = tmp_path / "atif"
    _write_strict_atif_trial(
        source,
        reward_source="heuristic.llm_rubric",
        evidence_kind="heuristic",
        truth_owner="heuristic",
        producer="ppl_synth_data.dataset.rich_reward_export",
    )
    scores_path = tmp_path / "scores.json"
    _write_rubric_scores(scores_path)

    with pytest.raises(ValueError, match="verifier-owned ELPD"):
        export_rich_reward_atif_dir(
            source,
            tmp_path / "rich",
            rubric_scores_path=scores_path,
            min_exported=1,
        )


@pytest.mark.narrow_unit_not_acceptance
def test_export_rich_reward_does_not_uplift_floor_clamped_base_reward(
    tmp_path: Path,
) -> None:
    source = tmp_path / "atif"
    _write_strict_atif_trial(source, training_log_reward=-20.0)
    scores_path = tmp_path / "scores.json"
    _write_rubric_scores(scores_path)
    out = tmp_path / "rich"

    export_rich_reward_atif_dir(
        source,
        out,
        rubric_scores_path=scores_path,
        heuristic_weight=0.2,
        process_weight=0.05,
        min_exported=1,
    )

    reward = RewardArtifact.model_validate_json(
        (out / "trial-1" / "reward.json").read_text(encoding="utf-8")
    )
    assert reward.training_log_reward == -20.0


@pytest.mark.narrow_unit_not_acceptance
def test_compose_live_rich_reward_updates_trial_reward_and_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    _write_strict_atif_trial(artifacts_dir, atif_id="trial-1")
    (artifacts_dir / "trial-1" / "reward.json").replace(artifacts_dir / "reward.json")
    (artifacts_dir / "trial-1" / "trajectory.json").replace(artifacts_dir / "trajectory.json")
    (artifacts_dir / "trial-1").rmdir()

    class _SingleBackend:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["artifact_dir"] == trial_dir / "rich_reward_openai_artifacts"

        def complete(self, request: object) -> LLMResult:
            assert request.text_format["type"] == "json_schema"
            return LLMResult(
                content=json.dumps(
                    {
                        "criterion_scores": {
                            "data_contract_use": 0.8,
                            "likelihood_structure": 0.7,
                            "uncertainty_reasoning": 0.6,
                            "submission_hygiene": 0.9,
                        },
                        "rationale": "live rubric score",
                        "confidence": 0.75,
                    }
                ),
                raw={"response_id": "resp-live-rubric"},
            )

    monkeypatch.setattr(rich_reward_module, "OpenAIBackend", _SingleBackend)

    manifest = compose_live_rich_reward(
        trial_dir=trial_dir,
        rubric_model="gpt-5.5",
        heuristic_weight=0.2,
        process_weight=0.05,
    )

    reward = RewardArtifact.model_validate_json(
        (artifacts_dir / "reward.json").read_text(encoding="utf-8")
    )
    atif = ATIFRecord.model_validate_json(
        (artifacts_dir / "trajectory.json").read_text(encoding="utf-8")
    )

    assert manifest["reward_source"] == "heuristic.composite_reward"
    assert manifest["component_names"] == [
        "holdout_elpd",
        "llm_trace_rubric",
        "process_information_gain_proxy",
    ]
    assert reward.reward_source == "heuristic.composite_reward"
    assert reward.provenance["rich_reward"]["rubric_model"] == "gpt-5.5"
    assert reward.provenance["rich_reward"]["rubric_backend"] == "openai_responses"
    assert atif.reward_artifact is not None
    assert atif.reward_artifact.reward_source == "heuristic.composite_reward"
    assert (trial_dir / "live_rich_reward_manifest.json").is_file()


@pytest.mark.narrow_unit_not_acceptance
def test_compose_live_rich_reward_uses_valid_agent_trajectory_when_artifact_copy_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    artifacts_dir = trial_dir / "artifacts"
    agent_dir = trial_dir / "agent"
    artifacts_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True)
    _write_strict_atif_trial(artifacts_dir, atif_id="trial-1")
    (artifacts_dir / "trial-1" / "reward.json").replace(artifacts_dir / "reward.json")
    valid_trajectory = (artifacts_dir / "trial-1" / "trajectory.json").read_text(encoding="utf-8")
    (agent_dir / "trajectory.json").write_text(valid_trajectory, encoding="utf-8")
    (artifacts_dir / "trajectory.json").write_text(
        valid_trajectory[:-20],
        encoding="utf-8",
    )
    (artifacts_dir / "trial-1" / "trajectory.json").unlink()
    (artifacts_dir / "trial-1").rmdir()

    class _SingleBackend:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["artifact_dir"] == trial_dir / "rich_reward_openai_artifacts"

        def complete(self, request: object) -> LLMResult:
            assert request.text_format["type"] == "json_schema"
            return LLMResult(
                content=json.dumps(
                    {
                        "criterion_scores": {
                            "data_contract_use": 0.8,
                            "likelihood_structure": 0.7,
                            "uncertainty_reasoning": 0.6,
                            "submission_hygiene": 0.9,
                        },
                        "rationale": "live rubric score",
                        "confidence": 0.75,
                    }
                ),
                raw={"response_id": "resp-live-rubric"},
            )

    monkeypatch.setattr(rich_reward_module, "OpenAIBackend", _SingleBackend)

    manifest = compose_live_rich_reward(
        trial_dir=trial_dir,
        rubric_model="gpt-5.5",
        heuristic_weight=0.2,
        process_weight=0.05,
    )

    assert manifest["base_trajectory_path"] == str(agent_dir / "trajectory.json")
    artifact_atif = ATIFRecord.model_validate_json(
        (artifacts_dir / "trajectory.json").read_text(encoding="utf-8")
    )
    agent_atif = ATIFRecord.model_validate_json(
        (agent_dir / "trajectory.json").read_text(encoding="utf-8")
    )
    assert artifact_atif.session_id == "trial-1"
    assert agent_atif.session_id == "trial-1"
    assert artifact_atif.reward_artifact is not None
    assert agent_atif.reward_artifact is not None

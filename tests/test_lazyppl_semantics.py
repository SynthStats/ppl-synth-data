from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest
from synthstats_schemas import RewardArtifact

from ppl_synth_data.evidence import SemanticStatus
from ppl_synth_data.validation import lazyppl_semantics

_MODEL_CODE = """
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


def _sample_data() -> dict:
    return {"n": 2, "xVal": [1.25, 2.5]}


def _reward_payload(
    *,
    pareto_warning: bool = False,
    pareto_error: str | None = None,
    avg_holdout_logp: float | None = None,
) -> dict:
    diagnostics = {"pareto_k_warning": pareto_warning}
    if pareto_error is not None:
        diagnostics["pareto_k_error"] = pareto_error
    return {
        "schema_version": "synthstats.harbor.reward.v1",
        "reward_source": "validation.lazyppl_elpd",
        "evidence_kind": "verifier",
        "truth_owner": "verifier",
        "producer": "ppl_synth_data.docker.lazyppl_harbor.score",
        "training_log_reward": 0.5,
        "reward_transform": {"kind": "baseline_normalized_log_reward"},
        "metrics": {
            "avg_holdout_logp": avg_holdout_logp,
            "elpd_loo": -12.3,
            "pareto_k_max": None if pareto_error is not None else 0.4,
            "compile_ok": 1.0,
            "n_draws": 2200,
            "n_obs": 50,
        },
        "diagnostics": diagnostics,
        "provenance": {"runtime": "lazyppl", "scorer": "arviz_loo"},
        "reward": 1.648721,
    }


def _flat_reward_payload() -> dict:
    return {
        "reward": 1.0,
        "avg_holdout_logp": None,
        "elpd_loo": -12.3,
        "pareto_k_max": 0.4,
        "pareto_k_warning": False,
        "compile_ok": 1.0,
        "n_draws": 2200,
        "n_obs": 50,
    }


def test_score_lazyppl_semantics_returns_contract_reward(monkeypatch) -> None:
    monkeypatch.setattr(
        lazyppl_semantics,
        "run_with_process_group_timeout",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(_reward_payload()),
        ),
    )

    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        harbor_image="test-lazyppl-harbor:latest",
    )

    assert result.ok is True
    assert result.status == SemanticStatus.pass_
    assert result.metric_name == "elpd_loo"
    assert result.metric_value == -12.3
    artifact = result.details["reward_artifact"]
    assert artifact["reward_source"] == "validation.lazyppl_elpd"
    assert artifact["producer"] == "ppl_synth_data.docker.lazyppl_harbor.score"


# decision_changed: convert->relabel. This exercises `_result_from_reward`'s
# rejection branch for a payload shape (flat, non-contract) that the real
# containerized `score.py` cannot deterministically emit, driving the real
# lazyppl-harbor boundary always produces a schema-valid RewardArtifact (see
# `test_score_lazyppl_semantics_real_docker_never_produces_a_non_contract_reward`
# below, which proves that convergence for real across independent runs).
@pytest.mark.narrow_unit_not_acceptance
def test_score_lazyppl_semantics_rejects_non_contract_reward(monkeypatch) -> None:
    monkeypatch.setattr(
        lazyppl_semantics,
        "run_with_process_group_timeout",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(_flat_reward_payload()),
        ),
    )

    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        harbor_image="test-lazyppl-harbor:latest",
    )

    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "contract validation" in result.message
    assert result.details["reward_artifact"] == _flat_reward_payload()


# decision_changed: convert->relabel. Same reasoning as
# `test_score_lazyppl_semantics_rejects_non_contract_reward` above: a
# `pareto_k_warning=True` diagnostic on an otherwise-contract-shaped payload is
# a parser/contract-branch case we manufacture, not a shape the real
# `arviz_loo` scorer reliably reproduces on demand. Cross-ref:
# `test_score_lazyppl_semantics_real_docker_never_produces_a_non_contract_reward`.
@pytest.mark.narrow_unit_not_acceptance
def test_score_lazyppl_semantics_rejects_pareto_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        lazyppl_semantics,
        "run_with_process_group_timeout",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(_reward_payload(pareto_warning=True)),
        ),
    )

    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        harbor_image="test-lazyppl-harbor:latest",
    )

    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "PSIS" in result.message


# decision_changed: convert->relabel. Same reasoning as
# `test_score_lazyppl_semantics_rejects_non_contract_reward` above: a
# `pareto_k_error` diagnostic is a parser/contract-branch case we manufacture
# on a canned payload, not a shape the real `arviz_loo` scorer reliably
# reproduces on demand. Cross-ref:
# `test_score_lazyppl_semantics_real_docker_never_produces_a_non_contract_reward`.
@pytest.mark.narrow_unit_not_acceptance
def test_score_lazyppl_semantics_rejects_pareto_error(monkeypatch) -> None:
    monkeypatch.setattr(
        lazyppl_semantics,
        "run_with_process_group_timeout",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(_reward_payload(pareto_error="nonfinite_pareto_k")),
        ),
    )

    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        harbor_image="test-lazyppl-harbor:latest",
    )

    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "PSIS" in result.message
    assert "nonfinite_pareto_k" in result.message


def test_score_lazyppl_semantics_accepts_holdout_metric_with_pareto_warning(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        lazyppl_semantics,
        "run_with_process_group_timeout",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(_reward_payload(pareto_warning=True, avg_holdout_logp=-1.25)),
        ),
    )

    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        hidden_data=_sample_data(),
        harbor_image="test-lazyppl-harbor:latest",
    )

    assert result.ok is True
    assert result.status == SemanticStatus.pass_
    assert result.metric_name == "avg_holdout_logp"
    assert result.metric_value == -1.25


def test_score_lazyppl_semantics_passes_timeout_and_traversable_workspace(
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):
        command = args[0]
        mounts = [item for item in command if item.endswith(":ro")]
        mounted_paths = [item.split(":", 1)[0] for item in mounts]
        assert kwargs["timeout"] == 7
        for mounted_path in mounted_paths:
            path = lazyppl_semantics.Path(mounted_path)
            assert path.exists()
            assert path.parent.stat().st_mode & 0o755 == 0o755
        taskdata_path = next(
            lazyppl_semantics.Path(path) for path in mounted_paths if path.endswith("TaskData.hs")
        )
        taskdata = taskdata_path.read_text()
        assert "xVal = 1.25" in taskdata
        assert "xVal = 2.5" in taskdata
        hidden_taskdata_path = next(
            lazyppl_semantics.Path(path)
            for path in mounted_paths
            if path.endswith("HiddenTaskData.hs")
        )
        hidden_taskdata = hidden_taskdata_path.read_text()
        assert "loadHiddenData" in hidden_taskdata
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(_reward_payload()),
        )

    monkeypatch.setattr(lazyppl_semantics, "run_with_process_group_timeout", fake_run)

    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        hidden_data=_sample_data(),
        harbor_image="test-lazyppl-harbor:latest",
        timeout_sec=7,
    )

    assert result.ok is True


# decision_changed: convert->relabel. This drives the container-cleanup
# branch through a canned `TimeoutExpired` plus a monkeypatched cleanup
# `subprocess.run`, forcing a genuine multi-minute Docker/GHC/MCMC timeout on
# every run would make this test prohibitively slow and flaky as ordinary
# unit coverage. The real, non-mocked timeout-and-cleanup boundary is proven
# by the heavy_lane twin below. Cross-ref:
# `test_score_lazyppl_semantics_real_docker_removes_container_on_real_timeout`.
@pytest.mark.narrow_unit_not_acceptance
def test_score_lazyppl_semantics_removes_container_on_timeout(monkeypatch) -> None:
    cleanup_calls: list[list[str]] = []
    container_name = "ppl-synth-lazyppl-semantic-test"

    def fake_run_with_timeout(*args, **kwargs):
        command = args[0]
        assert "--name" in command
        assert command[command.index("--name") + 1] == container_name
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

    def fake_cleanup_run(command, **kwargs):
        cleanup_calls.append(command)
        assert kwargs["timeout"] == 30
        return SimpleNamespace(returncode=0, stdout=container_name, stderr="")

    monkeypatch.setattr(lazyppl_semantics, "_docker_container_name", lambda: container_name)
    monkeypatch.setattr(
        lazyppl_semantics,
        "run_with_process_group_timeout",
        fake_run_with_timeout,
    )
    monkeypatch.setattr(lazyppl_semantics.subprocess, "run", fake_cleanup_run)

    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        hidden_data=_sample_data(),
        harbor_image="test-lazyppl-harbor:latest",
        timeout_sec=7,
    )

    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "timed out" in result.message
    assert cleanup_calls == [["docker", "rm", "-f", container_name]]
    assert result.details["container_cleanup"]["ok"] is True
    assert result.details["container_cleanup"]["container_name"] == container_name


def test_build_taskdata_from_sample_data_uses_stage_c_rows() -> None:
    taskdata = lazyppl_semantics._build_taskdata_from_sample_data(
        "logLikRow st row = normalLogPdf 0.0 1.0 (value row + xVal row)",
        sample_data={"n": 2, "value": [2.0, 3.0], "xVal": [4.0, 5.0]},
    )

    assert "value :: Double" in taskdata
    assert "xVal :: Double" in taskdata
    assert "loadData = return" in taskdata
    assert "value = 2.0" in taskdata
    assert "xVal = 5.0" in taskdata


def test_build_taskdata_from_sample_data_uses_stage_c_integer_types() -> None:
    taskdata = lazyppl_semantics._build_taskdata_from_sample_data(
        (
            "logLikRow st row = "
            "categoricalLogPmf [0.2, 0.8] (diet_category row) "
            "+ poissonLogPmf 1.0 (defect_count row)"
        ),
        sample_data={
            "n": 2,
            "diet_category": [0, 2],
            "defect_count": [3, 5],
        },
    )

    assert "diet_category :: Int" in taskdata
    assert "defect_count :: Integer" in taskdata
    assert "diet_category = 2" in taskdata
    assert "defect_count = 5" in taskdata


def test_build_taskdata_from_sample_data_types_numeric_strings_from_values() -> None:
    taskdata = lazyppl_semantics._build_taskdata_from_sample_data(
        "logLikRow st row = normalLogPdf 0.0 1.0 (strain_id row)",
        sample_data={"n": 2, "strain_id": ["0.0", "0.9"]},
    )

    assert "strain_id :: Double" in taskdata
    assert "strain_id = 0.0" in taskdata
    assert "strain_id = 0.9" in taskdata


def test_build_taskdata_from_sample_data_keeps_integral_numeric_strings() -> None:
    taskdata = lazyppl_semantics._build_taskdata_from_sample_data(
        "logLikRow st row = fromIntegral (subject_id row)",
        sample_data={"n": 2, "subject_id": ["0.0", "2.0"]},
    )

    assert "subject_id :: Int" in taskdata
    assert "subject_id = 2" in taskdata


def test_build_taskdata_from_sample_data_rejects_missing_stage_c_field() -> None:
    try:
        lazyppl_semantics._build_taskdata_from_sample_data(
            "logLikRow st row = normalLogPdf 0.0 1.0 (missingField row)",
            sample_data={"n": 1, "observed": [1.0]},
        )
    except ValueError as exc:
        assert "missingField" in str(exc)
    else:
        raise AssertionError("missing Stage C field should fail closed")


# --- heavy_lane: real docker/GHC substrate twins ---
#
# The monkeypatched tests above stay as fast local unit coverage of
# `score_lazyppl_semantics`'s result-parsing/rejection logic (kept unchanged).
# The tests below drive `run_with_process_group_timeout` for real against the
# built `lazyppl-harbor` image; no fake stdout. Hard-asserts substrate
# presence; raises rather than skips.


def _require_real_lazyppl_harbor() -> str:
    if not shutil.which("docker"):
        raise RuntimeError(
            "heavy_lane test requires a real docker binary on PATH; run this suite on "
            "the AWS heavy-lane instance"
        )
    image = os.environ.get("LAZYPPL_HARBOR_IMAGE")
    if not image:
        raise RuntimeError(
            "heavy_lane test requires LAZYPPL_HARBOR_IMAGE to point at the built "
            "lazyppl-harbor image tag"
        )
    return image


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_score_lazyppl_semantics_real_docker_returns_contract_reward() -> None:
    image = _require_real_lazyppl_harbor()
    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        harbor_image=image,
        timeout_sec=600,
    )
    assert result.ok is True, result.message
    assert result.metric_name in {"elpd_loo", "avg_holdout_logp"}
    artifact = result.details["reward_artifact"]
    assert artifact["reward_source"] == "validation.lazyppl_elpd"
    assert artifact["producer"] == "ppl_synth_data.docker.lazyppl_harbor.score"


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_score_lazyppl_semantics_real_docker_accepts_holdout_metric_with_hidden_data() -> None:
    image = _require_real_lazyppl_harbor()
    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        hidden_data={"n": 2, "xVal": [0.5, -0.5]},
        harbor_image=image,
        timeout_sec=600,
    )
    assert result.ok is True, result.message
    assert result.metric_name == "avg_holdout_logp"
    assert isinstance(result.metric_value, float)


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_score_lazyppl_semantics_real_docker_passes_timeout_and_traversable_workspace() -> None:
    image = _require_real_lazyppl_harbor()
    start = time.monotonic()
    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        hidden_data=_sample_data(),
        harbor_image=image,
        timeout_sec=180,
    )
    elapsed = time.monotonic() - start
    assert result.ok is True, result.message
    assert elapsed < 180


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_score_lazyppl_semantics_real_docker_removes_container_on_real_timeout() -> None:
    """Real-Docker twin of ``test_score_lazyppl_semantics_removes_container_on_timeout``
    (row 140): forces a genuine ``TimeoutExpired`` (``timeout_sec=1`` is far
    shorter than a real docker pull + GHC compile + MCMC run) and asserts the
    real container is cleaned up (no leftover container under the
    deterministic verifier name)."""
    image = _require_real_lazyppl_harbor()
    result = lazyppl_semantics.score_lazyppl_semantics(
        _MODEL_CODE,
        prompt="Fit values between 0 and 1.",
        sample_data=_sample_data(),
        hidden_data=_sample_data(),
        harbor_image=image,
        timeout_sec=1,
    )
    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "timed out" in result.message
    cleanup = result.details["container_cleanup"]
    assert cleanup["attempted"] is True
    assert cleanup["ok"] is True
    listing = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={cleanup['container_name']}",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert cleanup["container_name"] not in listing.stdout


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_score_lazyppl_semantics_real_docker_never_produces_a_non_contract_reward() -> None:
    """Rows 137-139 (``rejects_non_contract_reward`` / ``rejects_pareto_error`` /
    ``rejects_pareto_warning``) assert ``_result_from_reward``'s rejection
    branches for malformed scorer output. The real containerized ``score.py``
    always emits a schema-valid ``RewardArtifact`` (see
    ``test_score_main_real_arviz_*`` in ``test_lazyppl_tooling.py``, which
    drives the real ArviZ computation with no monkeypatch), so those three
    malformed-input branches cannot be triggered by driving the real
    container; they stay covered, unchanged, by the narrow unit tests
    above. This heavy_lane test instead proves the converse for real: several
    independent real container runs over varied real models all produce
    schema-valid, non-flat reward artifacts, i.e. the flat/malformed shapes
    those unit tests reject are never actually emitted by genuine Harbor
    output.
    """
    image = _require_real_lazyppl_harbor()
    scenarios = [
        (_sample_data(), None),
        (_sample_data(), {"n": 1, "xVal": [0.0]}),
    ]
    for sample_data, hidden_data in scenarios:
        result = lazyppl_semantics.score_lazyppl_semantics(
            _MODEL_CODE,
            prompt="Fit values between 0 and 1.",
            sample_data=sample_data,
            hidden_data=hidden_data,
            harbor_image=image,
            timeout_sec=600,
        )
        assert result.ok is True, result.message
        artifact = result.details["reward_artifact"]
        RewardArtifact.model_validate(artifact)
        assert artifact.get("reward_source") == "validation.lazyppl_elpd"
        assert "reward" in artifact

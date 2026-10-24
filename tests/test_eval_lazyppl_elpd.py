from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_harbor_command_mounts_inputs_and_runtime_flags(tmp_path: Path) -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_command_test", "scripts/eval_lazyppl_elpd.py")
    submission_path = tmp_path / "Submission.hs"
    taskdata_path = tmp_path / "TaskData.hs"
    hidden_taskdata_path = tmp_path / "HiddenTaskData.hs"

    command = eval_mod._build_harbor_command(
        submission_path=submission_path,
        taskdata_path=taskdata_path,
        hidden_taskdata_path=hidden_taskdata_path,
        harbor_image="custom-harbor:latest",
    )

    assert command[:5] == ["docker", "run", "--rm", "--network=none", "--memory=1g"]
    assert "custom-harbor:latest" in command
    assert f"{submission_path}:/tests/Submission.hs:ro" in command
    assert f"{taskdata_path}:/tests/TaskData.hs:ro" in command
    assert f"{hidden_taskdata_path}:/tests/HiddenTaskData.hs:ro" in command
    assert "LAZYPPL_BURNIN=" in " ".join(command)


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. A genuine Docker infra failure
# requires actually stopping the real Docker daemon/socket on a disposable
# instance (destructive; must never run on a shared machine); provisioned
# heavy_lane substrate, not something the default lane can honestly
# reproduce. Real-boundary coverage lives in
# test_evaluate_one_classifies_docker_infrastructure_failures_real_daemon_down.
def test_evaluate_one_classifies_docker_infrastructure_failures(monkeypatch) -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_infra_test", "scripts/eval_lazyppl_elpd.py")

    class _FakeRuntime:
        def extract_code(self, _completion: str) -> str:
            return "code"

    monkeypatch.setattr(eval_mod, "get_runtime", lambda _name: _FakeRuntime())
    monkeypatch.setattr(eval_mod, "_build_taskdata_realistic", lambda *_args, **_kwargs: "taskdata")
    monkeypatch.setattr(
        eval_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=125,
            stderr="Cannot connect to the Docker daemon",
            stdout="",
        ),
    )

    result = eval_mod.evaluate_one(
        {"completion": '<submit_program language="haskell">code</submit_program>', "metadata": {}},
        "record-1",
        "harbor:latest",
    )

    assert result["compile_ok"] is False
    assert result["error"] == "docker infrastructure: Cannot connect to the Docker daemon"


@pytest.mark.narrow_unit_not_acceptance
# decision_changed: convert->relabel. A genuine success round-trip requires
# the real built lazyppl-harbor Docker image, heavy_lane substrate that
# the default lane cannot provision. Real-boundary coverage lives in
# test_evaluate_one_reads_reward_payload_on_success_real_docker.
def test_evaluate_one_reads_reward_payload_on_success(monkeypatch) -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_success_test", "scripts/eval_lazyppl_elpd.py")

    class _FakeRuntime:
        def extract_code(self, _completion: str) -> str:
            return "code"

    reward_payload = {
        "compile_ok": 1.0,
        "elpd_loo": 1.25,
        "pareto_k_max": 0.4,
        "reward": 2.5,
        "n_draws": 2200,
        "n_obs": 50,
        "pareto_k_warning": None,
        "holdout_error": None,
    }

    monkeypatch.setattr(eval_mod, "get_runtime", lambda _name: _FakeRuntime())
    monkeypatch.setattr(eval_mod, "_build_taskdata_realistic", lambda *_args, **_kwargs: "taskdata")
    monkeypatch.setattr(
        eval_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(reward_payload),
        ),
    )

    result = eval_mod.evaluate_one(
        {
            "completion": '<submit_program language="haskell">code</submit_program>',
            "metadata": {"family": "gaussian"},
        },
        "record-2",
        "harbor:latest",
    )

    assert result["compile_ok"] is True
    assert result["elpd_loo"] == 1.25
    assert result["pareto_k_max"] == 0.4
    assert result["reward"] == 2.5
    assert result["n_draws"] == 2200
    assert result["n_obs"] == 50
    assert result["family"] == "gaussian"
    assert result["error"] is None


# --- heavy_lane: real docker substrate twins ---
#
# The monkeypatched tests above stay as fast local unit coverage of
# `evaluate_one`'s result-shaping logic (kept unchanged). The tests below
# drive `evaluate_one` for real against the built `lazyppl-harbor` image --
# no fake `subprocess.run`. Hard-asserts substrate presence; raises rather
# than skips.

_REAL_GAUSSIAN_COMPLETION = (
    '<submit_program language="haskell">\n'
    "module Submission where\n\n"
    "import LazyPPL\nimport LazyPPL.Distributions\nimport TaskData (Row(..))\n\n"
    "data State = State Double deriving (Show)\n\n"
    "model :: [Row] -> Meas State\n"
    "model rows = do\n"
    "  mu <- sample (normal 0.0 1.0)\n"
    "  let state = State mu\n"
    "  mapM_ (\\row -> score (exp (logLikRow state row))) rows\n"
    "  return state\n\n"
    "logLikRow :: State -> Row -> Double\n"
    "logLikRow (State mu) row = normalLogPdf mu 1.0 (value row)\n"
    "</submit_program>"
)


def _require_real_harbor_image() -> str:
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
def test_evaluate_one_reads_reward_payload_on_success_real_docker() -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_real_success_test", "scripts/eval_lazyppl_elpd.py")
    image = _require_real_harbor_image()

    result = eval_mod.evaluate_one(
        {"completion": _REAL_GAUSSIAN_COMPLETION, "metadata": {"family": "gaussian"}},
        "record-real-1",
        image,
        timeout=600,
    )

    assert result["compile_ok"] is True, result["error"]
    assert isinstance(result["elpd_loo"], float)
    assert result["family"] == "gaussian"
    assert result["error"] is None


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_evaluate_one_classifies_docker_infrastructure_failures_real_daemon_down() -> None:
    """Real-Docker twin of ``test_evaluate_one_classifies_docker_infrastructure_failures``
    (row 97): stops the real Docker daemon on this ephemeral single-purpose
    instance, drives a real ``docker run`` invocation through it, and asserts
    the real infra-failure classification; restarting the daemon afterward
    in a ``finally`` block. This is intentionally invasive and must only run
    on a disposable AWS heavy-lane instance, never a shared machine.
    """
    eval_mod = _load_module("eval_lazyppl_elpd_real_infra_test", "scripts/eval_lazyppl_elpd.py")
    image = _require_real_harbor_image()

    # `systemctl stop docker` alone leaves `docker.socket` listening: systemd
    # socket activation re-spawns `docker.service` on the next `docker`
    # invocation, so the daemon never actually goes down and this test's
    # premise (a real infra failure) silently doesn't hold. Stop the socket
    # unit too so `docker run` genuinely fails to reach a daemon.
    subprocess.run(["systemctl", "stop", "docker.socket", "docker.service"], check=True, timeout=60)
    try:
        result = eval_mod.evaluate_one(
            {"completion": _REAL_GAUSSIAN_COMPLETION, "metadata": {}},
            "record-real-infra-down",
            image,
            timeout=30,
        )
        assert result["compile_ok"] is False
        assert result["error"] is not None
        assert "docker infrastructure" in result["error"]
    finally:
        subprocess.run(
            ["systemctl", "start", "docker.socket", "docker.service"], check=True, timeout=60
        )
        deadline = time.monotonic() + 60
        recovered = False
        while time.monotonic() < deadline:
            probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
            if probe.returncode == 0:
                recovered = True
                break
            time.sleep(2)
        if not recovered:
            raise RuntimeError("docker daemon did not come back up after restart")


def test_aggregate_summarizes_family_breakdown() -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_aggregate_test", "scripts/eval_lazyppl_elpd.py")

    aggregate = eval_mod._aggregate(
        [
            {
                "id": "a",
                "family": "translated",
                "compile_ok": True,
                "holdout_error": None,
                "elpd_loo": 1.0,
            },
            {
                "id": "b",
                "family": "translated",
                "compile_ok": False,
                "holdout_error": "bad",
                "elpd_loo": None,
            },
            {
                "id": "c",
                "family": "native",
                "compile_ok": True,
                "holdout_error": None,
                "elpd_loo": 3.0,
            },
        ]
    )

    assert aggregate["summary"] == {
        "total": 3,
        "evaluated": 3,
        "compile_ok": 2,
        "holdout_errors": 1,
        "elpd_available": 2,
        "mean_elpd": 2.0,
        "median_elpd": 2.0,
    }
    assert aggregate["by_family"] == {
        "native": {
            "count": 1,
            "compile_ok": 1,
            "holdout_errors": 0,
            "elpd_available": 1,
            "mean_elpd": 3.0,
        },
        "translated": {
            "count": 2,
            "compile_ok": 1,
            "holdout_errors": 1,
            "elpd_available": 1,
            "mean_elpd": 1.0,
        },
    }

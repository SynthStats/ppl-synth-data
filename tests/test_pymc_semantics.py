from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import ppl_synth_data.validation.pymc_semantics as pymc_semantics
from ppl_synth_data.evidence import SemanticStatus
from ppl_synth_data.validation.pymc_semantics import (
    _extract_semantic_payload,
    _extract_timeout_phase,
    score_pymc_semantics,
)


def test_extract_semantic_payload_ignores_sampler_log_noise() -> None:
    payload = _extract_semantic_payload(
        "Only 50 samples per chain.\n"
        "NUTS: [intercept, beta]\n"
        '__PPL_SYNTH_DATA_SEMANTIC_JSON__={"ok": true, "status": "pass"}\n'
        "Sampling finished after structured payload\n"
    )

    assert payload == {"ok": True, "status": "pass"}


def test_extract_semantic_payload_ignores_later_bare_json() -> None:
    payload = _extract_semantic_payload(
        '__PPL_SYNTH_DATA_SEMANTIC_JSON__={"ok": true, "status": "pass"}\n'
        '{"ok": false, "status": "critical"}\n'
    )

    assert payload == {"ok": True, "status": "pass"}


def test_extract_semantic_payload_requires_sentinel() -> None:
    with pytest.raises(ValueError, match="semantic payload not found"):
        _extract_semantic_payload('{"ok": true, "status": "pass"}\n')


def test_extract_timeout_phase_uses_last_phase_marker() -> None:
    phase = _extract_timeout_phase(
        "noise\n"
        "__PPL_SYNTH_DATA_SEMANTIC_PHASE__=exec\n"
        "__PPL_SYNTH_DATA_SEMANTIC_PHASE__=posterior_sampling\n"
    )

    assert phase == "posterior_sampling"


@pytest.mark.narrow_unit_not_acceptance
def test_score_pymc_semantics_aligns_cpu_limit_with_timeout(monkeypatch) -> None:
    observed_limits: list[int] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        kwargs["preexec_fn"]()
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='__PPL_SYNTH_DATA_SEMANTIC_JSON__={"ok": true, "status": "pass"}\n',
            stderr="",
        )

    monkeypatch.setattr(pymc_semantics, "run_with_process_group_timeout", fake_run)
    monkeypatch.setattr(pymc_semantics.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        pymc_semantics,
        "set_resource_limits",
        lambda *, cpu_seconds, memory_gb=4: observed_limits.append(cpu_seconds),
    )

    result = score_pymc_semantics(
        "def model(data):\n    return None\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is True
    assert observed_limits == [70]


@pytest.mark.narrow_unit_not_acceptance
def test_score_pymc_semantics_omits_preexec_on_windows(monkeypatch) -> None:
    observed_has_preexec: list[bool] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        observed_has_preexec.append("preexec_fn" in kwargs)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='__PPL_SYNTH_DATA_SEMANTIC_JSON__={"ok": true, "status": "pass"}\n',
            stderr="",
        )

    monkeypatch.setattr(pymc_semantics, "run_with_process_group_timeout", fake_run)
    monkeypatch.setattr(pymc_semantics.platform, "system", lambda: "Windows")

    result = score_pymc_semantics(
        "def model(data):\n    return None\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is True
    assert observed_has_preexec == [False]


@pytest.mark.narrow_unit_not_acceptance
def test_score_pymc_semantics_uses_resolved_python_command(monkeypatch) -> None:
    observed_args: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        observed_args.append(list(args))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='__PPL_SYNTH_DATA_SEMANTIC_JSON__={"ok": true, "status": "pass"}\n',
            stderr="",
        )

    monkeypatch.setattr(
        pymc_semantics,
        "resolve_python_cmd",
        lambda: ["uv", "run", "--project", "/repo/ppl-synth-data", "python"],
    )
    monkeypatch.setattr(pymc_semantics, "run_with_process_group_timeout", fake_run)

    result = score_pymc_semantics(
        "def model(data):\n    return None\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is True
    assert len(observed_args) == 1
    assert observed_args[0][:5] == [
        "uv",
        "run",
        "--project",
        "/repo/ppl-synth-data",
        "python",
    ]
    assert observed_args[0][5] == "-m"
    assert observed_args[0][6] == "ppl_synth_data.validation._pymc_semantics_subprocess"
    assert Path(observed_args[0][7]).name == "semantic_payload.json"
    assert len(observed_args[0]) == 8


@pytest.mark.narrow_unit_not_acceptance
def test_score_pymc_semantics_sanitizes_non_json_subprocess_output(monkeypatch) -> None:
    def fake_run(args, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(
            args,
            1,
            stdout="/Users/example/private\n",
            stderr="AWS_SECRET_ACCESS_KEY=secret\n",
        )

    monkeypatch.setattr(pymc_semantics, "run_with_process_group_timeout", fake_run)

    result = score_pymc_semantics(
        "def model(data):\n    return None\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is False
    assert result.message == "semantic validation subprocess did not return a result payload"
    assert "/Users/" not in result.message
    assert "AWS_SECRET_ACCESS_KEY" not in result.message
    assert result.details["stdout_chars"] > 0
    assert result.details["stderr_chars"] > 0


def test_score_pymc_semantics_blocks_filesystem_builtins() -> None:
    result = score_pymc_semantics(
        "def model(data):\n    open('/etc/passwd').read()\n    return None\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "open" in result.message


def test_score_pymc_semantics_blocks_numpy_filesystem_apis() -> None:
    result = score_pymc_semantics(
        "def model(data):\n    np.fromfile('/etc/passwd')\n    return None\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "fromfile" in result.message


def test_score_pymc_semantics_blocks_object_graph_escape() -> None:
    result = score_pymc_semantics(
        (
            "def model(data):\n"
            "    for cls in ().__class__.__mro__[1].__subclasses__():\n"
            "        pass\n"
            "    return None\n"
        ),
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "unsafe attribute" in result.message


def test_score_pymc_semantics_blocks_direct_import_hook_use() -> None:
    result = score_pymc_semantics(
        "def model(data):\n    __import__('os')\n    return None\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is False
    assert result.status == SemanticStatus.critical
    assert "unsafe name" in result.message


def test_score_pymc_semantics_sanitizes_generated_exception_text() -> None:
    result = score_pymc_semantics(
        "def model(data):\n    raise ValueError('SENSITIVE_MARKER')\n",
        visible_data={"x": np.array([1.0])},
        timeout_sec=60,
    )

    assert result.ok is False
    assert result.message == "model(visible_data) raised: ValueError"
    assert "SENSITIVE_MARKER" not in result.message


def test_score_pymc_semantics_real_run() -> None:
    code = """
def model(data):
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    design = np.column_stack([x])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        mu = a * design[:, 0] + b
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
"""
    visible_data = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([1.0, 2.0, 3.0]),
    }
    hidden_data = {
        "x": np.array([3.0, 4.0]),
        "y": np.array([4.0, 5.0]),
    }

    result = score_pymc_semantics(
        code,
        visible_data=visible_data,
        hidden_data=hidden_data,
        draws=20,
        tune=20,
        timeout_sec=60,
    )

    assert result.ok is True
    assert result.status in {SemanticStatus.pass_, SemanticStatus.flag}
    assert result.metric_name == "avg_holdout_logp"
    assert isinstance(result.metric_value, float)
    assert result.details["hidden_data_present"] is True
    assert result.details["hidden_n_obs"] == 2
    assert "train_elpd_loo" in result.details


def test_score_pymc_semantics_allows_pytensor_internal_imports() -> None:
    code = """
def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    x_mean = x.mean()
    x_sd = x.std()
    x_sd = np.where(x_sd > 0, x_sd, 1.0)
    X = np.column_stack([(x - x_mean) / x_sd])

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=float(y.mean()), sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=2.0, shape=1)
        sigma = pm.HalfNormal("sigma", sigma=2.0)
        mu = alpha + pm.math.dot(X, beta)
        pm.Deterministic("x_effect_per_unit", beta[0] / x_sd)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        return model
"""
    x_visible = np.linspace(0.0, 5.0, 8)
    x_hidden = np.linspace(5.5, 7.0, 4)

    result = score_pymc_semantics(
        code,
        visible_data={"x": x_visible, "y": 1.5 * x_visible + 0.25},
        hidden_data={"x": x_hidden, "y": 1.5 * x_hidden + 0.25},
        draws=20,
        tune=20,
        timeout_sec=90,
    )

    assert result.ok is True
    assert result.status in {SemanticStatus.pass_, SemanticStatus.flag}
    assert result.metric_name == "avg_holdout_logp"
    assert result.details["observed_variables"] == ["obs"]


def test_score_pymc_semantics_emits_selection_diagnostics() -> None:
    # The selection verifier needs a per-row holdout vector (to bootstrap over
    # rows) plus the density-degeneracy signals. Verify a real run emits them.
    code = """
def model(data):
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        mu = a * x + b
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
"""
    rng = np.random.default_rng(0)
    x_vis = np.linspace(0.0, 5.0, 12)
    visible_data = {"x": x_vis, "y": 2.0 * x_vis + 1.0 + rng.normal(0, 0.5, x_vis.size)}
    x_hid = np.linspace(5.5, 8.0, 6)
    hidden_data = {"x": x_hid, "y": 2.0 * x_hid + 1.0 + rng.normal(0, 0.5, x_hid.size)}

    result = score_pymc_semantics(
        code,
        visible_data=visible_data,
        hidden_data=hidden_data,
        draws=40,
        tune=40,
        timeout_sec=120,
    )

    assert result.ok is True
    details = result.details
    # Per-row holdout vector: one value per held-out observation.
    assert "holdout_pointwise" in details
    assert len(details["holdout_pointwise"]) == 6
    assert all(isinstance(v, float) for v in details["holdout_pointwise"])
    # Density-degeneracy signals for the gate.
    assert "implied_min_pred_std" in details
    assert details["implied_min_pred_std"] > 0.0
    assert "data_std" in details
    assert details["data_std"] > 0.0
    # A healthy model's implied predictive scale should not be a near-Dirac
    # collapse relative to the data spread.
    assert details["implied_min_pred_std"] >= 0.01 * details["data_std"]

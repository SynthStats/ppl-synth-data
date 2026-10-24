#!/usr/bin/env python3
"""ELPD scorer for LazyPPL Harbor tasks.

Usage:
    score.py <log_lik.json> [hidden_eval.json]

log_lik.json schema (output from Main.hs):
    {
        "log_lik": [[float, ...], ...],   # shape (n_draws, n_obs)
        "holdout_log_lik": [[float, ...], ...],   # optional, same shape
        "n_draws": int,
        "n_obs": int
    }

Writes /tmp/reward.json as a synthstats-schemas RewardArtifact payload.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"\s*ArviZ is undergoing a major refactor.*",
        category=FutureWarning,
    )
    import arviz as az

import numpy as np

_EPS = 1e-8
_SCHEMA_VERSION = "synthstats.harbor.reward.v1"
_LOWER_LOG_REWARD_BOUND = -20.0
_PRODUCER = "ppl_synth_data.docker.lazyppl_harbor.score"
_REWARD_JSON = Path("/tmp/reward.json")


def _write_reward(payload: dict[str, Any]) -> None:
    _REWARD_JSON.write_text(json.dumps(payload, indent=2))


def _round_optional(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def _bounded_training_log_reward(log_reward: float) -> float:
    if not math.isfinite(log_reward):
        return _LOWER_LOG_REWARD_BOUND
    return max(float(log_reward), _LOWER_LOG_REWARD_BOUND)


def _reward_from_log_reward(log_reward: float) -> float:
    if not math.isfinite(log_reward):
        return 0.0
    return float(math.exp(min(log_reward, 20.0)))


def _reward_artifact(
    *,
    reward: float,
    reward_source: str,
    evidence_kind: str,
    truth_owner: str,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    provenance: dict[str, Any],
    reward_transform: dict[str, Any],
    training_log_reward: float,
    components: dict[str, float] | None = None,
    reward_components: list[dict[str, Any]] | None = None,
    scalarization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(training_log_reward):
        training_log_reward = _LOWER_LOG_REWARD_BOUND
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "reward": reward,
        "reward_source": reward_source,
        "evidence_kind": evidence_kind,
        "truth_owner": truth_owner,
        "producer": _PRODUCER,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "provenance": provenance,
        "reward_transform": reward_transform,
        "training_log_reward": training_log_reward,
    }
    if components:
        payload["components"] = components
    if reward_components:
        payload["reward_components"] = reward_components
    if scalarization:
        payload["scalarization"] = scalarization
    return payload


def _error_reward(reason: str) -> None:
    _write_reward(
        _reward_artifact(
            reward=0.0,
            reward_source="heuristic.lazyppl_elpd_error",
            evidence_kind="heuristic",
            truth_owner="heuristic",
            metrics={
                "avg_holdout_logp": None,
                "elpd_loo": None,
                "pareto_k_max": None,
                "compile_ok": 0.0,
                "n_draws": 0,
                "n_obs": 0,
            },
            diagnostics={"error": reason},
            provenance={"runtime": "lazyppl", "scorer": "arviz_loo"},
            reward_transform={
                "kind": "baseline_normalized_log_reward",
                "lower_log_reward_bound": _LOWER_LOG_REWARD_BOUND,
                "error_policy": "heuristic_floor",
            },
            training_log_reward=_LOWER_LOG_REWARD_BOUND,
            components={"lazyppl_elpd_error": 0.0},
            reward_components=[
                {
                    "name": "lazyppl_elpd_error",
                    "kind": "penalty",
                    "value": 0.0,
                    "source": "heuristic.lazyppl_elpd_error",
                    "evidence_kind": "heuristic",
                    "truth_owner": "heuristic",
                    "producer": _PRODUCER,
                    "weight": 1.0,
                    "log_contribution": _LOWER_LOG_REWARD_BOUND,
                    "metrics": {"compile_ok": 0.0},
                    "diagnostics": {"error": reason},
                    "provenance": {"runtime": "lazyppl", "scorer": "arviz_loo"},
                }
            ],
            scalarization={
                "method": "manual",
                "component_names": ["lazyppl_elpd_error"],
                "weights": {"lazyppl_elpd_error": 1.0},
                "lower_log_reward_bound": _LOWER_LOG_REWARD_BOUND,
            },
        )
    )


def _baseline_score(n_obs: int) -> float:
    # intercept-only baseline: log p(y|intercept) = -log(n_obs) per obs
    return -math.log(max(n_obs, 1)) * n_obs


def _loo_score(log_lik_matrix: np.ndarray) -> tuple[float, float]:
    # arviz PSIS-LOO → (elpd_loo, pareto_k_max)
    # arviz expects (chain, draw, obs): add a dummy chain dim
    log_lik_3d = log_lik_matrix[np.newaxis, :, :]  # (1, draws, obs)
    n_draws = log_lik_3d.shape[1]
    idata = az.from_dict(
        posterior={"_dummy": np.zeros((1, n_draws))},
        log_likelihood={"y": log_lik_3d},
    )
    loo = az.loo(idata, var_name="y", pointwise=True)
    elpd = float(loo.elpd_loo)
    k_max = float(np.max(loo.pareto_k))
    return elpd, k_max


def _avg_log_mean_exp(log_lik_matrix: np.ndarray) -> float:
    avg, _total = _holdout_log_predictive_scores(log_lik_matrix)
    return avg


def _holdout_log_predictive_scores(log_lik_matrix: np.ndarray) -> tuple[float, float]:
    if log_lik_matrix.ndim != 2 or log_lik_matrix.size == 0:
        raise ValueError(
            f"holdout_log_lik must be 2-D (draws x obs), got shape {log_lik_matrix.shape}"
        )
    safe = np.where(np.isfinite(log_lik_matrix), log_lik_matrix, -1e6)
    column_max = safe.max(axis=0, keepdims=True)
    stable = safe - column_max
    log_mean_exp = np.log(np.mean(np.exp(stable), axis=0)) + column_max[0]
    return float(np.mean(log_mean_exp)), float(np.sum(log_mean_exp))


def _resolve_holdout_payload(
    raw_payload: dict[str, Any],
    hidden_eval_path: Path | None,
) -> tuple[Any | None, str | None]:
    raw_holdout = raw_payload.get("holdout_log_lik")
    if raw_holdout is not None:
        return raw_holdout, None

    if hidden_eval_path is None or not hidden_eval_path.exists():
        return None, None

    try:
        hidden = json.loads(hidden_eval_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"failed to parse hidden_eval.json: {exc}"
    return hidden.get("holdout_log_lik"), None


_LOG_FLOOR = math.log(1e-300)  # the clamp value programs use as log(max 1e-300 p)


def likelihood_screen_metrics(log_lik: np.ndarray) -> dict[str, float | None]:
    """The numbers behind three soundness screens, from one posterior's draws x rows matrix.

    screen_lpd_sd: SD across rows of lpd_i = logsumexp over draws minus log S (0 up to rounding: the
    likelihood ignores
    its row; None with fewer than two rows); screen_floor_entry_share and screen_floor_row_share:
    entries, and rows
    whose median entry, within 1e-6 of log(1e-300); screen_max_log_lik: the largest entry (above 0
    on a discrete outcome
    is not a probability). The gate applies the thresholds, because only it knows the outcome
    values.
    """
    n_draws, n_obs = log_lik.shape
    lpd = np.logaddexp.reduce(log_lik, axis=0) - math.log(n_draws)
    at_floor = np.abs(log_lik - _LOG_FLOOR) <= 1e-6
    return {
        "screen_lpd_sd": float(np.std(lpd)) if n_obs >= 2 else None,
        "screen_floor_entry_share": float(at_floor.mean()),
        "screen_floor_row_share": float(
            (np.abs(np.median(log_lik, axis=0) - _LOG_FLOOR) <= 1e-6).mean()
        ),
        "screen_max_log_lik": float(np.max(log_lik)),
    }


def main() -> None:
    if len(sys.argv) < 2:
        _error_reward("no log_lik.json path provided")
        sys.exit(1)

    log_lik_path = Path(sys.argv[1])
    hidden_eval_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not log_lik_path.exists():
        _error_reward(f"log_lik file not found: {log_lik_path}")
        sys.exit(1)

    try:
        raw = json.loads(log_lik_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _error_reward(f"failed to parse log_lik.json: {exc}")
        sys.exit(1)

    if "log_lik" not in raw:
        _error_reward("log_lik.json missing 'log_lik' key")
        sys.exit(1)

    try:
        log_lik = np.array(raw["log_lik"], dtype=np.float64)
    except (ValueError, TypeError) as exc:
        _error_reward(f"could not convert log_lik to array: {exc}")
        sys.exit(1)

    if log_lik.ndim != 2 or log_lik.size == 0:
        _error_reward(f"log_lik must be 2-D (draws x obs), got shape {log_lik.shape}")
        sys.exit(1)

    # Replace NaN / ±Inf with a very negative number rather than failing.
    finite_mask = np.isfinite(log_lik)
    if not finite_mask.all():
        log_lik = np.where(finite_mask, log_lik, -1e6)

    n_draws, n_obs = log_lik.shape
    screens = likelihood_screen_metrics(log_lik)

    try:
        elpd_loo, pareto_k_max = _loo_score(log_lik)
    except Exception as exc:
        _error_reward(f"arviz LOO failed: {exc}")
        sys.exit(1)
    if not math.isfinite(elpd_loo):
        _error_reward("arviz LOO produced non-finite elpd_loo")
        sys.exit(1)

    avg_holdout_logp: float | None = None
    holdout_elpd: float | None = None
    n_holdout_obs = 0
    holdout_error: str | None = None
    holdout_ll, holdout_error = _resolve_holdout_payload(raw, hidden_eval_path)
    if holdout_ll is not None:
        try:
            holdout_arr = np.array(holdout_ll, dtype=np.float64)
            avg_holdout_logp, holdout_elpd = _holdout_log_predictive_scores(holdout_arr)
            n_holdout_obs = int(holdout_arr.shape[1])
            holdout_error = None
        except (TypeError, ValueError) as exc:
            avg_holdout_logp = None
            holdout_elpd = None
            n_holdout_obs = 0
            holdout_error = f"invalid holdout_log_lik: {exc}"

    # Pareto k diagnostic (Vehtari et al. 2024):
    # k < 0.5 reliable, 0.5-0.7 marginal, >= 0.7 unreliable PSIS estimate
    pareto_k_warning = math.isfinite(pareto_k_max) and pareto_k_max > 0.7
    pareto_k_metric = pareto_k_max if math.isfinite(pareto_k_max) else None

    # training_log_reward is the GFlowNet log-reward target. When a producer has
    # a gold ELPD, use it as the scale. Otherwise normalize by baseline
    # magnitude so absent gold scores do not collapse every successful program to
    # log_reward=0.
    if holdout_elpd is not None:
        score_metric_name = "holdout_elpd"
        score_value = holdout_elpd
        score_component_metrics = {
            "avg_holdout_logp": avg_holdout_logp,
            "n_holdout_obs": n_holdout_obs,
        }
        baseline = _baseline_score(n_holdout_obs)
    else:
        score_metric_name = "elpd_loo"
        score_value = elpd_loo
        score_component_metrics = {"pareto_k_max": pareto_k_metric}
        baseline = _baseline_score(n_obs)
    raw_gold_score = raw.get("gold_elpd")
    gold_score = float(raw_gold_score) if raw_gold_score is not None else None
    temperature = (
        abs(gold_score - baseline) + _EPS if gold_score is not None else max(abs(baseline), 1.0)
    )
    training_log_reward = _bounded_training_log_reward((score_value - baseline) / temperature)
    reward = _reward_from_log_reward(training_log_reward)

    diagnostics: dict[str, Any] = {"pareto_k_warning": pareto_k_warning}
    if pareto_k_metric is None:
        diagnostics["pareto_k_error"] = "nonfinite_pareto_k"
    if holdout_error is not None:
        diagnostics["holdout_error"] = holdout_error
    payload = _reward_artifact(
        reward=round(reward, 6),
        reward_source="validation.lazyppl_elpd",
        evidence_kind="verifier",
        truth_owner="verifier",
        metrics={
            "avg_holdout_logp": _round_optional(avg_holdout_logp, 6),
            "holdout_elpd": _round_optional(holdout_elpd, 4),
            "elpd_loo": round(elpd_loo, 4),
            "pareto_k_max": _round_optional(pareto_k_metric, 4),
            "compile_ok": 1.0,
            "n_draws": n_draws,
            "n_obs": n_obs,
            "n_holdout_obs": n_holdout_obs,
            **screens,
        },
        diagnostics=diagnostics,
        provenance={
            "runtime": "lazyppl",
            "scorer": ("holdout_log_predictive" if holdout_elpd is not None else "arviz_loo"),
        },
        reward_transform={
            "kind": "baseline_normalized_log_reward",
            "baseline_score": baseline,
            "gold_score": gold_score,
            "temperature": temperature,
            "lower_log_reward_bound": _LOWER_LOG_REWARD_BOUND,
        },
        training_log_reward=round(training_log_reward, 6),
        components={score_metric_name: round(score_value, 4)},
        reward_components=[
            {
                "name": score_metric_name,
                "kind": "terminal_verifier",
                "value": round(score_value, 4),
                "source": "validation.lazyppl_elpd",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
                "producer": _PRODUCER,
                "weight": 1.0,
                "log_contribution": round(training_log_reward, 6),
                "metrics": {
                    "baseline_score": baseline,
                    "temperature": temperature,
                    "n_draws": n_draws,
                    "n_obs": n_obs,
                    **score_component_metrics,
                },
                "diagnostics": diagnostics,
                "provenance": {
                    "runtime": "lazyppl",
                    "scorer": (
                        "holdout_log_predictive" if holdout_elpd is not None else "arviz_loo"
                    ),
                },
            }
        ],
        scalarization={
            "method": "weighted_sum",
            "component_names": [score_metric_name],
            "weights": {score_metric_name: 1.0},
            "lower_log_reward_bound": _LOWER_LOG_REWARD_BOUND,
            "temperature": temperature,
            "diagnostics": {
                "baseline_score": baseline,
                "gold_score": gold_score,
                "score_metric": score_metric_name,
            },
        },
    )
    _write_reward(payload)


if __name__ == "__main__":
    main()

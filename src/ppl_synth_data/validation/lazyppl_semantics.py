from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
import uuid
from collections.abc import Collection
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from synthstats_schemas import RewardArtifact

from ppl_synth_data.config import Settings
from ppl_synth_data.evidence import SemanticStatus
from ppl_synth_data.harbor.taskdata_gen import (
    generate_hidden_taskdata_hs,
    generate_taskdata_hs,
    parse_seed_row_fields,
)
from ppl_synth_data.runtimes.lazyppl_runtime import static_safety_error
from ppl_synth_data.validation.pymc_semantics import SemanticValidationResult
from ppl_synth_data.validation.subprocess_utils import run_with_process_group_timeout

log = logging.getLogger(__name__)

_MH_ENV = {
    "LAZYPPL_BURNIN": os.environ.get("LAZYPPL_BURNIN", "2000"),
    "LAZYPPL_THINNING": os.environ.get("LAZYPPL_THINNING", "50"),
    "LAZYPPL_NDRAWS": os.environ.get("LAZYPPL_NDRAWS", "2200"),
    "LAZYPPL_SEED": os.environ.get("LAZYPPL_SEED", "42"),
}
_REWARD_SOURCE = "validation.lazyppl_elpd"
_DEFAULT_TIMEOUT_SEC = int(os.environ.get("LAZYPPL_SEMANTIC_TIMEOUT_SEC", "600"))


def _completed_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _docker_workspace_parent() -> Path | None:
    home = Path.home()
    return home if str(home).startswith("/Users") else None


def _harbor_env_flags() -> list[str]:
    env_flags: list[str] = []
    for key, value in _MH_ENV.items():
        env_flags.extend(["-e", f"{key}={value}"])
    return env_flags


def _build_harbor_command(
    *,
    submission_path: Path,
    taskdata_path: Path,
    hidden_taskdata_path: Path,
    harbor_image: str,
    container_name: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network=none",
        "--memory=1g",
        "--cpus=2",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=100",
        *_harbor_env_flags(),
        "-v",
        f"{submission_path}:/tests/Submission.hs:ro",
        "-v",
        f"{taskdata_path}:/tests/TaskData.hs:ro",
        "-v",
        f"{hidden_taskdata_path}:/tests/HiddenTaskData.hs:ro",
        harbor_image,
        "bash",
        "-c",
        (
            # The hidden module must not be visible while Submission.hs is
            # type-checked: it is copied into its own directory only after the
            # type-check passes, and only the trusted Main.hs compile adds
            # that directory to the module path.
            "set -euo pipefail; "
            "cp /tests/TaskData.hs /app/TaskData.hs; "
            "cp /tests/Submission.hs /app/Submission.hs; "
            "ghc -fno-code -package-env /etc/ghc-pkg-env -i/app /app/Submission.hs >&2 && "
            "mkdir -p /app/hidden && "
            "cp /tests/HiddenTaskData.hs /app/hidden/HiddenTaskData.hs && "
            "ghc --make -O0 -package-env /etc/ghc-pkg-env "
            "-i/app -i/app/hidden /app/Main.hs -o /tmp/run-task >&2 && "
            "/tmp/run-task > /tmp/log_lik.json 2>&1 && "
            "python3 /app/score.py /tmp/log_lik.json >&2 && "
            "cat /tmp/reward.json"
        ),
    ]


def _docker_container_name() -> str:
    return f"ppl-synth-lazyppl-semantic-{uuid.uuid4().hex}"


def _cleanup_docker_container(container_name: str) -> dict[str, Any]:
    """Best-effort cleanup for verifier containers after host-side timeout."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed docker argv, no shell
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("LazyPPL verifier container cleanup failed for %s: %s", container_name, exc)
        return {
            "attempted": True,
            "ok": False,
            "container_name": container_name,
            "error": f"{type(exc).__name__}: {exc}",
        }

    stderr = _completed_text(proc.stderr).strip()
    stdout = _completed_text(proc.stdout).strip()
    # Docker returns nonzero if the named container has already exited and
    # --rm removed it. That is acceptable: the target invariant is no running
    # verifier container with this deterministic name.
    already_gone = proc.returncode != 0 and (
        "No such container" in stderr or "No such container" in stdout
    )
    ok = proc.returncode == 0 or already_gone
    if not ok:
        log.warning(
            "LazyPPL verifier container cleanup returned %s for %s: %s",
            proc.returncode,
            container_name,
            stderr or stdout,
        )
    return {
        "attempted": True,
        "ok": ok,
        "container_name": container_name,
        "returncode": proc.returncode,
        "stdout": stdout[:500],
        "stderr": stderr[:500],
    }


def _column_values(sample_data: dict[str, Any], field_name: str) -> list[Any]:
    if field_name not in sample_data:
        msg = f"Stage C sample data is missing field {field_name!r}"
        raise ValueError(msg)
    value = sample_data[field_name]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        msg = f"Stage C sample data field {field_name!r} is not row-indexable"
        raise ValueError(msg)
    return list(value)


def _sample_data_row_count(sample_data: dict[str, Any]) -> int:
    explicit_count = sample_data.get("n")
    if isinstance(explicit_count, int) and not isinstance(explicit_count, bool):
        if explicit_count <= 0:
            msg = "Stage C sample data must contain at least one row"
            raise ValueError(msg)
        return explicit_count
    for key in sample_data:
        if key == "n":
            continue
        values = _column_values(sample_data, key)
        if not values:
            continue
        return len(values)
    msg = "Stage C sample data must contain at least one row"
    raise ValueError(msg)


def _hidden_data_with_lag_pairs(
    hidden_data: dict[str, Any],
    accessed: set[str],
) -> dict[str, Any]:
    """Adapt the public dataset's raw held-out ``y`` series to lag pairs.

    Its LazyPPL time-series rows expose ``yPrev``/``yVal`` in visible data but
    store each separate held-out segment as a raw ``y`` series.
    """
    if not accessed.intersection({"yPrev", "yVal"}):
        return hidden_data
    if "yPrev" in hidden_data or "yVal" in hidden_data:
        return hidden_data
    if "y" not in hidden_data:
        return hidden_data

    y_values = _column_values(hidden_data, "y")
    if len(y_values) < 2:
        msg = "Stage C hidden data field 'y' must contain at least two values"
        raise ValueError(msg)

    return {
        "n": len(y_values) - 1,
        **{key: value for key, value in hidden_data.items() if key not in {"n", "y"}},
        "yPrev": y_values[:-1],
        "yVal": y_values[1:],
    }


def _python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _numeric_observed_values(values: list[Any]) -> list[float] | None:
    numeric: list[float] = []
    for value in values:
        scalar = _python_scalar(value)
        if scalar is None:
            continue
        if isinstance(scalar, bool):
            return None
        try:
            parsed = float(scalar)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        numeric.append(parsed)
    return numeric


def _all_integer_like(values: list[float]) -> bool:
    return all(value.is_integer() for value in values)


def _hs_type_from_stage_c_values(
    *,
    field_name: str,
    values: list[Any],
    fallback: str,
) -> str:
    observed = [_python_scalar(value) for value in values if _python_scalar(value) is not None]
    if not observed:
        return fallback
    if all(isinstance(value, bool) for value in observed):
        return "Bool"
    if all(isinstance(value, Integral) and not isinstance(value, bool) for value in observed):
        if field_name == "count" or field_name.endswith("_count"):
            return "Integer"
        return "Int"
    if all(isinstance(value, Real) and not isinstance(value, bool) for value in observed):
        return "Double"
    numeric = _numeric_observed_values(observed)
    if numeric is not None and numeric:
        if field_name == "count" or field_name.endswith("_count"):
            return "Integer" if _all_integer_like(numeric) else "Double"
        if fallback in {"Int", "Integer"} and _all_integer_like(numeric):
            return fallback
        return "Double"
    return fallback


def _build_taskdata_from_sample_data(
    code: str,
    *,
    sample_data: dict[str, Any],
) -> str:
    # The parser reads ``name r`` applications, so a candidate's own let-bound helper or
    # an argument written before
    # the row can read as a Row field; only columns of the data can be Row fields.
    parsed = parse_seed_row_fields(code)
    columns = [col for col in parsed if col["name"] in sample_data]
    if not columns:
        absent = sorted({col["name"] for col in parsed})
        msg = "LazyPPL semantic validation requires at least one Row field access"
        if absent:
            msg += f"; applied to the row but absent from the data: {', '.join(absent)}"
        raise ValueError(msg)

    row_count = _sample_data_row_count(sample_data)
    values_by_field = {col["name"]: _column_values(sample_data, col["name"]) for col in columns}
    columns = [
        {
            **col,
            "hs_type": _hs_type_from_stage_c_values(
                field_name=col["name"],
                values=values_by_field[col["name"]],
                fallback=col["hs_type"],
            ),
        }
        for col in columns
    ]

    rows: list[dict[str, float | int | bool | str]] = []
    for row_index in range(row_count):
        row: dict[str, float | int | bool | str] = {}
        for col in columns:
            name = col["name"]
            values = values_by_field[name]
            if row_index >= len(values):
                msg = (
                    f"Stage C sample data field {name!r} has {len(values)} rows, "
                    f"expected at least {row_count}"
                )
                raise ValueError(msg)
            row[name] = values[row_index]
        rows.append(row)

    return generate_taskdata_hs(columns, rows)


def _build_hidden_taskdata_from_sample_data(
    code: str,
    *,
    hidden_data: dict[str, Any] | None,
    row_fields: Collection[str],
) -> str:
    # Row fields are the columns of the task's visible data (see
    # _build_taskdata_from_sample_data). A visible column
    # missing from the held-out data still raises below, because that is a data defect,
    # not a candidate's.
    columns = [col for col in parse_seed_row_fields(code) if col["name"] in row_fields]
    if hidden_data is None:
        return generate_hidden_taskdata_hs(columns, [])

    accessed = {col["name"] for col in columns}
    hidden_data = _hidden_data_with_lag_pairs(hidden_data, accessed)
    row_count = _sample_data_row_count(hidden_data)
    values_by_field = {col["name"]: _column_values(hidden_data, col["name"]) for col in columns}
    columns = [
        {
            **col,
            "hs_type": _hs_type_from_stage_c_values(
                field_name=col["name"],
                values=values_by_field[col["name"]],
                fallback=col["hs_type"],
            ),
        }
        for col in columns
    ]

    rows: list[dict[str, float | int | bool | str]] = []
    for row_index in range(row_count):
        row: dict[str, float | int | bool | str] = {}
        for col in columns:
            name = col["name"]
            values = values_by_field[name]
            if row_index >= len(values):
                msg = (
                    f"Stage C hidden data field {name!r} has {len(values)} rows, "
                    f"expected at least {row_count}"
                )
                raise ValueError(msg)
            row[name] = values[row_index]
        rows.append(row)

    return generate_hidden_taskdata_hs(columns, rows)


def _parse_reward_payload(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return None, f"failed to parse reward JSON from stdout: {stdout[:200]}"
    if not isinstance(parsed, dict):
        return None, f"reward JSON must be an object, got {type(parsed).__name__}"
    return parsed, None


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    candidate = float(value)
    return candidate if math.isfinite(candidate) else None


def _metric(reward: RewardArtifact, name: str) -> float | None:
    return _finite_float(reward.metrics.get(name))


def _result_from_reward(reward_payload: dict[str, Any]) -> SemanticValidationResult:
    try:
        reward = RewardArtifact.model_validate(reward_payload)
    except Exception as exc:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=(
                f"LazyPPL reward artifact failed contract validation: {type(exc).__name__}: {exc}"
            ),
            details={"reward_artifact": reward_payload},
        )

    details = {
        "reward_artifact": reward.model_dump(mode="json"),
        "reward_source": reward.reward_source,
        "producer": reward.producer,
        **dict(reward.metrics),
        "diagnostics": dict(reward.diagnostics),
    }
    if reward.reward_source != _REWARD_SOURCE:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=f"LazyPPL scorer returned non-verifier reward source: {reward.reward_source}",
            details=details,
        )
    if reward.diagnostics.get("holdout_error") is not None:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=f"LazyPPL holdout scoring failed: {reward.diagnostics['holdout_error']}",
            details=details,
        )
    holdout_metric = _metric(reward, "avg_holdout_logp")
    has_holdout_metric = holdout_metric is not None
    if reward.diagnostics.get("pareto_k_error") is not None and not has_holdout_metric:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=(
                "LazyPPL semantic validation rejected unstable PSIS diagnostics: "
                f"{reward.diagnostics['pareto_k_error']}"
            ),
            metric_name="elpd_loo",
            metric_value=_metric(reward, "elpd_loo"),
            details=details,
        )
    if reward.diagnostics.get("pareto_k_warning") is True and not has_holdout_metric:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message="LazyPPL semantic validation rejected unstable PSIS diagnostics",
            metric_name="elpd_loo",
            metric_value=_metric(reward, "elpd_loo"),
            details=details,
        )

    metric_name = "avg_holdout_logp"
    metric_value = holdout_metric
    if metric_value is None:
        metric_name = "elpd_loo"
        metric_value = _metric(reward, metric_name)
    if metric_value is None:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message="LazyPPL scorer returned no finite ELPD metric",
            details=details,
        )

    return SemanticValidationResult(
        ok=True,
        status=SemanticStatus.pass_,
        message="LazyPPL semantic validation passed",
        metric_name=metric_name,
        metric_value=metric_value,
        details=details,
    )


def score_lazyppl_semantics(
    code: str,
    *,
    prompt: str,
    sample_data: dict[str, Any],
    hidden_data: dict[str, Any] | None = None,
    harbor_image: str | None = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
) -> SemanticValidationResult:
    """Score a LazyPPL submission with the Harbor ELPD verifier.

    The returned details include the contract-valid ``RewardArtifact`` emitted
    by the Harbor scorer so ATIF export can preserve the verifier-owned reward
    instead of re-scalarizing it as a PyMC holdout score.
    """
    static_error = static_safety_error(code)
    if static_error is not None:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=f"LazyPPL static safety filter rejected the submission: {static_error}",
            details={"static_safety_error": static_error},
        )
    resolved_image = harbor_image or Settings().lazyppl_harbor_image
    if not resolved_image:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=(
                "INFRA: LAZYPPL_HARBOR_IMAGE is not set; set it to an image built from "
                "docker/lazyppl-harbor"
            ),
        )
    try:
        taskdata_hs = _build_taskdata_from_sample_data(code, sample_data=sample_data)
        hidden_taskdata_hs = _build_hidden_taskdata_from_sample_data(
            code,
            hidden_data=hidden_data,
            row_fields=set(sample_data),
        )
    except ValueError as exc:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=f"LazyPPL semantic validation could not build TaskData: {exc}",
        )

    with tempfile.TemporaryDirectory(dir=_docker_workspace_parent()) as td:
        workspace = Path(td)
        submission_path = workspace / "Submission.hs"
        taskdata_path = workspace / "TaskData.hs"
        hidden_taskdata_path = workspace / "HiddenTaskData.hs"
        submission_path.write_text(code, encoding="utf-8")
        taskdata_path.write_text(taskdata_hs, encoding="utf-8")
        hidden_taskdata_path.write_text(hidden_taskdata_hs, encoding="utf-8")
        # tempfile creates 0700 directories. Docker must be able to resolve
        # the bind-mounted files even when the verifier image runs unprivileged.
        workspace.chmod(0o755)
        container_name = _docker_container_name()

        try:
            proc = run_with_process_group_timeout(
                _build_harbor_command(
                    submission_path=submission_path,
                    taskdata_path=taskdata_path,
                    hidden_taskdata_path=hidden_taskdata_path,
                    harbor_image=resolved_image,
                    container_name=container_name,
                ),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            cleanup = _cleanup_docker_container(container_name)
            return SemanticValidationResult(
                ok=False,
                status=SemanticStatus.critical,
                message=f"LazyPPL semantic validation timed out after {timeout_sec}s",
                details={"container_cleanup": cleanup},
            )

    if proc.returncode != 0:
        stderr = _completed_text(proc.stderr).strip()
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=stderr[:500] if stderr else f"LazyPPL Harbor exited {proc.returncode}",
            details={"stderr": stderr[:2000]},
        )

    stdout = _completed_text(proc.stdout)
    stderr = _completed_text(proc.stderr)
    reward_payload, parse_error = _parse_reward_payload(stdout.strip())
    if parse_error is not None or reward_payload is None:
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=parse_error or "LazyPPL scorer returned no reward payload",
            details={"stdout": stdout[:2000], "stderr": stderr[:2000]},
        )
    return _result_from_reward(reward_payload)


__all__ = [
    "score_lazyppl_semantics",
]

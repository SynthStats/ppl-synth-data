from __future__ import annotations

import json
import math
import platform
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ppl_synth_data.evidence import SemanticStatus
from ppl_synth_data.validation.subprocess_utils import (
    build_limited_env,
    resolve_python_cmd,
    run_with_process_group_timeout,
    set_resource_limits,
)

_JSON_SENTINEL = "__PPL_SYNTH_DATA_SEMANTIC_JSON__="
_PHASE_SENTINEL = "__PPL_SYNTH_DATA_SEMANTIC_PHASE__="


@dataclass(frozen=True)
class SemanticValidationResult:
    ok: bool
    status: SemanticStatus
    message: str
    metric_name: str | None = None
    metric_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


# Each diagnostic threshold below names its published source.

# Vehtari, Gelman, Simpson, Carpenter & Burkner, "Rank-normalization, folding,
# and localization: An improved R-hat" (arXiv:1903.08008). PyMC's own
# convergence module warns above this same value.
MAX_SPLIT_R_HAT: float = 1.01

# Betancourt, "Diagnosing Suboptimal Cotangent Disintegrations in Hamiltonian
# Monte Carlo" (arXiv:1604.00695). ArviZ's own `bfmi` docstring: "The current
# advice is that values smaller than 0.3 indicate poor sampling. However, this
# threshold is provisional and may change."
MIN_BFMI: float = 0.3

# Effective sample size per chain, from the same rank-normalization paper.
# PyMC's convergence module raises an error-severity warning below it, so a
# fit that fails this criterion is one PyMC itself considers unusable for
# reliable R-hat and ESS estimates.
MIN_ESS_PER_CHAIN: float = 100.0

# A divergent transition means the sampler could not follow the target
# geometry, so the draws are biased however good the score looks.
MAX_DIVERGENCES: int = 0

# Vehtari, Simpson, Gelman, Yao & Gabry, "Pareto smoothed importance sampling"
# (arXiv:1507.02646). Above 0.7 the importance-sampling correction behind
# PSIS-LOO is unreliable; ArviZ returns a non-finite value when it fails
# outright.
MAX_PARETO_K: float = 0.7

# RefineStat counts a program as reliable when at least five of the seven
# diagnostics pass.
MIN_PASSING_RELIABILITY_CHECKS: int = 5

RELIABILITY_CHECK_NAMES: tuple[str, ...] = (
    "split_r_hat",
    "bfmi",
    "ess_bulk",
    "ess_tail",
    "no_divergences",
    "pareto_k",
    "finite_elpd",
)


@dataclass(frozen=True)
class ReliabilityCheck:
    """One of the seven diagnostics, resolved to a pass/fail decision."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ReliabilityVerdict:
    """The reliability decision for one posterior fit.

    Deliberately separate from the score: `SemanticValidationResult.metric_value`
    says how well the program predicted, this says whether the posterior behind
    that number can be trusted. Neither is folded into the other.
    """

    passed: bool
    passed_count: int
    required_count: int
    checks: tuple[ReliabilityCheck, ...]

    @property
    def failing_names(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    def as_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "passed_count": self.passed_count,
            "required_count": self.required_count,
            "failing": list(self.failing_names),
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in self.checks
            ],
        }


@dataclass(frozen=True)
class SamplerDiagnostics:
    """The seven reliability diagnostics measured on one posterior fit.

    A value is None when the diagnostic could not be computed at all, with the
    reason under the same check name in `errors`. An uncomputable diagnostic is
    a FAILED check: the absence of a diagnostic is never evidence of
    reliability.
    """

    chains: int
    draws_per_chain: int
    split_r_hat_max: float | None = None
    bfmi_min: float | None = None
    ess_bulk_min: float | None = None
    ess_tail_min: float | None = None
    divergences: int | None = None
    pareto_k_max: float | None = None
    elpd_loo: float | None = None
    errors: Mapping[str, str] = field(default_factory=dict)

    @property
    def posterior_draws(self) -> int:
        return self.chains * self.draws_per_chain

    def as_payload(self) -> dict[str, Any]:
        return {
            "chains": self.chains,
            "draws_per_chain": self.draws_per_chain,
            "posterior_draws": self.posterior_draws,
            "split_r_hat_max": self.split_r_hat_max,
            "bfmi_min": self.bfmi_min,
            "ess_bulk_min": self.ess_bulk_min,
            "ess_tail_min": self.ess_tail_min,
            "divergences": self.divergences,
            "pareto_k_max": self.pareto_k_max,
            "elpd_loo": self.elpd_loo,
            "errors": dict(self.errors),
        }


def psis_tail_length(posterior_draws: int, relative_eff: float = 1.0) -> int:
    """Number of draws ArviZ puts in the PSIS right tail for `posterior_draws`.

    Not a diagnostic and not used to compute one. It exists to explain a
    non-finite Pareto-k: `arviz.stats.stats._psislw` sets k to infinity when
    the tail holds four or fewer draws, so below 21 posterior draws every
    model, correct or degenerate, reports an infinite Pareto-k. Reproducing
    ArviZ's cutoff arithmetic here lets the gate say which of the two it is
    looking at.
    """
    if posterior_draws <= 0:
        return 0
    return int(math.ceil(min(posterior_draws / 5.0, 3.0 * (posterior_draws / relative_eff) ** 0.5)))


def _format(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.4g}"


def evaluate_sampler_reliability(d: SamplerDiagnostics) -> ReliabilityVerdict:
    """Resolve measured diagnostics into the seven checks and a verdict.

    Pure: it interprets numbers the scorer measured with ArviZ and PyMC, and
    computes no statistic of its own.
    """
    ess_threshold = MIN_ESS_PER_CHAIN * d.chains
    checks: list[ReliabilityCheck] = []

    def add(name: str, passed: bool, detail: str) -> None:
        error = d.errors.get(name)
        if error is not None:
            checks.append(ReliabilityCheck(name=name, passed=False, detail=f"unavailable: {error}"))
        else:
            checks.append(ReliabilityCheck(name=name, passed=passed, detail=detail))

    add(
        "split_r_hat",
        d.split_r_hat_max is not None and d.split_r_hat_max <= MAX_SPLIT_R_HAT,
        f"max split R-hat={_format(d.split_r_hat_max)} (<= {MAX_SPLIT_R_HAT:g})",
    )
    add(
        "bfmi",
        d.bfmi_min is not None and d.bfmi_min >= MIN_BFMI,
        f"min BFMI={_format(d.bfmi_min)} (>= {MIN_BFMI:g})",
    )
    add(
        "ess_bulk",
        d.ess_bulk_min is not None and d.ess_bulk_min >= ess_threshold,
        f"min bulk ESS={_format(d.ess_bulk_min)} "
        f"(>= {ess_threshold:g} = {MIN_ESS_PER_CHAIN:g} * {d.chains} chains)",
    )
    add(
        "ess_tail",
        d.ess_tail_min is not None and d.ess_tail_min >= ess_threshold,
        f"min tail ESS={_format(d.ess_tail_min)} "
        f"(>= {ess_threshold:g} = {MIN_ESS_PER_CHAIN:g} * {d.chains} chains)",
    )
    add(
        "no_divergences",
        d.divergences is not None and d.divergences <= MAX_DIVERGENCES,
        f"divergent transitions={d.divergences} (<= {MAX_DIVERGENCES})",
    )

    pareto_detail = f"max Pareto-k={_format(d.pareto_k_max)} (<= {MAX_PARETO_K:g})"
    if d.pareto_k_max is not None and not math.isfinite(d.pareto_k_max):
        tail_length = psis_tail_length(d.posterior_draws)
        if tail_length <= 4:
            pareto_detail = (
                f"max Pareto-k={_format(d.pareto_k_max)}: PSIS tail holds {tail_length} of "
                f"{d.posterior_draws} posterior draws, at or below the four ArviZ needs to fit "
                "the generalised Pareto distribution, so this value carries no information "
                "about the model"
            )
    add(
        "pareto_k",
        d.pareto_k_max is not None
        and math.isfinite(d.pareto_k_max)
        and d.pareto_k_max <= MAX_PARETO_K,
        pareto_detail,
    )
    add(
        "finite_elpd",
        d.elpd_loo is not None and math.isfinite(d.elpd_loo),
        f"PSIS-LOO ELPD={_format(d.elpd_loo)} (finite)",
    )

    passed_count = sum(1 for check in checks if check.passed)
    return ReliabilityVerdict(
        passed=passed_count >= MIN_PASSING_RELIABILITY_CHECKS,
        passed_count=passed_count,
        required_count=MIN_PASSING_RELIABILITY_CHECKS,
        checks=tuple(checks),
    )


def _to_wire_value(value: Any) -> Any:
    """JSON-safe encoding for one visible/hidden data value.

    Mirrors `serialize_sample_data`'s duck-typed distinction (only values with
    a `.tolist()` method, i.e. numpy arrays, are marked for reconstruction as
    `np.array(...)` on the subprocess side; plain lists/scalars/strings pass
    through unchanged) so real callers that hand in plain `list[float]`
    columns (e.g. `agentic/domain.py`'s `_column_dict`) keep seeing plain
    lists inside the harness, not arrays.
    """
    if hasattr(value, "tolist"):
        return {"__ndarray__": value.tolist()}
    return value


def _to_wire_data(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    return {key: _to_wire_value(value) for key, value in data.items()}


def score_pymc_semantics(
    code: str,
    *,
    visible_data: dict[str, Any] | None,
    hidden_data: dict[str, Any] | None = None,
    draws: int = 50,
    tune: int = 50,
    timeout_sec: int = 60,
) -> SemanticValidationResult:
    cpu_limit_seconds = max(30, int(timeout_sec) + 10)
    request_payload = json.dumps(
        {
            "code": code,
            "visible_data": _to_wire_data(visible_data),
            "hidden_data": _to_wire_data(hidden_data),
            "draws": draws,
            "tune": tune,
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        payload_path = temp_path / "semantic_payload.json"
        payload_path.write_text(request_payload, encoding="utf-8")
        run_kwargs: dict[str, Any] = {}
        if platform.system() != "Windows":
            run_kwargs["preexec_fn"] = lambda: set_resource_limits(
                cpu_seconds=cpu_limit_seconds,
            )
        try:
            proc = run_with_process_group_timeout(
                [
                    *resolve_python_cmd(),
                    "-m",
                    "ppl_synth_data.validation._pymc_semantics_subprocess",
                    str(payload_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
                cwd=str(temp_path),
                env=build_limited_env(
                    extra={"PYTHONNOUSERSITE": "1"},
                    home_dir=temp_path / "home",
                ),
                **run_kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_phase = _extract_timeout_phase(exc.output)
            details = {"timeout_phase": timeout_phase} if timeout_phase else {}
            message = (
                f"semantic validation timed out during {timeout_phase}"
                if timeout_phase
                else "semantic validation timed out"
            )
            return SemanticValidationResult(
                ok=False,
                status=SemanticStatus.critical,
                message=message,
                details=details,
            )

    try:
        payload = _extract_semantic_payload(proc.stdout)
    except ValueError:
        stdout_text = _decode_process_output(proc.stdout)
        stderr_text = _decode_process_output(proc.stderr)
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message="semantic validation subprocess did not return a result payload",
            details={
                "returncode": proc.returncode,
                "stdout_chars": len(stdout_text),
                "stderr_chars": len(stderr_text),
            },
        )

    if not isinstance(payload, dict):
        return SemanticValidationResult(
            ok=False,
            status=SemanticStatus.critical,
            message=f"semantic validation returned {type(payload).__name__}, expected object",
        )

    payload_details = payload.get("details")
    return SemanticValidationResult(
        ok=bool(payload.get("ok", False)),
        status=SemanticStatus(str(payload.get("status", SemanticStatus.critical))),
        message=str(payload.get("message", "")),
        metric_name=(
            str(payload["metric_name"]) if payload.get("metric_name") is not None else None
        ),
        metric_value=(
            float(payload["metric_value"]) if payload.get("metric_value") is not None else None
        ),
        details=dict(payload_details) if isinstance(payload_details, dict) else {},
    )


def _decode_process_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _extract_semantic_payload(stdout: str | bytes | None) -> dict[str, Any]:
    text = _decode_process_output(stdout)
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if not stripped.startswith(_JSON_SENTINEL):
            continue
        stripped = stripped[len(_JSON_SENTINEL) :]
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("semantic payload is invalid JSON") from exc
        if isinstance(payload, dict):
            return payload
    raise ValueError("semantic payload not found")


def _extract_timeout_phase(output: object) -> str | None:
    if output is None:
        return None
    if isinstance(output, bytes):
        text = output.decode("utf-8", errors="replace")
    else:
        text = str(output)
    phase: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(_PHASE_SENTINEL):
            candidate = stripped[len(_PHASE_SENTINEL) :].strip()
            if candidate:
                phase = candidate
    return phase

"""Sandboxed PyMC semantic-validation harness.

The _pymc_semantics_subprocess module reads code, data and sampler parameters
and calls run_harness in a fresh process. The harness applies AST safety,
executes in a restricted namespace and scores held-out ELPD. ArviZ measures
split R-hat, BFMI, bulk and tail ESS, divergences, PSIS-LOO Pareto-k and finite
ELPD; evaluate_sampler_reliability records its verdict alongside the score.

The parent scorer invokes the module by its dotted path to keep PyMC and
ArviZ imports inside the subprocess.
"""

from __future__ import annotations

import ast
import warnings
from typing import Any

from ppl_synth_data.validation.pymc_semantics import (
    _PHASE_SENTINEL,
    MAX_PARETO_K,
    SamplerDiagnostics,
    evaluate_sampler_reliability,
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"\s*ArviZ is undergoing a major refactor.*",
        category=FutureWarning,
    )
    import arviz as az
import numpy as np
import pymc as pm


def _from_wire_value(value: Any) -> Any:
    """Reverse the parent process's _to_wire_value encoding.

    Only values marked with the __ndarray__ sentinel become NumPy arrays. Plain lists, scalars
    and strings pass through unchanged.
    """
    if isinstance(value, dict) and set(value) == {"__ndarray__"}:
        return np.array(value["__ndarray__"])
    return value


def _from_wire_data(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    return {key: _from_wire_value(value) for key, value in data.items()}


def _flatten_log_likelihood_group(group):
    arrays = []
    observed_names = []
    for name in group.data_vars:
        arr = np.asarray(group[name].values, dtype=np.float64)
        if arr.ndim < 2:
            raise ValueError(
                f"log_likelihood[{name}] must include chain/draw axes, got shape {arr.shape}"
            )
        arrays.append(arr.reshape(arr.shape[0], arr.shape[1], -1))
        observed_names.append(name)
    if not arrays:
        raise ValueError("no observed variables found in log_likelihood output")
    return np.concatenate(arrays, axis=2), observed_names


def _loo_all_observed(idata):
    """Apply az.loo to all observed variables stacked along the observation axis.

    One observed variable is passed through unchanged. Multiple observed variables are combined
    into one log-likelihood group so az.loo receives an unambiguous variable.
    """
    if len(idata.log_likelihood.data_vars) == 1:
        return az.loo(idata, pointwise=True)
    matrix, _ = _flatten_log_likelihood_group(idata.log_likelihood)
    stacked = az.from_dict(
        posterior={name: idata.posterior[name].values for name in idata.posterior.data_vars},
        log_likelihood={"all_observed": matrix},
    )
    return az.loo(stacked, pointwise=True)


def _pointwise_log_mean_exp(log_lik_matrix):
    safe = np.where(np.isfinite(log_lik_matrix), log_lik_matrix, -1e6)
    flattened = safe.reshape(-1, safe.shape[-1])
    column_max = flattened.max(axis=0, keepdims=True)
    stable = flattened - column_max
    return np.log(np.mean(np.exp(stable), axis=0)) + column_max[0]


def _reduce_over_free_vars(dataset, reducer):
    """Reduce a per-variable ArviZ diagnostic Dataset to one number.

    Mirrors PyMC's own reduction in `pymc.stats.convergence.run_convergence_checks`
    (`max(val.max() for val in rhat.values())`). Non-finite values propagate on
    purpose: an R-hat or ESS that came back NaN for a free variable means the
    sampler never moved that variable, and the check it feeds must fail rather
    than skip it.
    """
    per_variable = []
    for array in dataset.data_vars.values():
        values = np.asarray(array.values, dtype=np.float64)
        if values.size:
            per_variable.append(float(reducer(values)))
    if not per_variable:
        return None
    return float(reducer(np.asarray(per_variable, dtype=np.float64)))


def _collect_sampler_diagnostics(idata, model, loo):
    """Measure the seven reliability diagnostics on a completed posterior fit.

    Every diagnostic comes from a library function (`az.rhat`, `az.ess`,
    `az.bfmi`, the sampler's own divergence count, `az.loo`) because a subtly
    wrong R-hat is worse than no R-hat. A diagnostic that raises is recorded as
    unavailable with its exception, which the gate counts as a failed check.
    """
    posterior = idata.posterior
    chains = int(posterior.sizes["chain"])
    draws_per_chain = int(posterior.sizes["draw"])
    errors: dict[str, str] = {}
    values: dict[str, Any] = {}

    free_var_names = [rv.name for rv in model.free_RVs if rv.name in posterior]
    if not free_var_names:
        message = "no free model variables found in the posterior group"
        errors["split_r_hat"] = message
        errors["ess_bulk"] = message
        errors["ess_tail"] = message
    else:
        try:
            values["split_r_hat_max"] = _reduce_over_free_vars(
                az.rhat(idata, var_names=free_var_names, method="rank"), np.max
            )
        except Exception as exc:
            errors["split_r_hat"] = f"{type(exc).__name__}: {exc}"
        for name, method in (("ess_bulk_min", "bulk"), ("ess_tail_min", "tail")):
            try:
                values[name] = _reduce_over_free_vars(
                    az.ess(idata, var_names=free_var_names, method=method), np.min
                )
            except Exception as exc:
                errors[name.removesuffix("_min")] = f"{type(exc).__name__}: {exc}"

    try:
        bfmi = np.asarray(az.bfmi(idata), dtype=np.float64).ravel()
        values["bfmi_min"] = float(np.min(bfmi)) if bfmi.size else None
        if not bfmi.size:
            errors["bfmi"] = "sampler reported no per-chain energy"
    except Exception as exc:
        errors["bfmi"] = f"{type(exc).__name__}: {exc}"

    try:
        values["divergences"] = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    except Exception as exc:
        errors["no_divergences"] = f"{type(exc).__name__}: {exc}"

    pareto_k = np.asarray(loo.pareto_k, dtype=np.float64).ravel()
    if pareto_k.size:
        values["pareto_k_max"] = float(np.max(pareto_k))
    else:
        errors["pareto_k"] = "PSIS-LOO returned no pointwise Pareto-k values"

    values["elpd_loo"] = float(loo.elpd_loo)

    return SamplerDiagnostics(
        chains=chains,
        draws_per_chain=draws_per_chain,
        errors=errors,
        **values,
    )


def _exception_text(exc):
    # Kept in details["exception"] for visible-phase failures only, so failures can be
    # classified; the message stays
    # sanitised (type name only), and a held-out-phase failure keeps no exception text.
    return f"{type(exc).__name__}: {exc}"


def _payload(status, message, metric_name=None, metric_value=None, details=None, ok=True):
    return {
        "ok": ok,
        "status": status,
        "message": message,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "details": details or {},
    }


def _phase(name):
    print(_PHASE_SENTINEL + name, flush=True)


SAFE_BUILTINS = {
    # PyMC/PyTensor performs internal imports during graph construction. The
    # submitted source still cannot use imports: the AST gate rejects import
    # statements and direct dunder-name references before execution.
    "__import__": __import__,
    "Exception": Exception,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class AllowedNumpy:
    pass


ALLOWED_NUMPY = AllowedNumpy()
for _name in (
    "abs",
    "all",
    "any",
    "arange",
    "argmin",
    "argsort",
    "array",
    "asarray",
    "ceil",
    "clip",
    "column_stack",
    "concatenate",
    "exp",
    "float32",
    "float64",
    "floor",
    "full",
    "int64",
    "isfinite",
    "linspace",
    "log",
    "max",
    "mean",
    "min",
    "nan_to_num",
    "newaxis",
    "ones",
    "pi",
    "power",
    "ptp",
    "quantile",
    "round",
    "sqrt",
    "std",
    "sum",
    "unique",
    "where",
    "zeros",
):
    # Pure NumPy computations available to submitted programs.
    # np.empty stays out: it returns uninitialised memory, a possible read of held-out values in
    # this process.
    setattr(ALLOWED_NUMPY, _name, getattr(np, _name))


FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_ATTRIBUTES = {
    "__bases__",
    "__builtins__",
    "__class__",
    "__closure__",
    "__code__",
    "__dict__",
    "__func__",
    "__globals__",
    "__getattribute__",
    "__mro__",
    "__subclasses__",
    "fromfile",
    "genfromtxt",
    "load",
    "loadtxt",
    "memmap",
    "mro",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
}


def _unsafe_syntax_reason(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"syntax parse failed: {type(exc).__name__}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import statements are not allowed in submitted model code"
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                return f"unsafe name in submitted model code: {node.id}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                return f"unsafe attribute in submitted model code: {node.attr}"
    return None


def run_harness(
    *,
    code: str,
    visible_data: dict[str, Any],
    hidden_data: dict[str, Any] | None,
    draws: int,
    tune: int,
) -> dict[str, Any]:
    CODE = code
    VISIBLE_DATA = _from_wire_data(visible_data)
    HIDDEN_DATA = _from_wire_data(hidden_data)

    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__ppl_synth_submission__",
        "pm": pm,
        "np": ALLOWED_NUMPY,
        "numpy": ALLOWED_NUMPY,
    }
    unsafe_reason = _unsafe_syntax_reason(CODE)
    if unsafe_reason is not None:
        return _payload("critical", unsafe_reason, ok=False)
    try:
        _phase("exec")
        exec(CODE, namespace)  # noqa: S102 - isolated subprocess execution
    except Exception as exc:
        return _payload(
            "critical",
            f"exec failed: {type(exc).__name__}",
            details={"exception": _exception_text(exc)},
            ok=False,
        )

    model_fn = namespace.get("model")
    if not callable(model_fn):
        return _payload("critical", "model is not callable after exec", ok=False)

    try:
        _phase("model_visible")
        train_model = model_fn(VISIBLE_DATA)
    except Exception as exc:
        return _payload(
            "critical",
            f"model(visible_data) raised: {type(exc).__name__}",
            details={"exception": _exception_text(exc)},
            ok=False,
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\s*ArviZ is undergoing a major refactor.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"The effect of Potentials on other parameters is ignored "
                    r"during prior predictive sampling\..*"
                ),
                category=UserWarning,
            )
            with train_model:
                _phase("posterior_sampling")
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    # Two chains is the floor for a between-chain diagnostic.
                    # Given one chain, `az.rhat` warns "Shape validation
                    # failed: input_shape: (1, N), minimum_shape: (chains=2,
                    # draws=4)" and returns NaN for every variable, so split
                    # R-hat (one of the seven reliability checks below) is
                    # unmeasurable. Kept sequential (cores=1) so the whole fit
                    # stays inside this one resource-limited process.
                    chains=2,
                    cores=1,
                    progressbar=False,
                    # PyMC's own convergence pass only emits warnings and
                    # discards the numbers, so it cannot gate anything. The
                    # seven diagnostics are measured and recorded explicitly
                    # after sampling instead, and drive the reliability verdict.
                    compute_convergence_checks=False,
                    random_seed=0,
                    init="adapt_diag",
                    jitter_max_retries=1,
                    n_init=100,
                    idata_kwargs={"log_likelihood": True},
                    compile_kwargs={"mode": "FAST_COMPILE"},
                )
    except Exception as exc:
        return _payload(
            "critical",
            f"posterior sampling failed: {type(exc).__name__}",
            details={"exception": _exception_text(exc)},
            ok=False,
        )

    try:
        _phase("training_log_likelihood")
        loo = _loo_all_observed(idata)
        train_matrix, observed_names = _flatten_log_likelihood_group(idata.log_likelihood)
    except Exception as exc:
        return _payload(
            "critical",
            f"training log-likelihood scoring failed: {type(exc).__name__}",
            details={"exception": _exception_text(exc)},
            ok=False,
        )

    try:
        _phase("sampler_diagnostics")
        diagnostics = _collect_sampler_diagnostics(idata, train_model, loo)
    except Exception as exc:
        return _payload(
            "critical",
            f"sampler diagnostics failed: {type(exc).__name__}",
            details={"exception": _exception_text(exc)},
            ok=False,
        )
    reliability = evaluate_sampler_reliability(diagnostics)

    pareto_k = np.asarray(loo.pareto_k, dtype=np.float64).ravel()
    pareto_k_max = float(np.max(pareto_k)) if pareto_k.size else 0.0
    details = {
        "train_elpd_loo": float(loo.elpd_loo),
        "pareto_k_max": pareto_k_max,
        "pareto_k_warning": bool(pareto_k_max > MAX_PARETO_K),
        "sampler_diagnostics": diagnostics.as_payload(),
        "reliability": reliability.as_payload(),
        "n_draws": int(train_matrix.shape[0] * train_matrix.shape[1]),
        "visible_n_obs": int(train_matrix.shape[-1]),
        "observed_variables": observed_names,
        "hidden_data_present": HIDDEN_DATA is not None,
        "sampler": "pymc_nuts_adapt_diag_fast_compile",
        "sampler_compile_mode": "FAST_COMPILE",
    }

    metric_name = "train_elpd_loo"
    metric_value = float(loo.elpd_loo)

    if HIDDEN_DATA is not None:
        try:
            _phase("model_hidden")
            hidden_model = model_fn(HIDDEN_DATA)
            _phase("holdout_log_likelihood")
            holdout_group = pm.compute_log_likelihood(
                idata,
                model=hidden_model,
                extend_inferencedata=False,
                progressbar=False,
                compile_kwargs={"mode": "FAST_COMPILE"},
            )
            holdout_matrix, _ = _flatten_log_likelihood_group(holdout_group)
            holdout_pointwise = _pointwise_log_mean_exp(holdout_matrix)
            details["avg_holdout_logp"] = float(np.mean(holdout_pointwise))
            details["holdout_pointwise"] = [float(v) for v in holdout_pointwise]
            details["hidden_n_obs"] = int(holdout_matrix.shape[-1])
            metric_name = "avg_holdout_logp"
            metric_value = float(details["avg_holdout_logp"])

            # Scale reference + likelihood-family detection. Read the visible
            # target straight from the inference data's observed group (keyed by
            # the observed RV name), so it still works when the data-dict key
            # differs from the RV name.
            target_is_continuous = False
            try:
                target_name = observed_names[0] if observed_names else None
                if target_name is not None and "observed_data" in idata.groups():
                    obs_group = idata.observed_data
                    if target_name in obs_group:
                        target_vals = np.asarray(obs_group[target_name].values, dtype=float).ravel()
                        if target_vals.size > 1:
                            details["data_std"] = float(np.std(target_vals))
                        # Integer-valued targets are counts / binary; the Gaussian
                        # degeneracy inversion below does not apply to them.
                        target_is_continuous = not np.allclose(target_vals, np.round(target_vals))
            except Exception as exc:
                details["data_scale_error"] = type(exc).__name__

            # Degeneracy signal for the density gate, computed without extra
            # sampling. For a Gaussian likelihood, per-point log predictive
            # density obeys logp <= -0.5*log(2*pi*sigma^2), so
            #   sigma >= sqrt(exp(-2*logp)/(2*pi)).
            # A near-zero implied scale at the sharpest held-out point is a
            # sigma->0 degeneracy that can be selected by a max-score gate.
            # Only emitted for continuous targets: the inversion is meaningless
            # for Poisson / Bernoulli / categorical likelihoods and would
            # falsely flag healthy count or binary models.
            if target_is_continuous and holdout_pointwise.size:
                max_pointwise = float(np.max(holdout_pointwise))
                exponent = float(np.clip(-2.0 * max_pointwise, -700.0, 700.0))
                details["implied_min_pred_std"] = float(np.sqrt(np.exp(exponent) / (2.0 * np.pi)))
        except Exception as exc:
            return _payload(
                "critical",
                f"holdout log-likelihood scoring failed: {type(exc).__name__}",
                metric_name="train_elpd_loo",
                metric_value=float(loo.elpd_loo),
                details=details,
                ok=False,
            )
    else:
        details["hidden_n_obs"] = 0

    # The score and the reliability verdict stay separate: the metric is
    # reported as measured, and the status carries the reliability decision so a
    # consumer can see a good number came off a posterior that did not converge.
    flags = []
    if HIDDEN_DATA is None:
        flags.append("ran without hidden holdout data")
    if not reliability.passed:
        flags.append(
            f"unreliable sampling, {reliability.passed_count} of "
            f"{len(reliability.checks)} reliability checks passed and "
            f"{reliability.required_count} are required "
            f"(failing: {', '.join(reliability.failing_names)})"
        )
    status = "flag" if flags else "pass"
    message = (
        "semantic validation flagged: " + "; ".join(flags)
        if flags
        else "semantic validation passed"
    )

    return _payload(
        status,
        message,
        metric_name=metric_name,
        metric_value=metric_value,
        details=details,
        ok=True,
    )

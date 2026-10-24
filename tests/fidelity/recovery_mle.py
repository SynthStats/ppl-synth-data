"""Independent frequentist likelihoods. No production-module imports.

The numerical optimizer uses a reversible change of coordinates; reported beta
and its covariance are transformed back to the original raw input units.
NB2 is fitted as dispersion kappa=1/alpha, including the Poisson boundary.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


class UndefinedCoding(ValueError):
    pass


def design_columns(spec):
    columns = []
    for index, feature in enumerate(spec["features"]):
        law, name = feature["law"], feature["name"]
        if law["dist"] == "fixed_counts" and not (
            all(type(v) in (int, float) for v in law["levels"])
            or all(isinstance(v, str) for v in law["levels"])
        ):
            raise UndefinedCoding(
                f"/features/{index}/law/levels: mixed numeric and string levels "
                f"have no declared coding"
            )
        if law["dist"] == "categorical" or (
            law["dist"] == "fixed_counts" and all(isinstance(v, str) for v in law["levels"])
        ):
            columns.extend((f"{name}={level}", name, level) for level in law["levels"][1:])
        elif law["dist"] in {
            "normal",
            "truncated_normal",
            "discrete_uniform",
            "bernoulli",
            "fixed_counts",
        }:
            columns.append((name, name, None))
        else:
            raise UndefinedCoding(f"/features/{index}/law/dist: unsupported input law")
    expected = {name for name, _, _ in columns}
    if set(spec["response"]["coefficients"]) != expected:
        raise UndefinedCoding(
            
                '/response/coefficients: keys do not match raw'
                ' numeric / categorical-reference contract'
            
        )
    return columns


def design_matrix(spec, rows):
    columns = design_columns(spec)
    return np.column_stack(
        [np.ones(len(rows))]
        + [
            np.array(
                [row[feature] if level is None else float(row[feature] == level) for row in rows],
                dtype=float,
            )
            for _, feature, level in columns
        ]
    )


def parameter_names(spec):
    response = spec["response"]
    names = ["intercept_log_rate" if "intercept_log_rate" in response else "intercept"]
    names += [name for name, _, _ in design_columns(spec)]
    if response["family"] == "regression_continuous":
        names += ["noise_sigma"]
    if response.get("count_distribution") == "nb2":
        names += ["nb2_inverse_alpha"]
    if response["family"] == "zero_inflated_count":
        names += ["zero_inflation"]
    return names


def targets(spec):
    r = spec["response"]
    values = [r.get("intercept", r.get("intercept_log_rate"))]
    values += [r["coefficients"][name] for name, _, _ in design_columns(spec)]
    if r["family"] == "regression_continuous":
        values += [r["noise_sigma"]]
    if r.get("count_distribution") == "nb2":
        values += [1 / r["nb2_alpha"]]
    if r["family"] == "zero_inflated_count":
        values += [r["zero_inflation"]]
    return np.array(values)


@dataclass
class Fit:
    estimates: np.ndarray
    standard_errors: np.ndarray
    optimizer: str


def count_logpmf(y, eta, kappa):
    mu = np.exp(eta)
    poisson = y * eta - mu - gammaln(y + 1)
    # Taylor expansion is smooth at kappa=0, needed for the observed information
    # at a Poisson-boundary MLE. It is selected by the expansion's dimensionless
    # argument, not by the declared truth.
    if abs(kappa) * max(float(np.max(y)), float(np.max(mu)), 1.0) < 0.01:
        c1 = (y * (y - 1) - 2 * y * mu + mu**2) / 2
        c2 = -y * (y - 1) * (2 * y - 1) / 12 + y * mu**2 / 2 - mu**3 / 3
        c3 = (y * (y - 1)) ** 2 / 12 - y * mu**3 / 3 + mu**4 / 4
        c4 = -(y - 1) * y * (2 * y - 1) * (3 * y**2 - 3 * y - 1) / 120 + y * mu**4 / 4 - mu**5 / 5
        return poisson + kappa * (c1 + kappa * (c2 + kappa * (c3 + kappa * c4)))
    if kappa <= 0:
        return np.full(len(y), -np.inf)
    alpha = 1 / kappa
    return (
        gammaln(y + alpha)
        - gammaln(alpha)
        - gammaln(y + 1)
        + y * (eta + np.log(kappa))
        - (y + alpha) * np.log1p(mu * kappa)
    )


def fit(spec, x, y):
    """Estimate every response parameter; truth is never an optimizer input."""
    family = spec["response"]["family"]
    n, p = x.shape
    means, scales = x[:, 1:].mean(axis=0), x[:, 1:].std(axis=0)
    if np.any(scales == 0):
        raise ValueError("unidentified constant predictor")
    z = np.column_stack([np.ones(n), (x[:, 1:] - means) / scales])
    if np.linalg.matrix_rank(z) != p:
        raise ValueError("rank-deficient declared design")
    transform = np.eye(p)
    transform[0, 1:] = -means / scales
    transform[1:, 1:] = np.diag(1 / scales)
    if family == "regression_continuous":
        gamma = np.linalg.lstsq(z, y, rcond=None)[0]
        sigma = np.sqrt(np.mean((y - z @ gamma) ** 2))
        cov = transform @ np.linalg.inv(z.T @ z) @ transform.T * sigma**2
        return Fit(
            np.r_[transform @ gamma, sigma],
            np.r_[np.sqrt(np.diag(cov)), sigma / np.sqrt(2 * n)],
            "OLS Gaussian MLE",
        )
    nb = spec["response"].get("count_distribution") == "nb2"
    zi = family == "zero_inflated_count"
    start = np.zeros(p + nb + zi)
    start[0] = np.log(max(np.mean(y), 0.01))
    if family == "binary_choice":
        rate = np.clip(y.mean(), 1 / (n + 1), n / (n + 1))
        start[0] = np.log(rate / (1 - rate))
    if nb:
        start[p] = max((y.var() - y.mean()) / max(y.mean() ** 2, 0.01), 0.1)
    if zi:
        start[-1] = min(float(np.mean(y == 0)) / 2, 0.5)

    def objective(theta):
        eta = z @ theta[:p]
        if not np.isfinite(eta).all() or np.max(eta) > 700:
            return np.inf
        if family == "binary_choice":
            return float(np.mean(np.logaddexp(0, eta) - y * eta))
        logp = count_logpmf(y, eta, theta[p] if nb else 0.0)
        if zi:
            pi = theta[-1]
            zero = y == 0
            logp = logp.copy()
            logp[zero] = np.log(pi + (1 - pi) * np.exp(logp[zero]))
            logp[~zero] += np.log1p(-pi)
        return -float(np.mean(logp))

    bounds = [(None, None)] * p + ([(0.0, None)] if nb else []) + ([(0.0, 1 - 1e-9)] if zi else [])
    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 2000, "maxls": 50},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"MLE did not converge: {result.message}")
    theta = result.x
    d = len(theta)
    steps = np.full(d, 1e-4)
    # Count dispersion may be exactly zero. A very small symmetric extension of
    # the analytic boundary series is only used to estimate curvature.
    if nb:
        steps[p] = min(1e-4, 0.001 / max(float(y.max()), float(np.exp(z @ theta[:p]).max()), 1))
    basis = np.diag(steps)
    hessian = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            hessian[i, j] = (
                objective(theta + basis[i] + basis[j])
                - objective(theta + basis[i] - basis[j])
                - objective(theta - basis[i] + basis[j])
                + objective(theta - basis[i] - basis[j])
            ) / (4 * steps[i] * steps[j])
    if not np.isfinite(hessian).all() or np.linalg.eigvalsh(hessian).min() <= 0:
        raise ValueError("non-positive observed information: inference is not certified")
    back = np.eye(d)
    back[:p, :p] = transform
    covariance = back @ np.linalg.inv(n * hessian) @ back.T
    return Fit(back @ theta, np.sqrt(np.diag(covariance)), str(result.message))

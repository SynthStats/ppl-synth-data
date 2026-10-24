"""Render the realized regression DGPs without accepting program expressions."""

import math

import numpy as np

from ppl_synth_data.specs.models import CategoricalInput, FixedCountsInput, FourFamilySpec


def render_declared_gold(
    spec: FourFamilySpec,
    visible: dict | None = None,
) -> tuple[str, dict]:
    """Render a four-family gold using the visible rows to autoscale its priors."""
    if not isinstance(spec, FourFamilySpec):
        raise TypeError("declared gold requires a FourFamilySpec; template EnvSpec is unsupported")
    if visible is None:
        raise ValueError("four-family declared gold requires visible data to autoscale priors")
    return _render_four_family_gold(spec, visible)


_AUTOSCALE_SOURCE = "https://mc-stan.org/rstanarm/articles/priors.html"


def _render_four_family_gold(spec: FourFamilySpec, visible: dict) -> tuple[str, dict]:
    """Infer free parameters while preserving the declared conditional law.

    Priors are rstanarm's default autoscaled priors, computed once from the visible rows and
    written into the program as float literals; nothing reads the declared parameter values.
    """
    spec = FourFamilySpec.model_validate(spec.model_dump(mode="json"))
    response = spec.response
    required = [feature.name for feature in spec.features] + ["y"]
    missing = sorted(set(required) - set(visible))
    if missing:
        raise ValueError(f"visible data missing columns: {missing}")
    for name in required:
        values = np.asarray(visible[name])
        if values.dtype.kind in "biuf" and not np.isfinite(values).all():
            raise ValueError(f"visible data column {name!r} must be finite")
    lines = ["def model(data):", "    y = np.asarray(data['y'])"]
    coefficient_keys: list[str] = []
    columns = []
    for feature in spec.features:
        expression_values = f"np.asarray(data[{feature.name!r}])"
        visible_values = np.asarray(visible[feature.name])
        if feature.uses_reference_coding and isinstance(
            feature.law, (CategoricalInput, FixedCountsInput)
        ):
            terms = [
                (
                    f"{feature.name}={level}",
                    f"({expression_values} == {level!r})",
                    visible_values == level,
                )
                for level in feature.law.levels[1:]
            ]
        else:
            terms = [(feature.name, expression_values, visible_values)]
        for key, expression, column in terms:
            lines.append(f"    x_{len(coefficient_keys)} = {expression}")
            coefficient_keys.append(key)
            columns.append(np.asarray(column, dtype=float))
    centers = [float(np.mean(column)) for column in columns]
    x_scales = [float(np.std(column, ddof=1)) for column in columns]
    if response.family == "regression_continuous":
        y_visible = np.asarray(visible["y"], dtype=float)
        y_location, y_scale = float(np.mean(y_visible)), float(np.std(y_visible, ddof=1))
    else:
        y_location, y_scale = 0.0, 1.0
    if not all(math.isfinite(value) for value in [*centers, *x_scales, y_location, y_scale]):
        raise ValueError("visible data prior statistics must be finite")
    if y_scale == 0.0:
        raise ValueError("visible Gaussian response is constant; its prior scale is undefined")
    if response.family in {"regression_count", "zero_inflated_count"}:
        mean_count = float(np.mean(np.asarray(visible["y"], dtype=float)))
        if mean_count <= 0.0:
            raise ValueError(
                "visible counts have a zero or negative mean; log(mean count) is undefined"
            )
        y_location = math.log(mean_count)
    constant_in_visible = []
    for index, key in enumerate(coefficient_keys):
        if x_scales[index] == 0.0:
            x_scales[index] = 1.0
            constant_in_visible.append(key)
    intercept_sigma = 2.5 * y_scale
    coefficient_sigmas = [2.5 * y_scale / scale for scale in x_scales]
    noise_rate = 1.0 / y_scale
    constants = [y_location, intercept_sigma, *coefficient_sigmas, noise_rate]
    if not all(math.isfinite(value) for value in constants):
        raise ValueError("visible data prior constants must be finite")
    raw_intercept = "intercept_centered" + "".join(
        f" - coef_{index} * {centers[index]!r}" for index in range(len(coefficient_keys))
    )
    lines += [
        "    with pm.Model() as result:",
        f"        intercept_centered = pm.Normal('intercept_centered', "
        f"mu={y_location!r}, sigma={intercept_sigma!r})",
    ]
    for index in range(len(coefficient_keys)):
        lines.append(
            f"        coef_{index} = pm.Normal('coef_{index}', mu=0.0, "
            f"sigma={coefficient_sigmas[index]!r})"
        )
    lines.append(f"        intercept = pm.Deterministic('intercept', {raw_intercept})")
    lines.append(
        "        linear = intercept"
        + "".join(f" + coef_{index} * x_{index}" for index in range(len(coefficient_keys)))
    )
    if response.family == "regression_continuous":
        lines += [
            f"        sigma = pm.Exponential('sigma', lam={noise_rate!r})",
            "        pm.Normal('y', mu=linear, sigma=sigma, observed=y)",
        ]
    elif response.family == "binary_choice":
        lines.append("        pm.Bernoulli('y', logit_p=linear, observed=y)")
    else:
        lines.append("        mu = pm.math.exp(linear)")
        arguments = "mu=mu"
        if response.count_distribution == "nb2":
            lines.append("        alpha = pm.HalfNormal('alpha', sigma=10.0)")
            arguments += ", alpha=alpha"
            likelihood = "NegativeBinomial"
        else:
            likelihood = "Poisson"
        if response.family == "zero_inflated_count":
            lines.append("        zero_inflation = pm.Beta('zero_inflation', alpha=1.0, beta=1.0)")
            arguments += ", psi=1.0 - zero_inflation"
            likelihood = "ZeroInflated" + likelihood
        lines.append(f"        pm.{likelihood}('y', {arguments}, observed=y)")
    lines.append("    return result")
    return "\n".join(lines) + "\n", {
        "provenance": "declared-dgp-rendered",
        "classification": "case-1-typed-values-fixed-skeleton",
        "family": response.family,
        "coefficient_keys": coefficient_keys,
        "interpolation_points": {
            "columns": "validated feature names serialized with repr",
            "levels": "validated categorical or fixed-count levels serialized with repr",
            "likelihood": "closed response family and count distribution",
            "prior_constants": "finite floats computed from the visible rows, serialized with repr",
        },
        "prior_autoscale": {
            "rule": (
                f"rstanarm default autoscaled priors ({_AUTOSCALE_SOURCE}), computed once from "
                "the visible rows: the free intercept is the centered intercept, "
                "Normal(my, 2.5 * sy); each coefficient is Normal(0, 2.5 * sy / sx); the "
                "Gaussian noise scale is Exponential(1 / sy). For regression_continuous "
                "my = mean(y) and sy = sd(y); binary choice has my = 0 and sy = 1; count families "
                "have sy = 1 and my as stated below. "
                "sx is the standard deviation of the term's visible column (the indicator "
                "column for a reference-coded level). Standard deviations use ddof=1; a "
                "column that is constant in the visible rows uses sx = 1. The raw-scale "
                "intercept is the Deterministic intercept_centered - sum(coef * center). "
                "For count families the centered intercept's prior location is "
                "log(mean visible count), as rstanarm centers a Gaussian intercept at mean(y). "
                "This keeps the starting point at the visible mean, because PyMC's "
                "zero-inflated NB density underflows at mu = 1, and the validator cannot "
                "score a model with an initval."
            ),
            "y_location": y_location,
            "y_scale": y_scale,
            "centers": centers,
            "x_scales": x_scales,
            "constant_in_visible": constant_in_visible,
            "visible_rows": len(np.asarray(visible["y"])),
        },
        "parameter_treatment": (
            "Infer the centered intercept and independent raw-scale coefficients; the raw-scale "
            "intercept is a Deterministic of them; infer Gaussian noise, NB2 shape or "
            "structural-zero probability where declared. Numeric fixed-count levels use raw "
            "values; string and categorical levels use reference coding that omits the first "
            "level. No scaling of the data and no rate clipping. Intercept, coefficient and "
            "Gaussian noise priors are autoscaled from the visible rows (see prior_autoscale), "
            "never from the generating parameter values or held-out rows; the NB2 shape and "
            "structural-zero priors are fixed skeleton choices."
        ),
    }

"""Poisson and NB2 response draws using the caller's observation RNG.

This kernel does not choose feature laws, sample predictors, or fit a gold.
NB2 uses mean mu and shape alpha, with variance mu + mu**2 / alpha.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ppl_synth_data.specs.models import CountDeclaration, CountResponse, ZeroInflatedResponse


def count_mean(
    response: CountResponse | CountDeclaration | ZeroInflatedResponse, inputs: Mapping[str, float]
) -> float:
    """Evaluate the log link on named raw inputs, without clipping or scaling."""
    if inputs.keys() != response.coefficients.keys():
        raise ValueError("/features: query keys must equal /response/coefficients keys")
    for name, value in inputs.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"/features/{name}: query value must be a finite number")
    eta = response.intercept_log_rate + math.fsum(
        coefficient * inputs[name] for name, coefficient in response.coefficients.items()
    )
    try:
        mean = math.exp(eta)
    except OverflowError as exc:
        raise ValueError("/response: count mean exceeds finite numeric range") from exc
    if not math.isfinite(mean):
        raise ValueError("/response: count mean must be finite")
    return mean


def sample_count(
    response: CountResponse | CountDeclaration | ZeroInflatedResponse,
    inputs: Mapping[str, float],
    rng: np.random.Generator,
    *,
    size: int | None = None,
) -> int | np.ndarray:
    """Draw independent responses conditional on one supplied feature vector."""
    mean = count_mean(response, inputs)
    if response.count_distribution == "nb2":
        alpha = response.nb2_alpha
        assert alpha is not None
        return rng.negative_binomial(alpha, alpha / (alpha + mean), size=size)
    return rng.poisson(mean, size=size)

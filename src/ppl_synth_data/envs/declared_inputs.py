"""Draw an entire declared design before the visible/hidden partition."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm

from ppl_synth_data.specs.models import FourFamilySpec


def sample_inputs(spec: FourFamilySpec, *, seed: int) -> list[dict[str, float | int | str]]:
    """Sample independent feature streams and preserve exact design allocations."""
    n = spec.data_plan.n_observations
    streams = np.random.SeedSequence(seed).spawn(len(spec.features))
    columns: dict[str, list] = {}
    for index, (feature, stream) in enumerate(zip(spec.features, streams, strict=True)):
        rng = np.random.default_rng(stream)
        values: NDArray
        law = feature.law
        if law.dist == "normal":
            values = rng.normal(law.mean, law.sd, size=n)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"/features/{index}/law: sampled values exceed finite numeric range"
                )
        elif law.dist == "truncated_normal":
            a = -np.inf if law.low is None else (law.low - law.mean) / law.sd
            b = np.inf if law.high is None else (law.high - law.mean) / law.sd
            values = truncnorm.rvs(a, b, loc=law.mean, scale=law.sd, size=n, random_state=rng)
            if (
                not np.isfinite(values).all()
                or (law.low is not None and np.any(values < law.low))
                or (law.high is not None and np.any(values > law.high))
            ):
                raise ValueError(
                    f"/features/{index}/law: truncated draws exceed declared numeric support"
                )
        elif law.dist == "discrete_uniform":
            values = rng.integers(law.low, law.high, endpoint=True, size=n)
        elif law.dist == "bernoulli":
            values = rng.binomial(1, law.p, size=n)
        elif law.dist == "categorical":
            values = rng.choice(law.levels, size=n, p=law.probs)
        elif law.dist == "fixed_counts":
            # Object dtype preserves integer and string levels without coercion.
            values = np.repeat(np.asarray(law.levels, dtype=object), law.counts)
            rng.shuffle(values)
        else:
            raise ValueError(f"/features/{index}/law: unsupported input law {law.dist!r}")
        columns[feature.name] = values.tolist()
    return [{name: values[row] for name, values in columns.items()} for row in range(n)]

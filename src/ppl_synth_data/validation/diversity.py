"""Structural diversity measurement for PyMC programs.

A ShapeSignature normalizes names and constants in the program's AST.
Programs with the same structure share a signature; count and continuous
likelihoods, mixtures, hierarchies and censoring yield distinct signatures.
shape_diversity reports the effective number of structural modes in a corpus.

The analysis reads the AST without executing PyMC or Docker.
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from dataclasses import dataclass

# Distribution-name -> structural flag. A program carrying any of these names is
# flagged with the corresponding structural feature.
_FLAG_BY_DIST: dict[str, str] = {
    "Mixture": "mixture",
    "NormalMixture": "mixture",
    "Censored": "censored",
    "Truncated": "censored",
    "TruncatedNormal": "censored",
    "StudentT": "robust",
    "GaussianRandomWalk": "timeseries",
    "AR": "timeseries",
    "GARCH11": "timeseries",
    "NegativeBinomial": "overdispersed_count",
    "MvNormal": "multivariate",
    "MvStudentT": "multivariate",
    "Dirichlet": "multivariate",
    "Multinomial": "multivariate",
}

# Likelihood family buckets, so e.g. all the count likelihoods group together as
# the observed-likelihood discriminator.
_CONTINUOUS = {
    "Normal",
    "LogNormal",
    "StudentT",
    "Cauchy",
    "HalfCauchy",
    "HalfNormal",
    "Exponential",
    "Gamma",
    "InverseGamma",
    "Beta",
    "Uniform",
    "Laplace",
    "Weibull",
    "TruncatedNormal",
    "Truncated",
    "SkewNormal",
    "Gumbel",
}
_COUNT = {
    "Poisson",
    "NegativeBinomial",
    "Binomial",
    "BetaBinomial",
    "Geometric",
    "ZeroInflatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedBinomial",
}
_BINARY = {"Bernoulli"}
_CATEGORICAL = {"Categorical", "Multinomial", "OrderedLogistic"}


def _likelihood_bucket(name: str) -> str:
    if name.startswith("ZeroInflated"):
        return "count"
    if name in _COUNT:
        return "count"
    if name in _BINARY:
        return "binary"
    if name in _CATEGORICAL:
        return "categorical"
    if name in _CONTINUOUS:
        return "continuous"
    if name in ("MvNormal", "MvStudentT"):
        return "multivariate"
    return "other"


def _rv_count_bucket(n: int) -> str:
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"


@dataclass(frozen=True)
class ShapeSignature:
    """A program's structural mode, normalized of names and constants."""

    observed_likelihood: str  # bucket: continuous / count / binary / categorical / ...
    structural_flags: tuple[str, ...]  # sorted: mixture / censored / hierarchy / ...
    rv_count_bucket: str  # 1-2 / 3-5 / 6+
    has_hierarchy: bool

    def key(self) -> str:
        flags = ",".join(self.structural_flags) or "-"
        return (
            f"{self.observed_likelihood}|{flags}|rv={self.rv_count_bucket}"
            f"|hier={int(self.has_hierarchy)}"
        )


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _has_kwarg(node: ast.Call, *names: str) -> bool:
    return any(kw.arg in names for kw in node.keywords if kw.arg is not None)


def _is_gp(node: ast.Call) -> bool:
    # pm.gp.Marginal(...) / pm.gp.Latent(...) -> func is Attribute whose value is
    # an Attribute named "gp".
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "gp"
    )


def program_shape(code: str) -> ShapeSignature | None:
    """Extract the structural signature of a PyMC ``def model(data)`` program.

    Returns None if the code does not parse. Names and numeric constants are
    ignored by construction; only the structural use of distributions matters.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    dist_names: list[str] = []
    observed_dist: str | None = None
    n_prior_rvs = 0
    flags: set[str] = set()
    has_hierarchy = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_gp(node):
            flags.add("gp")
            continue
        name = _call_name(node)
        if name is None:
            continue
        if name in _FLAG_BY_DIST:
            flags.add(_FLAG_BY_DIST[name])
        if name.startswith("ZeroInflated"):
            flags.add("zero_inflated")
        # Distribution calls are PascalCase pm.* constructors with a string first
        # arg (the RV name) to identify distribution structure.
        is_dist = (
            bool(name)
            and name[0].isupper()
            and name
            not in (
                "Model",
                "Data",
                "Deterministic",
                "Potential",
                "MutableData",
                "ConstantData",
            )
        )
        if not is_dist:
            continue
        dist_names.append(name)
        if _has_kwarg(node, "observed"):
            observed_dist = name
        else:
            n_prior_rvs += 1
            # group-level / hierarchical effects usually carry a shape/dims kwarg.
            if _has_kwarg(node, "shape", "dims"):
                has_hierarchy = True

    if not dist_names:
        return None

    if has_hierarchy:
        flags.add("hierarchy")

    bucket = _likelihood_bucket(observed_dist) if observed_dist else "unknown"
    return ShapeSignature(
        observed_likelihood=bucket,
        structural_flags=tuple(sorted(flags)),
        rv_count_bucket=_rv_count_bucket(n_prior_rvs),
        has_hierarchy=has_hierarchy,
    )


def shape_diversity(codes: list[str]) -> dict[str, object]:
    """Effective number of distinct structural modes in a corpus of programs.

    ``effective_shapes`` = exp(Shannon entropy of the shape distribution): the
    number of equally-common modes the corpus is "worth". A corpus of 200
    programs that are all plain regression has effective_shapes ~ 1, regardless
    of how varied their prose is.
    """
    shapes = [s for s in (program_shape(c) for c in codes) if s is not None]
    n = len(shapes)
    hist = Counter(s.key() for s in shapes)
    if n == 0:
        return {
            "n_parsed": 0,
            "n_unparsed": len(codes),
            "unique_shapes": 0,
            "shape_entropy": 0.0,
            "effective_shapes": 0.0,
            "top_shape_share": 0.0,
            "histogram": {},
        }
    probs = [c / n for c in hist.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return {
        "n_parsed": n,
        "n_unparsed": len(codes) - n,
        "unique_shapes": len(hist),
        "shape_entropy": entropy,
        "effective_shapes": math.exp(entropy),
        "top_shape_share": max(hist.values()) / n,
        "histogram": dict(hist.most_common()),
    }

"""Tests for structural (semantic) diversity measurement."""

from __future__ import annotations

from ppl_synth_data.validation.diversity import (
    program_shape,
    shape_diversity,
)

_NORMAL_REG = """
def model(data):
    x = np.asarray(data["x"]); y = np.asarray(data["y"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1); b = pm.Normal("b", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("obs", mu=a * x + b, sigma=sigma, observed=y)
    return m
"""

# Same STRUCTURE, different nouns/names/constants -> must collapse to one mode.
_NORMAL_REG_NOUN_SWAP = """
def model(data):
    temp = np.asarray(data["temperature"]); rev = np.asarray(data["revenue"])
    with pm.Model() as m:
        slope = pm.Normal("slope", 5, 9); intercept = pm.Normal("intercept", 2, 3)
        noise = pm.HalfNormal("noise", 4)
        pm.Normal("revenue_obs", mu=slope * temp + intercept, sigma=noise, observed=rev)
    return m
"""

_POISSON_COUNT = """
def model(data):
    x = np.asarray(data["x"]); y = np.asarray(data["count"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1); b = pm.Normal("b", 0, 1)
        pm.Poisson("obs", mu=pm.math.exp(a * x + b), observed=y)
    return m
"""

_BERNOULLI = """
def model(data):
    x = np.asarray(data["x"]); y = np.asarray(data["choice"])
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1); b = pm.Normal("b", 0, 1)
        pm.Bernoulli("obs", p=pm.math.sigmoid(a * x + b), observed=y)
    return m
"""

_MIXTURE = """
def model(data):
    y = np.asarray(data["y"])
    with pm.Model() as m:
        w = pm.Dirichlet("w", [1, 1])
        mu = pm.Normal("mu", 0, 1, shape=2); sigma = pm.HalfNormal("sigma", 1, shape=2)
        pm.NormalMixture("obs", w=w, mu=mu, sigma=sigma, observed=y)
    return m
"""

_HIERARCHICAL = """
def model(data):
    g = np.asarray(data["group"]); y = np.asarray(data["y"])
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        a = pm.Normal("a", mu, 1, shape=5)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("obs", mu=a[g], sigma=sigma, observed=y)
    return m
"""


def test_noun_swap_collapses_to_one_shape():
    s1 = program_shape(_NORMAL_REG)
    s2 = program_shape(_NORMAL_REG_NOUN_SWAP)
    assert s1 is not None and s2 is not None
    assert s1.key() == s2.key()  # the whole point: structure, not names


def test_likelihood_family_discriminates():
    assert program_shape(_NORMAL_REG).observed_likelihood == "continuous"
    assert program_shape(_POISSON_COUNT).observed_likelihood == "count"
    assert program_shape(_BERNOULLI).observed_likelihood == "binary"


def test_distinct_likelihoods_are_distinct_shapes():
    keys = {program_shape(c).key() for c in (_NORMAL_REG, _POISSON_COUNT, _BERNOULLI)}
    assert len(keys) == 3


def test_mixture_flag_detected():
    s = program_shape(_MIXTURE)
    assert "mixture" in s.structural_flags


def test_hierarchy_detected():
    s = program_shape(_HIERARCHICAL)
    assert s.has_hierarchy is True
    assert "hierarchy" in s.structural_flags


def test_unparseable_code_is_none():
    assert program_shape("def model(data): this is not python !!!") is None


def test_diversity_of_homogeneous_corpus_is_near_one():
    # 10 plain regressions over different nouns -> effectively ONE mode.
    corpus = [_NORMAL_REG, _NORMAL_REG_NOUN_SWAP] * 5
    div = shape_diversity(corpus)
    assert div["unique_shapes"] == 1
    assert div["effective_shapes"] < 1.01
    assert div["top_shape_share"] == 1.0


def test_diversity_of_varied_corpus_is_higher():
    corpus = [_NORMAL_REG, _POISSON_COUNT, _BERNOULLI, _MIXTURE, _HIERARCHICAL]
    div = shape_diversity(corpus)
    assert div["unique_shapes"] >= 4
    assert div["effective_shapes"] > 3.0


def test_unparsed_counted_separately():
    div = shape_diversity([_NORMAL_REG, "not python !!!"])
    assert div["n_parsed"] == 1
    assert div["n_unparsed"] == 1

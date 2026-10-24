from __future__ import annotations

from ppl_synth_data.validation.leakage import (
    detect_banned_terms,
    is_public_source,
    public_source_or_sentinel,
)


def test_flags_distribution_name() -> None:
    prompt = "Model these counts with a Poisson process and estimate lambda."
    hits = detect_banned_terms(prompt, banned_terms=["poisson", "negative binomial"])
    assert "poisson" in hits


def test_scanner_accepts_phenomenon_description() -> None:
    prompt = "Observed daily counts are non-negative integers with occasional spikes."
    hits = detect_banned_terms(prompt, banned_terms=["poisson", "negative binomial"])
    assert hits == []


def test_scanner_matches_hyphenated_multiword_terms() -> None:
    prompt = "Use a negative-binomial style count model."
    hits = detect_banned_terms(prompt, banned_terms=["negative binomial"])
    assert hits == ["negative binomial"]


def test_flags_modeling_directives() -> None:
    prompt = "You should fit a regression to predict the outcome."
    hits = detect_banned_terms(prompt, banned_terms=["fit a regression", "hierarchical model"])
    assert "fit a regression" in hits


def test_flags_ppl_identifiers() -> None:
    prompt = "Define the model using pm.Normal for the likelihood."
    hits = detect_banned_terms(prompt, banned_terms=["pm.normal", "poisson"])
    assert "pm.normal" in hits


def test_scanner_accepts_non_modeling_data_descriptions() -> None:
    """Phenomenon terms should not be treated as modelling directives."""
    prompt = (
        "The counts are right-skewed with many zeros and the variance "
        "appears roughly proportional to the mean across groups. "
        "The equipment is usually run under normal operating conditions."
    )
    hits = detect_banned_terms(
        prompt,
        banned_terms=[
            "poisson",
            "normal distribution",
            "normal likelihood",
            "zero-inflated",
            "fit a regression",
        ],
    )
    assert hits == []


def test_flags_normal_only_when_used_as_modeling_family() -> None:
    prompt = "Use a normal likelihood for the continuous measurement."
    hits = detect_banned_terms(
        prompt,
        banned_terms=["normal likelihood", "normal distribution"],
    )
    assert hits == ["normal likelihood"]


_PRIVATE_PATH = "/private/non_public/source-plan/synthetic-supervision-transcript.md"


def test_public_url_is_public_source() -> None:
    assert is_public_source("https://github.com/pymc-devs/pymc-examples")
    assert is_public_source("http://probmods.org/")


def test_sentinel_is_safe_source_value() -> None:
    assert is_public_source("not_applicable_invented")


def test_local_paths_are_not_public_sources() -> None:
    assert not is_public_source(_PRIVATE_PATH)
    assert not is_public_source("file:///Users/example/private.md")
    assert not is_public_source("~/notes/private.md")
    assert not is_public_source("relative/path/notes.md")
    assert not is_public_source("")


def test_public_source_or_sentinel_strips_private_path() -> None:
    assert public_source_or_sentinel(_PRIVATE_PATH) == "not_applicable_invented"
    assert public_source_or_sentinel("file:///x.md") == "not_applicable_invented"


def test_public_source_or_sentinel_keeps_public_url() -> None:
    url = "https://github.com/pymc-devs/pymc-examples"
    assert public_source_or_sentinel(url) == url


def test_public_source_or_sentinel_passes_none() -> None:
    assert public_source_or_sentinel(None) is None

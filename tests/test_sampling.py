from __future__ import annotations

from ppl_synth_data.specs.registry import SUPPORTED_FAMILIES
from ppl_synth_data.specs.sampling import DOMAINS, sample_domain_family


def test_sample_domain_family_is_deterministic() -> None:
    a = sample_domain_family(12345)
    b = sample_domain_family(12345)
    assert a == b


def test_sample_domain_family_values_are_in_supported_sets() -> None:
    domain, family = sample_domain_family(987)
    assert domain in DOMAINS
    assert family in SUPPORTED_FAMILIES

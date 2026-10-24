from __future__ import annotations

import pytest

from ppl_synth_data.specs.registry import SUPPORTED_FAMILIES, validate_family


def test_validate_family_supported() -> None:
    for fam in SUPPORTED_FAMILIES:
        validate_family(fam)


def test_validate_family_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported generative_family"):
        validate_family("made_up_family")


def test_supported_families_non_empty() -> None:
    assert len(SUPPORTED_FAMILIES) > 0
    assert all(isinstance(f, str) for f in SUPPORTED_FAMILIES)

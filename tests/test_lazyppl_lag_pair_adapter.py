"""LazyPPL hidden-holdout lag-pair construction.

When visible row fields are yPrev and yVal and hidden data supplies a single
numeric series y, the builder derives yPrev=y[:-1] and yVal=y[1:].
The tests check the returned Haskell module.
"""

from __future__ import annotations

import pytest

from ppl_synth_data.validation import lazyppl_semantics

# A minimal seed snippet whose only Row field accesses are ``yPrev r`` and
# ``yVal r`` (field order of first appearance: yPrev, then yVal).
_LAG_MODEL_CODE = "logLikRow st r = normalLogPdf (yPrev r) 1.0 (yVal r)"

# Variants that access only one of the two lag fields.
_LAG_MODEL_CODE_YVAL_ONLY = "logLikRow st r = normalLogPdf (yVal r) 1.0 0.0"
_LAG_MODEL_CODE_YPREV_ONLY = "logLikRow st r = normalLogPdf (yPrev r) 1.0 0.0"


def test_build_hidden_taskdata_derives_lag_pairs_from_y_series() -> None:
    """Behaviour 1: yPrev/yVal derived from a single ``y`` series, n-1 rows."""
    hidden = lazyppl_semantics._build_hidden_taskdata_from_sample_data(
        _LAG_MODEL_CODE,
        hidden_data={"y": [0.5, 1.5, -2.0, 3.25]},
        row_fields={"yPrev", "yVal"},
    )

    assert hidden.count("Row {") == 3
    assert "Row { yPrev = 0.5, yVal = 1.5 }" in hidden
    assert "Row { yPrev = 1.5, yVal = -2.0 }" in hidden
    assert "Row { yPrev = -2.0, yVal = 3.25 }" in hidden


def test_build_hidden_taskdata_derives_lag_pairs_when_only_yval_accessed() -> None:
    """Behaviour 2 (yVal-only): both columns derived, only yVal emitted."""
    hidden = lazyppl_semantics._build_hidden_taskdata_from_sample_data(
        _LAG_MODEL_CODE_YVAL_ONLY,
        hidden_data={"y": [0.5, 1.5, -2.0, 3.25]},
        row_fields={"yPrev", "yVal"},
    )

    assert hidden.count("Row {") == 3
    assert "Row { yVal = 1.5 }" in hidden
    assert "Row { yVal = -2.0 }" in hidden
    assert "Row { yVal = 3.25 }" in hidden
    assert "yPrev" not in hidden


def test_build_hidden_taskdata_derives_lag_pairs_when_only_yprev_accessed() -> None:
    """Behaviour 2 (yPrev-only): both columns derived, only yPrev emitted."""
    hidden = lazyppl_semantics._build_hidden_taskdata_from_sample_data(
        _LAG_MODEL_CODE_YPREV_ONLY,
        hidden_data={"y": [0.5, 1.5, -2.0, 3.25]},
        row_fields={"yPrev", "yVal"},
    )

    assert hidden.count("Row {") == 3
    assert "Row { yPrev = 0.5 }" in hidden
    assert "Row { yPrev = 1.5 }" in hidden
    assert "Row { yPrev = -2.0 }" in hidden
    assert "yVal" not in hidden


def test_build_hidden_taskdata_uses_explicit_yprev_yval_columns_when_present() -> None:
    """Behaviour 3 (guard): explicit yPrev/yVal columns win, no derivation."""
    hidden = lazyppl_semantics._build_hidden_taskdata_from_sample_data(
        _LAG_MODEL_CODE,
        hidden_data={"n": 2, "yPrev": [10.0, 20.0], "yVal": [11.0, 21.0]},
        row_fields={"yPrev", "yVal"},
    )

    assert hidden.count("Row {") == 2
    assert "Row { yPrev = 10.0, yVal = 11.0 }" in hidden
    assert "Row { yPrev = 20.0, yVal = 21.0 }" in hidden


def test_build_hidden_taskdata_lag_pair_derivation_requires_at_least_two_y_values() -> None:
    """Behaviour 4: a ``y`` series shorter than 2 values fails closed, naming y."""
    with pytest.raises(ValueError, match=r"\by\b"):
        lazyppl_semantics._build_hidden_taskdata_from_sample_data(
            _LAG_MODEL_CODE,
            hidden_data={"y": [0.5]},
            row_fields={"yPrev", "yVal"},
        )


def test_build_hidden_taskdata_raises_missing_field_when_no_y_series_available() -> None:
    """Behaviour 5 (guard): no yPrev/yVal and no y -> existing missing-field error."""
    with pytest.raises(ValueError, match="missing field"):
        lazyppl_semantics._build_hidden_taskdata_from_sample_data(
            _LAG_MODEL_CODE,
            hidden_data={"n": 1, "other": [1.0]},
            row_fields={"yPrev", "yVal"},
        )

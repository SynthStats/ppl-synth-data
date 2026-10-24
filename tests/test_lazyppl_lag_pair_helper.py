"""Direct contracts of the held-out lag-pair helper.

The generated-module seam in test_lazyppl_lag_pair_adapter.py emits only the columns a program
reads, so it cannot observe the mapping the helper returns or the caller's input mapping.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from ppl_synth_data.validation import lazyppl_semantics

_LAG_FIELDS = {"yPrev", "yVal"}


def test_lag_pairs_replace_raw_y_with_adjacent_observations() -> None:
    """The helper drops raw y and keeps the direction of adjacent pairs."""
    y = [2.5, -1.0, 7.25, 0.5]
    result = lazyppl_semantics._hidden_data_with_lag_pairs({"y": y}, accessed=_LAG_FIELDS)

    assert "y" not in result
    assert result["n"] == len(y) - 1
    assert result["yPrev"] == y[:-1]
    assert result["yVal"] == y[1:]


def test_lag_pairs_leave_input_mapping_unchanged() -> None:
    """Creating lag pairs neither consumes nor rewrites the caller's held-out data."""
    hidden_data = {"y": [2.5, -1.0, 7.25, 0.5], "n": 99, "metadata": {"segment": [1, 2]}}
    before = deepcopy(hidden_data)

    lazyppl_semantics._hidden_data_with_lag_pairs(hidden_data, accessed=_LAG_FIELDS)

    assert hidden_data == before


def test_lag_pairs_reject_empty_y() -> None:
    """An empty held-out series fails with an error naming its field."""
    with pytest.raises(ValueError, match=r"\by\b"):
        lazyppl_semantics._hidden_data_with_lag_pairs({"y": []}, accessed=_LAG_FIELDS)


def test_lag_pairs_replace_existing_n_with_pair_count() -> None:
    """An existing row count does not override the number of derived lag pairs."""
    y = [2.5, -1.0, 7.25, 0.5]
    result = lazyppl_semantics._hidden_data_with_lag_pairs({"y": y, "n": 99}, accessed=_LAG_FIELDS)

    assert result["n"] == len(y) - 1

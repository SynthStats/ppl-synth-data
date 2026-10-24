"""A candidate's local bindings are not Row fields.

The builders retain parsed field accesses only when the names are columns
of the supplied data, excluding local helpers and preceding arguments.
"""

import pytest

from ppl_synth_data.validation.lazyppl_semantics import (
    _build_hidden_taskdata_from_sample_data,
    _build_taskdata_from_sample_data,
)

LET_BOUND_HELPER = """
model :: [Row] -> Meas Double
model rows = do
  mu <- sample (normal 0 1)
  let scoreRow r = score (normalPdf mu 1 (value r))
  mapM_ scoreRow rows
  return mu
"""

ARGUMENT_BEFORE_ROW = """
model :: [Row] -> Meas Double
model rows = do
  mu <- sample (normal 0 1)
  let initial = State mu
  mapM_ (\\row -> score (exp (logLikRow initial row))) rows
  return mu

logLikRow :: State -> Row -> Double
logLikRow st r = log (normalPdf (sMu st) 1 (value r))
"""


def test_a_let_bound_helper_is_not_a_row_field():
    hs = _build_taskdata_from_sample_data(LET_BOUND_HELPER, sample_data={"value": [1.0, 2.0, 3.0]})
    assert "scoreRow" not in hs
    assert "value" in hs


def test_an_argument_before_the_row_is_not_a_row_field():
    hs = _build_taskdata_from_sample_data(ARGUMENT_BEFORE_ROW, sample_data={"value": [1.0, 2.0]})
    assert "initial" not in hs
    assert "value" in hs


def test_the_hidden_builder_ignores_the_candidates_own_names_too():
    hs = _build_hidden_taskdata_from_sample_data(
        LET_BOUND_HELPER, hidden_data={"value": [0.5, 1.5]}, row_fields={"value"}
    )
    assert "scoreRow" not in hs
    assert "value" in hs


def test_a_program_that_reads_no_data_column_still_fails_loudly():
    code = "model rows = do\n  let helper r = 1.0\n  return 0\n"
    with pytest.raises(ValueError, match="at least one Row field access"):
        _build_taskdata_from_sample_data(code, sample_data={"value": [1.0]})


def test_a_visible_column_missing_from_the_held_out_data_still_raises():
    with pytest.raises(ValueError, match="missing field 'value'"):
        _build_hidden_taskdata_from_sample_data(
            LET_BOUND_HELPER, hidden_data={"other": [1.0]}, row_fields={"value"}
        )

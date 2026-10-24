"""Row-field parsing ignores Haskell comments."""

from __future__ import annotations

import pytest

from ppl_synth_data.harbor.taskdata_gen import _strip_haskell_comments, parse_seed_row_fields

_PROGRAM_WITH_PROSE_COMMENTS = """
module Submission where

import LazyPPL
import TaskData (Row(..))

-- Log likelihood for a single row given the parameters.
-- We treat each row as one observation; the row count is n.
{- Block comment mentioning the row and an observation row
   across two lines -}
model :: [Row] -> Meas Double
model rows = do
  lam <- sample $ gamma 2 2
  mapM_ (\\r -> score (poissonPdf lam (count r))) rows  -- one row at a time
  return lam
"""


def test_comment_prose_does_not_become_a_row_field() -> None:
    names = [f["name"] for f in parse_seed_row_fields(_PROGRAM_WITH_PROSE_COMMENTS)]
    assert names == ["count"]


def test_operator_containing_dashes_is_not_a_comment() -> None:
    source = "f = xVal r --> yVal r\n"
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["xVal", "yVal"]


def test_dashes_inside_string_literal_do_not_start_a_comment() -> None:
    source = 'label = "-- single row" ++ show (count r)\n'
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["count"]


def test_nested_block_comments_are_removed_as_a_whole() -> None:
    source = "{- outer {- inner row -} still comment: single row -}\ng = xVal r\n"
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["xVal"]


def test_line_comment_hides_a_block_opener_on_the_same_line() -> None:
    source = "-- note {-\ng = xVal r\n-- -}\n"
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["xVal"]


def test_char_literal_dash_is_not_a_comment() -> None:
    source = "sep = '-'\ng = yVal r -- trailing single row\n"
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["yVal"]


def test_block_comment_is_whitespace_between_tokens() -> None:
    source = "g = count{- observation -} r\n"
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["count"]


def test_block_comment_preserves_each_newline() -> None:
    source = "count{- first\nsecond\n-} r"
    assert _strip_haskell_comments(source) == "count \n\n r"


def test_unterminated_string_names_its_offset() -> None:
    with pytest.raises(ValueError, match=r"unterminated string literal at offset 8"):
        parse_seed_row_fields('label = "-- single row')


def test_unterminated_block_comment_names_its_offset() -> None:
    with pytest.raises(ValueError, match=r"unterminated block comment at offset 11"):
        parse_seed_row_fields("g = count r{- never closed")


def test_escaped_quote_literals_do_not_open_comments() -> None:
    source = r"""
apostrophe = '\''
quoted = "\""
g = count r -- prose row
"""
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["count"]


def test_string_gap_does_not_expose_comment_text() -> None:
    source = r"""
label = "prefix\
    \-- single row"
g = count r
"""
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["count"]


def test_unicode_operator_containing_dashes_is_not_a_comment() -> None:
    source = "f = xVal r ⊕-- yVal r\n"
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["xVal", "yVal"]


@pytest.mark.parametrize("literal", [r"'\DEL'", r"'\x41'", r"'\65'", r"'\n'"])
def test_named_and_numeric_char_escapes_do_not_open_comments(literal: str) -> None:
    source = f"character = {literal}\ng = count r -- prose row\n"
    names = [f["name"] for f in parse_seed_row_fields(source)]
    assert names == ["count"]

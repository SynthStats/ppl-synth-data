"""Generate per-task TaskData.hs modules for the LazyPPL Harbor pipeline.

Each seed program imports TaskData (Row(..)) and accesses specific fields on
Row values.  This module parses those field references from seed source code
and produces a complete Haskell module that defines the Row data type and
embeds observation data as a literal list.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Patterns that capture Row field access in Haskell seed programs.
# Covers:  fieldName r,  fieldName row,  r.fieldName  (GHC OverloadedRecordDot)
_FIELD_ACCESS_RE = re.compile(
    r"""
    (?:
        (?<![a-zA-Z0-9_])          # not preceded by identifier char
        ([a-z][a-zA-Z0-9_]*)       # capture group 1: field name (camelCase)
        [ \t]+                     # same-line whitespace
        (?:r\b|row\b|r\)|row\))    # applied to r or row (possibly before closing paren)
    |
        (?:r|row)\.([a-z][a-zA-Z0-9_]*)  # capture group 2: record dot syntax
    )
    """,
    re.VERBOSE,
)

_LOCAL_ROW_HELPER_SIG_RE = re.compile(r"(?m)^\s*([a-z][a-zA-Z0-9_]*)\s*::\s*Row\s*->")

_LOCAL_ROW_HELPER_BINDING_RE = re.compile(r"(?m)^\s*([a-z][a-zA-Z0-9_]*)[ \t]+(?:r|row|_)\b[^=]*=")
_LOCAL_BINDING_ARGS_RE = re.compile(
    r"(?m)^\s*(?P<name>[a-z][a-zA-Z0-9_]*)[ \t]+(?P<args>[^=\n]+?)\s*="
)
_LOCAL_GUARDED_BINDING_RE = re.compile(r"(?m)^\s*([a-z][a-zA-Z0-9_]*)[ \t]+[^=\n]+\n\s*\|")
_IDENTIFIER_RE = re.compile(r"\b[a-z][a-zA-Z0-9_]*\b")
_SIMPLE_HASKELL_ARGS_RE = re.compile(
    r"(?:[a-z_][a-zA-Z0-9_']*|_)(?:[ \t]+(?:[a-z_][a-zA-Z0-9_']*|_))*"
)

# Row field names that clash with Haskell keywords or common library names;
# we never extract these.
_IGNORE_NAMES = frozenset(
    {
        "return",
        "do",
        "let",
        "in",
        "where",
        "case",
        "of",
        "if",
        "then",
        "else",
        "module",
        "import",
        "data",
        "type",
        "class",
        "instance",
        "deriving",
        "score",
        "sample",
        "mapM_",
        "map",
        "const",
        "log",
        "exp",
        "max",
        "min",
        "length",
        "show",
        "read",
        "error",
        "fail",
        "print",
        "pure",
        "not",
        "otherwise",
        "fromIntegral",
        "abs",
        "signum",
        "sqrt",
        "pi",
        "sum",
        "head",
        "tail",
        "zip",
        "filter",
        "fst",
        "snd",
        "normalPdf",
        "normalLogPdf",
        "poissonPdf",
        "poissonLogPmf",
        "binomialLogPmf",
        "expPdf",
        "exponentialLogPdf",
        "gammaLogPdf",
        "bernoulliLogPmf",
        "sigmoid",
        # common parameter names in logLikRow signatures (not field accessors)
        "st",
        "state",
        "params",
        "s",
        "p",
        "model",
        "theta",
    }
)

# Haskell type defaults based on common field-name conventions.
_INTEGER_FIELD_NAMES = frozenset({"count"})
_INT_FIELD_NAMES = frozenset(
    {
        "nTrials",
        "nSuccesses",
        "timeIdx",
        "groupId",
        "subjectId",
        "attemptCount",
        "index",
        "idx",
        "t",
    }
)
_BOOL_FIELD_NAMES = frozenset(
    {
        "heads",
        "label",
        "isWeekday",
        "isAnomaly",
        "survived",
        "treated",
    }
)


def _infer_hs_type(field_name: str) -> str:
    """Pick a sensible Haskell type for a Row field based on its name."""
    if field_name in _INTEGER_FIELD_NAMES or field_name.endswith("_count"):
        return "Integer"
    if field_name in _INT_FIELD_NAMES:
        return "Int"
    if field_name.endswith("_id") or field_name.endswith("Id"):
        return "Int"
    if field_name in _BOOL_FIELD_NAMES:
        return "Bool"
    return "Double"


_HASKELL_SYMBOL_CHARS = frozenset("!#$%&*+./<=>?@\\^|~:-")


def _strip_haskell_comments(source: str) -> str:
    """Replace comments with whitespace and erase string contents.

    This is a small single-pass lexer because comments nest and both comment
    openers may occur inside literals. A comment becomes one space plus each
    newline it contained, while a string keeps only its delimiting quotes.
    """
    import unicodedata

    excluded_punctuation = frozenset("(),;[]`{}'\"")

    def is_symbol(char: str) -> bool:
        if char in _HASKELL_SYMBOL_CHARS:
            return True
        category = unicodedata.category(char)
        return category[0] in {"S", "P"} and char not in excluded_punctuation

    def char_literal_end(offset: int) -> int | None:
        content = offset + 1
        if content >= n or source[content] in "'\r\n":
            return None
        if source[content] != "\\":
            end = content + 1
            return end + 1 if end < n and source[end] == "'" else None

        escape = content + 1
        if escape >= n or source[escape] in "\r\n":
            return None
        if source[escape] == "^":
            end = escape + 2
        elif source[escape] == "x":
            end = escape + 1
            while end < n and source[end] in "0123456789abcdefABCDEF":
                end += 1
        elif source[escape] == "o":
            end = escape + 1
            while end < n and source[end] in "01234567":
                end += 1
        elif source[escape].isdigit():
            end = escape + 1
            while end < n and source[end].isdigit():
                end += 1
        elif source[escape].isalpha():
            end = escape + 1
            while end < n and source[end].isalpha():
                end += 1
        else:
            end = escape + 1
        return end + 1 if end < n and source[end] == "'" else None

    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == '"':
            start = i
            j = start + 1
            while j < n:
                if source[j] == '"':
                    break
                if source[j] == "\\":
                    j += 2
                else:
                    j += 1
            if j >= n:
                raise ValueError(f"unterminated string literal at offset {start}")
            out.append('""')
            i = j + 1
            continue
        if ch == "'":
            end = char_literal_end(i)
            if end is not None:
                out.append(source[i:end])
                i = end
                continue
        if source.startswith("{-", i):
            start = i
            depth = 1
            j = i + 2
            while j < n and depth:
                if source.startswith("{-", j):
                    depth += 1
                    j += 2
                elif source.startswith("-}", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth:
                raise ValueError(f"unterminated block comment at offset {start}")
            out.append(" ")
            out.extend(match.group() for match in re.finditer(r"\r\n|\r|\n", source[start:j]))
            i = j
            continue
        if source.startswith("--", i):
            j = i
            while j < n and source[j] == "-":
                j += 1
            preceded_by_symbol = i > 0 and is_symbol(source[i - 1])
            followed_by_symbol = j < n and is_symbol(source[j])
            if not preceded_by_symbol and not followed_by_symbol:
                while j < n and source[j] != "\n":
                    j += 1
                out.append(" ")
                i = j
                continue
            out.append(source[i:j])
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _find_locally_defined_row_helpers(seed_source: str) -> set[str]:
    """Return locally defined helper names that should not become Row fields.

    Some generated LazyPPL programs define helper functions like
    ``rowParkSize :: Row -> ParkSize`` and then call ``rowParkSize row``.
    Those are not TaskData accessors; treating them as fields creates
    conflicting record accessors in the generated ``TaskData.hs`` module.
    """
    local_helpers = set(_LOCAL_ROW_HELPER_SIG_RE.findall(seed_source))
    local_helpers.update(_LOCAL_ROW_HELPER_BINDING_RE.findall(seed_source))
    local_helpers.update(_LOCAL_GUARDED_BINDING_RE.findall(seed_source))
    for match in _LOCAL_BINDING_ARGS_RE.finditer(seed_source):
        name = match.group("name")
        args = match.group("args").strip()
        if name in _IGNORE_NAMES or not _SIMPLE_HASKELL_ARGS_RE.fullmatch(args):
            continue
        local_helpers.add(name)
        local_helpers.update(
            name for name in _IDENTIFIER_RE.findall(args) if name not in {"r", "row", "_"}
        )
    return local_helpers


def parse_seed_row_fields(seed_source: str) -> list[dict[str, str]]:
    """Extract Row field names and their Haskell types from a seed .hs file.

    Strategy: look for ``fieldName r`` / ``fieldName row`` patterns (the
    standard record-accessor application style in Haskell) and
    ``r.fieldName`` (OverloadedRecordDot).

    Returns a list of dicts ``[{"name": "xVal", "hs_type": "Double"}, ...]``
    ordered by first appearance, deduplicated.
    """
    seed_source = _strip_haskell_comments(seed_source)
    seen: set[str] = set()
    fields: list[dict[str, str]] = []
    local_helpers = _find_locally_defined_row_helpers(seed_source)

    for m in _FIELD_ACCESS_RE.finditer(seed_source):
        name = m.group(1) or m.group(2)
        if name is None or name in _IGNORE_NAMES or name in local_helpers or name in seen:
            continue
        seen.add(name)
        fields.append({"name": name, "hs_type": _infer_hs_type(name)})

    return fields


def _coerce_integral_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        parsed = float(value)
        if not parsed.is_integer():
            msg = f"value {value!r} is not integral"
            raise ValueError(msg) from None
        return int(parsed)


def _format_hs_value(value: float | int | str | bool, hs_type: str) -> str:
    """Format a Python value as a Haskell literal appropriate for *hs_type*."""
    if hs_type == "Bool":
        return "True" if value else "False"
    if hs_type == "Integer":
        return str(_coerce_integral_value(value))
    if hs_type == "Int":
        return str(_coerce_integral_value(value))
    if hs_type == "Double":
        fval = float(value)
        import math

        if math.isnan(fval):
            return "(0/0)"
        if math.isinf(fval):
            return "(1/0)" if fval > 0 else "(-(1/0))"
        return repr(fval)
    # fallback: treat as String
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def generate_taskdata_hs(
    columns: list[dict[str, str]],
    rows: list[dict[str, float | int | str]],
) -> str:
    """Generate a complete Haskell ``TaskData`` module.

    Parameters
    ----------
    columns:
        Field specifications, each ``{"name": ..., "hs_type": ...}``.
    rows:
        Observation data.  Keys must match the ``name`` entries in *columns*.

    Returns
    -------
    A string of valid Haskell source defining ``module TaskData``.
    """
    if not columns:
        msg = "columns must not be empty"
        raise ValueError(msg)

    lines: list[str] = [
        "module TaskData",
        "  ( Row(..)",
        "  , loadData",
        "  ) where",
        "",
    ]

    # data Row definition
    lines.append("data Row = Row")
    for i, col in enumerate(columns):
        prefix = "  { " if i == 0 else "  , "
        lines.append(f"{prefix}{col['name']} :: {col['hs_type']}")
    lines.append("  } deriving (Show)")
    lines.append("")

    # loadData function
    lines.append("loadData :: IO [Row]")
    if not rows:
        lines.append("loadData = return []")
    else:
        lines.append("loadData = return")
        for i, row in enumerate(rows):
            bracket = "  [ " if i == 0 else "  , "
            field_strs = [
                f"{col['name']} = {_format_hs_value(row[col['name']], col['hs_type'])}"
                for col in columns
            ]
            lines.append(f"{bracket}Row {{ {', '.join(field_strs)} }}")
        lines.append("  ]")

    lines.append("")  # trailing newline
    return "\n".join(lines)


def generate_hidden_taskdata_hs(
    columns: list[dict[str, str]],
    rows: list[dict[str, float | int | str]],
) -> str:
    """Generate a hidden holdout module using the public ``TaskData.Row`` type."""
    if rows and not columns:
        msg = "columns must not be empty when hidden rows are present"
        raise ValueError(msg)

    lines: list[str] = [
        "module HiddenTaskData",
        "  ( loadHiddenData",
        "  ) where",
        "",
        "import TaskData (Row(..))",
        "",
        "loadHiddenData :: IO [Row]",
    ]
    if not rows:
        lines.append("loadHiddenData = return []")
    else:
        lines.append("loadHiddenData = return")
        for i, row in enumerate(rows):
            bracket = "  [ " if i == 0 else "  , "
            field_strs = [
                f"{col['name']} = {_format_hs_value(row[col['name']], col['hs_type'])}"
                for col in columns
            ]
            lines.append(f"{bracket}Row {{ {', '.join(field_strs)} }}")
        lines.append("  ]")

    lines.append("")
    return "\n".join(lines)


def generate_taskdata_from_seed(
    seed_path: Path,
    data: list[dict[str, float | int | str]],
) -> str:
    """Parse fields from a seed file and generate the matching TaskData.hs.

    Convenience wrapper that combines :func:`parse_seed_row_fields` and
    :func:`generate_taskdata_hs`.
    """
    source = seed_path.read_text(encoding="utf-8")
    columns = parse_seed_row_fields(source)
    if not columns:
        msg = f"no Row fields found in {seed_path.name}"
        raise ValueError(msg)
    return generate_taskdata_hs(columns, data)

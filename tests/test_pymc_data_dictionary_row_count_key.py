"""The PyMC data dictionary of a task text lists every key of the dict that `model(data)` receives.

The verifier builds that dict as `{"n": number of rows}` plus one array per column. The schema
table of a task text has a row for `n`. The list under "Allowed `data` keys:" was built from the
columns alone, so it left `n` out and the text told the solver two different things. The
generator now writes the `n` line first in that list, and one rule adds the same line to a text
that was written before.
"""

from __future__ import annotations

import re
from pathlib import Path

from ppl_synth_data.harbor.export_pymc import add_row_count_key
from ppl_synth_data.pipeline.contracts import PYMC_ROW_COUNT_KEY_LINE
from ppl_synth_data.pipeline.io import read_jsonl
from ppl_synth_data.pipeline.stages import stage_d_prompts as stage_d
from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c
from tests.test_declared_stage_c import _prepared_paths

HEADING = "Allowed `data` keys:"

# The dictionary of one admitted task (earthworm counts), as it was written before the change.
OLD_TEXT = """\
## Data Description

We collected 256 soil cores over the past growing season.
## PyMC data dictionary

The verifier calls `def model(data)` with a Python dict built from the
exact materialized visible Stage C dataframe.

Observed target key: `y`.
Observed predictor keys:
- `tillage`
- `soil_moisture`

Allowed `data` keys:
- `tillage`
- `soil_moisture`
- `y`

Observed TaskData summaries from the exact rows used by the verifier:
- `tillage`: int64, min=0, q25=0, median=0.5, mean=0.5, q75=1, max=1, n=256.
- `y`: int64, min=0, q25=4, median=9, mean=10.8828, q75=14, max=59, n=256.

## Data Schema

| Key | Type | Description |
|-----|------|-------------|
| `n` | `int` | Number of observations |
| `tillage` | `np.ndarray` | Column from observed data |
"""


def _allowed_keys(text: str) -> list[str]:
    """Keys of the list that follows the heading directly, as an export gate reads them."""
    lines = text.splitlines()
    start = lines.index(HEADING)
    keys = []
    for line in lines[start + 1 :]:
        if not line.startswith("- `"):
            break
        keys.append(re.match(r"^- `([^`]+)`", line).group(1))
    return keys


def test_the_line_names_n_and_holds_no_dash_character():
    assert PYMC_ROW_COUNT_KEY_LINE.startswith("- `n` ")
    assert "—" not in PYMC_ROW_COUNT_KEY_LINE
    assert "–" not in PYMC_ROW_COUNT_KEY_LINE
    assert "\n" not in PYMC_ROW_COUNT_KEY_LINE
    # A dictionary summary line has the form "- `name`: int64, ..."; the key line must not
    # look like one.
    assert not re.match(r"- `[^`]+`: ", PYMC_ROW_COUNT_KEY_LINE)


def test_the_generated_dictionary_lists_n_before_the_columns(tmp_path: Path):
    paths, spec = _prepared_paths(tmp_path)
    run_stage_c(paths, seed=20260930, split_policy="random_row", visible_ratio=0.6)
    data_ref = read_jsonl(paths.stage_dirs["C"] / "data_refs.jsonl")[0]

    text = stage_d._pymc_data_schema_appendix(spec=spec, data_ref=data_ref, run_dir=paths.run_dir)

    lines = text.splitlines()
    assert lines[lines.index(HEADING) + 1] == PYMC_ROW_COUNT_KEY_LINE
    assert _allowed_keys(text)[0] == "n"
    assert set(_allowed_keys(text)[1:]) == {"month", "signal", "y"}
    # `n` is a key of the dict, not a predictor column.
    predictors = text.split("Observed predictor keys:", 1)[1].split(HEADING, 1)[0]
    assert "`n`" not in predictors


def test_the_rule_adds_the_line_first_and_changes_nothing_else():
    new_text = add_row_count_key(OLD_TEXT)

    assert _allowed_keys(new_text) == ["n", "tillage", "soil_moisture", "y"]
    assert new_text == OLD_TEXT.replace(
        HEADING + "\n", HEADING + "\n" + PYMC_ROW_COUNT_KEY_LINE + "\n", 1
    )
    # The predictor list above the heading is not the allowed-key list.
    assert new_text.count(PYMC_ROW_COUNT_KEY_LINE) == 1


def test_the_rule_is_idempotent():
    once = add_row_count_key(OLD_TEXT)

    assert add_row_count_key(once) == once


def test_a_text_without_the_list_is_returned_unchanged():
    text = "# Task: api latency\n\nThe table holds 3 requests.\n"

    assert add_row_count_key(text) == text


def test_a_list_that_already_names_n_is_returned_unchanged():
    # A table whose own column is called `n`: the list names the key already.
    text = OLD_TEXT.replace("- `soil_moisture`\n- `y`\n", "- `soil_moisture`\n- `n`\n- `y`\n")

    assert add_row_count_key(text) == text

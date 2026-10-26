"""Task text and row-count metadata in re-exported PyMC packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ppl_synth_data.pipeline.contracts import PYMC_ROW_COUNT_KEY_LINE
from tests.test_harbor_reexport_packages import (  # noqa: F401 - pytest fixtures
    TASK_ID,
    _package,
    _reexport,
)
from tests.test_harbor_reexport_packages import (
    recorded_fits as _shared_recorded_fits,
)
from tests.test_harbor_reexport_packages import (
    validator as _shared_validator,
)

validator = _shared_validator
del _shared_validator
recorded_fits = _shared_recorded_fits
del _shared_recorded_fits

TEXT = """\
# Task: api latency

## Data Description

The table holds 3 requests. Mean latency is 2.0 ms.
## PyMC data dictionary

Allowed `data` keys:
- `response_time_ms`
- `load`

Observed TaskData summaries from the exact rows used by the verifier:
- `load`: float64, min=0.1, q25=0.15, median=0.2, mean=0.2, q75=0.25, max=0.3, n=3.

## Data Schema

| Key | Type | Description |
|-----|------|-------------|
| `n` | `int` | Number of observations |
| `response_time_ms` | `np.ndarray` | Column from observed data |
| `load` | `np.ndarray` | Column from observed data |
"""
WITH_KEY = TEXT.replace(
    "Allowed `data` keys:\n", "Allowed `data` keys:\n" + PYMC_ROW_COUNT_KEY_LINE + "\n"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _source(root: Path, text: str = TEXT, task_id: str = TASK_ID) -> Path:
    task = _package(root, task_id)
    (task / "instruction.md").write_text(text)
    return task


def test_the_row_count_key_is_added_and_recorded(tmp_path, validator, recorded_fits):  # noqa: F811
    summary = _reexport(tmp_path, [_source(tmp_path / "old")], validator)

    assert (tmp_path / "new" / TASK_ID / "instruction.md").read_text() == WITH_KEY
    assert summary["instructions"] == {
        TASK_ID: {
            "source_sha256": _sha(TEXT),
            "sha256": _sha(WITH_KEY),
            "row_count_key_added": True,
        }
    }
    assert summary == json.loads((tmp_path / "new" / "export_summary.json").read_text())


def test_a_text_that_names_the_key_already_is_kept_byte_for_byte(
    tmp_path, validator, recorded_fits
):  # noqa: F811
    summary = _reexport(tmp_path, [_source(tmp_path / "old", WITH_KEY)], validator)

    assert (tmp_path / "new" / TASK_ID / "instruction.md").read_text() == WITH_KEY
    assert summary["instructions"][TASK_ID] == {
        "source_sha256": _sha(WITH_KEY),
        "sha256": _sha(WITH_KEY),
        "row_count_key_added": False,
    }

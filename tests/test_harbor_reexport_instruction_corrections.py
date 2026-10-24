"""Corrections to a task text when an existing package is exported again.

A re-export keeps a package's text byte for byte, with two exceptions that it records. One rule
adds the `n` key to the allowed-key list of the PyMC data dictionary. A table of exact
replacements, named by task, corrects a sentence that states something false about the supplied
rows. A replacement applies only when its `before` text occurs exactly once; otherwise the
export stops before any reference is fitted, so that a table written for another version of a
text cannot change the wrong sentence or silently do nothing.
"""

from __future__ import annotations

import hashlib
import hashlib as _h
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ppl_synth_data.cli import app
from ppl_synth_data.harbor import export_pymc
from ppl_synth_data.pipeline.contracts import PYMC_ROW_COUNT_KEY_LINE
from tests.test_harbor_reexport_packages import (  # noqa: F401 - pytest fixtures
    TASK_ID,
    _package,
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

OTHER_TASK = "pymc-held-task-00001"
TEXT = """\
# Task: api latency

## Data Description

The table holds 4 requests. Mean latency is 2.0 ms. Half of the requests are slow.
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
ROW_COUNT = {
    "before": "The table holds 4 requests.",
    "after": "The table holds the 3 supplied requests.",
}
SLOW_SHARE = {
    "before": "Half of the requests are slow.",
    "after": "One of the 3 supplied requests is slow.",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _source(root: Path, text: str = TEXT, task_id: str = TASK_ID) -> Path:
    task = _package(root, task_id)
    (task / "instruction.md").write_text(text)
    return task




def _fixture_role_manifest(tmp_path, sources):
    """A fixture manifest for these synthetic packages.

    The columns are read from each source's own visible split, so the manifest cannot miss a column
    the completeness check would then reject. The target is preserved, the row-count key is
    preserved under the sealed wording, and every other column is a replaced quantitative predictor.
    """
    import pandas as _pd

    packages = []
    for source in sources:
        source = Path(source)
        visible = _pd.read_parquet(source / "environment" / "workspace" / "data.parquet")
        # The re-export path reads the target from the package's own reference file, so the fixture
        # manifest must agree with it rather than assume "y".
        target = json.loads((source / "tests" / "elpd_reference.json").read_text())[(
                'target_name'
            )]
        columns = []
        # The export injects the row-count key into the data it hands the fit, so the
        # fixture manifest
        # must list it even where the parquet does not carry it. The sealed manifest
        # preserves it under
        # the same wording.
        names = sorted(set(visible.columns) | {"n"})
        for name in names:
            if name == target:
                columns.append(
                    {
                        "column": name,
                        "role": "TARGET",
                        "treatment": "preserve",
                        "reducer_type": None,
                    }
                )
            elif name == "n":
                columns.append(
                    {
                        "column": name,
                        "role": "UNUSED",
                        "treatment": "preserve (scalar row count, never replaced)",
                        "reducer_type": None,
                    }
                )
            else:
                columns.append(
                    {
                        "column": name,
                        "role": "PREDICTOR",
                        "treatment": "replace",
                        "reducer_type": "quantitative",
                    }
                )
        packages.append(
            {"package_id": source.name, "target": target, "columns": columns, "status": (
                'complete'
            )}
        )
    manifest = {
        "packages": packages,
        "package_count": len(packages),
        "complete": len(packages),
        "unresolved": 0,
    }
    path = tmp_path / "ROLE-MANIFEST.json"
    path.write_text(json.dumps(manifest))
    return path, _h.sha256(path.read_bytes()).hexdigest()


def _reexport(tmp_path, sources, validator, *, corrections=None, excluded=()):  # noqa: F811
    path, sha = validator
    manifest_path, manifest_sha = _fixture_role_manifest(tmp_path, sources)
    return export_pymc.reexport_harbor_tasks(
        sources,
        tmp_path / "new",
        validator_path=path,
        validator_sha256=sha,
        excluded=excluded,
        allow_unlocked_reference_runtime=True,
        instruction_corrections=corrections,
        role_manifest_path=manifest_path,
        role_manifest_sha256=manifest_sha,
        attempt_id="test-attempt",
    )


def test_the_row_count_key_is_added_and_recorded(tmp_path, validator, recorded_fits):  # noqa: F811
    summary = _reexport(tmp_path, [_source(tmp_path / "old")], validator)

    assert (tmp_path / "new" / TASK_ID / "instruction.md").read_text() == WITH_KEY
    assert summary["instructions"] == {
        TASK_ID: {
            "source_sha256": _sha(TEXT),
            "sha256": _sha(WITH_KEY),
            "row_count_key_added": True,
            "corrections_applied": 0,
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
        "corrections_applied": 0,
    }


def test_named_sentences_are_replaced_and_the_rest_is_unchanged(tmp_path, validator, recorded_fits):  # noqa: F811
    summary = _reexport(
        tmp_path,
        [_source(tmp_path / "old")],
        validator,
        corrections={TASK_ID: [ROW_COUNT, SLOW_SHARE]},
    )

    expected = WITH_KEY.replace(ROW_COUNT["before"], ROW_COUNT["after"]).replace(
        SLOW_SHARE["before"], SLOW_SHARE["after"]
    )
    assert (tmp_path / "new" / TASK_ID / "instruction.md").read_text() == expected
    assert summary["instructions"][TASK_ID] == {
        "source_sha256": _sha(TEXT),
        "sha256": _sha(expected),
        "row_count_key_added": True,
        "corrections_applied": 2,
    }
    # The source package is never written to.
    assert (tmp_path / "old" / TASK_ID / "instruction.md").read_text() == TEXT


@pytest.mark.parametrize(
    ("text", "occurrences"),
    [
        (TEXT.replace(ROW_COUNT["before"], "The table holds four requests."), 0),
        (TEXT.replace("Mean latency", ROW_COUNT["before"] + " Mean latency"), 2),
    ],
    ids=["no-match", "two-matches"],
)
def test_a_replacement_that_does_not_match_once_stops_the_export(
    tmp_path,
    validator,
    recorded_fits,
    text,
    occurrences,  # noqa: F811
):
    with pytest.raises(ValueError, match=TASK_ID) as refusal:
        _reexport(
            tmp_path,
            [_source(tmp_path / "old", text)],
            validator,
            corrections={TASK_ID: [ROW_COUNT]},
        )

    assert f"{occurrences} times" in str(refusal.value)
    assert ROW_COUNT["before"] in str(refusal.value)
    assert recorded_fits == []
    assert not (tmp_path / "new" / TASK_ID).exists()


def test_a_correction_for_a_task_that_is_not_exported_stops_the_export(
    tmp_path, validator, recorded_fits
):  # noqa: F811
    kept = _source(tmp_path / "old")
    held = _source(tmp_path / "old", task_id=OTHER_TASK)

    with pytest.raises(ValueError, match=OTHER_TASK):
        _reexport(
            tmp_path,
            [kept, held],
            validator,
            corrections={OTHER_TASK: [ROW_COUNT]},
            excluded={OTHER_TASK},
        )
    with pytest.raises(ValueError, match="pymc-no-such-task"):
        _reexport(tmp_path, [kept], validator, corrections={"pymc-no-such-task": [ROW_COUNT]})

    assert recorded_fits == []


@pytest.mark.parametrize(
    "entry",
    [
        {"before": "Mean latency is 2.0 ms.", "after": "Mean latency is 2.0 ms."},
        {"before": "", "after": "Mean latency is 2.0 ms."},
        {"before": "Mean latency is 2.0 ms.", "after": ""},
        {"before": "Mean latency is 2.0 ms."},
    ],
    ids=["same-text", "empty-before", "empty-after", "no-after"],
)
def test_a_replacement_must_change_the_text(tmp_path, validator, recorded_fits, entry):  # noqa: F811
    with pytest.raises(ValueError, match=TASK_ID):
        _reexport(tmp_path, [_source(tmp_path / "old")], validator, corrections={TASK_ID: [entry]})

    assert recorded_fits == []


def test_the_table_file_is_read_strictly(tmp_path):
    table = tmp_path / "corrections.json"
    table.write_text(
        json.dumps(
            {
                "description": "Sentences that stated something false about the supplied rows.",
                "tasks": {
                    TASK_ID: [{**ROW_COUNT, "class": "ROW-COUNT", "basis": (
                '3 supplied rows of 4'
            )}]
                },
            }
        )
    )

    assert export_pymc.load_instruction_corrections(table) == {TASK_ID: [ROW_COUNT]}

    table.write_text(json.dumps({"tasks": {TASK_ID: [{**ROW_COUNT, "afterr": "typo"}]}}))
    with pytest.raises(ValueError, match="afterr"):
        export_pymc.load_instruction_corrections(table)
    table.write_text(json.dumps({TASK_ID: [ROW_COUNT]}))
    with pytest.raises(ValueError, match="tasks"):
        export_pymc.load_instruction_corrections(table)


def test_the_cli_reads_the_table_and_forwards_it(monkeypatch, tmp_path):
    table = tmp_path / "corrections.json"
    table.write_text(json.dumps({"tasks": {TASK_ID: [ROW_COUNT]}}))
    captured = {}

    def record(sources, out_dir, **kwargs):
        captured.update(kwargs)
        return {"tasks_created": 1}

    monkeypatch.setattr(export_pymc, "reexport_harbor_tasks", record)
    result = CliRunner().invoke(
        app,
        [
            "reexport-harbor",
            "first",
            "--out",
            str(tmp_path / "out"),
            "--validator-path",
            "validator.py",
            "--validator-sha256",
            "abc",
            "--attempt-id",
            "attempt-1",
            "--role-manifest",
            str(tmp_path / "ROLE-MANIFEST.json"),
            "--role-manifest-sha256",
            "e" * 64,
            "--instruction-corrections",
            str(table),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["instruction_corrections"] == {TASK_ID: [ROW_COUNT]}

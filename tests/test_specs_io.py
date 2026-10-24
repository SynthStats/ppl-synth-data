from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppl_synth_data.harbor import export_lazyppl, export_pymc
from ppl_synth_data.pipeline import io as pipeline_io
from ppl_synth_data.pipeline.stages import (
    stage_b_compile,
    stage_c_data,
    stage_d_prompts,
    stage_g_dataset,
)
from ppl_synth_data.specs.io import read_envspecs

_MINIMAL_SPEC = {
    "env_id": "env_test_001",
    "env_name": "test_env",
    "domain": "statistics",
    "complexity": "simple",
    "model_type": "regression",
    "key_concepts": ["linear"],
    "io_schema": {
        "input_type": "float",
        "input_description": "predictor value",
        "output_type": "float",
        "target_name": "y",
        "feature_names": ["x"],
    },
    "data_plan": {"n_observations": 50},
    "generative_family": "linear_regression",
}


def test_read_envspecs_valid(tmp_path: Path) -> None:
    path = tmp_path / "specs.jsonl"
    path.write_text(json.dumps(_MINIMAL_SPEC) + "\n")
    specs = read_envspecs(path)
    assert len(specs) == 1
    assert specs[0].env_id == "env_test_001"


def test_read_envspecs_multiple(tmp_path: Path) -> None:
    s2 = {**_MINIMAL_SPEC, "env_id": "env_test_002", "env_name": "second"}
    path = tmp_path / "specs.jsonl"
    path.write_text(json.dumps(_MINIMAL_SPEC) + "\n" + json.dumps(s2) + "\n")
    specs = read_envspecs(path)
    assert len(specs) == 2


def test_read_envspecs_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert read_envspecs(path) == []


def test_read_envspecs_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "blanks.jsonl"
    path.write_text("\n" + json.dumps(_MINIMAL_SPEC) + "\n\n")
    assert len(read_envspecs(path)) == 1


def test_read_envspecs_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_envspecs(tmp_path / "no_such.jsonl")


def test_read_envspecs_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json\n")
    with pytest.raises(json.JSONDecodeError):
        read_envspecs(path)


def test_envspec_loader_call_sites_use_specs_module() -> None:
    from ppl_synth_data.specs.io import read_compile_inputs

    assert stage_b_compile.read_compile_inputs is read_compile_inputs
    assert stage_c_data.read_canonical_envspecs.__module__ == (
        "ppl_synth_data.pipeline.stages._envspecs"
    )
    assert stage_d_prompts.read_canonical_envspecs.__module__ == (
        "ppl_synth_data.pipeline.stages._envspecs"
    )
    assert stage_g_dataset.read_canonical_envspecs.__module__ == (
        "ppl_synth_data.pipeline.stages._envspecs"
    )
    assert export_pymc.read_envspecs is read_envspecs
    assert export_lazyppl.read_envspecs is read_envspecs


def test_pipeline_io_no_longer_exports_envspec_loader() -> None:
    assert not hasattr(pipeline_io, "read_envspecs")

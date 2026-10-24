"""Typed Stage D mapping on real compiled and materialized tables.

These check prompt assembly, not model generation or prompt-claim acceptance.
"""

from pathlib import Path

import pytest

from ppl_synth_data.pipeline.io import read_jsonl
from ppl_synth_data.pipeline.stages import stage_d_prompts as stage_d
from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c
from tests.test_declared_stage_c import _prepared_paths


@pytest.mark.parametrize("appendix", ["_pymc_data_schema_appendix", "_lazy_ppl_schema_appendix"])
def test_typed_schema_uses_visible_columns_and_rows(tmp_path: Path, appendix: str):
    paths, spec = _prepared_paths(tmp_path)
    run_stage_c(paths, seed=20260930, split_policy="random_row", visible_ratio=0.6)
    data_ref = read_jsonl(paths.stage_dirs["C"] / "data_refs.jsonl")[0]
    text = getattr(stage_d, appendix)(spec=spec, data_ref=data_ref, run_dir=paths.run_dir)
    assert "`y`" in text
    assert "`signal`" in text
    assert "`month`" in text
    assert "n=54" in text
    assert "n=90" not in text
    assert spec.scenario.source_row_sha256 not in text
    assert "hidden" not in text


def test_typed_prompt_record_preserves_lineage_without_inventing_metadata(tmp_path: Path):
    from ppl_synth_data.runtimes import get_runtime

    paths, spec = _prepared_paths(tmp_path)
    record = stage_d.build_prompt_record(
        request_id="prompt-000000",
        spec=spec,
        prompt_text="Fit a model of y using month and signal.",
        runtime=get_runtime("pymc"),
    )
    assert record.env_id == spec.env_id
    assert record.metadata["scenario"] == spec.scenario.model_dump(mode="json")
    assert record.metadata["data_origin"] == "regenerated"
    assert record.provenance.generative_family == "regression_count"
    assert record.provenance.complexity is None
    assert record.provenance.model_type is None
    assert record.provenance.key_concepts == []
    assert record.provenance.adaptation_mode == "regenerated"
    assert record.provenance.license_signal is None
    assert spec.scenario.source_row_sha256 not in record.prompt_text

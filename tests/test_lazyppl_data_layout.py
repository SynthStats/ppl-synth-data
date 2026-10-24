from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lazyppl_report_and_generated_artifacts_live_outside_data_dir() -> None:
    data_dir = REPO_ROOT / "data" / "lazyppl_synthesis"
    reports_dir = REPO_ROOT / "reports" / "lazyppl_synthesis"
    generated_dir = REPO_ROOT / "generated" / "lazyppl_synthesis"

    assert not (data_dir / "audit_report.md").exists()
    assert not (data_dir / "elpd_report.md").exists()
    assert not (data_dir / "elpd_results.json").exists()
    assert not (data_dir / "elpd_test.json").exists()

    if not reports_dir.exists():
        pytest.skip("Build artifacts are not distributed")
    assert (reports_dir / "audit_report.md").exists()
    assert (reports_dir / "elpd_report.md").exists()
    assert (generated_dir / "elpd_results.json").exists()
    assert (generated_dir / "elpd_test.json").exists()


def test_lazyppl_generated_inputs_live_outside_data_dir() -> None:
    data_dir = REPO_ROOT / "data" / "lazyppl_synthesis"
    generated_inputs_dir = REPO_ROOT / "generated" / "lazyppl_synthesis" / "inputs"

    for filename in (
        "train_translated.jsonl",
        "val_translated.jsonl",
        "test_translated.jsonl",
        "native_train.jsonl",
    ):
        assert not (data_dir / filename).exists()
        if not generated_inputs_dir.exists():
            pytest.skip("Build artifacts are not distributed")
        assert (generated_inputs_dir / filename).exists()


def test_lazyppl_legacy_intermediates_live_outside_data_dir() -> None:
    data_dir = REPO_ROOT / "data" / "lazyppl_synthesis"
    legacy_dir = REPO_ROOT / "generated" / "lazyppl_synthesis" / "legacy"

    for filename in (
        "train_pre_wrap.jsonl",
        "train_final.jsonl",
        "val_final.jsonl",
        "test_final.jsonl",
    ):
        assert not (data_dir / filename).exists()
        if not legacy_dir.exists():
            pytest.skip("Build artifacts are not distributed")
        assert (legacy_dir / filename).exists()


def test_eval_elpd_default_output_uses_generated_area() -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_layout_test", "scripts/eval_lazyppl_elpd.py")

    assert eval_mod._DEFAULT_OUTPUT == (
        REPO_ROOT / "generated" / "lazyppl_synthesis" / "elpd_results.json"
    )


def test_eval_elpd_usage_points_output_to_generated_results_json() -> None:
    eval_mod = _load_module("eval_lazyppl_elpd_usage_layout_test", "scripts/eval_lazyppl_elpd.py")

    doc = eval_mod.__doc__ or ""
    assert "generated/lazyppl_synthesis/elpd_results.json" in doc
    assert "generated/lazyppl_synthesis/elpd_batch.json" not in doc
    assert "data/lazyppl_synthesis/elpd_results.json" not in doc


def test_merge_script_default_input_uses_generated_inputs_area() -> None:
    merge_mod = _load_module("merge_lazyppl_merge_layout_test", "scripts/merge_lazyppl_splits.py")

    assert merge_mod.DEFAULT_INPUT_DIR == (REPO_ROOT / "generated" / "lazyppl_synthesis" / "inputs")


def test_canonical_lazyppl_train_split_excludes_wrapped_failures() -> None:
    train_path = REPO_ROOT / "data" / "lazyppl_synthesis" / "train.jsonl"

    records = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wrap_failures = [
        record
        for record in records
        if ((record.get("metadata") or {}).get("wrap_typecheck_ok") is False)
    ]

    assert records
    assert wrap_failures == []


def test_audit_script_usage_points_report_output_to_reports_area() -> None:
    audit_mod = _load_module(
        "audit_lazyppl_typecheck_layout_test",
        "scripts/audit_lazyppl_typecheck.py",
    )

    assert "reports/lazyppl_synthesis/audit_report.md" in (audit_mod.__doc__ or "")


def test_audit_script_usage_points_inputs_to_generated_area() -> None:
    audit_mod = _load_module(
        "audit_lazyppl_typecheck_input_layout_test",
        "scripts/audit_lazyppl_typecheck.py",
    )

    assert "generated/lazyppl_synthesis/inputs/train_translated.jsonl" in (audit_mod.__doc__ or "")


def test_translate_script_usage_points_inputs_to_generated_area() -> None:
    translate_mod = _load_module(
        "translate_pymc_to_lazyppl_layout_test",
        "scripts/translate_pymc_to_lazyppl.py",
    )

    assert "generated/lazyppl_synthesis/inputs/train_translated.jsonl" in (
        translate_mod.__doc__ or ""
    )


def test_wrap_submission_usage_points_legacy_area() -> None:
    wrap_mod = _load_module(
        "wrap_submission_layout_test",
        "scripts/wrap_submission.py",
    )

    doc = wrap_mod.__doc__ or ""
    assert "generated/lazyppl_synthesis/legacy/train_pre_wrap.jsonl" in doc
    assert "generated/lazyppl_synthesis/legacy/train_wrapped.jsonl" in doc


def test_boundary_inventory_records_legacy_retention_policy() -> None:
    note_path = REPO_ROOT / "artifacts" / "data-layout.md"
    if not note_path.exists():
        pytest.skip("Build documents are not distributed")
    note = note_path.read_text(encoding="utf-8")

    assert "## Retention decision" in note
    assert "keep `generated/lazyppl_synthesis/legacy/` tracked" in note
    assert "historical provenance" in note

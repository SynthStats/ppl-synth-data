from __future__ import annotations

from ppl_synth_data.specs.models import (
    CompletionRecord,
    PromptProvenance,
    ValidationInfo,
    ValidationTrace,
)
from ppl_synth_data.specs.record_contracts import normalize_completion_record


def test_normalize_completion_record_uses_explicit_validation_over_legacy_metadata() -> None:
    explicit_validation = ValidationInfo(ok=False, message="explicit", repaired=False)
    record = CompletionRecord(
        id="completion-001",
        prompt_id="prompt-001",
        prompt_text="prompt",
        raw_completion="completion",
        validation=explicit_validation,
        metadata={
            "validation": {"ok": True, "message": "legacy", "repaired": True},
            "env_id": "env-from-metadata",
            "split": "validation",
        },
    )

    normalized = normalize_completion_record(record)

    assert normalized.validation == explicit_validation
    assert normalized.env_id == "env-from-metadata"
    assert normalized.split == "validation"
    assert "validation" not in normalized.metadata


def test_normalize_completion_record_does_not_backfill_metadata_validation() -> None:
    record = CompletionRecord(
        id="completion-002",
        prompt_id="prompt-002",
        prompt_text="prompt",
        raw_completion="completion",
        provenance=PromptProvenance(runtime_name="lazyppl"),
        metadata={
            "validation": {
                "ok": True,
                "message": "legacy payload",
                "repaired": True,
                "leakage_rating": "CLEAN",
            },
            "env_id": "env-123",
            "split": "test",
            "lazyppl_version": "lazyppl-1.0",
            "extra": "keep-me",
        },
    )

    normalized = normalize_completion_record(record)

    assert normalized.validation is None
    assert normalized.env_id == "env-123"
    assert normalized.split == "test"
    assert normalized.provenance is not None
    assert normalized.provenance.runtime_name == "lazyppl"
    assert normalized.provenance.runtime_version == "lazyppl-1.0"
    assert normalized.metadata == {"extra": "keep-me"}


def test_normalize_completion_record_normalizes_runtime_aliases() -> None:
    record = CompletionRecord(
        id="completion-003",
        prompt_id="prompt-003",
        prompt_text="prompt",
        raw_completion="completion",
        provenance=PromptProvenance(runtime_name="haskell"),
        metadata={"lazyppl_version": "lazyppl-1.0"},
    )

    normalized = normalize_completion_record(record)

    assert normalized.provenance is not None
    assert normalized.provenance.runtime_name == "lazyppl"
    assert normalized.provenance.runtime_version == "lazyppl-1.0"


def test_normalize_completion_record_promotes_source_provenance_metadata() -> None:
    record = CompletionRecord(
        id="completion-004",
        prompt_id="prompt-004",
        prompt_text="prompt",
        raw_completion="completion",
        provenance=PromptProvenance(runtime_name="pymc"),
        metadata={
            "source_category": "pymc_examples",
            "source_url_or_repo": "https://github.com/pymc-devs/pymc-examples",
            "license_signal": "permissive",
            "adaptation_mode": "source-inspired",
            "source_retrieval_date": "2026-05-16",
            "pymc_version": "v5",
            "extra": "keep-me",
        },
    )

    normalized = normalize_completion_record(record)

    assert normalized.provenance is not None
    assert normalized.provenance.source_category == "pymc_examples"
    assert normalized.provenance.source_url_or_repo == "https://github.com/pymc-devs/pymc-examples"
    assert normalized.provenance.license_signal == "permissive"
    assert normalized.provenance.adaptation_mode == "source-inspired"
    assert normalized.provenance.source_retrieval_date == "2026-05-16"
    assert normalized.metadata == {"extra": "keep-me"}


def test_validation_trace_does_not_promote_taxonomy_from_metadata() -> None:
    trace = ValidationTrace.model_validate(
        {
            "steps": [
                {
                    "attempt": 0,
                    "phase": "runtime",
                    "ok": True,
                    "message": "ok",
                    "metadata": {
                        "evidence_kind": "verifier",
                        "truth_owner": "verifier",
                        "semantic_status": "not_scored",
                        "metric_name": "psis_loo",
                    },
                }
            ]
        }
    )

    assert trace.steps[0].evidence_kind is None
    assert trace.steps[0].truth_owner is None
    assert trace.steps[0].semantic_status is None
    assert trace.steps[0].metadata["metric_name"] == "psis_loo"

"""Tests for LLM-as-judge leakage detection."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from ppl_synth_data.llm.backend import BackendCapabilities, LLMResult
from ppl_synth_data.pipeline.validation.leakage_gate import (
    judge_leakage,
    judge_leakage_deterministic,
    prompt_text_for_leakage_scan,
)


def _make_mock_backend(judge_response: dict[str, Any]) -> MagicMock:
    backend = MagicMock()
    backend.capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=False,
    )
    backend.complete.return_value = LLMResult(
        content=json.dumps(judge_response),
        raw={},
    )
    return backend


@pytest.mark.narrow_unit_not_acceptance
def test_clean_prompt_passes() -> None:
    backend = _make_mock_backend(
        {"rating": "CLEAN", "evidence": "no leakage", "recovered_family": None},
    )
    result = judge_leakage(
        "Observed daily counts with seasonal patterns.",
        backend,
        "test-model",
    )
    assert result["leaked"] is False
    assert result["rating"] == "CLEAN"
    request = backend.complete.call_args.args[0]
    assert request.temperature is None


@pytest.mark.narrow_unit_not_acceptance
def test_leaked_prompt_rejected() -> None:
    backend = _make_mock_backend(
        {"rating": "LEAKED", "evidence": "names Poisson directly", "recovered_family": "poisson"},
    )
    result = judge_leakage(
        "Fit a Poisson regression model.",
        backend,
        "test-model",
    )
    assert result["leaked"] is True
    assert result["rating"] == "LEAKED"


@pytest.mark.narrow_unit_not_acceptance
def test_unknown_leakage_rating_fails_closed() -> None:
    backend = _make_mock_backend(
        {"rating": "MILD", "evidence": "exponential growth hint", "recovered_family": None},
    )
    result = judge_leakage(
        "Growth appears exponential.",
        backend,
        "test-model",
    )
    assert result["leaked"] is True
    assert result["rating"] == "INCONCLUSIVE"


@pytest.mark.narrow_unit_not_acceptance
def test_family_recovery_triggers_leak() -> None:
    backend = _make_mock_backend(
        {"rating": "CLEAN", "evidence": "subtle hint", "recovered_family": "Poisson"},
    )
    result = judge_leakage(
        "Count data",
        backend,
        "test-model",
        actual_family="poisson",
    )
    assert result["leaked"] is True


@pytest.mark.narrow_unit_not_acceptance
def test_wrong_family_recovery_ok() -> None:
    backend = _make_mock_backend(
        {"rating": "CLEAN", "evidence": "guessed wrong", "recovered_family": "normal"},
    )
    result = judge_leakage(
        "Count data",
        backend,
        "test-model",
        actual_family="poisson",
    )
    assert result["leaked"] is False


@pytest.mark.narrow_unit_not_acceptance
def test_missing_leakage_rating_fails_closed() -> None:
    backend = _make_mock_backend({"evidence": "no explicit rating"})
    result = judge_leakage("some prompt", backend, "test-model")

    assert result["leaked"] is True
    assert result["rating"] == "INCONCLUSIVE"


@pytest.mark.narrow_unit_not_acceptance
def test_malformed_recovered_family_fails_closed() -> None:
    backend = _make_mock_backend(
        {"rating": "CLEAN", "evidence": "bad family shape", "recovered_family": ["poisson"]},
    )
    result = judge_leakage("some prompt", backend, "test-model")

    assert result["leaked"] is True
    assert result["rating"] == "INCONCLUSIVE"


@pytest.mark.narrow_unit_not_acceptance
def test_leaked_rating_is_stripped_before_checking() -> None:
    backend = _make_mock_backend(
        {"rating": "LEAKED ", "evidence": "obvious", "recovered_family": "poisson"},
    )
    result = judge_leakage("some prompt", backend, "test-model")

    assert result["leaked"] is True
    assert result["rating"] == "LEAKED"


@pytest.mark.narrow_unit_not_acceptance
def test_handles_json_parse_failure() -> None:
    backend = MagicMock()
    backend.capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=False,
    )
    backend.complete.return_value = LLMResult(content="not json at all", raw={})
    result = judge_leakage("some prompt", backend, "test-model")
    assert result["leaked"] is True
    assert result["rating"] == "INCONCLUSIVE"


@pytest.mark.narrow_unit_not_acceptance
def test_handles_markdown_wrapped_json() -> None:
    wrapped = (
        '```json\n{"rating": "LEAKED", "evidence": "obvious", "recovered_family": "normal"}\n```'
    )
    backend = MagicMock()
    backend.capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=False,
    )
    backend.complete.return_value = LLMResult(content=wrapped, raw={})
    result = judge_leakage("Use Normal distribution", backend, "test-model")
    assert result["leaked"] is True
    assert result["rating"] == "LEAKED"


def test_deterministic_leakage_judge_marks_clean_prompt() -> None:
    result = judge_leakage_deterministic("Observed daily counts with seasonal patterns.")
    assert result["leaked"] is False
    assert result["rating"] == "CLEAN"
    assert result["method"] == "deterministic_banned_terms"


def test_deterministic_leakage_judge_marks_banned_terms() -> None:
    result = judge_leakage_deterministic("Use a Poisson likelihood.")
    assert result["leaked"] is True
    assert result["rating"] == "LEAKED"
    assert "poisson" in result["evidence"]


def test_leakage_scan_masks_trusted_runtime_contract_markers() -> None:
    prompt = (
        "A field team collected nitrate readings and rainfall covariates.\n\n"
        "## PyMC data dictionary\n"
        "The verifier calls `def model(data)` with exact keys.\n"
        "PyMC submission constraint: use the dictionary above.\n"
        "Final answer contract for this request:\n"
        "The submitted Python must return a `pm.Model`."
    )

    scan_text, markers = prompt_text_for_leakage_scan(prompt)

    assert "A field team collected nitrate readings and rainfall covariates." in scan_text
    assert "PyMC" not in scan_text
    assert "## PyMC data dictionary" in markers
    assert "PyMC submission constraint:" in markers
    assert "Final answer contract for this request:" in markers
    result = judge_leakage_deterministic(scan_text)
    assert result["leaked"] is False


def test_leakage_scan_preserves_scientific_family_leakage_before_runtime_contract() -> None:
    prompt = (
        "Use a Poisson likelihood for these observed daily counts.\n\n"
        "## PyMC data dictionary\n"
        "Allowed `data` keys: `count`, `day`."
    )

    scan_text, markers = prompt_text_for_leakage_scan(prompt)
    result = judge_leakage_deterministic(scan_text)

    assert "## PyMC data dictionary" in markers
    assert result["leaked"] is True
    assert "poisson" in result["evidence"]


def test_leakage_scan_preserves_scientific_family_leakage_after_runtime_marker() -> None:
    prompt = (
        "Observed daily counts.\n\n"
        "## PyMC data dictionary\n"
        "Allowed `data` keys: `count`, `day`.\n"
        "Use a Poisson likelihood for the submitted model."
    )

    scan_text, markers = prompt_text_for_leakage_scan(prompt)
    result = judge_leakage_deterministic(scan_text)

    assert "## PyMC data dictionary" in markers
    assert "PyMC" not in scan_text
    assert result["leaked"] is True
    assert "poisson" in result["evidence"]

from __future__ import annotations

from unittest.mock import patch

import pytest

from ppl_synth_data.llm.backend import (
    BackendCapabilities,
    LLMRequest,
    LLMResult,
)
from ppl_synth_data.runtimes.base import ValidationResult
from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime
from ppl_synth_data.validation.pipeline import validate_and_repair

pytestmark = pytest.mark.narrow_unit_not_acceptance

GOOD_CODE = """\
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=data["y"])
        return m
"""

BAD_SYNTAX = "def model(data:\n"

UNSAFE_CODE = "import os\nos.system('rm -rf /')\n"

# passes static check but fails runtime
RUNTIME_BAD = """\
def model(data):
    raise ValueError("broken")
"""

FENCED_GOOD_CODE = f"```python\n{GOOD_CODE}\n```"
FENCED_GOOD_CODE_COMPACT = f"```python\n{GOOD_CODE}```"

_RT = PyMCRuntime()
SAMPLE_DATA = {"n": 2, "y": [0.0, 1.0]}


class FakeBackend:
    capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=False,
    )

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResult:
        self.call_count += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResult(content=text, raw={})


def test_good_code_passes_without_repair() -> None:
    backend = FakeBackend([])
    ok, final, msg = validate_and_repair(
        GOOD_CODE,
        "prompt",
        backend,
        "model-x",
        max_retries=2,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert ok is True
    assert final == GOOD_CODE
    assert msg == "ok"
    assert backend.call_count == 0


def test_static_failure_triggers_repair() -> None:
    backend = FakeBackend([GOOD_CODE])
    ok, final, msg = validate_and_repair(
        UNSAFE_CODE,
        "prompt",
        backend,
        "model-x",
        max_retries=1,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert backend.call_count == 1


def test_syntax_error_triggers_repair() -> None:
    backend = FakeBackend([GOOD_CODE])
    ok, final, msg = validate_and_repair(
        BAD_SYNTAX,
        "prompt",
        backend,
        "model-x",
        max_retries=1,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert backend.call_count == 1


def test_runtime_failure_triggers_repair() -> None:
    backend = FakeBackend([GOOD_CODE])
    ok, final, msg = validate_and_repair(
        RUNTIME_BAD,
        "prompt",
        backend,
        "model-x",
        max_retries=1,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert backend.call_count == 1


def test_max_retries_exhausted_returns_failure() -> None:
    backend = FakeBackend([UNSAFE_CODE, UNSAFE_CODE, UNSAFE_CODE])
    ok, final, msg = validate_and_repair(
        UNSAFE_CODE,
        "prompt",
        backend,
        "model-x",
        max_retries=3,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert ok is False
    assert "syntax check failed" in msg
    assert backend.call_count == 3


def test_zero_retries_fails_immediately() -> None:
    backend = FakeBackend([])
    ok, final, msg = validate_and_repair(
        UNSAFE_CODE,
        "prompt",
        backend,
        "model-x",
        max_retries=0,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert ok is False
    assert backend.call_count == 0


def test_runtime_failure_exhausts_retries() -> None:
    backend = FakeBackend([RUNTIME_BAD, RUNTIME_BAD])
    ok, final, msg = validate_and_repair(
        RUNTIME_BAD,
        "prompt",
        backend,
        "model-x",
        max_retries=2,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert ok is False
    assert "runtime check failed" in msg
    assert backend.call_count == 2


@patch.object(PyMCRuntime, "validate_runtime")
@patch.object(
    PyMCRuntime,
    "validate_syntax",
    return_value=ValidationResult(ok=False, message="INFRA: missing verifier"),
)
def test_infra_syntax_failure_does_not_call_repair(mock_syntax, mock_runtime) -> None:
    backend = FakeBackend([GOOD_CODE])
    ok, final, msg = validate_and_repair(
        GOOD_CODE,
        "prompt",
        backend,
        "model-x",
        max_retries=1,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert ok is False
    assert final == GOOD_CODE
    assert "INFRA: missing verifier" in msg
    assert backend.call_count == 0
    mock_syntax.assert_called_once()
    mock_runtime.assert_not_called()


def test_fenced_repair_output_is_unwrapped_before_validation() -> None:
    backend = FakeBackend([FENCED_GOOD_CODE])
    ok, final, msg = validate_and_repair(
        UNSAFE_CODE,
        "prompt",
        backend,
        "model-x",
        max_retries=1,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert ok is True
    assert msg == "ok"
    assert backend.call_count == 1
    assert "```" not in final
    assert final.strip() == GOOD_CODE.strip()


def test_compact_fenced_repair_output_is_unwrapped() -> None:
    backend = FakeBackend([FENCED_GOOD_CODE_COMPACT])
    ok, final, msg = validate_and_repair(
        UNSAFE_CODE,
        "prompt",
        backend,
        "model-x",
        max_retries=1,
        sample_data=SAMPLE_DATA,
        runtime=_RT,
    )
    assert ok is True
    assert msg == "ok"
    assert backend.call_count == 1
    assert "```" not in final
    assert final.strip() == GOOD_CODE.strip()


@patch.object(PyMCRuntime, "validate_runtime", return_value=ValidationResult(ok=True, message="ok"))
def test_sample_data_is_forwarded_to_runtime_check(mock_runtime) -> None:
    backend = FakeBackend([])
    sample = {"n_parks": 5, "visitors": [10, 20, 30, 40, 50]}
    ok, final, msg = validate_and_repair(
        GOOD_CODE,
        "prompt",
        backend,
        "model-x",
        max_retries=0,
        sample_data=sample,
        runtime=_RT,
    )
    assert ok is True
    mock_runtime.assert_called_once_with(GOOD_CODE, sample_data=sample)

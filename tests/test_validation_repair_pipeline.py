from __future__ import annotations

import pytest

from ppl_synth_data.llm.backend import (
    BackendCapabilities,
    LLMRequest,
    LLMResult,
)
from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime
from ppl_synth_data.validation.pipeline import _normalize_repair_code, validate_and_repair

_RT = PyMCRuntime()


class FakeRepairBackend:
    capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=True,
    )

    def __init__(self, repair_code: str = "import pymc as pm\nmodel = pm.Model()") -> None:
        self._repair_code = repair_code
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResult:
        self.call_count += 1
        return LLMResult(content=self._repair_code, raw={})


def test_normalize_repair_code_plain() -> None:
    code = "x = 1"
    assert _normalize_repair_code(code) == "x = 1"


def test_normalize_repair_code_fenced() -> None:
    fenced = "```python\nx = 1\n```"
    assert _normalize_repair_code(fenced) == "x = 1"


def test_normalize_repair_code_fenced_no_lang() -> None:
    fenced = "```\ny = 2\n```"
    assert _normalize_repair_code(fenced) == "y = 2"


def test_normalize_repair_code_submit_program_tags() -> None:
    tagged = "<submit_program>\nz = 3\n</submit_program>"
    assert _normalize_repair_code(tagged) == "z = 3"


def test_normalize_repair_code_submit_program_tags_around_fence() -> None:
    tagged = '<submit_program language="python">\n```python\nz = 3\n```\n</submit_program>'
    assert _normalize_repair_code(tagged) == "z = 3"


@pytest.mark.narrow_unit_not_acceptance
def test_validate_and_repair_rejects_unsafe_code() -> None:
    code = "import os; os.system('rm -rf /')"
    backend = FakeRepairBackend(repair_code=code)
    ok, final, msg = validate_and_repair(
        code,
        "prompt",
        backend,
        model="m",
        max_retries=0,
        runtime=_RT,
    )
    assert ok is False
    assert "syntax check failed" in msg


@pytest.mark.narrow_unit_not_acceptance
def test_validate_and_repair_calls_repair_on_failure() -> None:
    bad_code = "import subprocess; subprocess.run(['ls'])"
    still_bad = "import subprocess; subprocess.call(['ls'])"
    backend = FakeRepairBackend(repair_code=still_bad)
    ok, final, msg = validate_and_repair(
        bad_code,
        "prompt",
        backend,
        model="m",
        max_retries=2,
        runtime=_RT,
    )
    assert ok is False
    assert backend.call_count >= 1

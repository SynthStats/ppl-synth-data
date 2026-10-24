"""Tests for dual-format extraction (submit_program + code fence)."""

from __future__ import annotations

from ppl_synth_data.runtimes.pymc_completion import (
    extract_python_code,
    replace_python_code,
    validate_completion_format,
)

SUBMIT_PROGRAM_COMPLETION = """\
<think>
Reasoning about data properties and model selection.
</think>

<submit_program>
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=data["y"])
        return m
</submit_program>
"""

CODE_FENCE_COMPLETION = """\
<think>
Reasoning about data.
</think>

```python
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        return m
```
"""

BOTH_FORMATS = """\
<think>reasoning</think>

<submit_program>
def model(data):
    return "submit_program version"
</submit_program>

```python
def model(data):
    return "fence version"
```
"""


def test_extract_prefers_submit_program() -> None:
    code = extract_python_code(BOTH_FORMATS)
    assert code is not None
    assert "submit_program version" in code


def test_extract_from_submit_program_tags() -> None:
    code = extract_python_code(SUBMIT_PROGRAM_COMPLETION)
    assert code is not None
    assert "def model(data):" in code
    assert "pm.Normal" in code


def test_extract_from_code_fence_fallback() -> None:
    code = extract_python_code(CODE_FENCE_COMPLETION)
    assert code is not None
    assert "def model(data):" in code


def test_extract_returns_none_for_plain_text() -> None:
    assert extract_python_code("just text, no code blocks") is None


def test_validate_submit_program_completion() -> None:
    ok, issues = validate_completion_format(SUBMIT_PROGRAM_COMPLETION)
    assert ok is True
    assert issues == []


def test_validate_code_fence_completion() -> None:
    ok, issues = validate_completion_format(CODE_FENCE_COMPLETION)
    assert ok is True
    assert issues == []


def test_replace_in_submit_program_tags() -> None:
    new_code = "def model(data):\n    return 'replaced'"
    result = replace_python_code(SUBMIT_PROGRAM_COMPLETION, new_code)
    assert "<submit_program>" in result
    assert "replaced" in result
    assert "</submit_program>" in result


def test_replace_in_code_fence() -> None:
    new_code = "def model(data):\n    return 'replaced'"
    result = replace_python_code(CODE_FENCE_COMPLETION, new_code)
    assert "```python" in result
    assert "replaced" in result


def test_replace_creates_submit_program_when_no_container() -> None:
    new_code = "def model(data):\n    return 'new'"
    result = replace_python_code("just text", new_code)
    assert "<submit_program>" in result
    assert "</submit_program>" in result
    assert "new" in result

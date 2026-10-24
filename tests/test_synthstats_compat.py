"""Integration tests for the shared SynthStats extraction contract."""

from __future__ import annotations

import re

from ppl_synth_data.runtimes.pymc_completion import extract_python_code

# matches synthstats extract_program_text() priority
_SYNTHSTATS_SUBMIT_RE = re.compile(r"<submit_program>\s*\n?(.*?)</submit_program>", re.DOTALL)
_SYNTHSTATS_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _synthstats_extract(text: str) -> str | None:
    """Mimics synthstats extract_program_text() priority order."""
    match = _SYNTHSTATS_SUBMIT_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _SYNTHSTATS_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


SUBMIT_PROGRAM_TEXT = """\
<think>Reasoning</think>

<submit_program>
import pymc as pm

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        return m
</submit_program>
"""

CODE_FENCE_TEXT = """\
<think>Reasoning</think>

```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        return m
```
"""


def test_static_records_extract_with_synthstats_logic() -> None:
    our_code = extract_python_code(SUBMIT_PROGRAM_TEXT)
    their_code = _synthstats_extract(SUBMIT_PROGRAM_TEXT)
    assert our_code == their_code


def test_code_fence_records_extract_with_synthstats_logic() -> None:
    our_code = extract_python_code(CODE_FENCE_TEXT)
    their_code = _synthstats_extract(CODE_FENCE_TEXT)
    assert our_code == their_code


def test_submit_program_tags_preferred() -> None:
    both = """\
<submit_program>
def model(data): return "submit"
</submit_program>

```python
def model(data): return "fence"
```
"""
    our_code = extract_python_code(both)
    their_code = _synthstats_extract(both)
    assert our_code is not None
    assert "submit" in our_code
    assert our_code == their_code

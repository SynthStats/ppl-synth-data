from __future__ import annotations

from pathlib import Path

from ppl_synth_data.runtimes.pymc_completion import validate_completion_format

PROMPT_DIR = Path(__file__).resolve().parents[1] / "src" / "ppl_synth_data" / "llm" / "prompts"

VALID_COMPLETION = """\
<think>
The data suggests a linear regression model with normal priors.
</think>

```python
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=data["y"])
        return m
```
"""


def test_valid_completion_passes() -> None:
    ok, issues = validate_completion_format(VALID_COMPLETION)
    assert ok is True
    assert issues == []


def test_missing_think_block() -> None:
    text = """\
```python
def model(data):
    pass
```
"""
    ok, issues = validate_completion_format(text)
    assert ok is False
    assert "missing <think> block" in issues


def test_missing_code_fence() -> None:
    text = """\
<think>reasoning</think>

def model(data):
    pass
"""
    ok, issues = validate_completion_format(text)
    assert ok is False
    assert any("missing code block" in i for i in issues)


def test_missing_model_def() -> None:
    text = """\
<think>reasoning</think>

```python
import pymc as pm
x = 1
```
"""
    ok, issues = validate_completion_format(text)
    assert ok is False
    assert "missing def model(data): definition" in issues


def test_all_issues_reported() -> None:
    ok, issues = validate_completion_format("just some text")
    assert ok is False
    assert len(issues) == 3


def test_think_block_can_be_multiline() -> None:
    text = """\
<think>
line one
line two
line three
</think>

```python
def model(data):
    pass
```
"""
    ok, issues = validate_completion_format(text)
    assert ok is True
    assert issues == []


def test_model_def_inside_code_fence() -> None:
    text = """\
<think>ok</think>

```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        return m
```
"""
    ok, issues = validate_completion_format(text)
    assert ok is True


def test_model_def_in_think_block_does_not_count() -> None:
    text = """\
<think>
I will define def model(data): later.
</think>

```python
import pymc as pm
x = 1
```
"""
    ok, issues = validate_completion_format(text)
    assert ok is False
    assert "missing def model(data): definition" in issues


def test_model_def_comment_does_not_count() -> None:
    text = """\
<think>ok</think>

```python
# def model(data):
import pymc as pm
```
"""
    ok, issues = validate_completion_format(text)
    assert ok is False
    assert "missing def model(data): definition" in issues


def test_model_def_must_be_top_level() -> None:
    text = """\
<think>ok</think>

```python
class Wrapper:
    def model(data):
        return data
```
"""
    ok, issues = validate_completion_format(text)
    assert ok is False
    assert "missing def model(data): definition" in issues


def test_pymc_generation_prompts_match_import_free_verifier_contract() -> None:
    for prompt_name in ("completion_gen.md", "repair_completion.md"):
        prompt = (PROMPT_DIR / prompt_name).read_text(encoding="utf-8")
        assert "import pymc as pm" not in prompt
        assert "import numpy as np" not in prompt
        assert "Do not include import statements." in prompt
        assert "verifier already provides `pm`, `np`" in prompt

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.narrow_unit_not_acceptance

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_wrap_one_repairs_invalid_code_and_strips_think(monkeypatch) -> None:
    wrap_mod = _load_module("wrap_submission_test", "scripts/wrap_submission.py")
    calls: list[list[dict[str, str]]] = []

    class _FakeRuntime:
        def extract_code(self, text: str):
            if "fixed-code" in text:
                return "fixed-code"
            if "broken-code" in text:
                return "broken-code"
            return None

        def validate_runtime(self, code: str):
            if code == "fixed-code":
                return SimpleNamespace(ok=True, message=None)
            return SimpleNamespace(ok=False, message="type error")

        def repair_system_prompt(self, *, prompt_text: str, broken_code: str, error_trace: str):
            assert prompt_text == "Prompt"
            assert broken_code == "broken-code"
            assert error_trace == "type error"
            return ("repair-system", "repair-user")

        def replace_code(self, raw_completion: str, new_code: str) -> str:
            assert "broken-code" in raw_completion
            return f'<submit_program language="haskell">{new_code}</submit_program>'

    def _fake_completion(*, messages, **_kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return _response(
                "<think>repair plan</think>\n"
                '<submit_program language="haskell">broken-code</submit_program>'
            )
        return _response('<submit_program language="haskell">fixed-code</submit_program>')

    monkeypatch.setattr(wrap_mod, "LAZYPPL", _FakeRuntime())
    monkeypatch.setattr(wrap_mod, "litellm_completion", _fake_completion)

    result = wrap_mod.wrap_one(
        prompt="Prompt",
        old_code="main = pure ()",
        llm_model="fake-model",
        max_repairs=2,
    )

    assert result == {
        "code": "fixed-code",
        "completion": '<submit_program language="haskell">fixed-code</submit_program>',
        "think": "repair plan",
        "repairs": 1,
        "typecheck_ok": True,
        "error": None,
    }
    assert calls[0][0]["content"] == wrap_mod.SUBMISSION_SYSTEM_PROMPT
    assert calls[1][0]["content"] == "repair-system"


def test_main_carries_forward_successful_prior_run_records(tmp_path: Path, monkeypatch) -> None:
    wrap_mod = _load_module("wrap_submission_main_test", "scripts/wrap_submission.py")
    input_path = tmp_path / "input.jsonl"
    prior_path = tmp_path / "prior.jsonl"
    output_path = tmp_path / "output.jsonl"

    carried = {
        "prompt": "prompt-a",
        "completion": '<submit_program language="haskell">carried</submit_program>',
        "metadata": {"wrap_typecheck_ok": True, "source": "previous"},
    }
    pending = {
        "prompt": "prompt-b",
        "completion": '<submit_program language="haskell">needs-wrap</submit_program>',
        "metadata": {},
    }
    input_path.write_text(
        "\n".join(json.dumps(record) for record in (carried, pending)) + "\n",
        encoding="utf-8",
    )
    prior_path.write_text(json.dumps(carried) + "\n", encoding="utf-8")

    args = SimpleNamespace(
        input=input_path,
        output=output_path,
        model="fake-model",
        max_repairs=3,
        temperature=0.3,
        limit=0,
        delay=0.0,
        no_skip_new=True,
        prior_run=prior_path,
    )
    wrap_calls: list[str] = []

    def _fake_parse_args(self):
        return args

    def _fake_wrap_one(
        *,
        prompt: str,
        old_code: str,
        llm_model: str,
        max_repairs: int,
        temperature: float,
    ):
        wrap_calls.append(prompt)
        assert old_code == "needs-wrap"
        assert llm_model == "fake-model"
        assert max_repairs == 3
        assert temperature == 0.3
        return {
            "code": "wrapped",
            "completion": '<submit_program language="haskell">wrapped</submit_program>',
            "think": None,
            "repairs": 0,
            "typecheck_ok": True,
            "error": None,
        }

    monkeypatch.setattr(wrap_mod.argparse.ArgumentParser, "parse_args", _fake_parse_args)
    monkeypatch.setattr(wrap_mod, "wrap_one", _fake_wrap_one)
    monkeypatch.setattr(wrap_mod.time, "sleep", lambda _seconds: None)

    wrap_mod.main()

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert wrap_calls == ["prompt-b"]
    assert rows[0] == carried
    assert rows[1]["completion"] == '<submit_program language="haskell">wrapped</submit_program>'
    assert rows[1]["metadata"]["wrapped_to_submission"] is True
    assert rows[1]["metadata"]["wrap_typecheck_ok"] is True
    assert rows[1]["metadata"]["wrap_repairs"] == 0

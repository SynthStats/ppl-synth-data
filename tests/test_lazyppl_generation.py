from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ppl_synth_data.runtimes.lazyppl_generation as generation
from ppl_synth_data.llm.prompt_loader import load_prompt_text


class _FakeBackend:
    def __init__(self, content: str) -> None:
        self._content = content

    def complete(self, _request: object) -> SimpleNamespace:
        return SimpleNamespace(content=self._content)


@pytest.mark.narrow_unit_not_acceptance
def test_generate_lazyppl_completion_reports_missing_code_block() -> None:
    result = generation.generate_lazyppl_completion(
        "prompt",
        _FakeBackend("plain text only"),
        "gpt-5.5",
    )

    assert result.code is None
    assert result.completion == "plain text only"
    assert result.think is None
    assert result.repaired is False
    assert result.typecheck_ok is False
    assert result.error == "no code block extracted"


@pytest.mark.narrow_unit_not_acceptance
def test_generate_lazyppl_completion_repairs_and_strips_think(monkeypatch) -> None:
    raw_completion = (
        "<think>reasoning</think>\n"
        '<submit_program language="haskell">\n'
        "module Submission where\n"
        "model :: [Row] -> Meas State\n"
        "model = undefined\n"
        "logLikRow :: State -> Row -> Double\n"
        "logLikRow = undefined\n"
        "</submit_program>"
    )
    repaired_code = (
        "module Submission where\n"
        "model :: [Row] -> Meas State\n"
        "model = pure undefined\n"
        "logLikRow :: State -> Row -> Double\n"
        "logLikRow = undefined\n"
    )

    monkeypatch.setattr(
        generation,
        "validate_and_repair",
        lambda code, prompt, backend, model, max_retries, runtime: (True, repaired_code, "ok"),
    )

    result = generation.generate_lazyppl_completion(
        "prompt",
        _FakeBackend(raw_completion),
        "gpt-5.5",
    )

    assert result.code == repaired_code
    assert result.repaired is True
    assert result.typecheck_ok is True
    assert result.error is None
    assert result.think == "reasoning"
    assert "<think>" not in result.completion
    assert "pure undefined" in result.completion


@pytest.mark.narrow_unit_not_acceptance
def test_generate_lazyppl_completion_uses_runtime_prompt_name(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class _PromptAwareBackend:
        def complete(self, request: object) -> SimpleNamespace:
            captured["system"] = request.system
            return SimpleNamespace(content="raw completion")

    class _FakeRuntime:
        def completion_system_prompt_name(self) -> str:
            return "custom_prompt.md"

        def extract_code(self, _completion: str) -> str:
            return "module Submission where"

        def replace_code(self, completion: str, new_code: str) -> str:
            return f"{completion}\n{new_code}"

    monkeypatch.setattr(generation, "load_prompt_text", lambda name: f"PROMPT::{name}")
    monkeypatch.setattr(
        generation,
        "validate_and_repair",
        lambda code, prompt, backend, model, max_retries, runtime: (True, code, "ok"),
    )

    generation.generate_lazyppl_completion(
        "prompt",
        _PromptAwareBackend(),
        "gpt-5.5",
        runtime=_FakeRuntime(),
    )

    assert captured["system"] == "PROMPT::custom_prompt.md"


@pytest.mark.narrow_unit_not_acceptance
def test_generate_lazyppl_completion_requests_fresh_default_runtime(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _DefaultRuntime:
        def completion_system_prompt_name(self) -> str:
            return "completion_gen_lazyppl.md"

        def extract_code(self, completion: str) -> str:
            captured["raw_completion"] = completion
            return "module Submission where"

        def replace_code(self, completion: str, new_code: str) -> str:
            return f"{completion}\n{new_code}"

    monkeypatch.setattr(generation, "get_runtime", lambda name: _DefaultRuntime())
    monkeypatch.setattr(
        generation,
        "validate_and_repair",
        lambda code, prompt, backend, model, max_retries, runtime: (True, code, "ok"),
    )

    result = generation.generate_lazyppl_completion(
        "prompt",
        _FakeBackend("raw completion"),
        "gpt-5.5",
    )

    assert captured["raw_completion"] == "raw completion"
    assert result.typecheck_ok is True


def test_build_lazyppl_record_merges_result_metadata() -> None:
    record = generation.build_lazyppl_record(
        prompt="prompt",
        complexity="medium",
        metadata={"source": "native_lazyppl"},
        result=generation.LazyPPLGenerationResult(
            code="module Submission where",
            completion="<submit_program>module Submission where</submit_program>",
            think="reasoning",
            repaired=True,
            typecheck_ok=False,
            error="typecheck failed",
        ),
    )

    assert record["prompt"] == "prompt"
    assert record["complexity"] == "medium"
    assert record["metadata"]["source"] == "native_lazyppl"
    assert record["metadata"]["repaired"] is True
    assert record["metadata"]["typecheck_ok"] is False
    assert record["metadata"]["think"] == "reasoning"
    assert record["metadata"]["error"] == "typecheck failed"


def test_write_jsonl_records_writes_atomically(tmp_path: Path) -> None:
    out_path = tmp_path / "out.jsonl"
    generation.write_jsonl_records(out_path, [{"a": 1}, {"b": 2}])

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]


def test_lazyppl_completion_prompt_uses_integer_for_poisson_log_pmf() -> None:
    prompt = load_prompt_text("completion_gen_lazyppl.md")

    assert "poissonLogPmf      :: Double -> Integer -> Double" in prompt
    assert "Wrong count type" in prompt
    assert ":: Integer" in prompt
    assert "poissonLogPmf      :: Double -> Int -> Double" not in prompt


def test_lazyppl_completion_prompt_prioritizes_taskdata_scale() -> None:
    prompt = load_prompt_text("completion_gen_lazyppl.md")

    assert "observed TaskData summaries" in prompt
    assert "exact rows used by the verifier" in prompt
    assert "compute empirical" in prompt
    assert "PSIS/LOO scoring is part of validation" in prompt


def test_lazyppl_completion_prompt_warns_max_min_are_binary() -> None:
    prompt = load_prompt_text("completion_gen_lazyppl.md")

    assert "Wrong `max` / `min` arity" in prompt
    assert "Haskell `max` and `min` each take exactly two" in prompt
    assert "max lo (min hi x)" in prompt
    assert "max lo x hi" in prompt

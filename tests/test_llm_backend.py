from __future__ import annotations

from ppl_synth_data.llm.backend import (
    BackendCapabilities,
    BatchHandle,
    LLMRequest,
    LLMResult,
)


def test_llm_request_defaults() -> None:
    req = LLMRequest(model="m", system="s", user="u")
    assert req.model == "m"
    assert req.reasoning_effort is None
    assert req.max_output_tokens is None
    assert req.temperature is None
    assert req.text_format is None


def test_llm_request_all_fields() -> None:
    req = LLMRequest(
        model="gpt-4",
        system="sys",
        user="usr",
        reasoning_effort="high",
        max_output_tokens=1024,
        temperature=0.7,
        text_format={"type": "json_schema"},
    )
    assert req.reasoning_effort == "high"
    assert req.max_output_tokens == 1024
    assert req.temperature == 0.7
    assert req.text_format == {"type": "json_schema"}


def test_llm_result() -> None:
    result = LLMResult(content="hello", raw={"k": "v"})
    assert result.content == "hello"
    assert result.raw == {"k": "v"}


def test_batch_handle_defaults() -> None:
    handle = BatchHandle(backend="openai", job_id="123")
    assert handle.request_ids == ()


def test_batch_handle_with_ids() -> None:
    handle = BatchHandle(backend="openai", job_id="123", request_ids=("a", "b"))
    assert handle.request_ids == ("a", "b")


def test_backend_capabilities() -> None:
    caps = BackendCapabilities(
        supports_batch=True,
        supports_json_schema=False,
        supports_reasoning_effort=True,
    )
    assert caps.supports_batch is True
    assert caps.supports_json_schema is False
    assert caps.supports_reasoning_effort is True

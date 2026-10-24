from __future__ import annotations

from ppl_synth_data.llm.backend import LLMRequest
from ppl_synth_data.llm.litellm_backend import LiteLLMBackend


def test_capabilities() -> None:
    b = LiteLLMBackend()
    assert b.capabilities.supports_batch is False
    assert b.capabilities.supports_json_schema is False
    assert b.capabilities.supports_reasoning_effort is False


def test_build_completion_kwargs_minimal() -> None:
    b = LiteLLMBackend()
    req = LLMRequest(model="gpt-4", system="sys", user="usr")
    kwargs = b._build_completion_kwargs(req)
    assert kwargs["model"] == "gpt-4"
    assert len(kwargs["messages"]) == 2
    assert kwargs["messages"][0]["role"] == "system"
    assert "api_base" not in kwargs
    assert "temperature" not in kwargs


def test_build_completion_kwargs_all_options() -> None:
    b = LiteLLMBackend(
        api_base="http://localhost:8000",
        api_key="sk-test",
        custom_llm_provider="openai",
        extra_headers={"X-Custom": "val"},
        drop_params=True,
        num_retries=3,
    )
    req = LLMRequest(
        model="local/model",
        system="sys",
        user="usr",
        temperature=0.5,
    )
    kwargs = b._build_completion_kwargs(req)
    assert kwargs["api_base"] == "http://localhost:8000"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["extra_headers"] == {"X-Custom": "val"}
    assert kwargs["temperature"] == 0.5
    assert "max_tokens" not in kwargs
    assert kwargs["drop_params"] is True
    assert kwargs["num_retries"] == 3

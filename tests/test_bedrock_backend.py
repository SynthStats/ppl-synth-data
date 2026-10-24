from __future__ import annotations

import os
from typing import Any

import pytest

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import LLMRequest
from ppl_synth_data.llm.bedrock_backend import BedrockBackend


@pytest.mark.paid_lane
def test_real_bedrock_invoke_acceptance(monkeypatch) -> None:
    """Call the real AWS Bedrock Converse API through litellm.

    Requires Bedrock-enabled AWS credentials and access to the configured cross-region inference
    profile. AWS_DEFAULT_REGION is explicitly supplied because the backend requires that
    variable.
    """
    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        raise RuntimeError(
            "AWS credentials are required but not set. "
            "Set AWS_PROFILE to a Bedrock-enabled profile (or AWS_ACCESS_KEY_ID/SECRET) "
            "before running the paid_lane."
        )
    monkeypatch.setenv("AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    settings = Settings()

    backend = BedrockBackend(settings=settings)
    req = LLMRequest(
        model="bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        system="Reply with exactly one word.",
        user="Say hello.",
        max_output_tokens=16,
    )

    out = backend.complete(req)

    assert out.content.strip()
    assert isinstance(out.raw, dict)
    assert out.raw.get("id")
    usage = out.raw.get("usage")
    assert isinstance(usage, dict)
    assert usage.get("total_tokens", usage.get("totalTokens", 0)) > 0

    print(
        {
            "response_id": out.raw.get("id"),
            "model": out.raw.get("model"),
            "usage": usage,
            "content": out.content,
        }
    )


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


@pytest.mark.narrow_unit_not_acceptance
def test_bedrock_converse_complete_uses_iam_region_without_bearer(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token")
    monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "mantle-token")

    captured: dict[str, Any] = {}
    env_seen_by_litellm: dict[str, str | None] = {}

    def _fake_completion(**kwargs: Any) -> _Response:
        captured.update(kwargs)
        env_seen_by_litellm["AWS_BEARER_TOKEN_BEDROCK"] = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        env_seen_by_litellm["BEDROCK_MANTLE_API_KEY"] = os.environ.get("BEDROCK_MANTLE_API_KEY")
        return _Response({"choices": [{"message": {"content": "hello from bedrock"}}]})

    monkeypatch.setattr("ppl_synth_data.llm.bedrock_backend.litellm.completion", _fake_completion)

    backend = BedrockBackend(settings=Settings())
    req = LLMRequest(
        model="bedrock/converse/us.anthropic.claude-sonnet-4-6",
        system="system",
        user="user",
        temperature=0.2,
    )

    out = backend.complete(req)

    assert out.content == "hello from bedrock"
    assert captured["model"] == req.model
    assert captured["aws_region_name"] == "us-east-1"
    assert "api_key" not in captured
    assert env_seen_by_litellm["AWS_BEARER_TOKEN_BEDROCK"] is None
    assert env_seen_by_litellm["BEDROCK_MANTLE_API_KEY"] is None
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "bedrock-token"
    assert os.environ["BEDROCK_MANTLE_API_KEY"] == "mantle-token"
    assert captured["messages"][0]["content"] == "system"
    assert captured["messages"][1]["content"] == "user"


@pytest.mark.narrow_unit_not_acceptance
def test_bedrock_mantle_complete_uses_mantle_api_base(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token")

    captured: dict[str, Any] = {}

    def _fake_completion(**kwargs: Any) -> _Response:
        captured.update(kwargs)
        return _Response({"choices": [{"message": {"content": "hello from mantle"}}]})

    monkeypatch.setattr("ppl_synth_data.llm.bedrock_backend.litellm.completion", _fake_completion)

    backend = BedrockBackend(settings=Settings())
    req = LLMRequest(
        model="bedrock_mantle/deepseek.v3.2",
        system="system",
        user="user",
    )

    out = backend.complete(req)

    assert out.content == "hello from mantle"
    assert captured["model"] == "bedrock_mantle/deepseek.v3.2"
    assert captured["api_key"] == "bedrock-token"
    assert captured["api_base"] == "https://bedrock-mantle.us-east-1.api.aws/v1"
    assert captured["aws_region_name"] == "us-east-1"


@pytest.mark.narrow_unit_not_acceptance
def test_bedrock_complete_requires_region(monkeypatch) -> None:
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    backend = BedrockBackend(settings=Settings())
    req = LLMRequest(
        model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        system="system",
        user="user",
    )

    with pytest.raises(RuntimeError, match="AWS_DEFAULT_REGION"):
        backend.complete(req)


@pytest.mark.narrow_unit_not_acceptance
def test_bedrock_complete_invalid_choices_shape_raises_runtime_error(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    def _fake_completion(**_kwargs: Any) -> _Response:
        return _Response({"choices": [123]})

    monkeypatch.setattr("ppl_synth_data.llm.bedrock_backend.litellm.completion", _fake_completion)

    backend = BedrockBackend(settings=Settings())
    req = LLMRequest(
        model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        system="system",
        user="user",
    )

    with pytest.raises(RuntimeError, match="unsupported choices format"):
        backend.complete(req)

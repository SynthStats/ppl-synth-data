from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    supports_batch: bool
    supports_json_schema: bool
    supports_reasoning_effort: bool


@dataclass(frozen=True)
class LLMRequest:
    model: str
    system: str
    user: str
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    text_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMResult:
    content: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class BatchHandle:
    backend: str
    job_id: str
    request_ids: tuple[str, ...] = ()


def build_chat_kwargs(request: LLMRequest) -> dict[str, Any]:
    """Build the common model/messages/temperature/max_tokens dict from a request."""
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ],
    }
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        kwargs["max_tokens"] = request.max_output_tokens
    return kwargs


class LLMBackend(Protocol):
    capabilities: BackendCapabilities

    def complete(self, request: LLMRequest) -> LLMResult: ...


class BatchCapableLLMBackend(LLMBackend, Protocol):
    """Extends LLMBackend for backends that support provider-native batching."""

    def submit_batch(self, requests: list[tuple[str, LLMRequest]]) -> BatchHandle: ...

    def collect_batch(
        self,
        handle: BatchHandle,
        *,
        allow_partial_failures: bool = False,
    ) -> dict[str, LLMResult]: ...

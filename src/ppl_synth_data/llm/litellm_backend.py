from __future__ import annotations

from typing import Any

import litellm

from ppl_synth_data.llm.backend import (
    BackendCapabilities,
    LLMBackend,
    LLMRequest,
    LLMResult,
    build_chat_kwargs,
)
from ppl_synth_data.llm.response_utils import extract_chat_text


class LiteLLMBackend(LLMBackend):
    capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=False,
    )

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        custom_llm_provider: str | None = None,
        extra_headers: dict[str, str] | None = None,
        drop_params: bool | None = None,
        num_retries: int | None = None,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.custom_llm_provider = custom_llm_provider
        self.extra_headers = extra_headers
        self.drop_params = drop_params
        self.num_retries = num_retries

    def _build_completion_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs = build_chat_kwargs(request)
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.custom_llm_provider:
            kwargs["custom_llm_provider"] = self.custom_llm_provider
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if self.drop_params is not None:
            kwargs["drop_params"] = self.drop_params
        if self.num_retries is not None:
            kwargs["num_retries"] = self.num_retries
        return kwargs

    def complete(self, request: LLMRequest) -> LLMResult:
        kwargs = self._build_completion_kwargs(request)
        response = litellm.completion(**kwargs)
        raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        content = extract_chat_text(raw, backend_name="LiteLLM")
        return LLMResult(content=content, raw=raw)

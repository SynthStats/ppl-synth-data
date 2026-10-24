from __future__ import annotations

import os

import litellm

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import (
    BackendCapabilities,
    LLMBackend,
    LLMRequest,
    LLMResult,
    build_chat_kwargs,
)
from ppl_synth_data.llm.response_utils import extract_chat_text


class BedrockBackend(LLMBackend):
    """Minimal Bedrock backend via LiteLLM."""

    capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=False,
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

    def _resolve_region(self) -> str:
        region = (self._settings.aws_default_region or "").strip()
        if not region:
            raise RuntimeError(
                "AWS_DEFAULT_REGION is required for Bedrock backend. "
                "Set it in your environment or .env."
            )
        return region

    def complete(self, request: LLMRequest) -> LLMResult:
        kwargs = build_chat_kwargs(request)
        region = self._resolve_region()
        kwargs["aws_region_name"] = region
        is_mantle = request.model.startswith("bedrock_mantle/")
        if is_mantle:
            kwargs["api_base"] = f"https://bedrock-mantle.{region}.api.aws/v1"

        bearer = (
            self._settings.aws_bearer_token_bedrock.get_secret_value()
            if self._settings.aws_bearer_token_bedrock
            else ""
        )
        if is_mantle and bearer.strip():
            kwargs["api_key"] = bearer

        if is_mantle:
            response = litellm.completion(**kwargs)
        else:
            api_key_env = {
                name: os.environ.pop(name)
                for name in ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_MANTLE_API_KEY")
                if name in os.environ
            }
            try:
                response = litellm.completion(**kwargs)
            finally:
                os.environ.update(api_key_env)
        raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        content = extract_chat_text(raw, backend_name="Bedrock")
        return LLMResult(content=content, raw=raw)

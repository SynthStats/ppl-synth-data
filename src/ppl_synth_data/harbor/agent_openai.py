"""OpenAI-compatible response parsing and client construction for the Harbor agent."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ppl_synth_data.harbor.agent_shared import _json_mapping


def build_openai_client(*, base_url: str, api_key: str) -> Any:
    """Construct an AsyncOpenAI client pointed at a vLLM/OpenAI-compatible endpoint."""

    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def render_openai_message(choice: Mapping[str, Any]) -> str:
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""

    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content)

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, Sequence):
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = _json_mapping(arguments) or arguments
            if isinstance(name, str):
                parts.append(
                    "<tool_call>"
                    + json.dumps({"name": name, "arguments": arguments}, sort_keys=True)
                    + "</tool_call>"
                )

    return "\n".join(parts)

from __future__ import annotations

from typing import Any


def _join_content_parts(content_val: list[Any]) -> str | None:
    parts: list[str] = []
    for item in content_val:
        if isinstance(item, dict) and "text" in item:
            text = item["text"]
            parts.append(text if isinstance(text, str) else str(text))
        elif isinstance(item, str):
            parts.append(item)
    return "".join(parts) if parts else None


def _content_value_to_text(content_val: Any) -> str | None:
    if isinstance(content_val, str):
        return content_val
    if isinstance(content_val, list):
        return _join_content_parts(content_val)
    return None


def _extract_message(raw: dict[str, Any], backend_name: str) -> dict[str, Any]:
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{backend_name} response missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"{backend_name} response has unsupported choices format")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"{backend_name} response has unsupported message format")
    return message


def extract_chat_text(raw: dict[str, Any], *, backend_name: str) -> str:
    message = _extract_message(raw, backend_name)
    result = _content_value_to_text(message.get("content"))
    if result is not None:
        return result
    raise RuntimeError(f"{backend_name} response has unsupported content format")


def extract_candidate_texts(raw: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    choices = raw.get("choices") if isinstance(raw, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            text = _content_value_to_text(message.get("content"))
            if isinstance(text, str) and text:
                texts.append(text)

    output_items = raw.get("output") if isinstance(raw, dict) else None
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            text = _content_value_to_text(item.get("content"))
            if isinstance(text, str) and text:
                texts.append(text)

    output_text = raw.get("output_text") if isinstance(raw, dict) else None
    if isinstance(output_text, str) and output_text:
        texts.append(output_text)

    seen: set[str] = set()
    deduped: list[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped

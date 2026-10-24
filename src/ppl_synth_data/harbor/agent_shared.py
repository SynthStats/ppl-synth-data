"""Primitive mapping and coercion helpers shared by the Harbor agent modules.

These are runtime-neutral building blocks used by more than one Harbor agent
concern (environment parsing, program extraction, OpenAI response handling,
tokenizer/EOS scoring, and token-metadata extraction); generic dict/sequence
coercion rather than domain logic owned by any single one of them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _json_mapping(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _nested(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if mapping is None:
        return None
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_choice(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("LLM response does not contain choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("LLM response choice is not an object")
    return choice


def _coerce_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[int] = []
    for item in value:
        if not isinstance(item, int):
            return None
        result.append(item)
    return result


def _coerce_token_ids(value: Any) -> list[int] | None:
    token_ids = _coerce_int_list(value)
    if token_ids is not None:
        return token_ids
    if isinstance(value, Mapping):
        return _coerce_int_list(value.get("input_ids"))
    return _coerce_int_list(getattr(value, "input_ids", None))


def _coerce_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            return None
        result.append(float(item))
    return result


def _coerce_optional_float_list(value: Any) -> list[float | None] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[float | None] = []
    for item in value:
        if item is None:
            result.append(None)
            continue
        if not isinstance(item, int | float):
            return None
        result.append(float(item))
    return result


def _batched(items: Sequence[list[int]], *, chunk_size: int) -> list[list[list[int]]]:
    return [list(items[index : index + chunk_size]) for index in range(0, len(items), chunk_size)]


def _content_logprobs(logprobs: Any) -> list[Mapping[str, Any]]:
    content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    if not isinstance(content, Sequence):
        return []
    return [item for item in content if isinstance(item, Mapping)]

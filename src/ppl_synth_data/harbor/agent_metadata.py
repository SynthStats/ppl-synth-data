"""Strict token-metadata extraction from OpenAI/vLLM-compatible LLM responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ppl_synth_data.harbor.agent_shared import (
    _coerce_float_list,
    _coerce_int_list,
    _content_logprobs,
    _first_choice,
    _nested,
)
from ppl_synth_data.harbor.agent_tokenizer import _encode_prompt_messages, _encode_text


def extract_generation_metadata(
    response: Mapping[str, Any],
    *,
    model_name: str,
    response_text: str,
    prompt_messages: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Extract strict token metadata from an OpenAI/vLLM-compatible response."""

    choice = _first_choice(response)
    logprobs_obj = choice.get("logprobs")
    content_logprobs = _content_logprobs(logprobs_obj)

    token_ids = (
        _coerce_int_list(choice.get("completion_token_ids"))
        or _coerce_int_list(choice.get("token_ids"))
        or _coerce_int_list(_nested(logprobs_obj, "completion_token_ids"))
        or _coerce_int_list(_nested(logprobs_obj, "token_ids"))
    )
    if token_ids is None and content_logprobs:
        token_ids = _coerce_int_list([item.get("token_id") for item in content_logprobs])
    if token_ids is None:
        token_ids = _encode_text(
            model_name,
            response_text,
            add_special_tokens=False,
        )

    logprobs = (
        _coerce_float_list(choice.get("completion_logprobs"))
        or _coerce_float_list(choice.get("logprob_values"))
        or _coerce_float_list(_nested(logprobs_obj, "completion_logprobs"))
        or _coerce_float_list(_nested(logprobs_obj, "logprob_values"))
    )
    if logprobs is None and content_logprobs:
        logprobs = _coerce_float_list([item.get("logprob") for item in content_logprobs])

    if logprobs is None or len(token_ids) != len(logprobs):
        raise ValueError("LLM response must include aligned completion token ids and logprobs")

    prompt_ids = (
        _coerce_int_list(choice.get("prompt_token_ids"))
        or _coerce_int_list(choice.get("prompt_ids"))
        or _coerce_int_list(_nested(logprobs_obj, "prompt_token_ids"))
        or _coerce_int_list(_nested(logprobs_obj, "prompt_ids"))
        or _encode_prompt_messages(model_name, prompt_messages)
    )

    eos_logprobs = (
        _coerce_float_list(choice.get("eos_logprobs"))
        or _coerce_float_list(_nested(logprobs_obj, "eos_logprobs"))
        or []
    )
    if eos_logprobs and len(eos_logprobs) != len(token_ids):
        raise ValueError("eos_logprobs must align with completion token ids")

    token_entropies = (
        _coerce_float_list(choice.get("token_entropies"))
        or _coerce_float_list(_nested(logprobs_obj, "token_entropies"))
        or []
    )
    if token_entropies and len(token_entropies) != len(token_ids):
        raise ValueError("token_entropies must align with completion token ids")

    metadata: dict[str, Any] = {
        "token_ids": token_ids,
        "prompt_ids": prompt_ids,
        "mask_ids": [1] * len(token_ids),
        "logprobs": logprobs,
        "stop_reason": str(choice.get("finish_reason") or "completed"),
    }
    if eos_logprobs:
        metadata["eos_logprobs"] = eos_logprobs
    if token_entropies:
        metadata["token_entropies"] = token_entropies
        metadata["entropy_source"] = "llm_response.token_entropies"
    return metadata

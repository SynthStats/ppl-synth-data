"""EOS prompt-logprob scoring for exact per-prefix flow potentials (SubTB)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ppl_synth_data.harbor.agent_env import parse_positive_int_env
from ppl_synth_data.harbor.agent_shared import (
    _as_mapping,
    _batched,
    _coerce_int_list,
    _coerce_optional_float_list,
    _nested,
)
from ppl_synth_data.harbor.agent_tokenizer import _decode_prompt_token_ids, _eos_token_id_for_model

_EOS_SCORE_CHUNK_SIZE = 16
_EOS_SCORE_CHUNK_SIZE_ENV = "SUBTB_EOS_SCORE_CHUNK_SIZE"


def _choice_prompt_token_ids(choice: Mapping[str, Any]) -> list[int] | None:
    logprobs_obj = choice.get("logprobs")
    return (
        _coerce_int_list(choice.get("prompt_token_ids"))
        or _coerce_int_list(choice.get("prompt_ids"))
        or _coerce_int_list(_nested(logprobs_obj, "prompt_token_ids"))
        or _coerce_int_list(_nested(logprobs_obj, "prompt_ids"))
    )


def _last_token_prompt_logprob(
    choice: Mapping[str, Any],
    *,
    token_id: int,
    prompt_length: int,
) -> float:
    logprobs_obj = choice.get("logprobs")
    token_logprobs = (
        _coerce_optional_float_list(_nested(logprobs_obj, "token_logprobs"))
        or _coerce_optional_float_list(choice.get("token_logprobs"))
        or []
    )
    prompt_index = prompt_length - 1
    if len(token_logprobs) > prompt_index:
        prompt_token_logprob = token_logprobs[prompt_index]
        if prompt_token_logprob is not None:
            return float(prompt_token_logprob)

    prompt_logprobs = (
        choice.get("prompt_logprobs") or _nested(logprobs_obj, "prompt_logprobs") or []
    )
    if isinstance(prompt_logprobs, Sequence) and not isinstance(
        prompt_logprobs, (str, bytes, bytearray)
    ):
        last = prompt_logprobs[prompt_index] if len(prompt_logprobs) > prompt_index else None
        if isinstance(last, Mapping):
            for key in (token_id, str(token_id), f"token_id:{token_id}"):
                item = last.get(key)
                if isinstance(item, Mapping):
                    value = item.get("logprob")
                    if isinstance(value, int | float):
                        return float(value)
                elif isinstance(item, int | float):
                    return float(item)

    raise ValueError("vLLM prompt scoring response omitted the requested token logprob")


async def score_eos_logprobs_with_prompt_logprobs(
    client: Any,
    *,
    model_name: str,
    prompt_ids: Sequence[int],
    token_ids: Sequence[int],
    eos_token_id: int,
    chunk_size: int = _EOS_SCORE_CHUNK_SIZE,
) -> list[float]:
    """Score log p(EOS | prefix) for each action state using vLLM prompt logprobs.

    Each scoring prompt is the original prompt plus a generated prefix plus an
    EOS token. vLLM's prompt-logprob surface returns the log probability of the
    actual prompt token, so the final prompt-token logprob is the exact EOS
    potential for that prefix without relying on top-k inclusion.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not token_ids:
        return []

    prompt_prefix = [int(token_id) for token_id in prompt_ids]
    action_ids = [int(token_id) for token_id in token_ids]
    scoring_prompts = [
        prompt_prefix + action_ids[:index] + [int(eos_token_id)] for index in range(len(action_ids))
    ]

    eos_logprobs: list[float] = []
    for prompt_token_batch in _batched(scoring_prompts, chunk_size=chunk_size):
        prompt_text_batch = [
            _decode_prompt_token_ids(model_name, prompt_ids) for prompt_ids in prompt_token_batch
        ]
        response = await client.completions.create(
            model=model_name,
            prompt=prompt_text_batch,
            max_tokens=0,
            echo=True,
            temperature=0.0,
            logprobs=1,
            extra_body={
                "prompt_logprobs": 1,
                "return_token_ids": True,
            },
        )
        response_mapping = _as_mapping(response)
        if response_mapping is None:
            raise TypeError("vLLM prompt scoring response must be mapping-like")
        choices = response_mapping.get("choices")
        if not isinstance(choices, Sequence) or len(choices) != len(prompt_token_batch):
            raise ValueError("vLLM prompt scoring response choice count mismatch")
        for choice, scored_prompt in zip(choices, prompt_token_batch, strict=True):
            if not isinstance(choice, Mapping):
                raise ValueError("vLLM prompt scoring choice is not an object")
            returned_prompt_ids = _choice_prompt_token_ids(choice)
            prompt_length = (
                len(returned_prompt_ids) if returned_prompt_ids is not None else len(scored_prompt)
            )
            if returned_prompt_ids is not None and returned_prompt_ids[-1:] != [int(eos_token_id)]:
                raise ValueError(
                    "vLLM prompt scoring retokenized EOS scoring prompt without terminal EOS token"
                )
            eos_logprobs.append(
                _last_token_prompt_logprob(
                    choice,
                    token_id=int(eos_token_id),
                    prompt_length=prompt_length,
                )
            )

    if len(eos_logprobs) != len(token_ids):
        raise ValueError("EOS prompt scoring did not return one potential per action")
    return eos_logprobs


async def complete_subtb_generation_metadata(
    client: Any,
    *,
    model_name: str,
    generation_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a terminal EOS action and exact per-prefix EOS potentials."""

    metadata = dict(generation_metadata)
    eos_token_id = _eos_token_id_for_model(model_name)
    token_ids = [int(token_id) for token_id in metadata["token_ids"]]
    logprobs = [float(value) for value in metadata["logprobs"]]
    prompt_ids = [int(token_id) for token_id in metadata["prompt_ids"]]
    mask_ids = [int(value) for value in metadata.get("mask_ids", [1] * len(token_ids))]

    if len(token_ids) != len(logprobs) or len(token_ids) != len(mask_ids):
        raise ValueError("generation metadata token_ids, logprobs, and mask_ids must align")

    if not token_ids or token_ids[-1] != eos_token_id:
        token_ids.append(eos_token_id)
        mask_ids.append(1)

    eos_logprobs = await score_eos_logprobs_with_prompt_logprobs(
        client,
        model_name=model_name,
        prompt_ids=prompt_ids,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        chunk_size=parse_positive_int_env(
            _EOS_SCORE_CHUNK_SIZE_ENV,
            default=_EOS_SCORE_CHUNK_SIZE,
        ),
    )
    if len(logprobs) < len(token_ids):
        logprobs.append(eos_logprobs[-1])

    metadata["token_ids"] = token_ids
    metadata["logprobs"] = logprobs
    metadata["mask_ids"] = mask_ids
    metadata["eos_logprobs"] = eos_logprobs
    metadata["terminal_eos_token_id"] = eos_token_id
    metadata["terminal_eos_token_count"] = 1
    metadata["terminal_eos_token_coverage"] = 1.0
    metadata["eos_potential_coverage"] = 1.0
    metadata["eos_logprob_source"] = "vllm.prompt_logprobs"
    return metadata

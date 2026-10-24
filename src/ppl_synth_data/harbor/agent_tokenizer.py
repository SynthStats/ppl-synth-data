"""Tokenizer cache, trust_remote_code allowlist, and token encode/decode helpers.

Setting trust_remote_code to true executes arbitrary Python shipped in a model
repo at load time, so every model defaults to trust_remote_code being false; a
model may load with remote code only if its exact id is pinned in
`_TRUST_REMOTE_CODE_ALLOWLIST` to a revision whose remote code has been
reviewed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ppl_synth_data.harbor.agent_shared import _coerce_int_list, _coerce_token_ids

_TOKENIZER_CACHE: dict[str, Any] = {}

# Exact model id -> pinned revision SHA. Setting trust_remote_code to true
# executes arbitrary Python shipped in a model repo at load time, so every
# model defaults to trust_remote_code=False; a model may load with remote code ONLY
# if its exact id is pinned here to a revision whose remote code has been
# reviewed. Empty by default: the project's pinned `transformers` git build
# (see synthstats/pyproject.toml) loads the Qwen3.5 family natively via
# AutoConfig/AutoModel, so no model currently in use needs remote code.
_TRUST_REMOTE_CODE_ALLOWLIST: dict[str, str] = {}


def _resolve_trust_remote_code(
    model_name: str, *, request_trust: bool = False
) -> tuple[bool, str | None]:
    """Resolve (trust_remote_code, revision) for a HF model load.

    Fails loudly rather than silently downgrading: if a caller explicitly asks
    for remote code (`request_trust=True`) on a model outside the allowlist,
    raise instead of quietly returning `False` for a request nobody vetted.
    """
    revision = _TRUST_REMOTE_CODE_ALLOWLIST.get(model_name)
    if revision is None:
        if request_trust:
            raise RuntimeError(
                f"trust_remote_code requested for {model_name!r}, which is not "
                "in _TRUST_REMOTE_CODE_ALLOWLIST. Add an exact model id -> "
                "reviewed revision SHA entry before enabling remote code for "
                "this model."
            )
        return False, None
    return True, revision


def _tokenizer_for_model(model_name: str) -> Any:
    tokenizer = _TOKENIZER_CACHE.get(model_name)
    if tokenizer is not None:
        return tokenizer

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LLM response omitted token ids and transformers is not installed for "
            "deterministic tokenizer fallback"
        ) from exc

    trust_remote_code, revision = _resolve_trust_remote_code(model_name)
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision is not None:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def _encode_text(model_name: str, text: str, *, add_special_tokens: bool) -> list[int]:
    tokenizer = _tokenizer_for_model(model_name)
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    token_ids = _coerce_int_list(encoded)
    if token_ids is None:
        raise TypeError(f"Tokenizer for {model_name!r} did not return integer token ids")
    return token_ids


def _encode_prompt_messages(
    model_name: str,
    prompt_messages: Sequence[Mapping[str, str]],
) -> list[int]:
    tokenizer = _tokenizer_for_model(model_name)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": False,
            "enable_thinking": False,
        }
        try:
            encoded = apply_chat_template(list(prompt_messages), **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                encoded = apply_chat_template(list(prompt_messages), **kwargs)
            except TypeError:
                kwargs.pop("return_dict", None)
                encoded = apply_chat_template(list(prompt_messages), **kwargs)
        token_ids = _coerce_token_ids(encoded)
        if token_ids is None:
            raise TypeError(
                f"Tokenizer chat template for {model_name!r} did not return integer token ids"
            )
        return token_ids

    prompt_text = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in prompt_messages
    )
    return _encode_text(model_name, prompt_text, add_special_tokens=True)


def _eos_token_id_for_model(model_name: str) -> int:
    tokenizer = _tokenizer_for_model(model_name)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise RuntimeError(f"Tokenizer for {model_name!r} does not expose eos_token_id")
    try:
        return int(eos_token_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Tokenizer for {model_name!r} exposes non-integer eos_token_id"
        ) from exc


def _decode_prompt_token_ids(model_name: str, token_ids: Sequence[int]) -> str:
    tokenizer = _tokenizer_for_model(model_name)
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError(f"Tokenizer for {model_name!r} does not expose decode()")
    text = decode(
        [int(token_id) for token_id in token_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(text, str):
        raise TypeError(f"Tokenizer for {model_name!r} returned non-string decoded text")
    return text

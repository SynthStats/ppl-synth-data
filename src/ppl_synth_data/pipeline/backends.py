from __future__ import annotations

import logging
from typing import Any, TypeVar

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import BackendCapabilities, LLMBackend
from ppl_synth_data.pipeline.config_defaults import PipelineDefaults, load_default_pipeline_defaults

log = logging.getLogger(__name__)
_BackendT = TypeVar("_BackendT")


def resolve_backend_name(
    name: str | None,
    *,
    defaults: PipelineDefaults | None = None,
) -> str:
    if name is not None and name.strip():
        return name.strip()
    resolved_defaults = defaults or load_default_pipeline_defaults()
    return resolved_defaults.default_backend


def resolve_judge_backend_name(
    name: str | None,
    *,
    defaults: PipelineDefaults | None = None,
) -> str:
    if name is not None and name.strip():
        return name.strip()
    resolved_defaults = defaults or load_default_pipeline_defaults()
    return resolved_defaults.judge_backend


def resolve_stage_model(
    stage: str,
    model: str | None,
    *,
    defaults: PipelineDefaults | None = None,
) -> str:
    if model is not None and model.strip():
        return model.strip()
    resolved_defaults = defaults or load_default_pipeline_defaults()
    _stage_model_map: dict[str, str] = {
        "envspec": resolved_defaults.stages.envspec.model,
        "prompts": resolved_defaults.stages.prompts.model,
        "completions": resolved_defaults.stages.completions.model,
        "repair": resolved_defaults.models.repair,
        "judge": resolved_defaults.models.judge,
    }
    try:
        return _stage_model_map[stage]
    except KeyError:
        raise ValueError(f"unsupported stage model key: {stage}") from None


_BACKEND_DEPS: dict[str, str] = {
    "openai": "openai",
    "litellm": "litellm",
    "bedrock": "litellm",
    "bedrock_converse": "boto3",
    "proxy": "litellm",
    "glm": "litellm",
    "minimax": "litellm",
    "kimi": "litellm",
}


def _litellm_common_config(settings: Settings) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if settings.litellm_num_retries is not None:
        config["num_retries"] = settings.litellm_num_retries
    return config


def _resolve_kimi_api_base(settings: Settings) -> str:
    api_key = settings.require_kimi_key()
    api_base = settings.kimi_api_base.rstrip("/")

    if api_base.endswith("/v1") and "api.kimi.com/coding" in api_base:
        return api_base[:-3]
    if "api.kimi.com/coding" in api_base:
        return api_base
    if api_key.startswith("sk-kimi-"):
        return "https://api.kimi.com/coding"
    return api_base


_LITELLM_CONFIG: dict[str, Any] = {
    "litellm": lambda s: _litellm_common_config(s),
    "proxy": lambda s: {
        **_litellm_common_config(s),
        "api_base": s.proxy_api_base,
        "api_key": s.require_proxy_key(),
        "custom_llm_provider": "openai",
        "drop_params": True,
    },
    "glm": lambda s: {
        **_litellm_common_config(s),
        "api_base": s.glm_api_base,
        "api_key": s.require_glm_key(),
        "custom_llm_provider": "anthropic",
        "drop_params": True,
    },
    "minimax": lambda s: {
        **_litellm_common_config(s),
        "api_base": s.minimax_api_base,
        "api_key": s.require_minimax_key(),
        "custom_llm_provider": "openai",
        "drop_params": True,
    },
    "kimi": lambda s: {
        **_litellm_common_config(s),
        "api_base": _resolve_kimi_api_base(s),
        "api_key": s.require_kimi_key(),
        "custom_llm_provider": "anthropic",
        "extra_headers": {
            "User-Agent": "ppl-synth-data/0.1.0 ",
        },
        "drop_params": True,
    },
}


def _import_litellm_backend():
    try:
        from ppl_synth_data.llm.litellm_backend import LiteLLMBackend
    except ImportError:
        raise RuntimeError(
            "LiteLLM-based backends require the litellm package. "
            "Install it with: uv pip install litellm"
        ) from None
    return LiteLLMBackend


def _log_backend_instance(name: str, backend: _BackendT) -> _BackendT:
    caps = getattr(backend, "capabilities", None)
    if not isinstance(caps, BackendCapabilities):
        log.info("built backend %s", name)
        return backend
    log.info(
        (
            "built backend %s (supports_batch=%s, supports_json_schema=%s, "
            "supports_reasoning_effort=%s)"
        ),
        name,
        caps.supports_batch,
        caps.supports_json_schema,
        caps.supports_reasoning_effort,
    )
    return backend


def build_backend(name: str | None, settings: Settings | None = None) -> LLMBackend:
    key = resolve_backend_name(name).strip().lower()
    if key == "bedrock_converse":
        from ppl_synth_data.llm.bedrock_converse_backend import BedrockConverseBackend

        return _log_backend_instance(key, BedrockConverseBackend())
    s = settings or Settings()
    if key == "openai":
        try:
            from ppl_synth_data.llm.openai_backend import OpenAIBackend
        except ImportError:
            raise RuntimeError(
                f"backend '{key}' requires the openai package. "
                "Install it with: uv pip install openai"
            ) from None
        return _log_backend_instance(key, OpenAIBackend(settings=s))
    if key == "bedrock":
        try:
            from ppl_synth_data.llm.bedrock_backend import BedrockBackend
        except ImportError:
            raise RuntimeError(
                f"backend '{key}' requires the litellm package. "
                "Install it with: uv pip install litellm"
            ) from None
        return _log_backend_instance(key, BedrockBackend(settings=s))
    if key in _LITELLM_CONFIG:
        LiteLLMBackend = _import_litellm_backend()
        return _log_backend_instance(key, LiteLLMBackend(**_LITELLM_CONFIG[key](s)))
    raise ValueError(f"unsupported backend: {name!r}")

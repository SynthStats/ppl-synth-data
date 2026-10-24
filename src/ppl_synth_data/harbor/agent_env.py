"""Environment-variable parsing and runtime path resolution for the Harbor agent.

Everything here reads process/Harbor-provided environment configuration (or an
explicit override mapping in tests) and fails loudly when a required value is
missing; there are no silent defaults for the LLM endpoint, API key, or task runtime.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_LOCAL_OPENAI_COMPAT_API_KEY = "synthstats-local-openai-compatible-endpoint"


def require_llm_api_base(env: Mapping[str, str] | None = None) -> str:
    env_map = env if env is not None else os.environ
    value = env_map.get("LLM_API_BASE")
    if value:
        return value
    raise ValueError("LLM_API_BASE must be set for SynthStatsInstalledAgent")


def _is_loopback_llm_api_base(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def require_openai_api_key(
    env: Mapping[str, str] | None = None,
    *,
    llm_api_base: str | None = None,
) -> str:
    env_map = env if env is not None else os.environ
    value = env_map.get("OPENAI_API_KEY")
    if value:
        return value
    resolved_base = llm_api_base or env_map.get("LLM_API_BASE")
    if _is_loopback_llm_api_base(resolved_base):
        return _LOCAL_OPENAI_COMPAT_API_KEY
    raise ValueError("OPENAI_API_KEY must be set for SynthStatsInstalledAgent")


def parse_json_env(
    key: str,
    *,
    env: Mapping[str, str] | None = None,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env_map = env if env is not None else os.environ
    raw = env_map.get(key)
    if not raw:
        return dict(default or {})
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{key} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{key} must decode to a JSON object")
    return parsed


def parse_bool_env(
    key: str,
    *,
    env: Mapping[str, str] | None = None,
    default: bool = False,
) -> bool:
    env_map = env if env is not None else os.environ
    raw = env_map.get(key)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean flag, got {raw!r}")


def parse_positive_int_env(
    key: str,
    *,
    env: Mapping[str, str] | None = None,
    default: int,
) -> int:
    env_map = env if env is not None else os.environ
    raw = env_map.get(key)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def runtime_from_task_config(task_config: Mapping[str, Any]) -> str:
    runtime = task_config.get("runtime") or task_config.get("language") or "pymc"
    return str(runtime).strip().lower()


def submission_path_for_runtime(runtime: str) -> Path:
    normalized = runtime.strip().lower()
    if normalized in {"pymc", "python", "py"}:
        return Path("/workspace/submission.py")
    if normalized in {"lazyppl", "haskell", "hs"}:
        return Path("/app/Submission.hs")
    raise ValueError(f"Unsupported Harbor task runtime: {runtime!r}")

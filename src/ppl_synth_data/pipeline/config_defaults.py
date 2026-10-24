from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

log = logging.getLogger(__name__)
_OMIT_REASONING_EFFORT = {"omit", "null", "off"}


@dataclass(frozen=True)
class ModelDefaults:
    envspec: str
    prompt: str
    completion: str
    repair: str
    judge: str


@dataclass(frozen=True)
class StageRequestDefaults:
    model: str
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class StageDefaults:
    envspec: StageRequestDefaults
    prompts: StageRequestDefaults
    completions: StageRequestDefaults


@dataclass(frozen=True)
class PipelineDefaults:
    run_root: str
    default_backend: str
    judge_backend: str
    models: ModelDefaults
    stages: StageDefaults


def _read_yaml(path: Path, *, tolerate_errors: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        if not tolerate_errors:
            raise
        log.warning("ignoring malformed config file: %s", path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_mappings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_str(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return fallback


def _as_opt_str(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _as_opt_int(value: Any, fallback: int | None = None) -> int | None:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return fallback


def effective_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip().lower() in _OMIT_REASONING_EFFORT:
        return None
    return value


def _configured_reasoning_effort(value: Any, fallback: str) -> str:
    raw = _as_opt_str(value)
    if raw is None:
        return fallback
    if effective_reasoning_effort(raw) is None:
        return "omit"
    return raw


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_default_config_layers(
    *,
    cwd: Path | None = None,
    repo_root: Path | None = None,
) -> tuple[Path, Path | None]:
    cwd_path = Path.cwd() if cwd is None else Path(cwd)
    local_dir = cwd_path / "configs"
    repo_dir = ((repo_root or _repo_root()) / "configs").resolve()

    has_local_default = (local_dir / "default.yaml").is_file()
    local_stage_dir = local_dir / "stages"
    has_local_stage_override = local_stage_dir.is_dir() and any(local_stage_dir.glob("*.yaml"))

    if repo_dir.exists() and repo_dir != local_dir.resolve():
        overlay_dir = local_dir if (has_local_default or has_local_stage_override) else None
        return repo_dir, overlay_dir

    return local_dir, None


def _build_pipeline_defaults(
    default_cfg: dict[str, Any],
    envspec_cfg: dict[str, Any],
    prompts_cfg: dict[str, Any],
    completions_cfg: dict[str, Any],
) -> PipelineDefaults:

    run_cfg = default_cfg.get("run", {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    providers_cfg = default_cfg.get("providers", {})
    if not isinstance(providers_cfg, dict):
        providers_cfg = {}
    openai_cfg = providers_cfg.get("openai", {})
    if not isinstance(openai_cfg, dict):
        openai_cfg = {}

    judge_cfg = providers_cfg.get("judge", {})
    if not isinstance(judge_cfg, dict):
        judge_cfg = {}

    models = ModelDefaults(
        envspec=_as_str(openai_cfg.get("model_envspec"), "gpt-5.5"),
        prompt=_as_str(openai_cfg.get("model_prompt"), "gpt-5.5"),
        completion=_as_str(openai_cfg.get("model_completion"), "gpt-5.5"),
        repair=_as_str(openai_cfg.get("model_repair"), "gpt-5.5"),
        judge=_as_str(judge_cfg.get("model"), "glm-5"),
    )

    stage_envspec = StageRequestDefaults(
        model=_as_str(envspec_cfg.get("model"), models.envspec),
        reasoning_effort=_configured_reasoning_effort(
            envspec_cfg.get("reasoning_effort"),
            "high",
        ),
        max_output_tokens=_as_opt_int(envspec_cfg.get("max_output_tokens")),
    )
    stage_prompts = StageRequestDefaults(
        model=_as_str(prompts_cfg.get("model"), models.prompt),
        reasoning_effort=_configured_reasoning_effort(
            prompts_cfg.get("reasoning_effort"),
            "medium",
        ),
        max_output_tokens=_as_opt_int(prompts_cfg.get("max_output_tokens")),
    )
    stage_completions = StageRequestDefaults(
        model=_as_str(completions_cfg.get("model"), models.completion),
        reasoning_effort=_configured_reasoning_effort(
            completions_cfg.get("reasoning_effort"),
            "high",
        ),
        max_output_tokens=_as_opt_int(completions_cfg.get("max_output_tokens")),
    )

    return PipelineDefaults(
        run_root=_as_str(run_cfg.get("root"), "artifacts/runs"),
        default_backend=_as_str(providers_cfg.get("default_backend"), "openai"),
        judge_backend=_as_str(judge_cfg.get("backend"), "glm"),
        models=models,
        stages=StageDefaults(
            envspec=stage_envspec,
            prompts=stage_prompts,
            completions=stage_completions,
        ),
    )


def _load_pipeline_defaults_from_layers(
    base_dir: Path,
    *,
    overlay_dir: Path | None = None,
    tolerate_overlay_errors: bool = False,
) -> PipelineDefaults:
    if not base_dir.is_dir():
        log.warning("config dir %s not found", base_dir)

    default_cfg = _read_yaml(base_dir / "default.yaml")
    envspec_cfg = _read_yaml(base_dir / "stages" / "envspec.yaml")
    prompts_cfg = _read_yaml(base_dir / "stages" / "prompts.yaml")
    completions_cfg = _read_yaml(base_dir / "stages" / "completions.yaml")

    if overlay_dir is not None:
        default_cfg = _merge_mappings(
            default_cfg,
            _read_yaml(
                overlay_dir / "default.yaml",
                tolerate_errors=tolerate_overlay_errors,
            ),
        )
        envspec_cfg = _merge_mappings(
            envspec_cfg,
            _read_yaml(
                overlay_dir / "stages" / "envspec.yaml",
                tolerate_errors=tolerate_overlay_errors,
            ),
        )
        prompts_cfg = _merge_mappings(
            prompts_cfg,
            _read_yaml(
                overlay_dir / "stages" / "prompts.yaml",
                tolerate_errors=tolerate_overlay_errors,
            ),
        )
        completions_cfg = _merge_mappings(
            completions_cfg,
            _read_yaml(
                overlay_dir / "stages" / "completions.yaml",
                tolerate_errors=tolerate_overlay_errors,
            ),
        )

    return _build_pipeline_defaults(default_cfg, envspec_cfg, prompts_cfg, completions_cfg)


def load_pipeline_defaults(config_dir: Path | str) -> PipelineDefaults:
    return _load_pipeline_defaults_from_layers(Path(config_dir))


def load_default_pipeline_defaults(
    *,
    cwd: Path | None = None,
    repo_root: Path | None = None,
) -> PipelineDefaults:
    if cwd is None and repo_root is None:
        return _load_default_pipeline_defaults_cached(
            cwd=str(Path.cwd().resolve()),
            repo_root=str(_repo_root().resolve()),
        )

    base_dir, overlay_dir = resolve_default_config_layers(cwd=cwd, repo_root=repo_root)
    return _load_pipeline_defaults_from_layers(
        base_dir,
        overlay_dir=overlay_dir,
        tolerate_overlay_errors=True,
    )


@lru_cache(maxsize=16)
def _load_default_pipeline_defaults_cached(*, cwd: str, repo_root: str) -> PipelineDefaults:
    base_dir, overlay_dir = resolve_default_config_layers(
        cwd=Path(cwd),
        repo_root=Path(repo_root),
    )
    return _load_pipeline_defaults_from_layers(
        base_dir,
        overlay_dir=overlay_dir,
        tolerate_overlay_errors=True,
    )

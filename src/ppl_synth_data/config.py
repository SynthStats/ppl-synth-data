from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_run_root() -> str:
    from ppl_synth_data.pipeline.config_defaults import (
        load_default_pipeline_defaults,
        resolve_default_config_layers,
    )

    base_dir, overlay_dir = resolve_default_config_layers()
    if not (base_dir / "default.yaml").exists() and not (
        overlay_dir is not None and (overlay_dir / "default.yaml").exists()
    ):
        raise RuntimeError(
            "configs/default.yaml not found. Create it with at least: run:\n  root: artifacts/runs"
        )

    value = load_default_pipeline_defaults().run_root
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "configs/default.yaml missing 'run.root' string. Add: run:\n  root: artifacts/runs"
        )
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_org_id: str | None = Field(default=None, validation_alias="OPENAI_ORG_ID")
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")

    aws_default_region: str | None = Field(default=None, validation_alias="AWS_DEFAULT_REGION")
    aws_bearer_token_bedrock: SecretStr | None = Field(
        default=None,
        validation_alias="AWS_BEARER_TOKEN_BEDROCK",
    )

    proxy_api_base: str | None = Field(default=None, validation_alias="PROXY_API_BASE")
    proxy_api_key: SecretStr | None = Field(default=None, validation_alias="PROXY_API_KEY")

    glm_api_base: str | None = Field(default=None, validation_alias="GLM_API_BASE")
    glm_api_key: SecretStr | None = Field(default=None, validation_alias="GLM_API_KEY")
    minimax_api_base: str = Field(
        default="https://api.minimax.io/v1/", validation_alias="MINIMAX_API_BASE"
    )
    minimax_api_key: SecretStr | None = Field(default=None, validation_alias="MINIMAX_API_KEY")
    kimi_api_base: str = Field(
        default="https://api.kimi.com/coding", validation_alias="KIMI_API_BASE"
    )
    kimi_api_key: SecretStr | None = Field(default=None, validation_alias="KIMI_API_KEY")
    litellm_num_retries: int | None = Field(
        default=None,
        ge=0,
        validation_alias="LITELLM_NUM_RETRIES",
    )

    lazyppl_harbor_image: str = Field(
        default="",
        validation_alias="LAZYPPL_HARBOR_IMAGE",
    )

    run_root: str = Field(default_factory=_default_run_root)

    def require_openai(self) -> str:
        val = self.openai_api_key.get_secret_value() if self.openai_api_key else ""
        if not val.strip():
            raise RuntimeError(
                "OPENAI_API_KEY is required but not set. Add it to .env or export it."
            )
        return val

    def require_hf_token(self) -> str:
        val = self.hf_token.get_secret_value() if self.hf_token else ""
        if not val.strip():
            raise RuntimeError("HF_TOKEN is required but not set. Add it to .env or export it.")
        return val

    def require_proxy_key(self) -> str:
        val = self.proxy_api_key.get_secret_value() if self.proxy_api_key else ""
        if not val.strip():
            raise RuntimeError(
                "PROXY_API_KEY is required but not set. Add it to .env or export it."
            )
        return val

    def require_glm_key(self) -> str:
        val = self.glm_api_key.get_secret_value() if self.glm_api_key else ""
        if not val.strip():
            raise RuntimeError("GLM_API_KEY is required but not set. Add it to .env or export it.")
        return val

    def require_minimax_key(self) -> str:
        val = self.minimax_api_key.get_secret_value() if self.minimax_api_key else ""
        if not val.strip():
            raise RuntimeError(
                "MINIMAX_API_KEY is required but not set. Add it to .env or export it."
            )
        return val

    def require_kimi_key(self) -> str:
        val = self.kimi_api_key.get_secret_value() if self.kimi_api_key else ""
        if not val.strip():
            raise RuntimeError("KIMI_API_KEY is required but not set. Add it to .env or export it.")
        return val

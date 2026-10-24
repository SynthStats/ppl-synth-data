"""Tests for provider backend factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ppl_synth_data.pipeline.backends import build_backend


@pytest.mark.narrow_unit_not_acceptance
def test_proxy_backend_factory() -> None:
    with patch("ppl_synth_data.pipeline.backends.Settings") as mock_settings:
        mock_settings.return_value.proxy_api_base = "https://example.com/v1"
        mock_settings.return_value.proxy_api_key = "test-key"
        mock_settings.return_value.litellm_num_retries = 4
        mock_settings.return_value.require_proxy_key.return_value = "test-key"

        backend = build_backend("proxy")

    from ppl_synth_data.llm.litellm_backend import LiteLLMBackend

    assert isinstance(backend, LiteLLMBackend)
    assert backend.api_base == "https://example.com/v1"
    assert backend.api_key == "test-key"
    assert backend.custom_llm_provider == "openai"
    assert backend.num_retries == 4


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="unsupported backend"):
        build_backend("nonexistent_backend")


def test_backend_deps_includes_new_providers() -> None:
    from ppl_synth_data.pipeline.backends import _BACKEND_DEPS

    assert "proxy" in _BACKEND_DEPS
    assert _BACKEND_DEPS["proxy"] == "litellm"

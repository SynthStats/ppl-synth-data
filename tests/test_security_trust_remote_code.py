"""The tokenizer loader disables remote code by default.

Only an exact model ID with a revision SHA in _TRUST_REMOTE_CODE_ALLOWLIST
may enable remote code. Tests replace transformers at the import boundary.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from ppl_synth_data.harbor import agent_tokenizer as agent_tokenizer_module
from ppl_synth_data.harbor.agent_tokenizer import _resolve_trust_remote_code, _tokenizer_for_model


def test_resolve_trust_remote_code_defaults_false_for_unknown_model() -> None:
    trust, revision = _resolve_trust_remote_code("org/some-unreviewed-model")

    assert trust is False
    assert revision is None


def test_resolve_trust_remote_code_raises_for_unallowlisted_explicit_request() -> None:
    with pytest.raises(RuntimeError, match="_TRUST_REMOTE_CODE_ALLOWLIST"):
        _resolve_trust_remote_code("org/some-unreviewed-model", request_trust=True)


def test_resolve_trust_remote_code_honors_allowlist_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_tokenizer_module,
        "_TRUST_REMOTE_CODE_ALLOWLIST",
        {"org/reviewed-model": "deadbeefcafe"},
    )

    trust, revision = _resolve_trust_remote_code("org/reviewed-model")

    assert trust is True
    assert revision == "deadbeefcafe"


def test_tokenizer_for_model_loads_with_trust_remote_code_false_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def _fake_from_pretrained(model_name: str, **kwargs: object) -> object:
        calls.update({"model_name": model_name, **kwargs})
        return types.SimpleNamespace(model_name=model_name)

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=_fake_from_pretrained)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(agent_tokenizer_module, "_TOKENIZER_CACHE", {})

    _tokenizer_for_model("org/unreviewed-model-security-test")

    assert calls["trust_remote_code"] is False
    assert "revision" not in calls


def test_tokenizer_for_model_uses_allowlisted_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_from_pretrained(model_name: str, **kwargs: object) -> object:
        calls.update({"model_name": model_name, **kwargs})
        return types.SimpleNamespace(model_name=model_name)

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=_fake_from_pretrained)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(agent_tokenizer_module, "_TOKENIZER_CACHE", {})
    monkeypatch.setattr(
        agent_tokenizer_module,
        "_TRUST_REMOTE_CODE_ALLOWLIST",
        {"org/reviewed-model-security-test": "deadbeefcafe"},
    )

    _tokenizer_for_model("org/reviewed-model-security-test")

    assert calls["trust_remote_code"] is True
    assert calls["revision"] == "deadbeefcafe"


def test_agent_source_has_no_unconditional_trust_remote_code_true() -> None:
    """Only allowlisted tokenizer calls enable remote code."""
    source = Path(agent_tokenizer_module.__file__).read_text(encoding="utf-8")

    assert re.search(r"trust_remote_code\s*=\s*True", source) is None

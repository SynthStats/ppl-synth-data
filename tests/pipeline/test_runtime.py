from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppl_synth_data.llm.backend import BackendCapabilities, BatchHandle, LLMRequest, LLMResult
from ppl_synth_data.pipeline.backends import build_backend, resolve_stage_model
from ppl_synth_data.pipeline.concurrency import execute_requests, execute_requests_raw
from ppl_synth_data.pipeline.env_loader import build_query, load_env_instance

from .helpers import make_numeric_spec


def test_build_query_uses_json_for_multi_feature_float_inputs() -> None:
    spec = make_numeric_spec(["x1", "x2", "x3"])

    query = build_query(spec, idx=4, rng=np.random.default_rng(0))

    payload = json.loads(query)
    assert list(payload) == ["x1", "x2", "x3"]
    assert all(isinstance(v, float) for v in payload.values())


def test_build_query_keeps_scalar_for_single_feature_float_inputs() -> None:
    spec = make_numeric_spec(["x"])

    query = build_query(spec, idx=4, rng=np.random.default_rng(0))

    assert isinstance(query, str)
    float(query)


def test_build_query_uses_json_for_multi_feature_number_inputs() -> None:
    spec = make_numeric_spec(["x1", "x2"], input_type="number")

    query = build_query(spec, idx=3, rng=np.random.default_rng(0))

    payload = json.loads(query)
    assert list(payload) == ["x1", "x2"]
    assert all(isinstance(v, float) for v in payload.values())


@pytest.mark.narrow_unit_not_acceptance
def test_build_backend_supports_bedrock(monkeypatch) -> None:
    marker = object()
    monkeypatch.setattr(
        "ppl_synth_data.llm.bedrock_backend.BedrockBackend",
        lambda settings: marker,
    )

    out = build_backend("bedrock")
    assert out is marker


def test_resolve_stage_model_strips_explicit_cli_override() -> None:
    assert resolve_stage_model("prompts", "  explicit-model  ") == "explicit-model"


class _PartialBatchBackend:
    capabilities = BackendCapabilities(
        supports_batch=True,
        supports_json_schema=False,
        supports_reasoning_effort=True,
    )

    def complete(self, request: LLMRequest) -> LLMResult:
        _ = request
        raise NotImplementedError

    def submit_batch(self, requests: list[tuple[str, LLMRequest]]) -> BatchHandle:
        _ = requests
        return BatchHandle(backend="stub", job_id="job-1")

    def collect_batch(
        self,
        handle: BatchHandle,
        *,
        allow_partial_failures: bool = False,
    ) -> dict[str, LLMResult]:
        _ = handle
        _ = allow_partial_failures
        return {"only-one": LLMResult(content="ok", raw={})}


class _ArtifactBatchBackend(_PartialBatchBackend):
    def __init__(self) -> None:
        self.artifact_dir: Path | None = None
        self.last_batch_artifacts: dict[str, object] = {
            "input_jsonl_sha256": "a" * 64,
            "output_jsonl_sha256": "b" * 64,
        }

    def enable_batch_artifact_preservation(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir


class _FailingArtifactBatchBackend(_PartialBatchBackend):
    def __init__(self) -> None:
        self.artifact_dir: Path | None = None
        self.last_batch_failures: dict[str, str] = {}
        self.last_batch_artifacts: dict[str, object] = {}

    def enable_batch_artifact_preservation(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir

    def submit_batch(self, requests: list[tuple[str, LLMRequest]]) -> BatchHandle:
        return BatchHandle(
            backend="openai",
            job_id="batch-1",
            request_ids=tuple(request_id for request_id, _ in requests),
        )

    def collect_batch(
        self,
        handle: BatchHandle,
        *,
        allow_partial_failures: bool = False,
    ) -> dict[str, LLMResult]:
        _ = handle
        _ = allow_partial_failures
        self.last_batch_failures = {
            "req-1": "{'code': 'server_is_overloaded'}",
            "req-2": "{'code': 'server_is_overloaded'}",
        }
        assert self.artifact_dir is not None
        error_path = self.artifact_dir / "openai_batch_errors_batch-1.jsonl"
        self.last_batch_artifacts = {
            "schema": "ppl_synth_data.openai_batch_artifacts.v1",
            "batch_id": "batch-1",
            "error_jsonl_path": str(error_path),
            "error_jsonl_sha256": "c" * 64,
            "request_counts": {"completed": 0, "failed": 2, "total": 2},
            "recovered_failure_count": 2,
        }
        raise RuntimeError(
            "OpenAI batch ended with status=completed without output_file_id; "
            f"error_jsonl_path={error_path}"
        )


@pytest.mark.narrow_unit_not_acceptance
def test_execute_requests_reports_missing_batch_ids() -> None:
    backend = _PartialBatchBackend()
    requests = [
        ("req-1", LLMRequest(model="m", system="", user="u1")),
        ("req-2", LLMRequest(model="m", system="", user="u2")),
    ]

    try:
        execute_requests(backend, requests=requests, use_batch=True)
    except RuntimeError as exc:
        assert "missing" in str(exc)
        assert "req-1" in str(exc) or "req-2" in str(exc)
    else:
        raise AssertionError("expected missing-id runtime error")


@pytest.mark.narrow_unit_not_acceptance
def test_execute_requests_raw_returns_available_partial_batch_results() -> None:
    backend = _PartialBatchBackend()
    requests = [
        ("req-1", LLMRequest(model="m", system="", user="u1")),
        ("only-one", LLMRequest(model="m", system="", user="u2")),
        ("req-3", LLMRequest(model="m", system="", user="u3")),
    ]

    results = execute_requests_raw(
        backend,
        requests=requests,
        use_batch=True,
        allow_partial_batch_failures=True,
    )

    assert list(results) == ["only-one"]
    assert results["only-one"].content == "ok"


@pytest.mark.narrow_unit_not_acceptance
def test_execute_requests_raw_writes_redacted_batch_receipt(tmp_path: Path) -> None:
    backend = _PartialBatchBackend()
    requests = [
        ("req-1", LLMRequest(model="m", system="system secret", user="user secret")),
        ("only-one", LLMRequest(model="m", system="", user="u2")),
        ("req-3", LLMRequest(model="m", system="", user="u3")),
    ]

    execute_requests_raw(
        backend,
        requests=requests,
        use_batch=True,
        state_dir=tmp_path,
        allow_partial_batch_failures=True,
    )

    receipt_path = tmp_path / "batch_receipt.json"
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["schema"] == "ppl_synth_data.batch_receipt.v1"
    assert receipt["backend"] == "stub"
    assert receipt["job_id"] == "job-1"
    assert receipt["request_count"] == 3
    assert receipt["success_count"] == 1
    assert receipt["missing_count"] == 2
    assert receipt["success_request_ids"] == ["only-one"]
    assert receipt["missing_request_ids"] == ["req-1", "req-3"]
    assert "user secret" not in receipt_text
    assert "system secret" not in receipt_text


@pytest.mark.narrow_unit_not_acceptance
def test_execute_requests_raw_enables_batch_artifact_preservation(
    tmp_path: Path,
) -> None:
    backend = _ArtifactBatchBackend()

    execute_requests_raw(
        backend,
        requests=[
            ("only-one", LLMRequest(model="m", system="system", user="user")),
        ],
        use_batch=True,
        state_dir=tmp_path,
        allow_partial_batch_failures=False,
    )

    assert backend.artifact_dir == tmp_path / "openai_batch_artifacts"
    receipt = json.loads((tmp_path / "batch_receipt.json").read_text(encoding="utf-8"))
    assert receipt["backend_artifacts"] == {
        "input_jsonl_sha256": "a" * 64,
        "output_jsonl_sha256": "b" * 64,
    }
    assert "system" not in json.dumps(receipt)
    assert "user" not in json.dumps(receipt)


@pytest.mark.narrow_unit_not_acceptance
def test_execute_requests_raw_writes_failure_receipt_with_batch_artifacts(
    tmp_path: Path,
) -> None:
    backend = _FailingArtifactBatchBackend()

    requests = [
        ("req-1", LLMRequest(model="m", system="system secret", user="user secret 1")),
        ("req-2", LLMRequest(model="m", system="system secret", user="user secret 2")),
    ]

    try:
        execute_requests_raw(
            backend,
            requests=requests,
            use_batch=True,
            state_dir=tmp_path,
            allow_partial_batch_failures=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected provider batch failure")

    receipt_path = tmp_path / "batch_receipt.json"
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["status"] == "failed"
    assert receipt["backend"] == "openai"
    assert receipt["job_id"] == "batch-1"
    assert receipt["request_count"] == 2
    assert receipt["success_count"] == 0
    assert receipt["missing_count"] == 2
    assert receipt["backend_failure_count"] == 2
    assert receipt["backend_failure_request_ids"] == ["req-1", "req-2"]
    assert receipt["backend_artifacts"]["recovered_failure_count"] == 2
    assert receipt["backend_artifacts"]["error_jsonl_sha256"] == "c" * 64
    assert "openai_batch_errors_batch-1.jsonl" in receipt["exception_message"]
    assert "user secret" not in receipt_text
    assert "system secret" not in receipt_text
    assert not (tmp_path / "pending_batch.json").exists()


@pytest.mark.narrow_unit_not_acceptance
def test_execute_requests_allows_json_object_without_schema_capability() -> None:
    class _SyncBackend:
        capabilities = BackendCapabilities(
            supports_batch=False,
            supports_json_schema=False,
            supports_reasoning_effort=True,
        )

        def complete(self, request: LLMRequest) -> LLMResult:
            assert request.text_format == {"type": "json_object"}
            return LLMResult(content='{"ok": true}', raw={})

    results = execute_requests(
        _SyncBackend(),
        requests=[
            (
                "req-1",
                LLMRequest(
                    model="m",
                    system="system",
                    user="user",
                    text_format={"type": "json_object"},
                ),
            )
        ],
        use_batch=False,
    )

    assert results == {"req-1": '{"ok": true}'}


def test_load_env_instance_rejects_module_outside_trusted_root(tmp_path: Path) -> None:
    module = tmp_path / "rogue.py"
    module.write_text("class RogueEnv:\n  pass\n", encoding="utf-8")
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir(parents=True, exist_ok=True)

    try:
        load_env_instance(module, env_name="rogue", seed=1, trusted_root=trusted_root)
    except RuntimeError as exc:
        assert "escapes trusted root" in str(exc)
    else:
        raise AssertionError("expected trusted-root rejection")


def test_load_env_instance_reports_missing_class_with_context(tmp_path: Path) -> None:
    module = tmp_path / "valid.py"
    module.write_text("class SomethingElse:\n  pass\n", encoding="utf-8")

    try:
        load_env_instance(module, env_name="expected_name", seed=1)
    except RuntimeError as exc:
        assert "missing expected class" in str(exc)
        assert "ExpectedNameEnv" in str(exc)
    else:
        raise AssertionError("expected class lookup failure")


def test_batch_state_roundtrip(tmp_path: Path) -> None:
    from ppl_synth_data.llm.backend import BatchHandle
    from ppl_synth_data.pipeline.concurrency import (
        _clear_pending_batch,
        _load_pending_batch,
        _save_pending_batch,
    )

    handle = BatchHandle(backend="openai", job_id="batch_abc123", request_ids=("r1", "r2"))

    assert _load_pending_batch(tmp_path) is None

    _save_pending_batch(tmp_path, handle)
    loaded = _load_pending_batch(tmp_path)
    assert loaded is not None
    assert loaded.backend == "openai"
    assert loaded.job_id == "batch_abc123"
    assert loaded.request_ids == ("r1", "r2")

    _clear_pending_batch(tmp_path)
    assert _load_pending_batch(tmp_path) is None


def test_batch_state_noop_when_state_dir_is_none() -> None:
    from ppl_synth_data.llm.backend import BatchHandle
    from ppl_synth_data.pipeline.concurrency import (
        _clear_pending_batch,
        _load_pending_batch,
        _save_pending_batch,
    )

    handle = BatchHandle(backend="openai", job_id="batch_xyz", request_ids=())
    assert _load_pending_batch(None) is None
    _save_pending_batch(None, handle)  # should not raise
    _clear_pending_batch(None)  # should not raise

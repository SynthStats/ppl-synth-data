from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

from ppl_synth_data.llm.backend import (
    BatchCapableLLMBackend,
    BatchHandle,
    LLMBackend,
    LLMRequest,
    LLMResult,
)
from ppl_synth_data.pipeline.io import atomic_write_text

log = logging.getLogger(__name__)

_BATCH_STATE_FILE = "pending_batch.json"
_BATCH_RECEIPT_FILE = "batch_receipt.json"


def execute_requests(
    backend: LLMBackend,
    *,
    requests: list[tuple[str, LLMRequest]],
    use_batch: bool,
    state_dir: Path | None = None,
) -> dict[str, str]:
    results = execute_requests_raw(
        backend,
        requests=requests,
        use_batch=use_batch,
        state_dir=state_dir,
    )
    return {request_id: result.content for request_id, result in results.items()}


def execute_requests_raw(
    backend: LLMBackend,
    *,
    requests: list[tuple[str, LLMRequest]],
    use_batch: bool,
    state_dir: Path | None = None,
    allow_partial_batch_failures: bool = False,
) -> dict[str, LLMResult]:
    needs_json_schema = any(
        isinstance(req.text_format, dict) and req.text_format.get("type") == "json_schema"
        for _, req in requests
    )
    if needs_json_schema and not backend.capabilities.supports_json_schema:
        raise ValueError(
            "requests require json_schema output but backend "
            "does not support it (supports_json_schema=False)"
        )

    if use_batch and backend.capabilities.supports_batch:
        batch_backend = cast(BatchCapableLLMBackend, backend)
        _enable_batch_artifact_preservation(batch_backend, state_dir)
        handle = _load_pending_batch(state_dir)
        if handle is None:
            handle = batch_backend.submit_batch(requests)
            _save_pending_batch(state_dir, handle)
        try:
            results = batch_backend.collect_batch(
                handle,
                allow_partial_failures=allow_partial_batch_failures,
            )
        except Exception as exc:
            _save_batch_receipt(
                state_dir,
                handle=handle,
                requests=requests,
                results={},
                missing_request_ids=[request_id for request_id, _ in requests],
                allow_partial_batch_failures=allow_partial_batch_failures,
                status="failed",
                exception=exc,
                backend=batch_backend,
            )
            _clear_pending_batch(state_dir)
            raise
        _clear_pending_batch(state_dir)
        missing = [request_id for request_id, _ in requests if request_id not in results]
        receipt_status = "completed" if not missing else "partial"
        _save_batch_receipt(
            state_dir,
            handle=handle,
            requests=requests,
            results=results,
            missing_request_ids=missing,
            allow_partial_batch_failures=allow_partial_batch_failures,
            status=receipt_status,
            exception=None,
            backend=batch_backend,
        )
        if missing:
            if not allow_partial_batch_failures:
                preview = ", ".join(missing[:5])
                raise RuntimeError(
                    f"batch response missing {len(missing)} request_ids; first missing: {preview}"
                )
            return {
                request_id: results[request_id]
                for request_id, _ in requests
                if request_id in results
            }
        return {request_id: results[request_id] for request_id, _ in requests}

    out: dict[str, LLMResult] = {}
    for request_id, req in requests:
        out[request_id] = backend.complete(req)
    return out


def _enable_batch_artifact_preservation(
    backend: BatchCapableLLMBackend,
    state_dir: Path | None,
) -> None:
    if state_dir is None:
        return
    configure = getattr(backend, "enable_batch_artifact_preservation", None)
    if callable(configure):
        configure(state_dir / "openai_batch_artifacts")


def _load_pending_batch(state_dir: Path | None) -> BatchHandle | None:
    if state_dir is None:
        return None
    state_path = state_dir / _BATCH_STATE_FILE
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return BatchHandle(
        backend=payload["backend"],
        job_id=payload["job_id"],
        request_ids=tuple(payload.get("request_ids", ())),
    )


def _save_pending_batch(state_dir: Path | None, handle: BatchHandle) -> None:
    if state_dir is None:
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": handle.backend,
        "job_id": handle.job_id,
        "request_ids": list(handle.request_ids),
    }
    atomic_write_text(state_dir / _BATCH_STATE_FILE, json.dumps(payload, indent=2))


def _save_batch_receipt(
    state_dir: Path | None,
    *,
    handle: BatchHandle,
    requests: list[tuple[str, LLMRequest]],
    results: dict[str, LLMResult],
    missing_request_ids: list[str],
    allow_partial_batch_failures: bool,
    status: str,
    exception: Exception | None,
    backend: BatchCapableLLMBackend,
) -> None:
    if state_dir is None:
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    backend_failures = getattr(backend, "last_batch_failures", {})
    if not isinstance(backend_failures, dict):
        backend_failures = {}
    backend_artifacts = getattr(backend, "last_batch_artifacts", {})
    if not isinstance(backend_artifacts, dict):
        backend_artifacts = {}
    payload = {
        "schema": "ppl_synth_data.batch_receipt.v1",
        "status": status,
        "backend": handle.backend,
        "job_id": handle.job_id,
        "request_count": len(requests),
        "request_ids": [request_id for request_id, _ in requests],
        "handle_request_ids": list(handle.request_ids),
        "success_count": len(results),
        "success_request_ids": sorted(results),
        "missing_count": len(missing_request_ids),
        "missing_request_ids": missing_request_ids,
        "allow_partial_batch_failures": allow_partial_batch_failures,
        "backend_failure_count": len(backend_failures),
        "backend_failure_request_ids": sorted(str(key) for key in backend_failures),
    }
    if backend_artifacts:
        payload["backend_artifacts"] = backend_artifacts
    if exception is not None:
        payload["exception_type"] = type(exception).__name__
        payload["exception_message"] = str(exception)[:500]
    atomic_write_text(
        state_dir / _BATCH_RECEIPT_FILE,
        json.dumps(payload, indent=2, sort_keys=True),
    )


def _clear_pending_batch(state_dir: Path | None) -> None:
    if state_dir is None:
        return
    state_path = state_dir / _BATCH_STATE_FILE
    if state_path.exists():
        state_path.unlink()

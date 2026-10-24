from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import (
    BackendCapabilities,
    BatchCapableLLMBackend,
    BatchHandle,
    LLMRequest,
    LLMResult,
)
from ppl_synth_data.llm.openai_batch import (
    create_batch,
    download_file_content,
    make_responses_batch_line,
    parse_batch_error_jsonl,
    parse_batch_output_jsonl,
    poll_batch_until_terminal,
    upload_batch_file,
    write_batch_jsonl,
)
from ppl_synth_data.pipeline.io import atomic_write_text

log = logging.getLogger(__name__)


class OpenAIBackend(BatchCapableLLMBackend):
    capabilities = BackendCapabilities(
        supports_batch=True,
        supports_json_schema=True,
        supports_reasoning_effort=True,
    )

    def __init__(
        self,
        settings: Settings | None = None,
        artifact_dir: Path | None = None,
        preserve_batch_artifacts: bool = False,
    ) -> None:
        self._settings = settings or Settings()
        self._artifact_dir = artifact_dir or Path("artifacts")
        self._artifact_dir_explicit = artifact_dir is not None
        self._preserve_batch_artifacts = preserve_batch_artifacts
        api_key = self._settings.require_openai()
        self._client = OpenAI(
            api_key=api_key,
            organization=self._settings.openai_org_id,
            timeout=httpx.Timeout(1200.0),  # 20 min for xhigh reasoning
        )
        self.last_batch_failures: dict[str, str] = {}
        self.last_batch_artifacts: dict[str, Any] = {}
        self._last_batch_status_snapshot: dict[str, Any] | None = None

    def enable_batch_artifact_preservation(self, artifact_dir: Path | None = None) -> None:
        if artifact_dir is not None and not self._artifact_dir_explicit:
            self._artifact_dir = artifact_dir
        self._preserve_batch_artifacts = True

    def complete(self, request: LLMRequest) -> LLMResult:
        body: dict[str, object] = {
            "model": request.model,
            "input": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            # Disable server-side response storage on every request.
            "store": False,
        }
        if request.reasoning_effort:
            body["reasoning"] = {"effort": request.reasoning_effort}
        if request.max_output_tokens is not None:
            body["max_output_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.text_format is not None:
            body["text"] = {"format": request.text_format}
        response = self._client.responses.create(**body)  # type: ignore[call-overload]
        raw = response.model_dump()
        content = response.output_text
        return LLMResult(content=content, raw=raw)

    def submit_batch(self, requests: list[tuple[str, LLMRequest]]) -> BatchHandle:
        self.last_batch_failures = {}
        lines = [
            make_responses_batch_line(
                custom_id=custom_id,
                model=req.model,
                system=req.system,
                user=req.user,
                reasoning_effort=req.reasoning_effort,
                max_output_tokens=req.max_output_tokens,
                temperature=req.temperature,
                text_format=req.text_format,
            )
            for custom_id, req in requests
        ]
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        input_path = self._artifact_dir / f"openai_batch_input_{uuid.uuid4().hex}.jsonl"
        write_batch_jsonl(lines, input_path)
        try:
            file_id = upload_batch_file(self._client, input_path)
        finally:
            if not self._preserve_batch_artifacts:
                input_path.unlink(missing_ok=True)
        batch_id = create_batch(self._client, input_file_id=file_id)
        request_ids = tuple(custom_id for custom_id, _ in requests)
        if self._preserve_batch_artifacts:
            self.last_batch_artifacts = {
                "schema": "ppl_synth_data.openai_batch_artifacts.v1",
                "backend": "openai",
                "batch_id": batch_id,
                "input_file_id": file_id,
                "input_jsonl_path": str(input_path),
                "input_jsonl_sha256": _sha256_file(input_path),
                "request_count": len(requests),
                "request_ids": list(request_ids),
                "model_ids": sorted({req.model for _, req in requests}),
                "request_hashes": {
                    custom_id: {
                        "system_sha256": _sha256_text(req.system),
                        "user_sha256": _sha256_text(req.user),
                    }
                    for custom_id, req in requests
                },
            }
            _write_artifact_metadata(self._artifact_dir, batch_id, self.last_batch_artifacts)
        else:
            self.last_batch_artifacts = {}
        return BatchHandle(backend="openai", job_id=batch_id, request_ids=request_ids)

    def _recover_failures(
        self,
        *,
        batch: Any,
        handle: BatchHandle,
        parsed_successes: dict[str, str],
        parsed_failures: dict[str, str],
    ) -> dict[str, str]:
        failures = dict(parsed_failures)
        request_counts = getattr(batch, "request_counts", None)
        failed_count = int(getattr(request_counts, "failed", 0)) if request_counts else 0
        if getattr(batch, "error_file_id", None):
            error_path = self._artifact_dir / f"openai_batch_errors_{handle.job_id}.jsonl"
            download_file_content(self._client, batch.error_file_id, error_path)
            failures.update(parse_batch_error_jsonl(error_path))
            if self._preserve_batch_artifacts:
                self.last_batch_artifacts["error_file_id"] = batch.error_file_id
                self.last_batch_artifacts["error_jsonl_path"] = str(error_path)
                self.last_batch_artifacts["error_jsonl_sha256"] = _sha256_file(error_path)
            else:
                error_path.unlink(missing_ok=True)

        if handle.request_ids:
            missing = [
                rid
                for rid in handle.request_ids
                if (
                    rid not in parsed_successes
                    and rid not in parsed_failures
                    and rid not in failures
                )
            ]
            if missing:
                failures["__missing__"] = ", ".join(missing[:20])

        if failed_count > 0 and len(failures) < failed_count:
            failures["__request_counts__"] = (
                f"batch reports {failed_count} failed request(s) but only "
                f"{len(failures)} failure record(s) were recovered"
            )
        return failures

    def collect_batch(
        self,
        handle: BatchHandle,
        *,
        allow_partial_failures: bool = False,
    ) -> dict[str, LLMResult]:
        self.last_batch_failures = {}
        batch = poll_batch_until_terminal(
            self._client,
            handle.job_id,
            on_poll=self._record_batch_status,
        )
        if batch.status != "completed" and not allow_partial_failures:
            failures = self._recover_failures(
                batch=batch,
                handle=handle,
                parsed_successes={},
                parsed_failures={},
            )
            self.last_batch_failures = failures
            self._record_batch_collection_failure(
                batch=batch,
                handle=handle,
                failures=failures,
            )
            raise RuntimeError(
                _format_batch_terminal_error(batch, failures, self.last_batch_artifacts)
            )
        if not batch.output_file_id:
            failures = self._recover_failures(
                batch=batch,
                handle=handle,
                parsed_successes={},
                parsed_failures={},
            )
            self.last_batch_failures = failures
            self._record_batch_collection_failure(
                batch=batch,
                handle=handle,
                failures=failures,
            )
            status = getattr(batch, "status", "unknown")
            raise RuntimeError(
                _format_missing_output_error(batch, failures, self.last_batch_artifacts)
                if failures
                else f"OpenAI batch ended with status={status} without output_file_id"
            )
        output_path = self._artifact_dir / f"openai_batch_output_{handle.job_id}.jsonl"
        download_file_content(self._client, batch.output_file_id, output_path)

        try:
            parsed = parse_batch_output_jsonl(output_path)
        finally:
            if not self._preserve_batch_artifacts:
                output_path.unlink(missing_ok=True)
        if self._preserve_batch_artifacts:
            self.last_batch_artifacts.update(
                {
                    "status": getattr(batch, "status", "unknown"),
                    "output_file_id": batch.output_file_id,
                    "output_jsonl_path": str(output_path),
                    "output_jsonl_sha256": _sha256_file(output_path),
                    "request_counts": _request_counts_payload(batch),
                    "success_count": len(parsed.successes),
                    "failure_count": len(parsed.failures),
                    "completion_hashes": {
                        custom_id: _sha256_text(content)
                        for custom_id, content in parsed.successes.items()
                    },
                }
            )
        failures = self._recover_failures(
            batch=batch,
            handle=handle,
            parsed_successes=parsed.successes,
            parsed_failures=parsed.failures,
        )
        self.last_batch_failures = failures
        if self._preserve_batch_artifacts:
            self.last_batch_artifacts["recovered_failure_count"] = len(failures)
            self.last_batch_artifacts["recovered_failure_request_ids"] = sorted(failures)
            _write_artifact_metadata(
                self._artifact_dir,
                handle.job_id,
                self.last_batch_artifacts,
            )

        if failures:
            for fid, reason in failures.items():
                log.warning("batch failure: %s = %s", fid, reason)
            if not allow_partial_failures:
                sample = next(iter(failures.items()))
                raise RuntimeError(
                    f"OpenAI batch had {len(failures)} failed/missing request(s); "
                    f"example {sample[0]}={sample[1]}"
                )
            if not parsed.successes:
                sample = next(iter(failures.items()))
                raise RuntimeError(
                    f"OpenAI batch had no successful request(s); example {sample[0]}={sample[1]}"
                )

        return {
            k: LLMResult(content=v, raw=parsed.raw_bodies.get(k, {}))
            for k, v in parsed.successes.items()
        }

    def _record_batch_collection_failure(
        self,
        *,
        batch: Any,
        handle: BatchHandle,
        failures: dict[str, str],
    ) -> None:
        if not self._preserve_batch_artifacts:
            return
        self.last_batch_artifacts.update(
            {
                "status": getattr(batch, "status", "unknown"),
                "output_file_id": getattr(batch, "output_file_id", None),
                "request_counts": _request_counts_payload(batch),
                "success_count": 0,
                "failure_count": 0,
                "recovered_failure_count": len(failures),
                "recovered_failure_request_ids": sorted(failures),
            }
        )
        _write_artifact_metadata(
            self._artifact_dir,
            handle.job_id,
            self.last_batch_artifacts,
        )

    def _record_batch_status(self, batch: Any) -> None:
        snapshot = _batch_status_snapshot(batch)
        self._last_batch_status_snapshot = snapshot
        if not self._preserve_batch_artifacts:
            return

        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        batch_id = str(snapshot["id"])
        metadata_path = self._artifact_dir / f"openai_batch_artifacts_{batch_id}.json"
        if not self.last_batch_artifacts and metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                self.last_batch_artifacts = loaded
        latest_path = self._artifact_dir / f"openai_batch_status_{batch_id}.json"
        history_path = self._artifact_dir / f"openai_batch_status_{batch_id}.jsonl"
        atomic_write_text(latest_path, json.dumps(snapshot, indent=2, sort_keys=True))
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, sort_keys=True) + "\n")
        self.last_batch_artifacts.update(
            {
                "status_snapshot_path": str(latest_path),
                "status_history_jsonl_path": str(history_path),
                "latest_status": snapshot["status"],
                "latest_status_checked_at": snapshot["checked_at_utc"],
            }
        )
        _write_artifact_metadata(
            self._artifact_dir,
            batch_id,
            self.last_batch_artifacts,
        )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_counts_payload(batch: Any) -> dict[str, int | None]:
    request_counts = getattr(batch, "request_counts", None)
    return {
        "completed": _optional_count(request_counts, "completed"),
        "failed": _optional_count(request_counts, "failed"),
        "total": _optional_count(request_counts, "total"),
    }


def _optional_count(request_counts: Any, key: str) -> int | None:
    if request_counts is None:
        return None
    value = getattr(request_counts, key, None)
    if value is None:
        return None
    return int(value)


def _format_batch_terminal_error(
    batch: Any,
    failures: dict[str, str],
    artifacts: dict[str, Any],
) -> str:
    status = getattr(batch, "status", "unknown")
    return _format_batch_collection_error(
        f"OpenAI batch ended with status={status}",
        batch,
        failures,
        artifacts,
    )


def _format_missing_output_error(
    batch: Any,
    failures: dict[str, str],
    artifacts: dict[str, Any],
) -> str:
    status = getattr(batch, "status", "unknown")
    return _format_batch_collection_error(
        f"OpenAI batch ended with status={status} without output_file_id",
        batch,
        failures,
        artifacts,
    )


def _format_batch_collection_error(
    prefix: str,
    batch: Any,
    failures: dict[str, str],
    artifacts: dict[str, Any],
) -> str:
    counts = _request_counts_payload(batch)
    parts = [
        prefix,
        (
            "request_counts="
            f"completed={counts['completed']} "
            f"failed={counts['failed']} "
            f"total={counts['total']}"
        ),
    ]
    if failures:
        sample_key = next(iter(failures))
        parts.append(f"recovered_failure_count={len(failures)}")
        parts.append(f"example {sample_key}={failures[sample_key]}")
    error_path = artifacts.get("error_jsonl_path")
    if isinstance(error_path, str):
        parts.append(f"error_jsonl_path={error_path}")
    return "; ".join(parts)


def _batch_status_snapshot(batch: Any) -> dict[str, Any]:
    request_counts = getattr(batch, "request_counts", None)
    errors = getattr(batch, "errors", None)
    return {
        "schema": "ppl_synth_data.openai_batch_status.v1",
        "checked_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "id": getattr(batch, "id", None),
        "status": getattr(batch, "status", "unknown"),
        "created_at": getattr(batch, "created_at", None),
        "in_progress_at": getattr(batch, "in_progress_at", None),
        "finalizing_at": getattr(batch, "finalizing_at", None),
        "completed_at": getattr(batch, "completed_at", None),
        "failed_at": getattr(batch, "failed_at", None),
        "expired_at": getattr(batch, "expired_at", None),
        "cancelled_at": getattr(batch, "cancelled_at", None),
        "expires_at": getattr(batch, "expires_at", None),
        "output_file_id_present": bool(getattr(batch, "output_file_id", None)),
        "error_file_id_present": bool(getattr(batch, "error_file_id", None)),
        "request_counts": None
        if request_counts is None
        else {
            "total": getattr(request_counts, "total", None),
            "completed": getattr(request_counts, "completed", None),
            "failed": getattr(request_counts, "failed", None),
        },
        "errors": None if errors is None else str(errors)[:2000],
    }


def _write_artifact_metadata(
    artifact_dir: Path,
    batch_id: str,
    payload: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        artifact_dir / f"openai_batch_artifacts_{batch_id}.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )

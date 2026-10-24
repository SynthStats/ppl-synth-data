from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from openai.types import Batch

from ppl_synth_data.pipeline.io import atomic_write_text

TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}


@dataclass(frozen=True)
class BatchOutputParse:
    successes: dict[str, str]
    raw_bodies: dict[str, dict[str, Any]]
    failures: dict[str, str]


def make_responses_batch_line(
    *,
    custom_id: str,
    model: str,
    system: str,
    user: str,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    text_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if text_format is not None:
        body["text"] = {"format": text_format}
    if extra_body:
        body.update(extra_body)
    # Set after extra_body so callers cannot re-enable server-side response storage.
    body["store"] = False
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def write_batch_jsonl(lines: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return output_path


def upload_batch_file(client: OpenAI, jsonl_path: Path) -> str:
    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    return uploaded.id


def create_batch(
    client: OpenAI,
    *,
    input_file_id: str,
    endpoint: Literal["/v1/responses"] = "/v1/responses",
    completion_window: Literal["24h"] = "24h",
    metadata: dict[str, str] | None = None,
) -> str:
    # thin wrapper: pins endpoint/completion_window defaults and returns
    # batch.id directly so callers don't handle the full Batch object
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {},
    )
    return batch.id


def poll_batch_until_terminal(
    client: OpenAI,
    batch_id: str,
    poll_seconds: float = 10.0,
    timeout_seconds: float = 86400.0,
    on_poll: Callable[[Batch], None] | None = None,
) -> Batch:
    start = time.monotonic()
    while True:
        batch = client.batches.retrieve(batch_id)
        if on_poll is not None:
            on_poll(batch)
        if batch.status in TERMINAL_BATCH_STATES:
            return batch
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"batch {batch_id} timed out; last status={batch.status}")
        time.sleep(poll_seconds)


def download_file_content(client: OpenAI, file_id: str, output_path: Path) -> Path:
    response = client.files.content(file_id)
    content = response.text if hasattr(response, "text") else str(response)
    atomic_write_text(output_path, content)
    return output_path


def _extract_text_parts_from_message(content_items: Any) -> list[str]:
    if not isinstance(content_items, list):
        return []
    parts: list[str] = []
    for content in content_items:
        if not isinstance(content, dict):
            continue
        if content.get("type") == "output_text" and "text" in content:
            text = content["text"]
            parts.append(text if isinstance(text, str) else str(text))
    return parts


def extract_output_text_from_responses_body(body: dict[str, Any]) -> str:
    output = body.get("output", []) if isinstance(body, dict) else []
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        parts.extend(_extract_text_parts_from_message(item.get("content", []) or []))
    return "".join(parts)


def _iter_jsonl_objects(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, None, exc.msg
                continue
            yield line_no, obj, None


def _extract_line_content(
    obj: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract content and body from a parsed batch line, or return an error string."""
    if obj.get("error") is not None:
        return None, str(obj["error"])
    response = obj.get("response", {})
    if not isinstance(response, dict):
        return None, "missing response payload"
    status_code = response.get("status_code")
    if not isinstance(status_code, int):
        return None, "missing status_code"
    if status_code >= 400:
        return None, f"status_code={status_code}"
    body = response.get("body", {})
    if not isinstance(body, dict):
        return None, "missing response body"
    content = extract_output_text_from_responses_body(body)
    if not content.strip():
        return None, "empty output_text"
    return {"content": content, "body": body}, None


def parse_batch_output_jsonl(path: Path) -> BatchOutputParse:
    successes: dict[str, str] = {}
    raw_bodies: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for line_no, obj, decode_error in _iter_jsonl_objects(path):
        if decode_error is not None:
            failures[f"__line_{line_no}__"] = f"invalid json: {decode_error}"
            continue
        assert obj is not None
        if not isinstance(obj, dict):
            failures[f"__line_{line_no}__"] = f"invalid json object: {type(obj).__name__}"
            continue
        custom_id = str(obj.get("custom_id", f"__line_{line_no}__"))
        result, error = _extract_line_content(obj)
        if error is not None:
            failures[custom_id] = error
        else:
            assert result is not None
            successes[custom_id] = result["content"]
            raw_bodies[custom_id] = result["body"]
    return BatchOutputParse(successes=successes, raw_bodies=raw_bodies, failures=failures)


def parse_batch_error_jsonl(path: Path) -> dict[str, str]:
    failures: dict[str, str] = {}
    for line_no, obj, decode_error in _iter_jsonl_objects(path):
        if decode_error is not None:
            failures[f"__line_{line_no}__"] = f"invalid json: {decode_error}"
            continue
        assert obj is not None
        if not isinstance(obj, dict):
            failures[f"__line_{line_no}__"] = f"invalid json object: {type(obj).__name__}"
            continue
        custom_id = str(obj.get("custom_id", f"__line_{line_no}__"))
        if obj.get("error") is not None:
            failures[custom_id] = str(obj["error"])
            continue
        response = obj.get("response", {})
        if not isinstance(response, dict):
            continue
        status_code = response.get("status_code")
        request_id = response.get("request_id")
        body = response.get("body", {})
        if isinstance(body, dict) and body.get("error") is not None:
            failures[custom_id] = _format_response_error(
                status_code=status_code,
                request_id=request_id,
                error=body["error"],
            )
        elif isinstance(status_code, int) and status_code >= 400:
            failures[custom_id] = _format_response_error(
                status_code=status_code,
                request_id=request_id,
                error=None,
            )
    return failures


def _format_response_error(
    *,
    status_code: Any,
    request_id: Any,
    error: Any,
) -> str:
    parts: list[str] = []
    if isinstance(status_code, int):
        parts.append(f"status_code={status_code}")
    if isinstance(request_id, str) and request_id:
        parts.append(f"request_id={request_id}")
    if error is not None:
        parts.append(f"error={error}")
    return " ".join(parts) if parts else "unknown response error"

#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ppl_synth_data.llm.backend import BatchHandle
from ppl_synth_data.llm.openai_backend import OpenAIBackend
from ppl_synth_data.pipeline.io import atomic_write_text


def recover_openai_batch(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    backend = OpenAIBackend(
        artifact_dir=artifact_dir,
        preserve_batch_artifacts=True,
    )
    backend.last_batch_artifacts = {
        "schema": "ppl_synth_data.openai_batch_artifacts.v1",
        "backend": "openai",
        "batch_id": args.batch_id,
        "proof_context": "recover_openai_batch",
        "recovered_at_utc": _utc_now(),
    }

    receipt: dict[str, Any] = {
        "schema": "ppl_synth_data.openai_batch_recovery_receipt.v1",
        "recovered_at_utc": _utc_now(),
        "batch_id": args.batch_id,
        "artifact_dir": str(artifact_dir),
        "expected_terminal_failure": args.expect_terminal_failure,
        "expected_failed_count": args.expected_failed_count,
    }

    try:
        results = backend.collect_batch(
            BatchHandle(
                backend="openai",
                job_id=args.batch_id,
                request_ids=tuple(args.request_id),
            )
        )
    except RuntimeError as exc:
        receipt["batch_terminal_failure"] = True
        receipt["exception_type"] = type(exc).__name__
        receipt["exception_message"] = str(exc)
        receipt["success_count"] = 0
        receipt["success_request_ids"] = []
    else:
        receipt["batch_terminal_failure"] = False
        receipt["success_count"] = len(results)
        receipt["success_request_ids"] = sorted(results)

    receipt["last_batch_failure_count"] = len(backend.last_batch_failures)
    receipt["last_batch_failure_ids_preview"] = sorted(backend.last_batch_failures)[:20]
    receipt["last_batch_artifacts"] = backend.last_batch_artifacts

    error_path = backend.last_batch_artifacts.get("error_jsonl_path")
    if isinstance(error_path, str) and Path(error_path).exists():
        receipt["error_jsonl_line_count"] = _nonblank_line_count(Path(error_path))

    if args.input_jsonl is not None:
        retry_jsonl = args.retry_jsonl or artifact_dir / "retry_input.jsonl"
        receipt["retry_input"] = _write_retry_input_jsonl(
            input_jsonl=args.input_jsonl,
            retry_jsonl=retry_jsonl,
            failed_request_ids=set(backend.last_batch_failures),
        )

    receipt["recovery_failures"] = _validate_recovery_expectations(receipt)
    receipt["recovery_status"] = "failed" if receipt["recovery_failures"] else "passed"
    atomic_write_text(args.receipt, json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _validate_recovery_expectations(receipt: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    terminal_failure = bool(receipt["batch_terminal_failure"])
    expected_terminal_failure = bool(receipt["expected_terminal_failure"])
    if terminal_failure != expected_terminal_failure:
        failures.append(
            "batch_terminal_failure="
            f"{terminal_failure} did not match expected_terminal_failure="
            f"{expected_terminal_failure}"
        )

    expected_failed_count = receipt.get("expected_failed_count")
    if expected_failed_count is not None:
        actual_failed_count = receipt["last_batch_failure_count"]
        if actual_failed_count != expected_failed_count:
            failures.append(
                f"expected {expected_failed_count} recovered failures, found {actual_failed_count}"
            )

        error_line_count = receipt.get("error_jsonl_line_count")
        if error_line_count != expected_failed_count:
            failures.append(
                f"expected {expected_failed_count} error JSONL rows, found {error_line_count}"
            )

    artifacts = receipt["last_batch_artifacts"]
    if terminal_failure and not artifacts.get("error_jsonl_path"):
        failures.append("terminal failure did not preserve error_jsonl_path")
    if terminal_failure and not artifacts.get("error_jsonl_sha256"):
        failures.append("terminal failure did not preserve error_jsonl_sha256")

    retry_input = receipt.get("retry_input")
    if isinstance(retry_input, dict):
        missing_failure_ids = retry_input.get("missing_failure_ids", [])
        if missing_failure_ids:
            failures.append(
                "retry input missing failed request ids: "
                f"{', '.join(str(item) for item in missing_failure_ids[:20])}"
            )
        if expected_failed_count is not None:
            retry_request_count = retry_input.get("retry_request_count")
            if retry_request_count != expected_failed_count:
                failures.append(
                    f"expected {expected_failed_count} retry rows, found {retry_request_count}"
                )
    return failures


def _write_retry_input_jsonl(
    *,
    input_jsonl: Path,
    retry_jsonl: Path,
    failed_request_ids: set[str],
) -> dict[str, Any]:
    source_rows: dict[str, str] = {}
    source_request_count = 0
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_request_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_jsonl}: invalid JSON on line {line_no}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{input_jsonl}: line {line_no} is not a JSON object")
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError(f"{input_jsonl}: line {line_no} missing custom_id")
            if custom_id in source_rows:
                raise ValueError(f"{input_jsonl}: duplicate custom_id {custom_id!r}")
            source_rows[custom_id] = line.rstrip("\n")

    retry_ids = [
        request_id for request_id in sorted(failed_request_ids) if request_id in source_rows
    ]
    retry_content = "".join(f"{source_rows[request_id]}\n" for request_id in retry_ids)
    atomic_write_text(retry_jsonl, retry_content)
    missing_failure_ids = sorted(failed_request_ids - set(retry_ids))
    return {
        "source_input_jsonl_path": str(input_jsonl),
        "source_input_jsonl_sha256": _sha256_file(input_jsonl),
        "source_request_count": source_request_count,
        "retry_jsonl_path": str(retry_jsonl),
        "retry_jsonl_sha256": _sha256_file(retry_jsonl),
        "retry_request_count": len(retry_ids),
        "retry_request_ids": retry_ids,
        "missing_failure_ids": missing_failure_ids,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonblank_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover OpenAI Batch status, output, and error artifacts.",
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--retry-jsonl", type=Path)
    parser.add_argument("--expect-terminal-failure", action="store_true")
    parser.add_argument("--expected-failed-count", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.receipt is None:
        args.receipt = args.artifact_dir / "recovery_receipt.json"
    receipt = recover_openai_batch(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(1 if receipt["recovery_status"] == "failed" else 0)


if __name__ == "__main__":
    main()

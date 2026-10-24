from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import BatchHandle, LLMRequest
from ppl_synth_data.llm.openai_backend import OpenAIBackend
from ppl_synth_data.llm.openai_batch import TERMINAL_BATCH_STATES


@pytest.mark.paid_lane
def test_real_openai_batch_acceptance(tmp_path: Path) -> None:
    """Submit a real OpenAI batch and check its returned identifier, status and usage.

    The API uses a 24-hour completion window. A completed batch must provide an output row and
    usage/request counts. A still-active batch must provide a valid accepted identifier and
    status, and its receipt records PENDING until completion can be observed.
    """
    settings = Settings()
    settings.require_openai()  # hard assert: no silent skip on missing credentials

    artifact_dir = tmp_path / "artifacts"
    backend = OpenAIBackend(artifact_dir=artifact_dir, preserve_batch_artifacts=True)

    req = LLMRequest(
        model="gpt-5.4-mini",
        system="Reply with exactly one word.",
        user="Say hello.",
        max_output_tokens=16,
    )
    handle = backend.submit_batch([("real-openai-batch", req)])
    assert handle.job_id
    assert handle.request_ids == ("real-openai-batch",)

    poll_seconds = 10.0
    budget_seconds = 420.0  # ~7 minutes; stays under the sandbox's per-command cap
    deadline = time.monotonic() + budget_seconds
    batch = backend._client.batches.retrieve(handle.job_id)
    while batch.status not in TERMINAL_BATCH_STATES and time.monotonic() < deadline:
        time.sleep(poll_seconds)
        batch = backend._client.batches.retrieve(handle.job_id)

    receipt = {
        "job_id": handle.job_id,
        "status": batch.status,
        "request_counts": (
            dict(batch.request_counts.model_dump()) if batch.request_counts else None
        ),
    }

    if batch.status == "completed":
        results = backend.collect_batch(handle, allow_partial_failures=False)
        assert "real-openai-batch" in results
        result = results["real-openai-batch"]
        assert result.content.strip()
        usage = result.raw.get("usage")
        assert isinstance(usage, dict)
        assert usage.get("total_tokens", 0) > 0
        receipt["outcome"] = "completed"
        receipt["content"] = result.content
        receipt["response_id"] = result.raw.get("id")
        receipt["usage"] = usage
    else:
        assert batch.status in {"validating", "in_progress", "finalizing"}, (
            f"unexpected non-terminal-but-also-non-progressing status: {batch.status}"
        )
        receipt["outcome"] = "pending_real_boundary_confirmed"

    (tmp_path / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps(receipt, indent=2, default=str))  # captured by -s for the run receipt


class _DummyClient:
    pass


def _poll_returns(batch):
    def poll(*_args, **kwargs):
        on_poll = kwargs.get("on_poll")
        if on_poll is not None:
            on_poll(batch)
        return batch

    return poll


@pytest.mark.narrow_unit_not_acceptance
def test_submit_batch_includes_temperature(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    captured = {}

    def fake_write(lines, output_path):
        captured["lines"] = lines
        path = tmp_path / "input.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return path

    monkeypatch.setattr("ppl_synth_data.llm.openai_backend.write_batch_jsonl", fake_write)
    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.upload_batch_file",
        lambda *_: "file-1",
    )
    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.create_batch",
        lambda *_args, **_kwargs: "batch-1",
    )

    backend = OpenAIBackend()
    backend_any: Any = backend
    backend_any._client = _DummyClient()

    req = LLMRequest(
        model="gpt-5.2",
        system="S",
        user="U",
        reasoning_effort="high",
        temperature=0.7,
    )
    handle = backend.submit_batch([("id-1", req)])

    line = captured["lines"][0]
    assert line["body"]["temperature"] == 0.7
    assert handle.request_ids == ("id-1",)


@pytest.mark.narrow_unit_not_acceptance
def test_submit_batch_preserves_input_jsonl_and_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    uploaded_paths: list[Path] = []

    def fake_upload(_client, path: Path) -> str:
        uploaded_paths.append(path)
        assert path.exists()
        return "file-1"

    monkeypatch.setattr("ppl_synth_data.llm.openai_backend.upload_batch_file", fake_upload)
    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.create_batch",
        lambda *_args, **_kwargs: "batch-1",
    )

    backend = OpenAIBackend(artifact_dir=tmp_path, preserve_batch_artifacts=True)
    backend_any: Any = backend
    backend_any._client = _DummyClient()

    req = LLMRequest(model="gpt-5.5-pro", system="S", user="U")
    handle = backend.submit_batch([("id-1", req)])

    assert handle.job_id == "batch-1"
    assert len(uploaded_paths) == 1
    input_path = uploaded_paths[0]
    assert input_path.exists()
    metadata = json.loads(
        (tmp_path / "openai_batch_artifacts_batch-1.json").read_text(encoding="utf-8")
    )
    assert metadata["schema"] == "ppl_synth_data.openai_batch_artifacts.v1"
    assert metadata["input_file_id"] == "file-1"
    assert metadata["input_jsonl_path"] == str(input_path)
    assert len(metadata["input_jsonl_sha256"]) == 64
    assert metadata["model_ids"] == ["gpt-5.5-pro"]
    assert set(metadata["request_hashes"]["id-1"]) == {"system_sha256", "user_sha256"}
    assert "S" not in json.dumps(metadata)
    assert "U" not in json.dumps(metadata)


@pytest.mark.narrow_unit_not_acceptance
def test_collect_batch_raises_when_failed_requests_present(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    backend = OpenAIBackend()
    backend_any: Any = backend
    backend_any._client = _DummyClient()

    batch = SimpleNamespace(
        id="batch-1",
        status="completed",
        output_file_id="out-file",
        error_file_id="err-file",
        request_counts=SimpleNamespace(failed=1),
    )

    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.poll_batch_until_terminal",
        _poll_returns(batch),
    )

    def fake_download(_client, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if file_id == "out-file":
            output_path.write_text(
                '{"custom_id":"ok-1","response":{"status_code":200,"body":{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}}}\n',
                encoding="utf-8",
            )
        else:
            output_path.write_text(
                '{"custom_id":"bad-1","error":{"message":"rate limit"}}\n',
                encoding="utf-8",
            )
        return output_path

    monkeypatch.setattr("ppl_synth_data.llm.openai_backend.download_file_content", fake_download)

    with pytest.raises(RuntimeError):
        backend.collect_batch(
            BatchHandle(
                backend="openai",
                job_id="batch-1",
                request_ids=("ok-1", "bad-1"),
            )
        )


@pytest.mark.narrow_unit_not_acceptance
def test_collect_batch_raises_when_request_counts_failures_not_recoverable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    backend = OpenAIBackend()
    backend_any: Any = backend
    backend_any._client = _DummyClient()

    batch = SimpleNamespace(
        status="completed",
        output_file_id="out-file",
        error_file_id=None,
        request_counts=SimpleNamespace(failed=1),
    )

    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.poll_batch_until_terminal",
        _poll_returns(batch),
    )

    def fake_download(_client, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assert file_id == "out-file"
        output_path.write_text(
            '{"custom_id":"ok-1","response":{"status_code":200,"body":{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}}}\n',
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr("ppl_synth_data.llm.openai_backend.download_file_content", fake_download)

    with pytest.raises(RuntimeError, match="failed request\\(s\\)"):
        backend.collect_batch(BatchHandle(backend="openai", job_id="batch-1"))


@pytest.mark.narrow_unit_not_acceptance
def test_collect_batch_recovers_partial_expired_outputs_when_allowed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    backend = OpenAIBackend(artifact_dir=tmp_path)
    backend_any: Any = backend
    backend_any._client = _DummyClient()

    batch = SimpleNamespace(
        status="expired",
        output_file_id="out-file",
        error_file_id="err-file",
        request_counts=SimpleNamespace(failed=1),
    )

    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.poll_batch_until_terminal",
        _poll_returns(batch),
    )

    def fake_download(_client, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if file_id == "out-file":
            output_path.write_text(
                '{"custom_id":"ok-1","response":{"status_code":200,"body":{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}}}\n',
                encoding="utf-8",
            )
        else:
            output_path.write_text(
                '{"custom_id":"bad-1","error":{"message":"timeout"}}\n',
                encoding="utf-8",
            )
        return output_path

    monkeypatch.setattr("ppl_synth_data.llm.openai_backend.download_file_content", fake_download)

    results = backend.collect_batch(
        BatchHandle(
            backend="openai",
            job_id="batch-1",
            request_ids=("ok-1", "bad-1"),
        ),
        allow_partial_failures=True,
    )

    assert list(results) == ["ok-1"]
    assert results["ok-1"].content == "ok"
    assert backend.last_batch_failures == {"bad-1": "{'message': 'timeout'}"}


@pytest.mark.narrow_unit_not_acceptance
def test_collect_batch_preserves_output_error_jsonl_and_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    backend = OpenAIBackend(artifact_dir=tmp_path, preserve_batch_artifacts=True)
    backend_any: Any = backend
    backend_any._client = _DummyClient()
    backend.last_batch_artifacts = {
        "schema": "ppl_synth_data.openai_batch_artifacts.v1",
        "backend": "openai",
        "batch_id": "batch-1",
    }

    batch = SimpleNamespace(
        id="batch-1",
        status="completed",
        output_file_id="out-file",
        error_file_id="err-file",
        request_counts=SimpleNamespace(completed=1, failed=1, total=2),
    )

    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.poll_batch_until_terminal",
        _poll_returns(batch),
    )

    def fake_download(_client, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if file_id == "out-file":
            output_path.write_text(
                '{"custom_id":"ok-1","response":{"status_code":200,"body":{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}}}\n',
                encoding="utf-8",
            )
        else:
            output_path.write_text(
                '{"custom_id":"bad-1","error":{"message":"timeout"}}\n',
                encoding="utf-8",
            )
        return output_path

    monkeypatch.setattr("ppl_synth_data.llm.openai_backend.download_file_content", fake_download)

    results = backend.collect_batch(
        BatchHandle(
            backend="openai",
            job_id="batch-1",
            request_ids=("ok-1", "bad-1"),
        ),
        allow_partial_failures=True,
    )

    assert list(results) == ["ok-1"]
    output_path = tmp_path / "openai_batch_output_batch-1.jsonl"
    error_path = tmp_path / "openai_batch_errors_batch-1.jsonl"
    assert output_path.exists()
    assert error_path.exists()
    metadata = json.loads(
        (tmp_path / "openai_batch_artifacts_batch-1.json").read_text(encoding="utf-8")
    )
    assert metadata["output_file_id"] == "out-file"
    assert metadata["error_file_id"] == "err-file"
    assert metadata["output_jsonl_path"] == str(output_path)
    assert metadata["error_jsonl_path"] == str(error_path)
    assert len(metadata["output_jsonl_sha256"]) == 64
    assert len(metadata["error_jsonl_sha256"]) == 64
    assert metadata["completion_hashes"].keys() == {"ok-1"}
    assert metadata["recovered_failure_request_ids"] == ["bad-1"]
    assert metadata["latest_status"] == "completed"
    status_path = tmp_path / "openai_batch_status_batch-1.json"
    status_history_path = tmp_path / "openai_batch_status_batch-1.jsonl"
    assert status_path.exists()
    assert status_history_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["schema"] == "ppl_synth_data.openai_batch_status.v1"
    assert status["status"] == "completed"
    assert status["request_counts"] == {"completed": 1, "failed": 1, "total": 2}
    assert status["output_file_id_present"] is True
    assert "out-file" not in json.dumps(status)


@pytest.mark.narrow_unit_not_acceptance
def test_collect_batch_preserves_error_jsonl_when_output_file_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    backend = OpenAIBackend(artifact_dir=tmp_path, preserve_batch_artifacts=True)
    backend_any: Any = backend
    backend_any._client = _DummyClient()
    backend.last_batch_artifacts = {
        "schema": "ppl_synth_data.openai_batch_artifacts.v1",
        "backend": "openai",
        "batch_id": "batch-1",
    }

    batch = SimpleNamespace(
        id="batch-1",
        status="completed",
        output_file_id=None,
        error_file_id="err-file",
        request_counts=SimpleNamespace(completed=0, failed=2, total=2),
    )

    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.poll_batch_until_terminal",
        _poll_returns(batch),
    )

    def fake_download(_client, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assert file_id == "err-file"
        output_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "custom_id": "bad-1",
                            "error": {
                                "message": "The server is overloaded.",
                                "code": "server_is_overloaded",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "custom_id": "bad-2",
                            "error": {
                                "message": "The server is overloaded.",
                                "code": "server_is_overloaded",
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr("ppl_synth_data.llm.openai_backend.download_file_content", fake_download)

    with pytest.raises(RuntimeError) as exc_info:
        backend.collect_batch(
            BatchHandle(
                backend="openai",
                job_id="batch-1",
                request_ids=("bad-1", "bad-2"),
            )
        )

    message = str(exc_info.value)
    assert "without output_file_id" in message
    assert "failed=2" in message
    assert "openai_batch_errors_batch-1.jsonl" in message
    assert "bad-1" in message

    error_path = tmp_path / "openai_batch_errors_batch-1.jsonl"
    assert error_path.exists()
    metadata = json.loads(
        (tmp_path / "openai_batch_artifacts_batch-1.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "completed"
    assert metadata["output_file_id"] is None
    assert metadata["error_file_id"] == "err-file"
    assert metadata["error_jsonl_path"] == str(error_path)
    assert len(metadata["error_jsonl_sha256"]) == 64
    assert metadata["request_counts"] == {"completed": 0, "failed": 2, "total": 2}
    assert metadata["success_count"] == 0
    assert metadata["failure_count"] == 0
    assert metadata["recovered_failure_count"] == 2
    assert metadata["recovered_failure_request_ids"] == ["bad-1", "bad-2"]
    assert backend.last_batch_failures["bad-1"].find("server_is_overloaded") >= 0


@pytest.mark.narrow_unit_not_acceptance
def test_collect_batch_rejects_expired_outputs_when_partial_not_allowed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    backend = OpenAIBackend(artifact_dir=tmp_path)
    backend_any: Any = backend
    backend_any._client = _DummyClient()

    batch = SimpleNamespace(
        status="expired",
        output_file_id="out-file",
        error_file_id=None,
        request_counts=SimpleNamespace(failed=0),
    )

    monkeypatch.setattr(
        "ppl_synth_data.llm.openai_backend.poll_batch_until_terminal",
        _poll_returns(batch),
    )

    with pytest.raises(RuntimeError, match="status=expired"):
        backend.collect_batch(BatchHandle(backend="openai", job_id="batch-1"))


@pytest.mark.narrow_unit_not_acceptance
def test_complete_sends_store_false(monkeypatch) -> None:
    # Disable server-side response storage on every request.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, Any] = {}

    class _Responses:
        def create(self, **body: Any) -> Any:
            captured.update(body)
            return SimpleNamespace(model_dump=lambda: {"id": "resp-1"}, output_text="ok")

    backend = OpenAIBackend()
    backend_any: Any = backend
    backend_any._client = SimpleNamespace(responses=_Responses())

    result = backend.complete(LLMRequest(model="gpt-5.2", system="S", user="U"))

    assert captured["store"] is False
    assert result.content == "ok"

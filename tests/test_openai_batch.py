from __future__ import annotations

from pathlib import Path

import pytest

from ppl_synth_data.llm.openai_batch import (
    extract_output_text_from_responses_body,
    make_responses_batch_line,
    parse_batch_error_jsonl,
    parse_batch_output_jsonl,
    poll_batch_until_terminal,
)


def test_make_responses_batch_line_includes_required_fields() -> None:
    line = make_responses_batch_line(
        custom_id="spec-0001",
        model="gpt-5.2",
        system="system msg",
        user="user msg",
        reasoning_effort="high",
        text_format={
            "type": "json_schema",
            "name": "EnvSpec",
            "schema": {"type": "object"},
            "strict": True,
        },
    )

    assert line["custom_id"] == "spec-0001"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/responses"
    body = line["body"]
    assert body["model"] == "gpt-5.2"
    assert body["reasoning"]["effort"] == "high"
    assert "max_output_tokens" not in body
    assert body["text"]["format"]["name"] == "EnvSpec"
    assert body["input"][0]["role"] == "system"
    assert body["input"][1]["role"] == "user"


def test_extract_output_text_from_responses_body() -> None:
    body = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello "},
                    {"type": "output_text", "text": "world"},
                ],
            }
        ]
    }

    assert extract_output_text_from_responses_body(body) == "hello world"


def test_parse_batch_output_jsonl_collects_failures(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        (
            '{"custom_id":"ok-1","response":{"status_code":200,"body":{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}}}\n'
            '{"custom_id":"bad-1","response":{"status_code":500,"body":{}}}\n'
            '{"custom_id":"bad-2","error":{"message":"rate limit"}}\n'
        ),
        encoding="utf-8",
    )
    parsed = parse_batch_output_jsonl(path)
    assert parsed.successes == {"ok-1": "ok"}
    assert set(parsed.failures) == {"bad-1", "bad-2"}


def test_parse_batch_output_jsonl_flags_malformed_status(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        ('{"custom_id":"bad-1","response":{"body":{}}}\nnot-json\n'),
        encoding="utf-8",
    )
    parsed = parse_batch_output_jsonl(path)
    assert parsed.successes == {}
    assert parsed.failures["bad-1"] == "missing status_code"
    assert "__line_2__" in parsed.failures


def test_parse_batch_output_jsonl_flags_empty_output_text(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        (
            '{"custom_id":"bad-1","response":{"status_code":200,'
            '"body":{"output":[{"type":"message","content":'
            '[{"type":"output_text","text":"   "}]}]}}}\n'
        ),
        encoding="utf-8",
    )
    parsed = parse_batch_output_jsonl(path)
    assert parsed.successes == {}
    assert parsed.failures["bad-1"] == "empty output_text"


def test_parse_batch_output_jsonl_handles_non_object_records(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        ('[1, 2, 3]\n"oops"\n'),
        encoding="utf-8",
    )
    parsed = parse_batch_output_jsonl(path)
    assert parsed.successes == {}
    assert parsed.failures["__line_1__"] == "invalid json object: list"
    assert parsed.failures["__line_2__"] == "invalid json object: str"


def test_parse_batch_output_jsonl_non_dict_content_items(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        (
            '{"custom_id":"bad-1","response":{"status_code":200,'
            '"body":{"output":[{"type":"message","content":["oops"]}]}}}\n'
        ),
        encoding="utf-8",
    )
    parsed = parse_batch_output_jsonl(path)
    assert parsed.successes == {}
    assert parsed.failures["bad-1"] == "empty output_text"


def test_parse_batch_error_jsonl_handles_non_object_records(tmp_path: Path) -> None:
    path = tmp_path / "err.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    parsed = parse_batch_error_jsonl(path)
    assert parsed["__line_1__"] == "invalid json object: list"


def test_parse_batch_error_jsonl_extracts_response_body_errors(tmp_path: Path) -> None:
    path = tmp_path / "err.jsonl"
    path.write_text(
        (
            '{"custom_id":"spec-000000","error":null,'
            '"response":{"status_code":503,"request_id":"req_123",'
            '"body":{"error":{"message":"Our servers are currently overloaded.",'
            '"type":"service_unavailable_error","code":"server_is_overloaded",'
            '"param":null}}}}\n'
        ),
        encoding="utf-8",
    )

    parsed = parse_batch_error_jsonl(path)

    assert set(parsed) == {"spec-000000"}
    assert "status_code=503" in parsed["spec-000000"]
    assert "server_is_overloaded" in parsed["spec-000000"]
    assert "req_123" in parsed["spec-000000"]


def test_poll_batch_until_terminal_times_out() -> None:
    class _Batches:
        @staticmethod
        def retrieve(_batch_id: str):
            class _Batch:
                status = "in_progress"

            return _Batch()

    class _Client:
        batches = _Batches()

    with pytest.raises(TimeoutError):
        poll_batch_until_terminal(
            _Client(),  # type: ignore[arg-type]
            "batch-1",
            poll_seconds=0.001,
            timeout_seconds=0.005,
        )


def test_poll_batch_until_terminal_invokes_status_callback() -> None:
    class _Batches:
        def __init__(self) -> None:
            self.statuses = ["validating", "in_progress", "completed"]

        def retrieve(self, _batch_id: str):
            status = self.statuses.pop(0)

            class _Batch:
                id = "batch-1"

            _Batch.status = status
            return _Batch()

    class _Client:
        def __init__(self) -> None:
            self.batches = _Batches()

    seen: list[str] = []
    batch = poll_batch_until_terminal(
        _Client(),  # type: ignore[arg-type]
        "batch-1",
        poll_seconds=0.001,
        on_poll=lambda item: seen.append(item.status),
    )

    assert batch.status == "completed"
    assert seen == ["validating", "in_progress", "completed"]


def test_poll_batch_default_timeout_is_24h() -> None:
    import inspect

    sig = inspect.signature(poll_batch_until_terminal)
    default = sig.parameters["timeout_seconds"].default
    assert default == 86400.0


def test_make_responses_batch_line_sets_store_false() -> None:
    # Disable server-side response storage on every request.
    # The Responses API stores by default, so every batch line must say store=False.
    line = make_responses_batch_line(
        custom_id="spec-0002",
        model="gpt-5.2",
        system="S",
        user="U",
    )

    assert line["body"]["store"] is False


def test_make_responses_batch_line_extra_body_cannot_enable_store() -> None:
    line = make_responses_batch_line(
        custom_id="spec-0003",
        model="gpt-5.2",
        system="S",
        user="U",
        extra_body={"store": True},
    )

    assert line["body"]["store"] is False

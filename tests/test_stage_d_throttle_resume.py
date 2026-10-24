from __future__ import annotations

import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from ppl_synth_data.llm.bedrock_converse_backend import BedrockConverseBackend, result_from_converse
from ppl_synth_data.pipeline.io import write_jsonl
from ppl_synth_data.pipeline.stages import stage_d_prompts as stage_d
from ppl_synth_data.runtimes import get_runtime
from tests.test_declared_stage_c import _prepared_paths


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "provider rejected call"}}, "Converse")


@pytest.mark.parametrize("code", ["ThrottlingException", "ServiceUnavailableException"])
def test_retry_recovers_with_capped_exponential_full_jitter_and_logs_each_attempt(code):
    from ppl_synth_data.llm.bedrock_converse_backend import retry_converse_call

    calls, sleeps, ranges, events = [], [], [], []
    response = {"completed": True}

    def operation():
        calls.append(1)
        if len(calls) <= 4:
            raise _client_error(code)
        return response

    def uniform(low, high):
        ranges.append((low, high))
        return high / 2

    actual = retry_converse_call(
        operation,
        on_throttle=events.append,
        max_attempts=5,
        base_delay=1.0,
        max_delay=3.0,
        sleep=sleeps.append,
        uniform=uniform,
    )
    assert actual is response
    assert len(calls) == 5
    assert ranges == [(0, 1.0), (0, 2.0), (0, 3.0), (0, 3.0)]
    assert sleeps == [0.5, 1.0, 1.5, 1.5]
    assert events == [
        {"attempt": 1, "wait": 0.5, "error_code": code},
        {"attempt": 2, "wait": 1.0, "error_code": code},
        {"attempt": 3, "wait": 1.5, "error_code": code},
        {"attempt": 4, "wait": 1.5, "error_code": code},
    ]


def test_retry_exhaustion_logs_final_failure_without_sleep_and_reraises_same_error():
    from ppl_synth_data.llm.bedrock_converse_backend import retry_converse_call

    error = _client_error("ThrottlingException")
    calls, sleeps, events = [], [], []

    def operation():
        calls.append(1)
        raise error

    with pytest.raises(ClientError) as caught:
        retry_converse_call(
            operation,
            on_throttle=events.append,
            max_attempts=3,
            sleep=sleeps.append,
            uniform=lambda low, high: high,
        )
    assert caught.value is error
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]
    assert events[-1] == {"attempt": 3, "wait": 0, "error_code": "ThrottlingException"}
    assert len(events) == 3


@pytest.mark.parametrize("error", [_client_error("AccessDeniedException"), ValueError("bad input")])
def test_retry_does_not_retry_other_provider_or_local_errors(error):
    from ppl_synth_data.llm.bedrock_converse_backend import retry_converse_call

    calls, sleeps, events = [], [], []

    def operation():
        calls.append(1)
        raise error

    with pytest.raises(type(error)) as caught:
        retry_converse_call(operation, on_throttle=events.append, sleep=sleeps.append)
    assert caught.value is error
    assert len(calls) == 1
    assert sleeps == events == []


def _cached_prompt(tmp_path):
    paths, spec = _prepared_paths(tmp_path)
    record = stage_d.build_prompt_record(
        request_id="prompt-000000",
        spec=spec,
        prompt_text="Fit y using the visible month and signal measurements. Café.",
        runtime=get_runtime("pymc"),
    )
    directory = paths.stage_dirs["D"]
    directory.mkdir(parents=True, exist_ok=True)
    prompts_path = directory / "prompts.jsonl"
    receipts_path = directory / "receipts.jsonl"
    write_jsonl(prompts_path, [record.model_dump(mode="json")])
    fixture = Path(__file__).parent / "fixtures/bedrock_receipt/converse-sonnet-4-6.json"
    result = result_from_converse(json.loads(fixture.read_text()), "us.anthropic.claude-sonnet-4-6")
    stage_d.persist_prompt_receipt(receipts_path, spec.env_id, record.prompt_text, result)
    return paths, record, prompts_path, receipts_path


def test_resume_loads_verified_prompt_as_prompt_record(tmp_path):
    paths, record, prompts, receipts = _cached_prompt(tmp_path)
    actual = stage_d.load_resumable_prompts(prompts, receipts)
    assert actual == {record.env_id: record}


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_receipt",
        "other_env",
        "wrong_hash",
        "missing_request_id",
        "blank_request_id",
        "missing_tokens",
        "wrong_provider",
        "empty_prompt",
    ],
)
def test_resume_rejects_missing_or_mismatching_provider_evidence(tmp_path, mutation):
    paths, record, prompts, receipts = _cached_prompt(tmp_path)
    receipt = json.loads(receipts.read_text())
    if mutation == "missing_receipt":
        receipts.unlink()
    elif mutation == "empty_prompt":
        row = record.model_dump(mode="json")
        row["prompt_text"] = "   "
        write_jsonl(prompts, [row])
    else:
        if mutation == "other_env":
            receipt["env_id"] = "other-env"
        elif mutation == "wrong_hash":
            receipt["prompt_sha256"] = "0" * 64
        elif mutation == "missing_request_id":
            receipt.pop("request_id")
        elif mutation == "blank_request_id":
            receipt["request_id"] = " "
        elif mutation == "missing_tokens":
            receipt.pop("output_tokens")
        elif mutation == "wrong_provider":
            receipt["provider"] = "litellm"
        write_jsonl(receipts, [receipt])
    assert stage_d.load_resumable_prompts(prompts, receipts) == {}


def test_resume_missing_prompt_file_has_no_reusable_prompts(tmp_path):
    assert (
        stage_d.load_resumable_prompts(
            tmp_path / "prompts.jsonl",
            tmp_path / "receipts.jsonl",
        )
        == {}
    )


def test_fully_cached_resume_uses_real_backend_without_credentials_and_preserves_ledgers(tmp_path):
    from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c

    paths, record, prompts, receipts = _cached_prompt(tmp_path)
    run_stage_c(paths, seed=20260930, split_policy="random_row", visible_ratio=0.6)
    before_prompts, before_receipts = prompts.read_bytes(), receipts.read_bytes()
    backend = BedrockConverseBackend()
    result = stage_d.run_stage_d(
        paths,
        backend_name="bedrock_converse",
        backend=backend,
        runtime=get_runtime("pymc"),
        use_batch=False,
        model="us.anthropic.claude-sonnet-4-6",
        resume=True,
    )
    assert result.status == "success"
    assert result.outputs["count"] == 1
    assert prompts.read_bytes() == before_prompts
    assert receipts.read_bytes() == before_receipts
    assert json.loads(prompts.read_text()) == record.model_dump(mode="json")
    assert backend._client is None


def test_throttle_events_are_appended_to_real_ledger_before_sleep(tmp_path):
    from functools import partial

    from ppl_synth_data.llm.bedrock_converse_backend import retry_converse_call

    ledger = tmp_path / "stage_D_prompts" / "throttle-log.jsonl"
    attempts = []

    def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise _client_error("ThrottlingException")
        return "finished"

    def check_persisted_before_sleep(wait):
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert len(rows) == len(attempts)
        assert rows[-1] == {
            "env_id": "env-a",
            "attempt": len(attempts),
            "wait": wait,
            "error_code": "ThrottlingException",
        }

    assert (
        retry_converse_call(
            operation,
            on_throttle=partial(stage_d.persist_throttle_event, ledger, "env-a"),
            sleep=check_persisted_before_sleep,
        )
        == "finished"
    )
    assert len(ledger.read_text().splitlines()) == 2
    assert not ledger.with_name("receipts.jsonl").exists()


def test_resume_preserves_receipt_history_when_one_receipt_is_incomplete(tmp_path):
    paths, record, prompts, receipts = _cached_prompt(tmp_path)
    row = json.loads(receipts.read_text())
    row.pop("request_id")
    with receipts.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    before = receipts.read_bytes()
    assert stage_d.load_resumable_prompts(prompts, receipts) == {record.env_id: record}
    assert receipts.read_bytes() == before

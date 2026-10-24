from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ppl_synth_data.llm.backend import LLMResult
from ppl_synth_data.llm.bedrock_converse_backend import result_from_converse
from ppl_synth_data.pipeline.stages.stage_d_prompts import persist_prompt_receipt

FIXTURES = Path(__file__).parent / "fixtures" / "bedrock_receipt"
MODEL_ID = "us.anthropic.claude-sonnet-4-6"
EXPECTED_RECEIPT = {
    "provider": "aws-bedrock-converse",
    "model_id": MODEL_ID,
    "request_id": "5a3cadcf-e9fe-44f6-b7d2-1ed094c0414f",
    "input_tokens": 31,
    "output_tokens": 5,
    "stop_reason": "end_turn",
    "latency_ms": 979,
}


def captured_converse():
    return json.loads((FIXTURES / "converse-sonnet-4-6.json").read_text())


def test_captured_converse_preserves_text_and_provider_receipt():
    result = result_from_converse(captured_converse(), MODEL_ID)
    assert isinstance(result, LLMResult)
    assert result.content == "11"
    assert result.raw["receipt"] == EXPECTED_RECEIPT


@pytest.mark.parametrize("field", ["RequestId", "inputTokens", "outputTokens"])
@pytest.mark.parametrize("replacement", ["missing", None, ""])
def test_converse_refuses_missing_request_id_or_token_counts(field, replacement):
    response = captured_converse()
    section = response["ResponseMetadata"] if field == "RequestId" else response["usage"]
    if replacement == "missing":
        section.pop(field)
    else:
        section[field] = replacement
    with pytest.raises((ValueError, RuntimeError)):
        result_from_converse(response, MODEL_ID)


def test_zero_tokens_are_real_counts_not_missing_counts():
    response = captured_converse()
    response["usage"]["inputTokens"] = 0
    response["usage"]["outputTokens"] = 0
    receipt = result_from_converse(response, MODEL_ID).raw["receipt"]
    assert receipt["input_tokens"] == 0
    assert receipt["output_tokens"] == 0


def test_litellm_completion_id_cannot_replace_aws_request_id():
    response = json.loads((FIXTURES / "litellm-bedrock-sonnet-4-6-raw.json").read_text())
    assert response["id"].startswith("chatcmpl-")
    with pytest.raises((ValueError, RuntimeError)):
        result_from_converse(response, MODEL_ID)


def test_stage_d_appends_one_receipt_for_each_prompt_with_final_text_hash(tmp_path):
    path = tmp_path / "stage_D_prompts" / "receipts.jsonl"
    result = result_from_converse(captured_converse(), MODEL_ID)
    prompts = [("env-a", "11\nFinal schema: y."), ("env-b", "11\nFinal schema: café.")]
    for env_id, prompt_text in prompts:
        persist_prompt_receipt(path, env_id, prompt_text, result)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    for row, (env_id, prompt_text) in zip(rows, prompts, strict=True):
        assert row == {
            **EXPECTED_RECEIPT,
            "env_id": env_id,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        }


@pytest.mark.parametrize("field", ["request_id", "input_tokens", "output_tokens"])
@pytest.mark.parametrize("replacement", ["missing", None, ""])
def test_stage_d_refuses_incomplete_receipt_before_appending(tmp_path, field, replacement):
    path = tmp_path / "receipts.jsonl"
    good_result = result_from_converse(captured_converse(), MODEL_ID)
    persist_prompt_receipt(path, "already-recorded", "11", good_result)
    before = path.read_bytes()
    raw = copy.deepcopy(good_result.raw)
    if replacement == "missing":
        raw["receipt"].pop(field)
    else:
        raw["receipt"][field] = replacement
    incomplete_result = LLMResult(content=good_result.content, raw=raw)
    with pytest.raises((ValueError, RuntimeError)):
        persist_prompt_receipt(path, "refused", "11", incomplete_result)
    assert path.read_bytes() == before


def test_stage_d_rejects_real_litellm_response_without_receipt(tmp_path):
    raw = json.loads((FIXTURES / "litellm-bedrock-sonnet-4-6-raw.json").read_text())
    path = tmp_path / "receipts.jsonl"
    result = LLMResult(content=raw["choices"][0]["message"]["content"], raw=raw)
    with pytest.raises((ValueError, RuntimeError)):
        persist_prompt_receipt(path, "refused", result.content, result)
    assert not path.exists()


@pytest.mark.parametrize("field", ["inputTokens", "outputTokens"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "31"])
def test_converse_refuses_invalid_token_counts(field, value):
    response = captured_converse()
    response["usage"][field] = value
    with pytest.raises((ValueError, RuntimeError)):
        result_from_converse(response, MODEL_ID)


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "31"])
def test_stage_d_refuses_invalid_token_counts_before_write(tmp_path, field, value):
    raw = copy.deepcopy(result_from_converse(captured_converse(), MODEL_ID).raw)
    raw["receipt"][field] = value
    path = tmp_path / "receipts.jsonl"
    with pytest.raises((ValueError, RuntimeError)):
        persist_prompt_receipt(path, "refused", "11", LLMResult(content="11", raw=raw))
    assert not path.exists()


@pytest.mark.parametrize("field", ["stopReason", "latencyMs"])
def test_converse_refuses_missing_stop_reason_or_latency(field):
    response = captured_converse()
    if field == "stopReason":
        response.pop(field)
    else:
        response["metrics"].pop(field)
    with pytest.raises((ValueError, RuntimeError)):
        result_from_converse(response, MODEL_ID)


def test_missing_environment_credentials_fail_before_client_creation(tmp_path):
    import subprocess
    import sys

    script = """
from ppl_synth_data.llm.backend import LLMRequest
from ppl_synth_data.pipeline.backends import build_backend
backend = build_backend("bedrock_converse")
try:
    backend.complete(LLMRequest(model="us.anthropic.claude-sonnet-4-6", system="", user="11"))
except RuntimeError as exc:
    assert "AWS_ACCESS_KEY_ID" in str(exc)
    assert "AWS_SECRET_ACCESS_KEY" in str(exc)
else:
    raise AssertionError("Missing environment credentials must fail")
"""
    run = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={},
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_registry_builds_converse_without_environment_configuration(tmp_path):
    import subprocess
    import sys

    script = """
from ppl_synth_data.llm.bedrock_converse_backend import BedrockConverseBackend
from ppl_synth_data.pipeline.backends import build_backend
backend = build_backend("bedrock_converse")
assert isinstance(backend, BedrockConverseBackend)
assert backend._client is None
"""
    run = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={},
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr

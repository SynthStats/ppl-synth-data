"""One paid live call; run explicitly with caller-provided AWS environment."""

from __future__ import annotations

import hashlib
import json
import os

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import LLMRequest
from ppl_synth_data.pipeline.backends import build_backend
from ppl_synth_data.pipeline.stages.stage_d_prompts import persist_prompt_receipt


def test_live_bedrock_converse_receipt(tmp_path):
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    access_keys = bool(
        os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
    if not region or not access_keys:
        raise RuntimeError(
            "Live Bedrock test requires caller environment AWS region and credentials; "
            "no profile or key-file lookup"
        )
    model_id = os.environ.get("BEDROCK_TEST_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    backend = build_backend("bedrock_converse", settings=Settings(_env_file=None))
    result = backend.complete(
        LLMRequest(
            model=model_id,
            system="Answer the arithmetic question with only its integer result.",
            user="What is 5 + 6?",
            max_output_tokens=32,
            temperature=0,
        )
    )
    path = tmp_path / "stage_D_prompts" / "receipts.jsonl"
    persist_prompt_receipt(path, "live-bedrock-converse", result.content, result)
    assert result.content.strip() == "11"
    receipt = result.raw["receipt"]
    assert receipt["provider"] == "aws-bedrock-converse"
    assert receipt["model_id"] == model_id
    assert isinstance(receipt["request_id"], str) and receipt["request_id"].strip()
    assert isinstance(receipt["input_tokens"], int) and receipt["input_tokens"] > 0
    assert isinstance(receipt["output_tokens"], int) and receipt["output_tokens"] > 0
    assert receipt["stop_reason"]
    assert isinstance(receipt["latency_ms"], (int, float)) and receipt["latency_ms"] >= 0
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {
            **receipt,
            "env_id": "live-bedrock-converse",
            "prompt_sha256": hashlib.sha256(result.content.encode("utf-8")).hexdigest(),
        }
    ]

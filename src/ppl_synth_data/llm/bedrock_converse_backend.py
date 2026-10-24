from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from ppl_synth_data.llm.backend import BackendCapabilities, LLMRequest, LLMResult

_ResultT = TypeVar("_ResultT")


def retry_converse_call(
    operation: Callable[[], _ResultT],
    *,
    on_throttle: Callable[[dict[str, Any]], None],
    max_attempts: int = 8,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    uniform: Callable[[float, float], float] = random.uniform,
) -> _ResultT:
    """Retry only explicit service throttling/unavailability, with capped full jitter.

    The caller persists each event before sleeping. SDK retries must be disabled
    so each attempt here represents exactly one Converse call.
    """
    from botocore.exceptions import ClientError

    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if any(not math.isfinite(value) or value <= 0 for value in (base_delay, max_delay)):
        raise ValueError("retry delays must be finite and positive")
    ceiling = min(base_delay, max_delay)
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"ThrottlingException", "ServiceUnavailableException"}:
                raise
            wait = uniform(0.0, ceiling) if attempt < max_attempts else 0.0
            on_throttle({"attempt": attempt, "wait": wait, "error_code": code})
            if attempt == max_attempts:
                raise
            sleep(wait)
            ceiling = min(max_delay, ceiling * 2)
    raise AssertionError("unreachable retry state")


def validate_converse_receipt(receipt: Any) -> dict[str, Any]:
    """Validate the provider evidence before a result can be persisted."""
    if not isinstance(receipt, dict) or receipt.get("provider") != "aws-bedrock-converse":
        raise RuntimeError("Bedrock Converse response requires an aws-bedrock-converse receipt")
    for field in ("model_id", "request_id", "stop_reason"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise RuntimeError(f"Bedrock Converse receipt missing {field}")
    for field in ("input_tokens", "output_tokens", "latency_ms"):
        value = receipt.get(field)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"Bedrock Converse receipt requires nonnegative integer {field}")
    return {
        key: receipt[key]
        for key in (
            "provider",
            "model_id",
            "request_id",
            "input_tokens",
            "output_tokens",
            "stop_reason",
            "latency_ms",
        )
    }


def result_from_converse(response: dict[str, Any], model_id: str) -> LLMResult:
    """Extract text and AWS response evidence without fabricating missing usage."""
    receipt = validate_converse_receipt(
        {
            "provider": "aws-bedrock-converse",
            "model_id": model_id,
            "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
            "input_tokens": response.get("usage", {}).get("inputTokens"),
            "output_tokens": response.get("usage", {}).get("outputTokens"),
            "stop_reason": response.get("stopReason"),
            "latency_ms": response.get("metrics", {}).get("latencyMs"),
        }
    )
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    content = "".join(block["text"] for block in blocks if "text" in block)
    return LLMResult(content=content, raw={"receipt": receipt})


class BedrockConverseBackend:
    """Direct Converse transport, with credentials supplied only by the caller's environment."""

    capabilities = BackendCapabilities(
        supports_batch=False,
        supports_json_schema=False,
        supports_reasoning_effort=False,
    )

    def __init__(self) -> None:
        self._client: Any = None

    def _create_client(self) -> Any:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            raise RuntimeError(
                "bedrock_converse requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "in the caller's environment; credential files and metadata are not consulted"
            )
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not region or not region.strip():
            raise RuntimeError("bedrock_converse requires AWS_REGION or AWS_DEFAULT_REGION")

        import boto3
        import botocore.session
        from botocore.config import Config

        # Keep configuration and credential-file discovery out of this transport.
        core = botocore.session.Session()
        core.set_config_variable("config_file", os.devnull)
        core.set_config_variable("credentials_file", os.devnull)
        core.get_component("config_store").set_config_variable("profile", None)
        core.set_credentials(access_key, secret_key, os.environ.get("AWS_SESSION_TOKEN"))
        session = boto3.Session(botocore_session=core, region_name=region)
        return session.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(signature_version="v4", retries={"total_max_attempts": 1}),
        )

    def complete(self, request: LLMRequest) -> LLMResult:
        if request.text_format is not None:
            raise ValueError("bedrock_converse does not support structured text_format")
        if not request.model.strip() or request.model.startswith(("bedrock/", "bedrock_mantle/")):
            raise ValueError("bedrock_converse requires a native Bedrock model ID or ARN")
        if self._client is None:
            self._client = self._create_client()
        kwargs: dict[str, Any] = {
            "modelId": request.model,
            "messages": [{"role": "user", "content": [{"text": request.user}]}],
        }
        if request.system:
            kwargs["system"] = [{"text": request.system}]
        inference: dict[str, Any] = {}
        if request.max_output_tokens is not None:
            inference["maxTokens"] = request.max_output_tokens
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if inference:
            kwargs["inferenceConfig"] = inference
        return result_from_converse(self._client.converse(**kwargs), request.model)

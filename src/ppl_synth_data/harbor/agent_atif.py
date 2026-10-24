"""Prompt-message construction and ATIF trajectory-record building."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def build_messages(instruction: str, task_config: Mapping[str, Any]) -> list[dict[str, str]]:
    system = (
        "You synthesize one probabilistic program for the Harbor task. "
        "Do not output hidden reasoning or <think> tags. "
        "Submit only the final program as a fenced code block or a "
        "<submit_program> block."
    )
    user = f"/no_think\n\n{instruction.strip()}"
    if task_config:
        user = f"{user}\n\nTask configuration:\n{json.dumps(dict(task_config), sort_keys=True)}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_atif_record(
    *,
    messages: Sequence[Mapping[str, str]],
    response_text: str,
    generation_metadata: Mapping[str, Any],
    task_config: Mapping[str, Any],
    task_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    metrics: dict[str, Any] = {
        "prompt_token_ids": list(generation_metadata["prompt_ids"]),
        "completion_token_ids": list(generation_metadata["token_ids"]),
        "logprobs": list(generation_metadata["logprobs"]),
    }
    eos_logprobs = generation_metadata.get("eos_logprobs")
    if isinstance(eos_logprobs, list):
        metrics["eos_logprobs"] = list(eos_logprobs)
    for key in (
        "eos_potential_coverage",
        "terminal_eos_token_count",
        "terminal_eos_token_coverage",
    ):
        value = generation_metadata.get(key)
        if isinstance(value, int | float):
            metrics[key] = float(value)
    token_entropies = generation_metadata.get("token_entropies")
    if isinstance(token_entropies, list):
        metrics["token_entropies"] = list(token_entropies)

    steps: list[dict[str, Any]] = [
        {
            "step_id": f"step-prompt-{index}",
            "source": str(message.get("role", "user")),
            "message": str(message.get("content", "")),
        }
        for index, message in enumerate(messages)
    ]
    steps.append(
        {
            "step_id": "step-agent",
            "source": "agent",
            "message": response_text,
            "metrics": metrics,
            "metadata": {
                "stop_reason": generation_metadata.get("stop_reason", "completed"),
            },
        }
    )

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": session_id,
        "metadata": {
            "task_config": dict(task_config),
            "task_provenance": dict(task_provenance),
        },
        "steps": steps,
    }


def write_agent_result_log(result: Mapping[str, Any], *, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(result), indent=2, sort_keys=True), encoding="utf-8")

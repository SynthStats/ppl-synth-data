"""Submitted-program extraction from LLM text, and writing it into the sandbox.

Covers both local writes (used in tests / off-sandbox tooling) and the
chunked, shell-safe remote writes used against a live Harbor ``environment``.
"""

from __future__ import annotations

import base64
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ppl_synth_data.harbor.agent_env import submission_path_for_runtime
from ppl_synth_data.harbor.agent_shared import _json_mapping

_REMOTE_WRITE_CHUNK_BYTES = 6 * 1024

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_SUBMIT_PROGRAM_PATTERN = re.compile(
    r"<submit_program(?:\s+[^>]*)?>\s*(.*?)\s*</submit_program>",
    re.DOTALL,
)
_PROGRAM_TAG_PATTERN = re.compile(r"<program>\s*(.*?)\s*</program>", re.DOTALL)
_FENCE_PATTERN = re.compile(r"```(?P<lang>[A-Za-z0-9_+.-]*)\s*\n(?P<code>.*?)```", re.DOTALL)


def _extract_tool_call_program(text: str) -> str | None:
    for match in _TOOL_CALL_PATTERN.finditer(text):
        payload = _json_mapping(match.group(1))
        if payload is None:
            continue
        function = payload.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            name = payload.get("name")
            arguments = payload.get("arguments")
        if name not in {"submit_program", "submit", "final_answer"}:
            continue
        if isinstance(arguments, str):
            arguments = _json_mapping(arguments)
        if not isinstance(arguments, Mapping):
            continue
        code = arguments.get("code") or arguments.get("program") or arguments.get("answer")
        if isinstance(code, str) and code.strip():
            return code.strip()
    return None


def extract_submitted_program(text: str, *, runtime: str) -> str:
    """Extract the submitted program from a model response.

    Supported surfaces are explicit tool calls, ``<program>`` tags, and fenced
    code blocks. Raw prose is accepted only when no structured program wrapper is
    present; the verifier remains the authority on whether the content is valid.
    """

    tool_program = _extract_tool_call_program(text)
    if tool_program is not None:
        return tool_program

    for tag_pattern in (_SUBMIT_PROGRAM_PATTERN, _PROGRAM_TAG_PATTERN):
        tag_match = tag_pattern.search(text)
        if tag_match and tag_match.group(1).strip():
            return tag_match.group(1).strip()

    preferred_languages = (
        {"pymc", "python", "py"} if runtime in {"pymc", "python", "py"} else {"haskell", "hs"}
    )
    fallback_code: str | None = None
    for match in _FENCE_PATTERN.finditer(text):
        language = match.group("lang").strip().lower()
        code = match.group("code").strip()
        if not code:
            continue
        fallback_code = fallback_code or code
        if language in preferred_languages:
            return code
    if fallback_code is not None:
        return fallback_code

    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM response did not contain a submitted program")
    return stripped


def write_submission_program(
    program: str,
    *,
    runtime: str,
    root: Path | None = None,
) -> Path:
    path = submission_path_for_runtime(runtime)
    if root is not None:
        path = root / path.relative_to("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(program, encoding="utf-8")
    return path


def _write_text_commands(path: Path, text: str) -> list[str]:
    target = shlex.quote(str(path))
    parent = shlex.quote(str(path.parent))
    data = text.encode("utf-8")
    if not data:
        return [f"mkdir -p {parent} && : > {target}"]

    commands: list[str] = []
    for offset in range(0, len(data), _REMOTE_WRITE_CHUNK_BYTES):
        chunk = data[offset : offset + _REMOTE_WRITE_CHUNK_BYTES]
        payload = shlex.quote(base64.b64encode(chunk).decode("ascii"))
        prefix = f"mkdir -p {parent} && " if offset == 0 else ""
        redirect = ">" if offset == 0 else ">>"
        commands.append(f"{prefix}printf %s {payload} | base64 -d {redirect} {target}")
    return commands


async def write_text_to_environment(
    agent: Any,
    environment: Any,
    *,
    path: Path,
    text: str,
) -> None:
    for command in _write_text_commands(path, text):
        await agent.exec_as_root(environment, command=command)

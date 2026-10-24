from __future__ import annotations

import ast
import re

from ppl_synth_data.llm.completion_markup import THINK_RE, code_fence_re

_SUBMIT_PROGRAM_RE = re.compile(r"<submit_program>\s*\n?(.*?)</submit_program>", re.DOTALL)
_CODE_FENCE_RE = code_fence_re("python")


def validate_completion_format(completion: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not THINK_RE.search(completion):
        issues.append("missing <think> block")
    code = extract_python_code(completion)
    if code is None:
        issues.append("missing code block (no <submit_program> or python fence)")
        issues.append("missing def model(data): definition")
    elif not _has_top_level_model_def(code):
        issues.append("missing def model(data): definition")
    return (len(issues) == 0, issues)


def extract_python_code(completion: str) -> str | None:
    match = _SUBMIT_PROGRAM_RE.search(completion)
    if match:
        return match.group(1).strip()
    match = _CODE_FENCE_RE.search(completion)
    if match:
        return match.group(1).strip()
    return None


def replace_python_code(completion: str, python_code: str) -> str:
    normalized = python_code.rstrip() + "\n"

    match = _SUBMIT_PROGRAM_RE.search(completion)
    if match:
        start, end = match.span(1)
        return completion[:start] + "\n" + normalized + completion[end:]

    match = _CODE_FENCE_RE.search(completion)
    if match:
        start, end = match.span(1)
        return completion[:start] + normalized + completion[end:]

    think = (
        "<think>\n"
        "Repaired program chosen to satisfy compilation and safety constraints.\n"
        "</think>\n\n"
    )
    return f"{think}<submit_program>\n{normalized}</submit_program>"


def _has_top_level_model_def(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "model":
            if len(node.args.args) != 1:
                return False
            return node.args.args[0].arg == "data"
    return False

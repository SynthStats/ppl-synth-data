from __future__ import annotations

import re

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def code_fence_re(lang: str) -> re.Pattern[str]:
    return re.compile(rf"```{re.escape(lang)}\s*\n(.*?)```", re.DOTALL)


def extract_think_text(completion: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    return match.group(1).strip() if match else None


def strip_think_block(completion: str) -> str:
    return THINK_RE.sub("", completion).strip()

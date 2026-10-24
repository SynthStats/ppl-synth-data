from __future__ import annotations

from importlib import resources


def strip_html_comments(text: str, *, source: str) -> str:
    parts: list[str] = []
    start = 0
    while (comment_start := text.find("<!--", start)) != -1:
        comment_end = text.find("-->", comment_start + 4)
        if comment_end == -1:
            raise ValueError(f"Unterminated HTML comment in {source}")
        parts.append(text[start:comment_start])
        start = comment_end + 3
    parts.append(text[start:])
    return "".join(parts)


def load_prompt_text(name: str) -> str:
    resource = resources.files("ppl_synth_data").joinpath("llm/prompts", name)
    return strip_html_comments(resource.read_text(encoding="utf-8"), source=name)

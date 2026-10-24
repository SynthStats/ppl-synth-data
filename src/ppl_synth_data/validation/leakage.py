from __future__ import annotations

import re
from pathlib import Path
from typing import Final


def _normalize_text(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"[_-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def load_banned_terms(path: Path) -> list[str]:
    terms: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and not text.startswith("#"):
                terms.append(_normalize_text(text))
    return terms


def detect_banned_terms(text: str, banned_terms: list[str]) -> list[str]:
    normalized = _normalize_text(text)
    hits: list[str] = []
    for term in banned_terms:
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
        if pat.search(normalized):
            hits.append(term)
    return sorted(set(hits))


# Source-provenance leakage: a `source_url_or_repo` is only safe to embed in a
# model prompt if it points at a public location. A local filesystem path,
# file:// URI, or bare/relative path is private and must never reach a prompt
# (it leaks the author's machine layout and the existence of private files).
_PUBLIC_SOURCE_SCHEMES: Final = ("https://", "http://")
_PROVENANCE_SENTINELS: Final = frozenset({"not_applicable_invented", "not_applicable"})


def is_public_source(value: str) -> bool:
    """True only for public web URLs and the not-applicable sentinels."""
    candidate = value.strip()
    if not candidate:
        return False
    if candidate in _PROVENANCE_SENTINELS:
        return True
    return candidate.lower().startswith(_PUBLIC_SOURCE_SCHEMES)


def public_source_or_sentinel(
    value: str | None, *, sentinel: str = "not_applicable_invented"
) -> str | None:
    """Pass through publishable values (public URL or sentinel); replace any
    private value (local path, file://, relative path) with the sentinel so it
    cannot leak into an EnvSpec prompt. None is preserved (means "no source")."""
    if value is None:
        return None
    return value if is_public_source(value) else sentinel

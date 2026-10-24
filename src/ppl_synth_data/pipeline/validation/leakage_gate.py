from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from importlib import resources
from typing import Any

from ppl_synth_data.llm.backend import LLMBackend, LLMRequest
from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.specs.models import CompletionRecord
from ppl_synth_data.validation.leakage import detect_banned_terms, load_banned_terms

log = logging.getLogger(__name__)

DETERMINISTIC_JUDGE_BACKENDS = {"deterministic", "local", "scanner"}
LEAKAGE_SCAN_SECTION_MARKERS = (
    "## PyMC data dictionary",
    "## LazyPPL TaskData schema",
    "PyMC submission constraint:",
    "LazyPPL submission constraint:",
    "Final answer contract for this request:",
    "Data dictionary contract for",
    "Data dictionary contract for the submitted",
)


@lru_cache(maxsize=1)
def default_banned_terms() -> tuple[str, ...]:
    resource = resources.files("ppl_synth_data").joinpath("resources/banned_terms.txt")
    with resources.as_file(resource) as resource_path:
        return tuple(load_banned_terms(resource_path))


def judge_leakage(
    prompt_text: str,
    backend: LLMBackend,
    model: str,
    actual_family: str | None = None,
) -> dict[str, Any]:
    system = load_prompt_text("leakage_judge.md")
    request = LLMRequest(
        model=model,
        system=system,
        user=prompt_text,
    )
    result = backend.complete(request)
    raw = result.content.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        end = len(lines)
        while end > 1 and not lines[end - 1].strip():
            end -= 1
        if end > 1 and lines[end - 1].startswith("```"):
            end -= 1
        raw = "\n".join(lines[1:end])

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("leakage judge returned non-JSON response (%d chars)", len(raw))
        return {
            "rating": "INCONCLUSIVE",
            "evidence": "judge parse error",
            "recovered_family": None,
            "leaked": True,
        }

    rating_value = parsed.get("rating")
    if not isinstance(rating_value, str):
        return {
            "rating": "INCONCLUSIVE",
            "evidence": "judge response missing string rating",
            "recovered_family": parsed.get("recovered_family"),
            "leaked": True,
        }
    rating = rating_value.strip().upper()
    if rating not in {"CLEAN", "LEAKED", "INCONCLUSIVE"}:
        return {
            "rating": "INCONCLUSIVE",
            "evidence": "judge response has invalid rating value",
            "recovered_family": parsed.get("recovered_family"),
            "leaked": True,
        }
    recovered = parsed.get("recovered_family")
    if recovered is not None and not isinstance(recovered, str):
        return {
            "rating": "INCONCLUSIVE",
            "evidence": "judge response has invalid recovered_family",
            "recovered_family": None,
            "leaked": True,
        }
    leaked = rating != "CLEAN"

    if actual_family and recovered and recovered.lower() == actual_family.lower():
        leaked = True

    return {
        "rating": rating,
        "evidence": parsed.get("evidence", ""),
        "recovered_family": recovered,
        "leaked": leaked,
    }


def judge_leakage_deterministic(
    prompt_text: str,
    actual_family: str | None = None,
) -> dict[str, Any]:
    hits = detect_banned_terms(prompt_text, list(default_banned_terms()))
    leaked = bool(hits)
    return {
        "rating": "LEAKED" if leaked else "CLEAN",
        "evidence": ", ".join(hits),
        "recovered_family": hits[0] if hits else None,
        "leaked": leaked,
        "method": "deterministic_banned_terms",
        "actual_family": actual_family,
    }


def prompt_text_for_leakage_scan(prompt_text: str) -> tuple[str, tuple[str, ...]]:
    """Return prompt text with trusted runtime-contract boilerplate masked."""
    present_markers = tuple(
        marker for marker in LEAKAGE_SCAN_SECTION_MARKERS if marker in prompt_text
    )
    if not present_markers:
        return prompt_text, ()

    scan_text = prompt_text
    for marker in present_markers:
        scan_text = scan_text.replace(marker, "")
    scan_text = re.sub(r"\bPyMC\b", "PPL", scan_text, flags=re.IGNORECASE)
    scan_text = re.sub(r"\n{3,}", "\n\n", scan_text).strip()
    return scan_text, present_markers


def completion_generative_family(rec: CompletionRecord) -> str | None:
    if rec.provenance is not None and rec.provenance.generative_family:
        return rec.provenance.generative_family
    metadata_family = rec.metadata.get("generative_family")
    if isinstance(metadata_family, str) and metadata_family.strip():
        return metadata_family
    return None

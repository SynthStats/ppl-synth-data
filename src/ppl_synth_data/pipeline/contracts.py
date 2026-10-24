from __future__ import annotations

import re

PYMC_ROW_COUNT_KEY_LINE = "- `n` (integer number of rows; metadata, not a predictor)"

STAGE_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
STAGE_DIR_NAMES: dict[str, str] = {
    "A": "stage_A_envspec",
    "B": "stage_B_envs",
    "C": "stage_C_data",
    "D": "stage_D_prompts",
    "E": "stage_E_completions",
    "F": "stage_F_validation",
    "G": "stage_G_dataset",
}
VALID_STAGE_STATUSES: frozenset[str] = frozenset(
    {"success", "completed_with_refusals", "skipped", "failed"}
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

REQUIRED_STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "A": ("manifest.json", "envspecs.jsonl"),
    "B": ("manifest.json", "envs"),
    "C": ("manifest.json", "observations.jsonl", "summaries.jsonl", "data_refs.jsonl"),
    "D": ("manifest.json", "prompts.jsonl"),
    "E": ("manifest.json", "completions.jsonl"),
    "F": ("manifest.json", "validated.jsonl", "rejected.jsonl", "traces.jsonl"),
    "G": (
        "manifest.json",
        "dropped.jsonl",
    ),
}

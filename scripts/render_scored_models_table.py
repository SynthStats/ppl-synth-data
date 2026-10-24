#!/usr/bin/env python3
"""Render a scored-models table (CSV + self-contained HTML) from real
reward_artifact.json files (schema synthstats.harbor.reward.v1).

Each positional directory is scanned recursively for reward_artifact.json
files; every file found is one candidate row. Arm receipts supply candidate
paths, environment identity, target and container wall time. Copied references
join to a unique task under the same run's tasks directory. Harbor result.json
supplies trial identity and trial wall time separately.

Missing identities and scores render as "absent", never as 0. Missing
diagnostics render as "not retained" or "not evaluated" from fit evidence.
Rows preserve input and artifact-path order unless --order names an arm order.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ABSENT = "absent"

# (column key, display label)
COLUMNS: list[tuple[str, str]] = [
    ("task_id", "Task ID"),
    ("task_directory", "Task directory"),
    ("env_id", "Environment ID"),
    ("env_name", "Environment name"),
    ("target_name", "Target"),
    ("arm", "Arm"),
    ("candidate", "Candidate path"),
    ("candidate_index", "Candidate index"),
    ("safepymc_verdict", "SafePyMC L3"),
    ("layers_passed", "Layers passed"),
    ("reward", "Reward"),
    ("training_log_reward", "Training log reward"),
    ("avg_holdout_logp", "Mean holdout logp"),
    ("holdout_elpd", "Holdout ELPD"),
    ("train_elpd_loo", "Train LOO ELPD"),
    ("pareto_k_max", "Pareto k maximum"),
    ("density_gate_status", "Density gate status"),
    ("pareto_k_warning", "Pareto k warning"),
    ("ess_bulk", "Bulk effective sample size"),
    ("ess_tail", "Tail effective sample size"),
    ("bfmi", "Energy diagnostic (BFMI)"),
    ("divergences", "Divergences"),
    ("r_hat", "R-hat"),
    ("container_wall_seconds", "Container wall time (s)"),
    ("wall_seconds", "Trial wall time (s)"),
    ("task_identity_source", "Task identity receipt"),
    ("split_receipt", "Environment identity receipt"),
    ("reference_receipt", "Reference receipt"),
    ("reward_receipt", "Reward receipt"),
    ("wall_time_receipt", "Container timing receipt"),
    ("failed_layer", "Failed layer"),
    ("failure_reason", "Failure reason"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return ABSENT
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # repr is the shortest round-trip form, so the table shows the number
        # exactly as the artifact recorded it (no trailing-zero collapse).
        return repr(value)
    return str(value)


def discover_artifacts(directory: Path) -> list[Path]:
    """All reward_artifact.json files under directory, sorted by path."""
    return sorted(directory.rglob("reward_artifact.json"))


def _find_result_json(artifact_path: Path, stop: Path) -> Path | None:
    """Nearest result.json at or above the artifact, not above stop's parent."""
    node = artifact_path.parent
    stop = stop.resolve()
    while True:
        candidate = node / "result.json"
        if candidate.is_file():
            return candidate
        if node == stop or node.parent == node:
            return None
        node = node.parent
        if stop not in node.resolve().parents and node.resolve() != stop:
            return None


def _receipt_directory(artifact_path: Path) -> Path:
    directory = artifact_path.parent
    while directory.name in ("verifier", "logs"):
        directory = directory.parent
    return directory


def _identity_receipts(directory: Path) -> dict[str, str]:
    """Read arm receipts; match copied references only within this run's tasks."""
    fields = dict.fromkeys(
        [
            "env_id",
            "env_name",
            "target_name",
            "task_id",
            "task_directory",
            "task_identity_source",
            "split_receipt",
            "reference_receipt",
        ],
        ABSENT,
    )
    split = directory / "split_policy.txt"
    if split.is_file():
        fields["split_receipt"] = str(split)
        for line in split.read_text().splitlines():
            _, marker, payload = line.partition("stage_C_data_refs.jsonl provenance line: ")
            if marker:
                provenance = json.loads(payload)
                fields["env_id"] = _fmt(provenance.get("env_id"))
                fields["env_name"] = _fmt(provenance.get("env_name"))
    reference_path = directory / "elpd_reference.json"
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text())
        fields["reference_receipt"] = str(reference_path)
        fields["target_name"] = _fmt(reference.get("target_name"))
        # The gate copies tasks/<task>/tests/elpd_reference.json into each arm.
        # Match the full payload, never a guessed task name or a score alone.
        matches = [
            path
            for path in (directory.parent.parent / "tasks").glob("*/tests/elpd_reference.json")
            if reference and json.loads(path.read_text()) == reference
        ]
        if len(matches) == 1:
            task = matches[0].parent.parent
            fields["task_id"] = task.name
            fields["task_directory"] = str(task)
            fields["task_identity_source"] = str(matches[0])
    return fields


def _wall_seconds(result: dict[str, Any]) -> str:
    started = result.get("started_at")
    finished = result.get("finished_at")
    if not started or not finished:
        return ABSENT
    try:
        t0 = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
    except ValueError:
        return ABSENT
    return _fmt(round((t1 - t0).total_seconds(), 3))


def _container_wall_seconds(path: Path) -> str:
    if not path.is_file():
        return ABSENT
    records = [
        line.split() for line in path.read_text().splitlines() if line.split()[:1] == ["real"]
    ]
    if len(records) != 1 or len(records[0]) != 2:
        raise ValueError(f"{path}: expected exactly one real time record")
    value = records[0][1]
    match = re.fullmatch(r"(?:(\d+)m)?(\d+(?:\.\d+)?)s", value)
    try:
        seconds = (60 * int(match[1] or 0) + float(match[2])) if match else float(value)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid real time {value!r}") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{path}: real time must be finite and nonnegative")
    return _fmt(round(seconds, 3))


def _verdict(diagnostics: dict[str, Any]) -> str:
    details = diagnostics.get("details")
    if not isinstance(details, dict):
        return ABSENT
    for name, layer in details.items():
        if str(name).startswith("L3") and isinstance(layer, dict) and "passed" in layer:
            return "pass" if layer["passed"] else "fail"
    return ABSENT


def _layers(metrics: dict[str, Any]) -> str:
    passed = metrics.get("passed_layers")
    total = metrics.get("total_layers")
    if passed is None or total is None:
        return ABSENT
    return f"{passed}/{total}"


def _diagnostic_fields(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, str]:
    fitted = metrics.get("train_elpd_loo") is not None or diagnostics.get("elpd_status") == "scored"
    missing = "not retained" if fitted else "not evaluated"
    fields = {}
    for key in [
        "pareto_k_max",
        "pareto_k_warning",
        "density_gate_status",
        "ess_bulk",
        "ess_tail",
        "bfmi",
        "divergences",
        "r_hat",
    ]:
        value = diagnostics.get(key)
        if value is None:
            value = metrics.get(key)
        fields[key] = missing if value is None else _fmt(value)
    return fields


def build_row(
    artifact_path: Path,
    *,
    root: Path,
    arm: str,
    candidate_index: int,
) -> dict[str, str]:
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") != "synthstats.harbor.reward.v1":
        raise RuntimeError(
            f"{artifact_path}: schema_version {artifact.get('schema_version')!r} "
            "is not synthstats.harbor.reward.v1"
        )
    metrics = artifact.get("metrics") or {}
    diagnostics = artifact.get("diagnostics") or {}

    result_path = _find_result_json(artifact_path, root)
    result: dict[str, Any] = {}
    if result_path is not None:
        result = json.loads(result_path.read_text())

    receipt_dir = _receipt_directory(artifact_path)
    identity = _identity_receipts(receipt_dir)
    candidate_path = receipt_dir / "candidate.py"
    candidate = str(candidate_path) if candidate_path.is_file() else ABSENT
    if result.get("task_name") is not None:
        identity["task_id"] = _fmt(result["task_name"])
        identity["task_directory"] = _fmt((result.get("task_id") or {}).get("path"))
        identity["task_identity_source"] = str(result_path)

    return {
        **identity,
        "arm": arm,
        "candidate": candidate,
        "candidate_index": str(candidate_index),
        "safepymc_verdict": _verdict(diagnostics),
        "layers_passed": _layers(metrics),
        "reward": _fmt(artifact.get("reward")),
        "training_log_reward": _fmt(artifact.get("training_log_reward")),
        "avg_holdout_logp": _fmt(metrics.get("avg_holdout_logp")),
        "holdout_elpd": _fmt(metrics.get("holdout_elpd")),
        "train_elpd_loo": _fmt(metrics.get("train_elpd_loo")),
        **_diagnostic_fields(metrics, diagnostics),
        "wall_seconds": _wall_seconds(result),
        "container_wall_seconds": _container_wall_seconds(receipt_dir / "wall-time.txt"),
        "reward_receipt": str(artifact_path),
        "wall_time_receipt": (
            str(receipt_dir / "wall-time.txt")
            if (receipt_dir / "wall-time.txt").is_file()
            else ABSENT
        ),
        "failed_layer": _fmt(diagnostics.get("failed_layer")),
        "failure_reason": _fmt(diagnostics.get("failure_reason")),
    }


def build_rows(
    directories: list[Path],
    *,
    arms: dict[str, str] | None = None,
    order: list[str] | None = None,
) -> list[dict[str, str]]:
    arms = arms or {}

    rows: list[dict[str, str]] = []
    labels: set[str] = set()
    for directory in directories:
        input_name = directory.name
        directory = directory.resolve()
        if not directory.is_dir():
            raise RuntimeError(f"input directory does not exist: {directory}")
        if input_name in ("", ".."):
            input_name = directory.name
        arm = arms.get(input_name, input_name)
        labels.add(arm)
        for index, artifact_path in enumerate(discover_artifacts(directory)):
            rows.append(
                build_row(
                    artifact_path,
                    root=directory,
                    arm=arm,
                    candidate_index=index,
                )
            )

    if order is not None:
        if len(order) != len(set(order)) or set(order) != labels:
            raise ValueError(f"--order must name each arm exactly once: {sorted(labels)}")
        positions = {arm: index for index, arm in enumerate(order)}
        rows.sort(key=lambda row: positions[row["arm"]])
    return rows


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[label for _, label in COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow({label: row[key] for key, label in COLUMNS})


_PAGE_STYLE = """
:root { color-scheme: light dark; --bg: #fff; --fg: #18202b; --muted: #505c6e;
        --line: #c9d1dc; --tint: #f1f4f8; --link: #174fa2; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #151b24; --fg: #eef2f8; --muted: #b6c2d3;
          --line: #46566c; --tint: #222e3e; --link: #99c4ff; }
}
* { box-sizing: border-box; }
body { font: 16px/1.5 -apple-system, Helvetica, Arial, sans-serif;
       margin: 0; padding: 28px; background: var(--bg); color: var(--fg); }
main { max-width: 1680px; margin: auto; }
h1 { font-size: 28px; margin: 0 0 8px; }
h2 { font-size: 22px; margin-top: 32px; }
a { color: var(--link); text-underline-offset: 3px; }
a:focus-visible, summary:focus-visible, .table-wrap:focus-visible {
  outline: 3px solid var(--link); outline-offset: 3px;
}
.table-wrap { overflow-x: auto; margin: 28px 0 12px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
caption { text-align: left; font-size: 22px; font-weight: 650; padding-bottom: 12px; }
th, td { border-bottom: 1px solid var(--line); padding: 10px 12px;
         text-align: left; vertical-align: top; font-variant-numeric: tabular-nums; }
th { background: var(--tint); }
.task-id { min-width: 160px; max-width: 240px; overflow-wrap: anywhere; }
.absent, .note { color: var(--muted); }
.absent { font-style: italic; }
.note { max-width: 90ch; font-size: 14px; }
details { border: 1px solid var(--line); padding: 14px 18px; margin: 12px 0; }
summary { cursor: pointer; font-weight: 650; }
dl { display: grid; grid-template-columns: minmax(120px, 200px) minmax(0, 1fr);
     gap: 8px 20px; font-size: 14px; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
footer { margin-top: 32px; border-top: 1px solid var(--line); padding-top: 16px; }
footer code { overflow-wrap: anywhere; }
@media (max-width: 640px) {
  body { padding: 16px; } dl { grid-template-columns: 1fr; gap: 4px; }
  dd { margin-bottom: 12px; }
}
"""


def write_html(
    rows: list[dict[str, str]],
    out_path: Path,
    *,
    input_dirs: list[Path],
    csv_path: Path,
) -> None:
    esc = html.escape
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Scored models table</title>",
        f"<style>{_PAGE_STYLE}</style>",
        "</head><body><main>",
        "<h1>Scored models table</h1>",
        f"<p>{len(rows)} candidates. Arm order follows the supplied inputs or --order.</p>",
    ]
    labels = dict(COLUMNS)
    score_keys = [
        "arm",
        "task_id",
        "safepymc_verdict",
        "layers_passed",
        "avg_holdout_logp",
        "holdout_elpd",
        "train_elpd_loo",
        "reward",
        "training_log_reward",
        "container_wall_seconds",
    ]
    if any(row["wall_seconds"] != ABSENT for row in rows):
        score_keys.append("wall_seconds")
    diagnostic_keys = [
        "arm",
        "pareto_k_max",
        "pareto_k_warning",
        "density_gate_status",
        "ess_bulk",
        "ess_tail",
        "bfmi",
        "divergences",
        "r_hat",
    ]
    for table_id, title, keys in [
        ("scores", "Scores", score_keys),
        ("diagnostics", "Diagnostics", diagnostic_keys),
    ]:
        parts.append(f'<div class="table-wrap" role="region" aria-label="{title}" tabindex="0">')
        parts.append(f'<table id="{table_id}"><caption>{title}</caption><thead><tr>')
        parts.extend(f'<th scope="col">{esc(labels[key])}</th>' for key in keys)
        parts.append("</tr></thead><tbody>")
        for index, row in enumerate(rows):
            parts.append("<tr>")
            for key in keys:
                value = row[key]
                cls = (
                    ' class="absent"' if value in (ABSENT, "not retained", "not evaluated") else ""
                )
                if key == "task_id":
                    cls = ' class="task-id"'
                rendered = esc(value)
                if key == "arm":
                    rendered = f'<a href="#receipt-{index}">{rendered}</a>'
                parts.append(f"<td{cls}>{rendered}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
    parts.append(
        '<p class="note">Container wall time is the full container invocation. '
        "Trial wall time, when recorded, covers the Harbor trial. "
        "SafePyMC L3 reports the recorded L3 safety check only.</p>"
    )
    parts.append(
        '<p class="note">Missing values after a completed fit are "not retained"; '
        'diagnostics before a completed fit are "not evaluated". '
        "ELPD is expected log predictive density; LOO denotes leave-one-out. "
        "Each diagnostic is reported separately.</p>"
    )
    parts.append("<h2>Candidate files and source receipts</h2>")
    receipt_keys = [
        "task_id",
        "task_directory",
        "env_id",
        "env_name",
        "target_name",
        "candidate",
        "task_identity_source",
        "split_receipt",
        "reference_receipt",
        "reward_receipt",
        "wall_time_receipt",
    ]
    for index, row in enumerate(rows):
        parts.append(
            f'<details id="receipt-{index}"><summary>{esc(row["arm"])} '
            f"| {esc(row['task_id'])}</summary><dl>"
        )
        for key in receipt_keys:
            value = row[key]
            rendered = esc(value)
            if value != ABSENT and Path(value).is_absolute():
                rendered = f'<a href="{esc(Path(value).as_uri())}">{rendered}</a>'
            parts.append(f"<dt>{esc(labels[key])}</dt><dd>{rendered}</dd>")
        for key in ["failed_layer", "failure_reason"]:
            if row[key] != ABSENT:
                parts.append(f"<dt>{esc(labels[key])}</dt><dd>{esc(row[key])}</dd>")
        parts.append("</dl></details>")
    parts.append("<footer>")
    parts.append("<p>Input directories:</p><ul>")
    for directory in input_dirs:
        parts.append(f"<li><code>{esc(str(directory.resolve()))}</code></li>")
    parts.append("</ul>")
    parts.append(f"<p>CSV output: <code>{esc(str(csv_path.resolve()))}</code></p>")
    parts.append(f"<p>HTML output: <code>{esc(str(out_path.resolve()))}</code></p>")
    parts.append(
        "<p>Generated by "
        + esc(str(Path(__file__).resolve()))
        + '. Missing identities, paths and scores render as "absent".</p>'
    )
    parts.append("</footer></main></body></html>")
    out_path.write_text("\n".join(parts) + "\n")


def _parse_arms(values: list[str]) -> dict[str, str]:
    arms: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--arms entry must be name=label, got: {value!r}")
        name, label = value.split("=", 1)
        arms[name] = label
    return arms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Directories scanned recursively for reward_artifact.json",
    )
    parser.add_argument(
        "--arms",
        action="append",
        default=[],
        metavar="NAME=LABEL",
        help="Map an input directory name to an arm label (repeatable)",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        metavar="ARM",
        help="Complete arm-label order; default is input directory order",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = build_rows(
        args.directories,
        arms=_parse_arms(args.arms),
        order=args.order,
    )
    write_csv(rows, args.out_csv)
    write_html(
        rows,
        args.out_html,
        input_dirs=args.directories,
        csv_path=args.out_csv,
    )
    print(f"wrote {len(rows)} rows")
    print(f"csv: {args.out_csv.resolve()}")
    print(f"html: {args.out_html.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

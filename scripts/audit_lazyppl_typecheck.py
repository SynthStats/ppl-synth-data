#!/usr/bin/env python3
"""Audit LazyPPL JSONL records: typecheck rate, conformance, per-family breakdown.

Reads one or more JSONL files, validates each record via Docker-based GHC typecheck,
and produces a markdown audit report.

Usage:
    uv run python scripts/audit_lazyppl_typecheck.py \
        generated/lazyppl_synthesis/inputs/train_translated.jsonl \
        generated/lazyppl_synthesis/inputs/native_train.jsonl \
        --report reports/lazyppl_synthesis/audit_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ppl_synth_data.runtimes.lazyppl_runtime import LazyPPLRuntime

log = logging.getLogger(__name__)
rt = LazyPPLRuntime()


def _coerce_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            records.append(record)
    return records


def _load_sample_data(path: Path, n_records: int) -> list[dict[str, Any] | None]:
    """Load per-record sample data aligned positionally with the audited records.

    The translated records carry no stable id, so alignment is by position: the
    file must hold exactly ``n_records`` JSONL entries in the same order as the
    concatenated input files. Each entry is either the field-values dict itself
    ({"trials": [10, 12], "successes": [3, 4]}) or a wrapper {"sample_data": {...}}.
    An empty entry (null or {}) leaves that record on the schema-guessing path.
    """
    entries: list[dict[str, Any] | None] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
            if entry is None or entry == {}:
                entries.append(None)
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"expected JSON object or null in {path}:{line_number}")
            sample = entry.get("sample_data", entry)
            if not isinstance(sample, dict):
                raise ValueError(f"sample_data must be an object in {path}:{line_number}")
            entries.append(sample or None)
    if len(entries) != n_records:
        raise ValueError(
            f"sample-data file {path} has {len(entries)} entries but {n_records} records "
            "were loaded; positional alignment requires one entry per record "
            "(use null or {} for records without sample data)"
        )
    return entries


def audit_record(
    record: dict[str, Any],
    sample_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completion = _coerce_text(record.get("completion"))
    code = rt.extract_code(completion)
    meta = record.get("metadata")
    if not isinstance(meta, dict):
        meta = {}

    info: dict[str, Any] = {
        "id": meta.get("original_id", meta.get("id", "?")),
        "source": meta.get("source", "unknown"),
        "family": meta.get("family", meta.get("generative_family", "unknown")),
        "has_code": code is not None,
        "has_logLikRow": "logLikRow" in completion,
        "has_module_submission": "module Submission" in completion,
        "has_model": "model ::" in completion,
        "meta_typecheck_ok": meta.get("typecheck_ok"),
        "used_sample_data": bool(sample_data),
    }

    if code is not None:
        # With real sample data the harness builds a schema-correct TaskData module;
        # without it, validate_syntax guesses the Row schema and undercounts.
        if sample_data is not None:
            result = rt.validate_static(code, sample_data=sample_data)
        else:
            result = rt.validate_syntax(code)
        info["typecheck_ok"] = result.ok
        info["typecheck_msg"] = result.message if not result.ok else ""
    else:
        info["typecheck_ok"] = False
        info["typecheck_msg"] = "no code extracted"

    return info


def categorize_error(msg: str) -> str:
    lower = msg.lower()
    if "could not find module" in lower:
        return "missing_module"
    if "not in scope" in lower:
        return "not_in_scope"
    if "couldn't match type" in lower or "couldn't match expected type" in lower:
        return "type_mismatch"
    if "parse error" in lower:
        return "parse_error"
    if "ambiguous" in lower:
        return "ambiguous"
    if "no instance" in lower:
        return "missing_instance"
    if "ghc" in lower and "not found" in lower:
        return "ghc_unavailable"
    if "docker" in lower:
        return "docker_error"
    return "other"


def generate_report(results: list[dict], output: Path | None) -> str:
    total = len(results)
    passing = sum(1 for r in results if r["typecheck_ok"])
    conforming = sum(
        1 for r in results if r["has_logLikRow"] and r["has_module_submission"] and r[(
                                                                                          'has_model'
                                                                                      )]
    )
    with_sample = sum(1 for r in results if r.get("used_sample_data"))

    if with_sample == total and total:
        mode = "sample-data-informed (schema-correct TaskData for every record)"
    elif with_sample:
        mode = (
            f"mixed: {with_sample}/{total} sample-data-informed, "
            f"{total - with_sample}/{total} schema-guessed"
        )
    else:
        mode = "schema-guessed (no sample data; Row schema inferred heuristically)"

    lines = [
        "# LazyPPL Typecheck Audit Report\n",
        f"**Total records**: {total}",
        (
            f"**Typecheck pass**: {passing}/{total} ({100 * passing / total:.1f}%)"
            if total
            else "**No records**"
        ),
        f"**Conforming** (logLikRow + module Submission + model): {conforming}/{total}",
        f"**Validation mode**: {mode}",
        "",
    ]

    # per-source breakdown
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)

    lines.append("## By source\n")
    lines.append("| Source | Total | Pass | Rate |")
    lines.append("|--------|-------|------|------|")
    for src, recs in sorted(by_source.items()):
        n = len(recs)
        p = sum(1 for r in recs if r["typecheck_ok"])
        lines.append(f"| {src} | {n} | {p} | {100 * p / n:.0f}% |")

    # per-family breakdown
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_family[r["family"]].append(r)

    lines.append("\n## By family\n")
    lines.append("| Family | Total | Pass | Rate |")
    lines.append("|--------|-------|------|------|")
    for fam, recs in sorted(by_family.items()):
        n = len(recs)
        p = sum(1 for r in recs if r["typecheck_ok"])
        lines.append(f"| {fam} | {n} | {p} | {100 * p / n:.0f}% |")

    # error categories
    errors = [r for r in results if not r["typecheck_ok"]]
    if errors:
        categories: Counter[str] = Counter()
        for r in errors:
            categories[categorize_error(r["typecheck_msg"])] += 1

        lines.append("\n## Error categories\n")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, cnt in categories.most_common():
            lines.append(f"| {cat} | {cnt} |")

        lines.append("\n## Failed records\n")
        for r in errors[:20]:
            msg = r["typecheck_msg"][:120].replace("\n", " ")
            lines.append(f"- **{r['id']}** ({r['source']}/{r['family']}): {msg}")
        if len(errors) > 20:
            lines.append(f"\n... and {len(errors) - 20} more failures")

    report = "\n".join(lines)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        log.info("Report written to %s", output)

    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Audit LazyPPL JSONL records")
    parser.add_argument("files", nargs="+", type=Path, help="JSONL files to audit")
    parser.add_argument("--report", type=Path, default=None, help="Output markdown report path")
    parser.add_argument(
        "--sample-data",
        type=Path,
        default=None,
        help=(
            "JSONL file of per-record sample data, aligned positionally with the audited "
            "records (one entry per record; null or {} for records without data). Each "
            'entry is a {field: [values]} dict or a {"sample_data": {...}} wrapper. When '
            "provided, records with data are type-checked against a schema-correct TaskData "
            "module instead of the heuristic Row-schema guess."
        ),
    )
    args = parser.parse_args()

    all_records = []
    for path in args.files:
        if not path.exists():
            log.warning("File not found: %s (skipping)", path)
            continue
        all_records.extend(_load_records(path))

    if not all_records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    if args.sample_data is not None:
        if not args.sample_data.exists():
            raise FileNotFoundError(f"sample-data file not found: {args.sample_data}")
        sample_data_per_record = _load_sample_data(args.sample_data, len(all_records))
    else:
        sample_data_per_record = [None] * len(all_records)

    log.info("Auditing %d records...", len(all_records))
    results = []
    for i, record in enumerate(all_records):
        info = audit_record(record, sample_data=sample_data_per_record[i])
        status = "OK" if info["typecheck_ok"] else "FAIL"
        print(
            f"[{i + 1}/{len(all_records)}] "
            f"{info['id']} ({info['source']}/{info['family']}): {status}"
        )
        results.append(info)

    report = generate_report(results, args.report)
    print("\n" + report)


if __name__ == "__main__":
    main()

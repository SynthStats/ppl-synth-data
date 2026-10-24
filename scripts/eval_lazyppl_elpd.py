#!/usr/bin/env python3
"""Batch ELPD-LOO evaluation for LazyPPL synthesis records.

Runs each typecheck-passing record through the lazyppl-harbor Docker container,
collects ELPD-LOO scores from the in-container score.py, and writes an aggregate
results JSON.

Usage:
    uv run python scripts/eval_lazyppl_elpd.py
    uv run python scripts/eval_lazyppl_elpd.py \
        --input data/lazyppl_synthesis/train.jsonl --workers 4
    uv run python scripts/eval_lazyppl_elpd.py \
        --limit 10 --output generated/lazyppl_synthesis/elpd_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import statistics
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ppl_synth_data.harbor.taskdata_gen import (
    generate_hidden_taskdata_hs,
    generate_taskdata_hs,
    parse_seed_row_fields,
)
from ppl_synth_data.pipeline.io import atomic_write_text
from ppl_synth_data.runtimes import get_runtime
from ppl_synth_data.runtimes.lazyppl_runtime import static_safety_error

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_INPUT = REPO_ROOT / "data" / "lazyppl_synthesis" / "train.jsonl"
_DEFAULT_OUTPUT = REPO_ROOT / "generated" / "lazyppl_synthesis" / "elpd_results.json"

_DEFAULT_HARBOR_IMAGE = os.environ.get("LAZYPPL_HARBOR_IMAGE", "lazyppl-harbor:latest")

_MH_ENV = {
    # Vehtari 2024: S >= 2200 for full 0.7 Pareto k threshold
    "LAZYPPL_BURNIN": os.environ.get("LAZYPPL_BURNIN", "2000"),
    "LAZYPPL_THINNING": os.environ.get("LAZYPPL_THINNING", "50"),
    "LAZYPPL_NDRAWS": os.environ.get("LAZYPPL_NDRAWS", "2200"),
    "LAZYPPL_SEED": os.environ.get("LAZYPPL_SEED", "42"),
}


def _docker_workspace_parent() -> Path | None:
    home = Path.home()
    return home if str(home).startswith("/Users") else None


def _harbor_env_flags() -> list[str]:
    env_flags: list[str] = []
    for key, value in _MH_ENV.items():
        env_flags.extend(["-e", f"{key}={value}"])
    return env_flags


def _build_harbor_command(
    *,
    submission_path: Path,
    taskdata_path: Path,
    hidden_taskdata_path: Path,
    harbor_image: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--memory=1g",
        "--cpus=2",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=100",
        *_harbor_env_flags(),
        "-v",
        f"{submission_path}:/tests/Submission.hs:ro",
        "-v",
        f"{taskdata_path}:/tests/TaskData.hs:ro",
        "-v",
        f"{hidden_taskdata_path}:/tests/HiddenTaskData.hs:ro",
        harbor_image,
        "bash",
        "-c",
        (
            # Same isolation as validation.lazyppl_semantics: the hidden module
            # becomes visible only after Submission.hs type-checks without it,
            # and only to the trusted Main.hs compile.
            "set -euo pipefail; "
            "cp /tests/TaskData.hs /app/TaskData.hs; "
            "cp /tests/Submission.hs /app/Submission.hs; "
            "ghc -fno-code -package-env /etc/ghc-pkg-env -i/app /app/Submission.hs >&2 && "
            "mkdir -p /app/hidden && "
            "cp /tests/HiddenTaskData.hs /app/hidden/HiddenTaskData.hs && "
            "ghc --make -O0 -package-env /etc/ghc-pkg-env "
            "-i/app -i/app/hidden /app/Main.hs -o /tmp/run-task >&2 && "
            "/tmp/run-task > /tmp/log_lik.json 2>&1 && "
            "python3 /app/score.py /tmp/log_lik.json >&2 && "
            "cat /tmp/reward.json"
        ),
    ]


def _parse_reward_payload(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(stdout), None
    except (json.JSONDecodeError, IndexError):
        return None, f"failed to parse reward JSON from stdout: {stdout[:200]}"


def _write_output_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _metric_or_top_level(reward_data: dict[str, Any], key: str) -> Any:
    metrics = reward_data.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    return reward_data.get(key)


def _classify_family(record: dict) -> str:
    meta = record.get("metadata", {})
    family = meta.get("family")
    if family and family != "NONE":
        return family
    source = meta.get("source", "")
    if source == "translated_from_pymc":
        return "translated"
    if source == "native_lazyppl":
        kc = meta.get("key_concepts", [])
        return kc[0] if kc else "native_unknown"
    return "unknown"


# regex to extract numeric ranges like "15-35", "from 0.1 to 1.5", "between 10 and 100"
_NUM_RANGE_RE = re.compile(
    r"(?:rang[ei]|between|from)\s+(?:of\s+)?"
    r"([-+]?\d+(?:\.\d+)?)\s*(?:to|and|[-\u2013])\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_numeric_range(prompt: str) -> tuple[float, float]:
    """Extract a plausible numeric range from the prompt text."""
    matches = _NUM_RANGE_RE.findall(prompt)
    if matches:
        lo, hi = float(matches[0][0]), float(matches[0][1])
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)
    return (-5.0, 5.0)


_N_OBS = 50  # Vehtari: n_obs >= 50 for reliable LOO SE


def _build_taskdata_realistic(
    code: str,
    prompt: str,
    *,
    n_obs: int = _N_OBS,
    seed: int = 42,
) -> str:
    """Parses Row field names from the code and infers plausible value ranges
    from the prompt text. Generates *n_obs* rows of synthetic observations.
    """
    rng = random.Random(seed)
    columns = parse_seed_row_fields(code)
    if not columns:
        columns = [{"name": "_stub", "hs_type": "Double"}]

    lo, hi = _parse_numeric_range(prompt)

    rows: list[dict[str, float | int | bool]] = []
    for _ in range(n_obs):
        row: dict[str, float | int | bool] = {}
        for col in columns:
            if col["hs_type"] == "Bool":
                row[col["name"]] = rng.random() < 0.5
            elif col["hs_type"] == "Int":
                row[col["name"]] = rng.randint(0, max(1, int(hi)))
            else:  # Double
                row[col["name"]] = round(rng.uniform(lo, hi), 4)
        rows.append(row)

    return generate_taskdata_hs(columns, rows)


def _new_result(record_id: str, family: str) -> dict[str, Any]:
    return {
        "id": record_id,
        "family": family,
        "elpd_loo": None,
        "pareto_k_max": None,
        "reward": None,
        "compile_ok": False,
        "error": None,
        "holdout_error": None,
        "n_draws": None,
        "n_obs": None,
        "pareto_k_warning": None,
    }


def evaluate_one(
    record: dict[str, Any],
    record_id: str,
    harbor_image: str,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run one record through the Docker harbor chain and return the ELPD result."""
    runtime = get_runtime("lazyppl")
    result = _new_result(record_id, _classify_family(record))

    code = runtime.extract_code(record.get("completion", ""))
    if code is None:
        result["error"] = "no code block found in completion"
        return result
    static_error = static_safety_error(code)
    if static_error is not None:
        result["error"] = f"static safety filter: {static_error}"
        return result

    prompt = record.get("prompt", "")
    taskdata_hs = _build_taskdata_realistic(code, prompt)
    hidden_taskdata_hs = generate_hidden_taskdata_hs(parse_seed_row_fields(code), [])

    with tempfile.TemporaryDirectory(dir=_docker_workspace_parent()) as td:
        workspace = Path(td)
        submission_path = workspace / "Submission.hs"
        taskdata_path = workspace / "TaskData.hs"
        hidden_taskdata_path = workspace / "HiddenTaskData.hs"

        submission_path.write_text(code, encoding="utf-8")
        taskdata_path.write_text(taskdata_hs, encoding="utf-8")
        hidden_taskdata_path.write_text(hidden_taskdata_hs, encoding="utf-8")
        cmd = _build_harbor_command(
            submission_path=submission_path,
            taskdata_path=taskdata_path,
            hidden_taskdata_path=hidden_taskdata_path,
            harbor_image=harbor_image,
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result["error"] = f"docker timed out ({timeout}s)"
            return result

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            # detect infrastructure vs code errors
            lower = stderr.lower()
            if any(
                pat in lower
                for pat in (
                    "unable to find image",
                    "no such image",
                    "cannot connect to the docker daemon",
                    "is the docker daemon running",
                )
            ):
                result["error"] = f"docker infrastructure: {stderr[:300]}"
            else:
                result["error"] = stderr[:500] if stderr else f"exit code {proc.returncode}"
            return result

        reward_data, parse_error = _parse_reward_payload(proc.stdout.strip())
        if parse_error is not None or reward_data is None:
            result["error"] = parse_error
            return result

        result["compile_ok"] = _metric_or_top_level(reward_data, "compile_ok") == 1.0
        result["elpd_loo"] = _finite_or_none(_metric_or_top_level(reward_data, "elpd_loo"))
        result["pareto_k_max"] = _finite_or_none(_metric_or_top_level(reward_data, "pareto_k_max"))
        result["reward"] = _finite_or_none(reward_data.get("reward"))
        result["n_draws"] = _metric_or_top_level(reward_data, "n_draws")
        result["n_obs"] = _metric_or_top_level(reward_data, "n_obs")
        diagnostics = reward_data.get("diagnostics")
        result["pareto_k_warning"] = (
            diagnostics.get("pareto_k_warning")
            if isinstance(diagnostics, dict) and "pareto_k_warning" in diagnostics
            else reward_data.get("pareto_k_warning")
        )
        result["holdout_error"] = (
            diagnostics.get("holdout_error")
            if isinstance(diagnostics, dict) and "holdout_error" in diagnostics
            else reward_data.get("holdout_error")
        )

        if reward_data.get("error"):
            result["error"] = reward_data["error"]

    return result


def _load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _aggregate(results: list[dict]) -> dict[str, Any]:
    """Build summary statistics from individual evaluation results."""
    total = len(results)
    compile_ok_count = sum(1 for r in results if r["compile_ok"])
    holdout_error_count = sum(1 for r in results if r.get("holdout_error"))
    elpd_values = [r["elpd_loo"] for r in results if r["elpd_loo"] is not None]

    summary: dict[str, Any] = {
        "total": total,
        "evaluated": total,
        "compile_ok": compile_ok_count,
        "holdout_errors": holdout_error_count,
        "elpd_available": len(elpd_values),
        "mean_elpd": round(statistics.mean(elpd_values), 4) if elpd_values else None,
        "median_elpd": round(statistics.median(elpd_values), 4) if elpd_values else None,
    }

    # per-family breakdown
    by_family: dict[str, dict[str, Any]] = {}
    for r in results:
        fam = r["family"]
        if fam not in by_family:
            by_family[fam] = {
                "count": 0,
                "compile_ok": 0,
                "holdout_errors": 0,
                "elpd_values": [],
            }
        by_family[fam]["count"] += 1
        if r["compile_ok"]:
            by_family[fam]["compile_ok"] += 1
        if r.get("holdout_error"):
            by_family[fam]["holdout_errors"] += 1
        if r["elpd_loo"] is not None:
            by_family[fam]["elpd_values"].append(r["elpd_loo"])

    by_family_clean: dict[str, dict[str, Any]] = {}
    for fam, info in sorted(by_family.items()):
        vals = info["elpd_values"]
        by_family_clean[fam] = {
            "count": info["count"],
            "compile_ok": info["compile_ok"],
            "holdout_errors": info["holdout_errors"],
            "elpd_available": len(vals),
            "mean_elpd": round(statistics.mean(vals), 4) if vals else None,
        }

    return {"summary": summary, "by_family": by_family_clean}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ELPD-LOO evaluation for LazyPPL synthesis records.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="JSONL file with LazyPPL records (default: data/lazyppl_synthesis/train.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Output JSON file for results (default: generated/lazyppl_synthesis/elpd_results.json)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel Docker containers (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N records (for testing)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per Docker run in seconds (default: 600)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=_DEFAULT_HARBOR_IMAGE,
        help="Docker image name (default: lazyppl-harbor:latest or LAZYPPL_HARBOR_IMAGE env)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input.exists():
        log.error("input file not found: %s", args.input)
        raise SystemExit(1)

    records = _load_records(args.input)
    log.info("loaded %d records from %s", len(records), args.input)

    eligible = [
        (i, r) for i, r in enumerate(records) if r.get("metadata", {}).get("typecheck_ok", False)
    ]
    log.info("%d / %d records have typecheck_ok=True", len(eligible), len(records))

    if args.limit is not None:
        eligible = eligible[: args.limit]
        log.info("limiting to first %d records", args.limit)

    def record_id(idx: int, rec: dict) -> str:
        oid = rec.get("metadata", {}).get("original_id")
        if oid:
            return oid
        return f"record#{idx}"

    results: list[dict] = []
    n_total = len(eligible)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                evaluate_one,
                rec,
                record_id(idx, rec),
                args.image,
                args.timeout,
            ): (idx, record_id(idx, rec), _classify_family(rec))
            for idx, rec in eligible
        }

        for i, future in enumerate(as_completed(futures), 1):
            idx, rid, family = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _new_result(rid, family)
                result["error"] = f"worker exception: {exc}"
            results.append(result)

            status = "ok" if result["compile_ok"] else "FAIL"
            elpd_str = (
                f"elpd={result['elpd_loo']:.2f}" if result["elpd_loo"] is not None else "no elpd"
            )
            message = result.get("error") or result.get("holdout_error")
            err_str = f" [{message[:60]}]" if message else ""
            log.info(
                "[%d/%d] %s %s %s%s",
                i,
                n_total,
                result["id"],
                status,
                elpd_str,
                err_str,
            )

    # sort results by id for stable output
    results.sort(key=lambda r: r["id"])

    agg = _aggregate(results)
    output = {
        "summary": agg["summary"],
        "by_family": agg["by_family"],
        "records": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_output_json(args.output, output)
    log.info("results written to %s", args.output)

    # print summary table
    s = agg["summary"]
    print(f"\n{'=' * 60}")
    print("ELPD-LOO Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"  Total records:    {s['total']}")
    print(f"  Compile OK:       {s['compile_ok']}")
    print(f"  Holdout errors:   {s['holdout_errors']}")
    print(f"  ELPD available:   {s['elpd_available']}")
    print(f"  Mean ELPD:        {s['mean_elpd']}")
    print(f"  Median ELPD:      {s['median_elpd']}")
    print()
    print(f"  {'Family':<20} {'Count':>6} {'Compile':>8} {'ELPD':>6} {'Mean ELPD':>12}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 12}")
    for fam, info in sorted(agg["by_family"].items()):
        mean_str = f"{info['mean_elpd']:.2f}" if info["mean_elpd"] is not None else "n/a"
        print(
            f"  {fam:<20} {info['count']:>6} {info['compile_ok']:>8} "
            f"{info['elpd_available']:>6} {mean_str:>12}"
        )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

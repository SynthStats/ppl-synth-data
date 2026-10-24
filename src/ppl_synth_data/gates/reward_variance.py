"""Reward-variance gate over the SFT bootstrap rows.

The four stages persist independent artifacts so policy sampling and verifier
scoring can run on different machines and resume without repeating completed
work.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import statistics
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ppl_synth_data.agentic.domain import TRAINING_LOG_REWARD_FLOOR

DATASET = "SynthStats/ppl-synthesis-sft-bootstrap"
VISIBLE_MARKERS = ("Observed data JSON:", "Observed row fields JSON:")
RUNTIMES = ("pymc", "lazyppl")
_SAMPLE_MANIFEST_FIELDS = (
    "dataset",
    "revision",
    "model",
    "base_url_host",
    "k",
    "temperature",
    "max_tokens",
    "thinking",
    "system_prompt_sha256",
)
_SCORE_MANIFEST_DRIFT_FIELDS = (
    "data_shape",
    "pymc_draws",
    "pymc_tune",
    "pymc_timeout_sec",
    "lazyppl_timeout_sec",
    "lazyppl_harbor_image",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _required_jsonl(out_dir: Path, filename: str) -> list[dict[str, Any]]:
    path = out_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"missing required input {path}")
    return read_jsonl(path)


def visible_data_from_prompt(prompt: str) -> dict[str, Any]:
    """Return the observed-data JSON object embedded in an SFT prompt."""
    decoder = json.JSONDecoder()
    for marker in VISIBLE_MARKERS:
        start = prompt.find(marker)
        if start < 0:
            continue
        brace = prompt.find("{", start)
        if brace < 0:
            raise ValueError(f"marker {marker!r} present but no JSON object follows")
        obj, _ = decoder.raw_decode(prompt[brace:])
        if not isinstance(obj, dict):
            raise ValueError("visible data is not a JSON object")
        return obj
    raise ValueError(f"prompt carries none of {VISIBLE_MARKERS}: {prompt[:120]!r}")


def strip_null_columns(data: dict[str, Any]) -> dict[str, Any]:
    """Drop the null-padded columns the datasets loader adds to a union schema."""
    return {key: value for key, value in data.items() if value is not None}


def stage_pull(*, out_dir: Path, revision: str) -> None:
    from datasets import load_dataset

    rows_path = out_dir / "rows.jsonl"
    if rows_path.exists():
        raise FileExistsError(f"{rows_path} exists; delete it to re-pull")

    heldout: dict[str, dict[str, Any]] = {}
    for runtime in RUNTIMES:
        config = f"{runtime}_heldout"
        for record in load_dataset(DATASET, config, revision=revision)["heldout"]:
            heldout[record["row_id"]] = strip_null_columns(record["heldout_data"])

    rows: list[dict[str, Any]] = []
    for runtime in RUNTIMES:
        dataset = load_dataset(DATASET, runtime, revision=revision)
        for split in dataset.keys():
            for record in dataset[split]:
                meta = record["metadata"]
                row_id = meta["row_id"]
                if meta.get("hidden_data_sha256") is not None and row_id not in heldout:
                    raise ValueError(
                        f"row {row_id!r} declares hidden_data_sha256 but has no held-out record"
                    )
                rows.append(
                    {
                        "row_id": row_id,
                        "runtime": runtime,
                        "split": split,
                        "data_source": meta["data_source"],
                        "structural_class": meta.get("structural_class"),
                        "prompt": record["prompt"],
                        "visible_data": visible_data_from_prompt(record["prompt"]),
                        "heldout_data": heldout.get(row_id),
                        "dataset": DATASET,
                        "revision": revision,
                    }
                )
    append_jsonl(rows_path, rows)
    scorable = sum(1 for row in rows if row["heldout_data"] is not None)
    print(f"pulled {len(rows)} rows at {revision[:8]}; {scorable} carry held-out data")


@dataclass(frozen=True)
class Completion:
    row_id: str
    sample_idx: int
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


def _single_row_value(rows: list[dict[str, Any]], field: str) -> Any:
    values = {row.get(field) for row in rows}
    if len(values) != 1 or None in values:
        raise ValueError(f"scorable rows must carry one non-null {field!r}")
    return values.pop()


def sample_manifest(
    *,
    rows: list[dict[str, Any]],
    model: str,
    base_url: str,
    k: int,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    system_prompt: str | None,
) -> dict[str, Any]:
    host = urlparse(base_url).hostname
    if host is None:
        raise ValueError(f"base_url has no host: {base_url!r}")
    prompt_hash = (
        hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        if system_prompt is not None
        else None
    )
    return {
        "dataset": _single_row_value(rows, "dataset"),
        "revision": _single_row_value(rows, "revision"),
        "model": model,
        "base_url_host": host,
        "k": k,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "system_prompt_sha256": prompt_hash,
    }


def ensure_sample_manifest(out_dir: Path, expected: dict[str, Any]) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        _write_json(manifest_path, expected)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest is not a JSON object: {manifest_path}")
    for field in _SAMPLE_MANIFEST_FIELDS:
        if manifest.get(field) != expected[field]:
            raise ValueError(
                f"manifest field {field!r} differs: "
                f"existing={manifest.get(field)!r}, requested={expected[field]!r}"
            )


def missing_sample_requests(
    rows: list[dict[str, Any]],
    completions: list[dict[str, Any]],
    *,
    k: int,
) -> list[tuple[dict[str, Any], tuple[int, ...]]]:
    present = {(item["row_id"], int(item["sample_idx"])) for item in completions}
    requests: list[tuple[dict[str, Any], tuple[int, ...]]] = []
    for row in rows:
        missing = tuple(idx for idx in range(k) if (row["row_id"], idx) not in present)
        if missing:
            requests.append((row, missing))
    return requests


async def _sample_row(
    client: Any,
    *,
    model: str,
    row: dict[str, Any],
    sample_indices: tuple[int, ...],
    temperature: float,
    max_tokens: int,
    thinking: bool,
    system_prompt: str | None,
    semaphore: asyncio.Semaphore,
) -> list[Completion]:
    extra_body = {} if thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": row["prompt"]})
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            n=len(sample_indices),
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
    choices = sorted(response.choices, key=lambda choice: choice.index)
    if len(choices) != len(sample_indices):
        raise RuntimeError(
            f"policy returned {len(choices)} choices for {len(sample_indices)} requested samples"
        )
    usage = response.usage
    completions: list[Completion] = []
    for sample_idx, choice in zip(sample_indices, choices, strict=True):
        content = choice.message.content
        if content is None or content == "":
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            raise TypeError(f"completion content is not text: {type(content).__name__}")
        completions.append(
            Completion(
                row_id=row["row_id"],
                sample_idx=sample_idx,
                text=text,
                finish_reason=choice.finish_reason,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        )
    return completions


async def _sample_all(
    *,
    requests: list[tuple[dict[str, Any], tuple[int, ...]]],
    out_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    thinking: bool,
    system_prompt: str | None,
) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=900.0, max_retries=3)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            _sample_row(
                client,
                model=model,
                row=row,
                sample_indices=sample_indices,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking,
                system_prompt=system_prompt,
                semaphore=semaphore,
            )
        )
        for row, sample_indices in requests
    ]
    done = 0
    started = time.time()
    for coroutine in asyncio.as_completed(tasks):
        completions = await coroutine
        append_jsonl(out_path, (asdict(completion) for completion in completions))
        done += 1
        if done % 25 == 0 or done == len(tasks):
            elapsed = time.time() - started
            print(f"sampled {done}/{len(tasks)} rows in {elapsed:.0f}s", flush=True)


def stage_sample(
    *,
    out_dir: Path,
    k: int,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    model: str,
    base_url: str,
    thinking: bool,
    system_prompt_file: Path | None,
) -> None:
    system_prompt = system_prompt_file.read_text(encoding="utf-8") if system_prompt_file else None
    rows = [row for row in read_jsonl(out_dir / "rows.jsonl") if row["heldout_data"] is not None]
    if not rows:
        raise RuntimeError("no scorable rows; run `pull` first")

    ensure_sample_manifest(
        out_dir,
        sample_manifest(
            rows=rows,
            model=model,
            base_url=base_url,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            system_prompt=system_prompt,
        ),
    )
    completions_path = out_dir / "completions.jsonl"
    completions = read_jsonl(completions_path)
    requests = missing_sample_requests(rows, completions, k=k)
    present = k * len(rows) - sum(len(indices) for _, indices in requests)
    missing = sum(len(indices) for _, indices in requests)
    print(
        f"{len(rows)} scorable rows, {present} samples present, "
        f"{missing} samples to request (k={k})"
    )
    if not requests:
        return

    api_key = os.environ.get("GATE_POLICY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GATE_POLICY_API_KEY is not set; refusing to sample without the real endpoint"
        )
    asyncio.run(
        _sample_all(
            requests=requests,
            out_path=completions_path,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency=concurrency,
            thinking=thinking,
            system_prompt=system_prompt,
        )
    )


@dataclass(frozen=True)
class VerifierSettings:
    data_shape: str
    pymc_draws: int
    pymc_tune: int
    pymc_timeout_sec: int
    lazyppl_timeout_sec: int
    lazyppl_harbor_image: str


@dataclass(frozen=True)
class Score:
    row_id: str
    sample_idx: int
    runtime: str
    data_shape: str
    training_log_reward: float
    metric_value: float | None
    reward_source: str
    semantic_status: str
    reason: str
    wall_sec: float
    diagnostic: str | None = None  # visible-phase exception text; never a reason or feedback
    screens: dict[str, float | None] | None = (
        None  # LazyPPL soundness-screen numbers (gates/likelihood_screens.py)
    )


# The numbers the LazyPPL scorer reports for the soundness screens (score.py,
# likelihood_screen_metrics).
_SCREEN_KEYS = (
    "screen_lpd_sd",
    "screen_floor_entry_share",
    "screen_floor_row_share",
    "screen_max_log_lik",
)


# The harness's visible-phase failure messages. A held-out-phase failure never fills
# diagnostic, because
# its exception text could print held-out values.
_VISIBLE_PHASE_FAILURES = (
    "exec failed",
    "model(visible_data) raised",
    "posterior sampling failed",
    "training log-likelihood scoring failed",
    "sampler diagnostics failed",
)


def _diagnostic(result: Any) -> str | None:
    details = getattr(result, "details", None) or {}
    text = details.get("exception")
    if not text or not str(result.message).startswith(_VISIBLE_PHASE_FAILURES):
        return None
    return str(text)[:300]


def _pymc_scorer_defaults() -> tuple[int, int, int]:
    """The PyMC scorer's own draws, tune and timeout defaults, read without ``Settings()``."""
    from ppl_synth_data.validation.pymc_semantics import score_pymc_semantics

    pymc = inspect.signature(score_pymc_semantics).parameters
    return (
        int(pymc["draws"].default),
        int(pymc["tune"].default),
        int(pymc["timeout_sec"].default),
    )


def verifier_settings() -> VerifierSettings:
    from ppl_synth_data.config import Settings
    from ppl_synth_data.validation.lazyppl_semantics import score_lazyppl_semantics

    pymc_draws, pymc_tune, pymc_timeout_sec = _pymc_scorer_defaults()
    lazyppl = inspect.signature(score_lazyppl_semantics).parameters
    return VerifierSettings(
        data_shape="lists",
        pymc_draws=pymc_draws,
        pymc_tune=pymc_tune,
        pymc_timeout_sec=pymc_timeout_sec,
        lazyppl_timeout_sec=int(lazyppl["timeout_sec"].default),
        lazyppl_harbor_image=Settings().lazyppl_harbor_image,
    )


def _record_score_manifest(out_dir: Path, settings: VerifierSettings) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing sample manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest is not a JSON object: {manifest_path}")
    expected = asdict(settings)
    recorded = manifest.get("score")
    if read_jsonl(out_dir / "scores.jsonl"):
        for field in _SCORE_MANIFEST_DRIFT_FIELDS:
            existing = recorded.get(field) if isinstance(recorded, dict) else None
            if existing != expected[field]:
                raise ValueError(
                    f"score manifest field {field!r} differs: "
                    f"existing={existing!r}, requested={expected[field]!r}"
                )
    manifest["score"] = expected
    _write_json(manifest_path, manifest)


def to_log_reward(metric_name: str | None, metric_value: float | None) -> float:
    """Floor anything that is not a finite held-out average log density."""
    if metric_value is None or metric_name != "avg_holdout_logp" or not math.isfinite(metric_value):
        return TRAINING_LOG_REWARD_FLOOR
    return max(TRAINING_LOG_REWARD_FLOOR, float(metric_value))


def _floor_score(
    row: dict[str, Any],
    completion: dict[str, Any],
    *,
    started: float,
    data_shape: str,
    reason: str,
    metric_value: float | None = None,
    semantic_status: str = "critical",
    diagnostic: str | None = None,
) -> Score:
    return Score(
        row_id=row["row_id"],
        sample_idx=completion["sample_idx"],
        runtime=row["runtime"],
        data_shape=data_shape,
        training_log_reward=TRAINING_LOG_REWARD_FLOOR,
        metric_value=metric_value,
        reward_source="floor",
        semantic_status=semantic_status,
        reason=reason,
        wall_sec=time.time() - started,
        diagnostic=diagnostic,
    )


def _pymc_data_shape(
    data: dict[str, Any] | None,
    *,
    data_shape: str,
) -> dict[str, Any] | None:
    if data is None or data_shape == "lists":
        return data
    if data_shape != "arrays":
        raise ValueError(f"unknown PyMC data shape {data_shape!r}")

    import numpy as np

    return {
        key: np.asarray(value) if isinstance(value, list) else value for key, value in data.items()
    }


def score_one(
    row: dict[str, Any],
    completion: dict[str, Any],
    settings: VerifierSettings,
) -> Score:
    started = time.time()
    runtime = row["runtime"]
    if completion["text"] == "":
        return _floor_score(
            row,
            completion,
            started=started,
            data_shape=settings.data_shape,
            reason="empty completion",
        )

    screens: dict[str, float | None] | None = None
    if runtime == "pymc":
        from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime

        code = PyMCRuntime().extract_code(completion["text"])
    elif runtime == "lazyppl":
        from ppl_synth_data.runtimes.lazyppl_runtime import LazyPPLRuntime

        code = LazyPPLRuntime().extract_code(completion["text"])
    else:
        raise ValueError(f"unknown runtime {runtime!r}")

    if code is None:
        return _floor_score(
            row,
            completion,
            started=started,
            data_shape=settings.data_shape,
            reason="no extractable program",
        )

    if runtime == "pymc":
        from ppl_synth_data.validation.pymc_semantics import score_pymc_semantics

        result = score_pymc_semantics(
            code,
            visible_data=_pymc_data_shape(
                row["visible_data"],
                data_shape=settings.data_shape,
            ),
            hidden_data=_pymc_data_shape(
                row["heldout_data"],
                data_shape=settings.data_shape,
            ),
            draws=settings.pymc_draws,
            tune=settings.pymc_tune,
            timeout_sec=settings.pymc_timeout_sec,
        )
        metric_value = float(result.metric_value) if result.metric_value is not None else None
        if not (
            result.ok
            and result.metric_name == "avg_holdout_logp"
            and metric_value is not None
            and math.isfinite(metric_value)
        ):
            return _floor_score(
                row,
                completion,
                started=started,
                data_shape=settings.data_shape,
                reason=result.message[:300],
                metric_value=metric_value,
                semantic_status=str(result.status),
                diagnostic=_diagnostic(result),
            )
        training_log_reward = to_log_reward(result.metric_name, metric_value)
        reward_source = "validation.pymc_elpd"
    else:
        from ppl_synth_data.validation.lazyppl_semantics import score_lazyppl_semantics

        result = score_lazyppl_semantics(
            code,
            prompt=row["prompt"],
            sample_data=row["visible_data"],
            hidden_data=row["heldout_data"],
            harbor_image=settings.lazyppl_harbor_image,
            timeout_sec=settings.lazyppl_timeout_sec,
        )
        metric_value = float(result.metric_value) if result.metric_value is not None else None
        artifact = result.details.get("reward_artifact") if result.details else None
        artifact_reward: float | None = None
        if isinstance(artifact, dict) and artifact.get("training_log_reward") is not None:
            try:
                artifact_reward = float(artifact["training_log_reward"])
            except (TypeError, ValueError):
                artifact_reward = None
        if not (
            result.ok
            and isinstance(artifact, dict)
            and artifact.get("reward_source") == "validation.lazyppl_elpd"
            and artifact_reward is not None
            and math.isfinite(artifact_reward)
        ):
            return _floor_score(
                row,
                completion,
                started=started,
                data_shape=settings.data_shape,
                reason=result.message[:300],
                metric_value=metric_value,
                semantic_status=str(result.status),
            )
        training_log_reward = artifact_reward
        reward_source = str(artifact["reward_source"])
        screens = (
            {key: result.details[key] for key in _SCREEN_KEYS}
            if all(key in result.details for key in _SCREEN_KEYS)
            else None
        )

    return Score(
        row_id=row["row_id"],
        sample_idx=completion["sample_idx"],
        runtime=runtime,
        data_shape=settings.data_shape,
        training_log_reward=training_log_reward,
        metric_value=metric_value,
        reward_source=reward_source,
        semantic_status=str(result.status),
        reason=result.message[:300],
        wall_sec=time.time() - started,
        screens=screens,
    )


def _score_job(
    payload: tuple[dict[str, Any], dict[str, Any], VerifierSettings],
) -> dict[str, Any]:
    row, completion, settings = payload
    return asdict(score_one(row, completion, settings))


def stage_score(
    *,
    out_dir: Path,
    workers: int,
    runtime_filter: str | None,
    settings: VerifierSettings,
) -> None:
    rows = {row["row_id"]: row for row in _required_jsonl(out_dir, "rows.jsonl")}
    completions = _required_jsonl(out_dir, "completions.jsonl")
    if not completions:
        raise RuntimeError("no completions; run `sample` first")
    _record_score_manifest(out_dir, settings)

    scores_path = out_dir / "scores.jsonl"
    already = {(score["row_id"], score["sample_idx"]) for score in read_jsonl(scores_path)}
    jobs = [
        (rows[completion["row_id"]], completion, settings)
        for completion in completions
        if (completion["row_id"], completion["sample_idx"]) not in already
        and (runtime_filter is None or rows[completion["row_id"]]["runtime"] == runtime_filter)
    ]
    print(
        f"{len(completions)} completions, {len(already)} scored, "
        f"{len(jobs)} to score with {workers} workers"
    )
    if not jobs:
        return
    if any(row["runtime"] == "lazyppl" for row, _, _ in jobs):
        _require_lazyppl_image(settings.lazyppl_harbor_image)

    done = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score_job, job) for job in jobs]
        for future in as_completed(futures):
            append_jsonl(scores_path, [future.result()])
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"scored {done}/{len(jobs)} in {time.time() - started:.0f}s", flush=True)


def _require_lazyppl_image(image: str) -> None:
    import subprocess

    if not image:
        raise RuntimeError("Set LAZYPPL_HARBOR_IMAGE to an image built from docker/lazyppl-harbor.")
    proc = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"LazyPPL verifier image {image!r} is not present in the local docker store; "
            "build it from docker/lazyppl-harbor or set LAZYPPL_HARBOR_IMAGE"
        )


@dataclass(frozen=True)
class RowGate:
    row_id: str
    runtime: str
    split: str
    data_source: str
    structural_class: str | None
    rewards: list[float]
    n_floor: int
    variance: float
    spread: float
    verdict: str


def classify(rewards: list[float], *, floor: float, eps: float) -> tuple[str, float, float, int]:
    """Return verdict, population variance, spread, and floor count."""
    if len(rewards) < 2:
        raise ValueError("classify requires at least 2 rewards")
    if not all(math.isfinite(reward) for reward in rewards):
        raise ValueError("rewards must all be finite")
    if not (math.isfinite(eps) and eps >= 0):
        raise ValueError("eps must be a finite non-negative number")

    n_floor = sum(1 for reward in rewards if reward <= floor)
    variance = statistics.pvariance(rewards)
    spread = max(rewards) - min(rewards)
    if n_floor == len(rewards):
        return "DEAD", variance, spread, n_floor
    if spread <= eps:
        return "FLAT", variance, spread, n_floor
    return "LIVE", variance, spread, n_floor


def stage_report(*, out_dir: Path, k: int, eps: float) -> int:
    row_items = _required_jsonl(out_dir, "rows.jsonl")
    _required_jsonl(out_dir, "completions.jsonl")
    score_items = _required_jsonl(out_dir, "scores.jsonl")
    rows = {row["row_id"]: row for row in row_items}

    by_row: dict[str, dict[int, dict[str, Any]]] = {}
    for score in score_items:
        by_row.setdefault(score["row_id"], {})[int(score["sample_idx"])] = score

    scorable = [row for row in rows.values() if row["heldout_data"] is not None]
    gates: list[RowGate] = []
    incomplete: list[str] = []
    for row in sorted(scorable, key=lambda item: item["row_id"]):
        row_id = row["row_id"]
        scores = by_row.get(row_id, {})
        required = [scores.get(sample_idx) for sample_idx in range(k)]
        if any(score is None for score in required):
            incomplete.append(row_id)
            continue
        rewards = [float(score["training_log_reward"]) for score in required if score is not None]
        verdict, variance, spread, n_floor = classify(
            rewards,
            floor=TRAINING_LOG_REWARD_FLOOR,
            eps=eps,
        )
        gates.append(
            RowGate(
                row_id=row_id,
                runtime=row["runtime"],
                split=row["split"],
                data_source=row["data_source"],
                structural_class=row.get("structural_class"),
                rewards=rewards,
                n_floor=n_floor,
                variance=variance,
                spread=spread,
                verdict=verdict,
            )
        )

    _write_jsonl(out_dir / "gate.jsonl", (asdict(gate) for gate in gates))

    unscorable = [row for row in rows.values() if row["heldout_data"] is None]
    lines = [
        f"# Reward-variance gate, k={k}, eps={eps}",
        "",
        f"rows in dataset: {len(rows)}; without held-out data (unscorable by the training reward): "
        f"{len(unscorable)}; scored with all {k} samples: {len(gates)}; "
        f"incomplete: {len(incomplete)}",
        "",
        "| runtime | LIVE | FLAT | DEAD | total |",
        "|---|---|---|---|---|",
    ]
    for runtime in RUNTIMES:
        subset = [gate for gate in gates if gate.runtime == runtime]
        counts = {
            verdict: sum(1 for gate in subset if gate.verdict == verdict)
            for verdict in ("LIVE", "FLAT", "DEAD")
        }
        lines.append(
            f"| {runtime} | {counts['LIVE']} | {counts['FLAT']} | {counts['DEAD']} "
            f"| {len(subset)} |"
        )
    lines += ["", "## By structural class (LIVE / total)", ""]
    classes: dict[str, list[RowGate]] = {}
    for gate in gates:
        classes.setdefault(f"{gate.runtime}:{gate.structural_class}", []).append(gate)
    for key, subset in sorted(classes.items()):
        live = sum(1 for gate in subset if gate.verdict == "LIVE")
        lines.append(f"- {key}: {live} / {len(subset)}")
    lines += ["", "## Unscorable rows by source", ""]
    sources: dict[str, int] = {}
    for row in unscorable:
        key = f"{row['runtime']}:{row['data_source']}"
        sources[key] = sources.get(key, 0) + 1
    for key, count in sorted(sources.items()):
        lines.append(f"- {key}: {count}")
    if incomplete:
        lines += ["", f"## Incomplete rows ({len(incomplete)})", "", ", ".join(incomplete)]

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    keep_path = out_dir / "keep_row_ids.txt"
    if incomplete:
        keep_path.unlink(missing_ok=True)
    else:
        keep = [gate.row_id for gate in gates if gate.verdict == "LIVE"]
        keep_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if incomplete else 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _sample_count(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="stage", required=True)

    pull = sub.add_parser("pull")
    pull.add_argument("--revision", required=True, help="full dataset revision sha")

    sample = sub.add_parser("sample")
    sample.add_argument("--k", type=_sample_count, default=4)
    sample.add_argument("--temperature", type=float, default=0.7)
    sample.add_argument("--max-tokens", type=_positive_int, default=2048)
    sample.add_argument("--concurrency", type=_positive_int, default=16)
    sample.add_argument("--model", required=True)
    sample.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL of the policy endpoint",
    )
    sample.add_argument(
        "--thinking",
        action="store_true",
        help="sample with the chat template's thinking mode on (default: answer-only)",
    )
    sample.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="optional system message sent before every prompt (e.g. the language contract)",
    )

    score = sub.add_parser("score")
    pymc_draws, pymc_tune, pymc_timeout_sec = _pymc_scorer_defaults()
    score.add_argument("--workers", type=_positive_int, default=max(1, (os.cpu_count() or 2) - 2))
    score.add_argument("--runtime", choices=RUNTIMES, default=None)
    score.add_argument("--data-shape", choices=("lists", "arrays"), default="lists")
    score.add_argument("--pymc-draws", type=_positive_int, default=pymc_draws)
    score.add_argument("--pymc-tune", type=_positive_int, default=pymc_tune)
    score.add_argument(
        "--pymc-timeout-sec",
        type=_positive_int,
        default=pymc_timeout_sec,
    )

    report = sub.add_parser("report")
    report.add_argument("--k", type=_sample_count, default=4)
    report.add_argument("--eps", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "pull":
        stage_pull(out_dir=args.out_dir, revision=args.revision)
    elif args.stage == "sample":
        stage_sample(
            out_dir=args.out_dir,
            k=args.k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            model=args.model,
            base_url=args.base_url,
            thinking=args.thinking,
            system_prompt_file=args.system_prompt_file,
        )
    elif args.stage == "score":
        defaults = verifier_settings()
        stage_score(
            out_dir=args.out_dir,
            workers=args.workers,
            runtime_filter=args.runtime,
            settings=VerifierSettings(
                data_shape=args.data_shape,
                pymc_draws=args.pymc_draws,
                pymc_tune=args.pymc_tune,
                pymc_timeout_sec=args.pymc_timeout_sec,
                lazyppl_timeout_sec=defaults.lazyppl_timeout_sec,
                lazyppl_harbor_image=defaults.lazyppl_harbor_image,
            ),
        )
    elif args.stage == "report":
        return stage_report(out_dir=args.out_dir, k=args.k, eps=args.eps)
    return 0

#!/usr/bin/env python3
"""Batch-translate PyMC synthesis records to LazyPPL Haskell.

Reads PyMC train/val/test JSONL records, sends each prompt through GPT-5.5
with the LazyPPL completion system prompt, validates via GHC type-check,
and optionally repairs up to 3 rounds. Active outputs belong under
generated/lazyppl_synthesis/inputs/.

Usage:
    uv run python scripts/translate_pymc_to_lazyppl.py \
        --input data/pymc_synthesis/train.jsonl \
        --output generated/lazyppl_synthesis/inputs/train_translated.jsonl \
        --model gpt-5.5 \
        --max-repairs 3

Skip quarantined records (known-bad PyMC):
    --skip-ids 'test#7,val#3,val#4,val#7,train#96,train#110,train#125,
                train#127,train#129,train#137,train#146'
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from litellm import completion as litellm_completion

from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.runtimes import get_runtime

LAZYPPL = get_runtime("lazyppl")
SYSTEM_PROMPT = load_prompt_text("completion_gen_lazyppl.md")


def translate_one(
    prompt: str,
    model: str,
    max_repairs: int = 3,
    temperature: float = 0.4,
    strip_thinking: bool = True,
) -> dict[str, Any]:
    resp = litellm_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        drop_params=True,
    )
    raw_completion = resp.choices[0].message.content or ""

    code = LAZYPPL.extract_code(raw_completion)
    if code is None:
        return {
            "code": None,
            "completion": raw_completion,
            "think": None,
            "repairs": 0,
            "typecheck_ok": False,
            "error": "no code block extracted",
        }

    think_match = re.search(r"<think>(.*?)</think>", raw_completion, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else None

    result = LAZYPPL.validate_runtime(code)
    repairs = 0

    while not result.ok and repairs < max_repairs:
        repairs += 1
        sys_prompt, user_msg = LAZYPPL.repair_system_prompt(
            prompt_text=prompt,
            broken_code=code,
            error_trace=result.message,
        )
        repair_resp = litellm_completion(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            drop_params=True,
        )
        repair_text = repair_resp.choices[0].message.content
        new_code = LAZYPPL.extract_code(repair_text)
        if new_code is not None:
            code = new_code
            raw_completion = LAZYPPL.replace_code(raw_completion, code)
        result = LAZYPPL.validate_runtime(code)

    completion_out = raw_completion
    if strip_thinking and think_text:
        completion_out = re.sub(
            r"<think>.*?</think>\s*", "", completion_out, flags=re.DOTALL
        ).strip()

    return {
        "code": code,
        "completion": completion_out,
        "think": think_text,
        "repairs": repairs,
        "typecheck_ok": result.ok,
        "error": None if result.ok else result.message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate PyMC records to LazyPPL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument(
        "--skip-ids",
        type=str,
        default="",
        help="Comma-separated IDs to skip, e.g. 'test#7,train#96'",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N records (0 = all)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    args = parser.parse_args()

    skip_set = set()
    if args.skip_ids:
        for sid in args.skip_ids.split(","):
            sid = sid.strip()
            if sid:
                skip_set.add(sid)

    split_name = args.input.stem  # train, val, test
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            record_id = f"{split_name}#{i}"
            if record_id in skip_set:
                print(f"SKIP {record_id} (quarantined)")
                continue
            rec = json.loads(line)
            records.append((i, record_id, rec))

    if args.limit > 0:
        records = records[: args.limit]

    total = len(records)
    ok_count = 0
    fail_count = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, (_orig_idx, record_id, rec) in enumerate(records):
            print(f"[{idx + 1}/{total}] {record_id}...", end=" ", flush=True)

            result = translate_one(
                prompt=rec["prompt"],
                model=args.model,
                max_repairs=args.max_repairs,
                temperature=args.temperature,
            )

            out_rec = {
                "prompt": rec["prompt"],
                "completion": result["completion"],
                "metadata": {
                    **(rec.get("metadata") or {}),
                    "lazyppl_version": "lazyppl-1.0",
                    "source": "translated_from_pymc",
                    "original_id": record_id,
                    "typecheck_ok": result["typecheck_ok"],
                    "repairs": result["repairs"],
                },
                "complexity": rec.get("complexity", "medium"),
            }

            if result["think"]:
                out_rec["metadata"]["think"] = result["think"]

            if result["error"]:
                out_rec["metadata"]["error"] = result["error"]

            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            status = "OK" if result["typecheck_ok"] else "FAIL"
            repairs_info = f" ({result['repairs']} repairs)" if result["repairs"] else ""
            print(f"{status}{repairs_info}")

            if result["typecheck_ok"]:
                ok_count += 1
            else:
                fail_count += 1

            if args.delay > 0 and idx < total - 1:
                time.sleep(args.delay)

    print(f"\nDone. {ok_count}/{total} typecheck OK, {fail_count} failed.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

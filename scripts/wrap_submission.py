#!/usr/bin/env python3
"""Wrap old translated LazyPPL records into the Submission module contract.

Old translated records have standalone code with embedded data and a main function.
This script sends each through GPT to refactor into the Submission contract:
    module Submission where
    model :: [Row] -> Meas State
    logLikRow :: State -> Row -> Double

Usage:
    uv run python scripts/wrap_submission.py \
        --input generated/lazyppl_synthesis/legacy/train_pre_wrap.jsonl \
        --output generated/lazyppl_synthesis/legacy/train_wrapped.jsonl \
        --model gpt-5.5 \
        --max-repairs 3

    # Dry run (first 5 records):
    uv run python scripts/wrap_submission.py --input ... --output ... --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from litellm import completion as litellm_completion

from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.runtimes import get_runtime

LAZYPPL = get_runtime("lazyppl")
SUBMISSION_SYSTEM_PROMPT = load_prompt_text("completion_gen_lazyppl.md")

WRAP_INSTRUCTION = """\
Below is a LazyPPL probabilistic model that uses embedded observations and a
standalone `main` function. Refactor it into the Submission module contract.

## What to change

1. Add `module Submission where` at the top.
2. Replace embedded observations with `import TaskData (Row(..), loadData)`.
   The `Row` record will have the same fields that the original code observes.
   Access fields via the Row accessor functions (e.g., `fieldName row`).
3. Refactor the inference logic into `model :: [Row] -> Meas State`.
   - Keep all priors and scoring logic.
   - Replace references to embedded data with the `rows` argument.
   - Do NOT call `mh`, `mhirreducible`, `wis`, or any IO action.
4. Define `logLikRow :: State -> Row -> Double` that returns the per-row
   log-likelihood. Extract this from the scoring pattern in the original code.
   If the original uses `score (normalPdf mu sigma x)`, the log-likelihood is
   `normalLogPdf mu sigma x`. If it uses `score (exp ll)`, the log-likelihood is `ll`.
5. Remove the `main` function entirely.
6. Define `type State = ...` or `data State = ... deriving (Show)` for whatever
   latent parameters the model infers.

## Important rules

- `logLikRow` must be the log-density for ONE row, not the sum over all rows.
- Guard against log(0) and division by zero (use `max 1e-300` floors).
- `score` takes raw likelihood (positive), not log-likelihood.
- Both `model` and `logLikRow` MUST have explicit type signatures.
- Do NOT import or use `System.IO`, `Text.Printf`, or any IO modules.
- Do NOT define `main`. The harness handles IO.

## Original prompt (for context)

{prompt}

## Original code to refactor

```haskell
{code}
```

Output ONLY the refactored code inside `<submit_program language="haskell">` tags.
A brief `<think>` block first is fine but optional.
"""


def wrap_one(
    prompt: str,
    old_code: str,
    llm_model: str,
    max_repairs: int = 3,
    temperature: float = 0.3,
) -> dict:
    """Refactor one old record into Submission contract."""
    user_msg = WRAP_INSTRUCTION.format(prompt=prompt, code=old_code)

    resp = litellm_completion(
        model=llm_model,
        messages=[
            {"role": "system", "content": SUBMISSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
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
        sys_prompt, user_msg_repair = LAZYPPL.repair_system_prompt(
            prompt_text=prompt,
            broken_code=code,
            error_trace=result.message,
        )
        repair_resp = litellm_completion(
            model=llm_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg_repair},
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

    # strip thinking from final completion
    completion_out = raw_completion
    if think_text:
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
    parser = argparse.ArgumentParser(
        description="Wrap old LazyPPL records into Submission contract"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N old records (0 = all)"
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument(
        "--no-skip-new",
        action="store_true",
        default=False,
        help="Process ALL records, even those that already have module Submission",
    )
    parser.add_argument(
        "--prior-run",
        type=Path,
        default=None,
        help="Path to prior wrapped output; skip records that already succeeded",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _prompt_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    # build skip set from prior run, keyed by prompt hash (not positional index)
    skip_keys: set[str] = set()
    prior_by_key: dict[str, dict] = {}
    if args.prior_run and args.prior_run.exists():
        with open(args.prior_run, encoding="utf-8") as pf:
            for line in pf:
                prec = json.loads(line)
                key = _prompt_key(prec.get("prompt", ""))
                meta = prec.get("metadata", {})
                if meta.get("wrap_typecheck_ok") is True:
                    skip_keys.add(key)
                    prior_by_key[key] = prec
                elif meta.get("source") == "native_lazyppl":
                    skip_keys.add(key)
                    prior_by_key[key] = prec

    records: list[tuple[int, dict]] = []
    all_records: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            all_records.append(rec)
            key = _prompt_key(rec.get("prompt", ""))
            if key in skip_keys:
                continue
            completion = rec.get("completion", "")
            if not args.no_skip_new and "module Submission" in completion:
                continue
            records.append((i, rec))

    if args.limit > 0:
        records = records[: args.limit]

    total = len(records)
    ok_count = 0
    fail_count = 0
    indices_to_update: dict[int, dict] = {}

    print(f"Wrapping {total} old records (of {len(all_records)} total)")
    print(f"Model: {args.model}, max repairs: {args.max_repairs}")
    print()

    for idx, (orig_idx, rec) in enumerate(records):
        print(f"[{idx + 1}/{total}] record #{orig_idx}...", end=" ", flush=True)

        old_code = rec.get("completion", "")
        # extract just the haskell code from submit_program tags
        code_match = re.search(
            r"<submit_program[^>]*>(.*?)</submit_program>",
            old_code,
            re.DOTALL,
        )
        code_body = code_match.group(1).strip() if code_match else old_code

        result = wrap_one(
            prompt=rec["prompt"],
            old_code=code_body,
            llm_model=args.model,
            max_repairs=args.max_repairs,
            temperature=args.temperature,
        )

        status = "OK" if result["typecheck_ok"] else "FAIL"
        repairs_info = f" ({result['repairs']} repairs)" if result["repairs"] else ""
        print(f"{status}{repairs_info}")

        if result["typecheck_ok"]:
            ok_count += 1
        else:
            fail_count += 1

        indices_to_update[orig_idx] = result

        if args.delay > 0 and idx < total - 1:
            time.sleep(args.delay)

    # write output atomically: temp file + rename
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(all_records):
            if i in indices_to_update:
                result = indices_to_update[i]
                rec["completion"] = result["completion"]
                meta = rec.get("metadata", {})
                meta["wrapped_to_submission"] = True
                meta["wrap_typecheck_ok"] = result["typecheck_ok"]
                meta["wrap_repairs"] = result["repairs"]
                if result["error"]:
                    meta["wrap_error"] = result["error"]
                elif "wrap_error" in meta:
                    del meta["wrap_error"]
                rec["metadata"] = meta
            else:
                key = _prompt_key(rec.get("prompt", ""))
                if key in prior_by_key:
                    # carry forward successful records from prior run
                    rec = prior_by_key[key]
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(args.output)

    print(f"\nDone. {ok_count}/{total} typecheck OK, {fail_count} failed.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ppl_synth_data.pipeline.io import read_jsonl

# placeholders: {batch_num}, {start}, {end} (filled by build_batch_text)
DEFAULT_SYSTEM_PROMPT = """\
# Dataset Audit: Batch {batch_num} (entries {start}-{end})

## Your task

You are a Bayesian statistics expert and PyMC v5 practitioner. For each numbered \
entry below, verify:

**A. Prompt quality (score 1-5)**
- Unambiguous and well-posed problem description
- No distribution names leaked (Poisson, Bernoulli, Beta, Gamma, Normal, \
Exponential, Dirichlet, etc.)
- No method names leaked (hierarchical, MCMC, Bayesian, prior, posterior, \
Markov chain, variational inference, etc.)
- Describes observable data a practitioner would actually have

**B. Completion correctness (score 1-5)**
- `<think>` block correctly identifies the right statistical approach
- Distribution choice appropriate for the data type and problem
- Priors are reasonable for the domain and plausible scale (not wildly off)
- PyMC v5 API correct (parameter names, function signatures, observed-data
  containers such as `pm.Data`, etc.)
- Variable names consistent with what the prompt describes
- `def model(data):` signature present; `data` dict keys used in code match \
what would realistically be available from the described data

**C. Metadata accuracy**
- `model_type` (inferential/predictive) matches the code's purpose
- `complexity` (simple/medium/complex) matches the actual code length and \
number of modelling components
- `key_concepts` list matches statistical concepts actually used in the code

## Output format (strict: one block per entry, copy exactly)

```
Entry N: [PASS | FLAG | CRITICAL]
Prompt score: X/5
Completion score: X/5
Metadata: [OK | <issue description>]
Issues: [None | <precise description of what is wrong and the correct version>]
```

Then a separator `---` between entries.

After all entries, write a summary section:

```
## Summary
- PASS: N
- FLAG: N
- CRITICAL: N
- Most common issue categories: [list]
```

Use PASS when everything is correct or only trivially cosmetic.
Use FLAG for meaningful errors that affect training quality but have a clear fix.
Use CRITICAL for errors that would teach the model wrong statistics \
(wrong distribution family, broken API call, banned-term leakage in prompt).

---

## Entries

"""


@dataclass
class BatchSpec:
    batch_num: int
    start: int  # global 1-based index of first entry
    end: int  # global 1-based index of last entry
    entries: list[tuple[int, dict, str]]  # (global_idx, entry, split)


def build_entry_list(data_dir: Path) -> list[tuple[int, dict, str]]:
    all_entries: list[tuple[int, dict, str]] = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            for entry in read_jsonl(path):
                all_entries.append((len(all_entries) + 1, entry, split))
    return all_entries


def make_batches(
    entries: list[tuple[int, dict, str]],
    batch_size: int,
) -> list[BatchSpec]:
    batches: list[BatchSpec] = []
    for i in range(0, len(entries), batch_size):
        chunk = entries[i : i + batch_size]
        n = len(batches) + 1
        batches.append(
            BatchSpec(
                batch_num=n,
                start=chunk[0][0],
                end=chunk[-1][0],
                entries=chunk,
            )
        )
    return batches


def _format_entry(global_idx: int, entry: dict, split: str) -> str:
    prompt = entry.get("prompt", "").strip()
    completion = entry.get("completion", "").strip()
    meta = entry.get("metadata", {})
    top_complexity = entry.get("complexity", meta.get("complexity", "?"))

    model_type = meta.get("model_type", "?")
    complexity = meta.get("complexity", top_complexity)
    key_concepts = meta.get("key_concepts", [])
    domain = meta.get("domain", "?")
    concepts_str = ", ".join(key_concepts) if key_concepts else "none listed"

    return "\n".join(
        [
            f"### Entry {global_idx} (split: {split})",
            f"**Prompt:** {prompt}",
            "",
            "**Completion:**",
            "```",
            completion,
            "```",
            "",
            f"**Metadata:** model_type={model_type}, complexity={complexity}, "
            f"domain={domain}, key_concepts=[{concepts_str}]",
            "",
            "---",
            "",
        ]
    )


def build_batch_text(spec: BatchSpec, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    header = system_prompt.format(
        batch_num=spec.batch_num,
        start=spec.start,
        end=spec.end,
    )
    body = "\n".join(_format_entry(idx, entry, split) for idx, entry, split in spec.entries)
    return header + body

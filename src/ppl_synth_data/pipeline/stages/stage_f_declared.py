"""Build declared four-family golds on visible data, without fitting or LLM calls."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ppl_synth_data.harbor.declared_gold import render_declared_gold
from ppl_synth_data.pipeline.io import read_jsonl, write_json, write_jsonl
from ppl_synth_data.pipeline.types import RunPaths, StageResult
from ppl_synth_data.pipeline.validation.data_loading import load_sample_data
from ppl_synth_data.specs.io import read_envspecs
from ppl_synth_data.specs.models import (
    CompletionRecord,
    FourFamilySpec,
    PromptProvenance,
    PromptRecord,
    ValidationInfo,
)


def execute_declared_gold(spec: FourFamilySpec, parquet: Path) -> tuple[str, dict]:
    """Execute the trusted renderer and evaluate its initial density; never sample."""
    import pymc as pm

    data = load_sample_data(parquet)
    if data is None:
        raise ValueError(f"empty visible data parquet: {parquet}")
    required = {feature.name for feature in spec.features} | {"y"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"visible data missing columns: {sorted(missing)}")
    for name in required:
        values = np.asarray(data[name])
        if values.dtype.kind in "biuf" and not np.isfinite(values).all():
            raise ValueError(f"nonfinite visible data in {name}")
    y = np.asarray(data["y"])
    if spec.generative_family == "binary_choice" and not np.isin(y, [0, 1]).all():
        raise ValueError("binary observations must be 0 or 1")
    if spec.generative_family in {"regression_count", "zero_inflated_count"}:
        if not ((y >= 0) & (y == np.floor(y))).all():
            raise ValueError("count observations must be nonnegative integers")
    code, audit = render_declared_gold(spec, data)
    namespace = {"np": np, "pm": pm}
    # Only code from the closed typed renderer reaches this execution point.
    exec(compile(code, f"declared:{spec.env_id}", "exec"), namespace)
    model = namespace["model"](data)
    if not isinstance(model, pm.Model) or len(model.observed_RVs) != 1:
        raise ValueError("declared gold must build one observed PyMC likelihood")
    initial_logp = float(model.compile_logp()(model.initial_point()))
    if not np.isfinite(initial_logp):
        raise ValueError("declared gold has nonfinite initial log density on visible data")
    return code, {
        "gold_source": "declared",
        "gold_program_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "visible_data_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "visible_rows": data["n"],
        "initial_logp": initial_logp,
        "declared_gold_audit": audit,
        "validation_scope": "visible_model_build_and_initial_logp",
        "references": "PENDING",
    }


def run_stage_f_declared(paths: RunPaths) -> StageResult:
    """Require exact compiled-spec/prompt coverage and persist genuine build results."""
    specs = read_envspecs(paths.stage_dirs["B"] / "envspecs.jsonl")
    if not specs:
        raise ValueError("declared Stage F requires nonempty canonical Stage B specs")
    spec_ids = set()
    declared_specs: list[FourFamilySpec] = []
    for spec in specs:
        if not isinstance(spec, FourFamilySpec):
            raise ValueError(f"{spec.env_id}: declared Stage F requires FourFamilySpec")
        if spec.env_id in spec_ids:
            raise ValueError(f"duplicate canonical env_id: {spec.env_id}")
        spec_ids.add(spec.env_id)
        declared_specs.append(spec)
    prompts = [
        PromptRecord.model_validate(row)
        for row in read_jsonl(paths.stage_dirs["D"] / "prompts.jsonl")
    ]
    prompt_by_env = {}
    prompt_ids = set()
    for prompt in prompts:
        if prompt.env_id in prompt_by_env or prompt.id in prompt_ids:
            raise ValueError(f"duplicate prompt coverage: {prompt.env_id} ({prompt.id})")
        if prompt.provenance and prompt.provenance.runtime_name not in {None, "pymc"}:
            raise ValueError(f"{prompt.env_id}: prompt runtime must be pymc")
        prompt_by_env[prompt.env_id] = prompt
        prompt_ids.add(prompt.id)
    missing = spec_ids - prompt_by_env.keys()
    extra = prompt_by_env.keys() - spec_ids
    if missing or extra:
        raise ValueError(
            f"Stage D coverage incomplete: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    stage_dir = paths.stage_dirs["F"]
    validated, rejected = [], []
    for index, spec in enumerate(declared_specs):
        parquet = paths.stage_dirs["C"] / "data" / f"{spec.env_id}.parquet"
        try:
            code, metadata = execute_declared_gold(spec, parquet)
        except Exception as exc:
            rejected.append(
                {
                    "env_id": spec.env_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "visible_data": str(parquet),
                }
            )
            continue
        prompt = prompt_by_env[spec.env_id]
        provenance = prompt.provenance.model_dump() if prompt.provenance else {}
        provenance["runtime_name"] = "pymc"
        record = CompletionRecord(
            id=f"declared-{index:06d}",
            prompt_id=prompt.id,
            env_id=spec.env_id,
            split=prompt.split,
            prompt_text=prompt.prompt_text,
            raw_completion=code,
            parsed_program=code,
            metadata=metadata,
            provenance=PromptProvenance.model_validate(provenance),
            validation=ValidationInfo(
                ok=True,
                message=(
                    "Declared model executed on visible data with finite initial log density; "
                    "posterior fits and held-out references not run."
                ),
            ),
        )
        validated.append(record.model_dump(mode="json"))
    write_jsonl(stage_dir / "validated.jsonl", validated)
    write_jsonl(stage_dir / "rejected.jsonl", rejected)
    write_jsonl(
        stage_dir / "traces.jsonl",
        [
            {
                "env_id": row["env_id"],
                "gold_source": "declared",
                "validation_scope": row["metadata"]["validation_scope"],
            }
            for row in validated
        ],
    )
    manifest = {
        "status": "failed" if rejected else "success",
        "runtime": "pymc",
        "gold_source": "declared",
        "count_total": len(specs),
        "count_validated": len(validated),
        "count_rejected": len(rejected),
        "references": "PENDING",
        "posterior_fit_performed": False,
    }
    write_json(stage_dir / "manifest.json", manifest)
    if rejected:
        raise RuntimeError(
            "declared gold execution failed:\n"
            + "\n".join(f"{row['env_id']}: {row['reason']}" for row in rejected)
        )
    return StageResult(stage="F", status="success", outputs=manifest)

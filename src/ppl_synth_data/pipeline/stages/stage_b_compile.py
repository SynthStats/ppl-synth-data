from __future__ import annotations

import re
import shutil
from typing import Literal

from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.io import write_json, write_jsonl
from ppl_synth_data.pipeline.stages._envspecs import canonical_envspecs_path
from ppl_synth_data.pipeline.types import RunPaths, StageResult
from ppl_synth_data.specs.io import read_compile_inputs
from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec

_UNSAFE_ENV_NAME_SUFFIX = re.compile(r"[^a-z0-9_]+")


def _env_name_suffix(value: str, fallback: str) -> str:
    suffix = _UNSAFE_ENV_NAME_SUFFIX.sub("_", value.lower()).strip("_")
    return suffix or fallback


def _canonicalize_env_names(
    specs: list[EnvSpec | FourFamilySpec],
) -> tuple[list[EnvSpec | FourFamilySpec], list[dict[str, str]]]:
    used: set[str] = set()
    seen: dict[str, int] = {}
    canonical: list[EnvSpec | FourFamilySpec] = []
    renamed: list[dict[str, str]] = []

    for idx, spec in enumerate(specs):
        if isinstance(spec, FourFamilySpec):
            canonical.append(spec)
            continue
        seen_count = seen.get(spec.env_name, 0)
        seen[spec.env_name] = seen_count + 1
        if seen_count == 0 and spec.env_name not in used:
            used.add(spec.env_name)
            canonical.append(spec)
            continue

        suffix = _env_name_suffix(spec.env_id, fallback=f"env_{idx:06d}")
        candidate = f"{spec.env_name}_{suffix}"
        collision_idx = 2
        while candidate in used:
            candidate = f"{spec.env_name}_{suffix}_{collision_idx}"
            collision_idx += 1

        canonical_spec = spec.model_copy(update={"env_name": candidate})
        canonical.append(canonical_spec)
        used.add(candidate)
        renamed.append(
            {
                "env_id": spec.env_id,
                "original_env_name": spec.env_name,
                "env_name": candidate,
            }
        )

    return canonical, renamed


def _canonicalize_io_schemas(
    specs: list[EnvSpec | FourFamilySpec],
) -> tuple[list[EnvSpec | FourFamilySpec], list[dict[str, object]]]:
    canonical: list[EnvSpec | FourFamilySpec] = []
    changed: list[dict[str, object]] = []
    for spec in specs:
        if isinstance(spec, FourFamilySpec):
            # The closed declaration validates its feature keys without repairs.
            canonical.append(spec)
            continue
        seen: set[str] = set()
        feature_names: list[str] = []
        removed: list[str] = []
        for feature_name in spec.io_schema.feature_names:
            if feature_name == spec.io_schema.target_name or feature_name in seen:
                removed.append(feature_name)
                continue
            seen.add(feature_name)
            feature_names.append(feature_name)
        if removed:
            io_schema = spec.io_schema.model_copy(update={"feature_names": feature_names})
            spec = spec.model_copy(update={"io_schema": io_schema})
            changed.append(
                {
                    "env_id": spec.env_id,
                    "env_name": spec.env_name,
                    "target_name": spec.io_schema.target_name,
                    "removed_feature_names": removed,
                    "feature_names": feature_names,
                }
            )
        canonical.append(spec)
    return canonical, changed


def run_stage_b(paths: RunPaths) -> StageResult:
    stage = "B"
    stage_dir = paths.stage_dirs[stage]
    stage_dir.mkdir(parents=True, exist_ok=True)
    env_out_dir = stage_dir / "envs"
    if env_out_dir.exists():
        shutil.rmtree(env_out_dir)
    env_out_dir.mkdir(parents=True, exist_ok=True)

    specs, refusals = read_compile_inputs(paths.stage_dirs["A"] / "envspecs.jsonl")
    env_ids = [spec.env_id for spec in specs]
    if len(set(env_ids)) != len(env_ids):
        raise ValueError("Stage B requires unique env_id values across EnvSpecs")

    specs, renamed_envs = _canonicalize_env_names(specs)
    specs, canonicalized_io_schemas = _canonicalize_io_schemas(specs)
    compiler = TemplateCompiler()
    compiled_paths: list[str] = []
    compiled_specs = []
    skipped_envs: list[dict[str, str]] = [
        {
            "env_id": item["env_id"],
            "error": "; ".join(r["missing_semantics"] for r in item["refusals"]),
            "exception_type": "ValidationError",
        }
        for item in refusals
    ]
    for spec in specs:
        try:
            compiled = compiler.compile_envspec(spec, out_dir=env_out_dir)
        except Exception as exc:
            skipped_envs.append(
                {
                    "env_id": spec.env_id,
                    "env_name": spec.env_name,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
            continue
        compiled_paths.append(str(compiled))
        compiled_specs.append(spec)

    write_jsonl(
        canonical_envspecs_path(paths), [spec.model_dump(mode="json") for spec in compiled_specs]
    )
    write_jsonl(stage_dir / "refusals.jsonl", refusals)

    status: Literal["completed_with_refusals", "success"] = (
        "completed_with_refusals" if skipped_envs else "success"
    )
    manifest = {
        "status": status,
        "resumable": not skipped_envs,
        "count": len(specs) + len(refusals),
        "refused_count": len(refusals),
        "compiled_count": len(compiled_paths),
        "skipped_env_count": len(skipped_envs),
        "skipped_envs": skipped_envs,
        "renamed_env_name_count": len(renamed_envs),
        "renamed_env_names": renamed_envs,
        "canonicalized_io_schema_count": len(canonicalized_io_schemas),
        "canonicalized_io_schemas": canonicalized_io_schemas,
        "canonical_envspecs": str(canonical_envspecs_path(paths)),
        "compiled_modules": compiled_paths,
    }
    write_json(stage_dir / "manifest.json", manifest)
    write_jsonl(stage_dir / "skipped_envs.jsonl", skipped_envs)
    if skipped_envs and not compiled_paths:
        raise RuntimeError(
            "stage B could not compile any generated environments; "
            f"first skipped env: {skipped_envs[0]}"
        )
    return StageResult(stage=stage, status=status, outputs=manifest)

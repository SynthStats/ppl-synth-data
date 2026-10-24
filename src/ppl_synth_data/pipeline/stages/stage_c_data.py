from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppl_synth_data.envs.base import InputValidationError
from ppl_synth_data.envs.declared_inputs import sample_inputs
from ppl_synth_data.pipeline.data_summary import summarize_numeric
from ppl_synth_data.pipeline.env_loader import (
    build_query,
    env_module_filename,
    load_env_instance,
    seed_for_env,
    validate_feature_types,
)
from ppl_synth_data.pipeline.io import write_json, write_jsonl
from ppl_synth_data.pipeline.serialization import json_safe
from ppl_synth_data.pipeline.split_policy import (
    SplitPolicy,
    SplitPolicyIneligibleError,
    split_visible_hidden,
    validate_visible_ratio,
)
from ppl_synth_data.pipeline.stages._envspecs import read_canonical_envspecs
from ppl_synth_data.pipeline.types import RunPaths, StageResult
from ppl_synth_data.specs.models import FourFamilySpec


def _run_declared_query(spec: Any, env: Any, idx: int, *, query: str) -> dict[str, Any]:
    try:
        result, success = env.run_experiment(query)
    except InputValidationError as exc:
        raise RuntimeError(
            f"declared query rejected for env_id={spec.env_id!r} at observation {idx}: {exc}"
        ) from exc
    if not success:
        raise RuntimeError(
            f"declared query failed for env_id={spec.env_id!r} "
            f"at observation {idx}: query={query!r}, result={json_safe(result)!r}"
        )
    return {
        "query": query,
        "result": json_safe(result),
        "success": True,
    }


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def run_stage_c(
    paths: RunPaths,
    *,
    split_policy: SplitPolicy | str,
    seed: int = 0,
    limit: int | None = None,
    visible_ratio: float,
) -> StageResult:
    split_policy = SplitPolicy(split_policy)
    validate_visible_ratio(visible_ratio)
    stage = "C"
    stage_dir = paths.stage_dirs[stage]
    specs = read_canonical_envspecs(paths)
    if limit is not None:
        specs = specs[:limit]

    attempt_prefix = f".{stage_dir.name}.attempt-"
    attempt_dir = stage_dir.with_name(f"{attempt_prefix}active")
    data_dir = attempt_dir / "data"

    observations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    data_refs: list[dict[str, Any]] = []
    skipped_envs: list[dict[str, str]] = []
    split_policy_counts: dict[str, int] = {}

    try:
        stage_dir.parent.mkdir(parents=True, exist_ok=True)
        for stale_attempt in stage_dir.parent.iterdir():
            if stale_attempt.name.startswith(attempt_prefix):
                _remove_path(stale_attempt)
        _remove_path(stage_dir)
        data_dir.mkdir(parents=True)

        # Invalid feature declarations invalidate the run, including mixed batches.
        # Check outside the per-environment skip handler before generating any data.
        for spec in specs:
            if not isinstance(spec, FourFamilySpec):
                validate_feature_types(spec)

        for spec in specs:
            env_seed = seed_for_env(seed, spec.env_id)
            # Separate design, outcome, and split streams prevent shared draws.
            # Reuse one design generator across rows for reproducible covariates.
            design_rng = np.random.default_rng(seed_for_env(seed, f"{spec.env_id}:inputs"))
            try:
                env_module_path = paths.stage_dirs["B"] / "envs" / env_module_filename(spec)
                env = load_env_instance(
                    env_module_path,
                    env_name=spec.env_name,
                    seed=env_seed,
                    trusted_root=paths.stage_dirs["B"] / "envs",
                )
                n_obs = int(spec.data_plan.n_observations)
                if isinstance(spec, FourFamilySpec):
                    design = sample_inputs(spec, seed=seed_for_env(seed, f"{spec.env_id}:inputs"))
                    queries = (json.dumps(row, ensure_ascii=False) for row in design)
                else:
                    queries = (build_query(spec, idx, rng=design_rng) for idx in range(n_obs))
                env_observations: list[dict[str, Any]] = []
                for idx, query in enumerate(queries):
                    query_result = _run_declared_query(spec, env, idx, query=query)
                    env_observations.append(
                        {
                            "env_id": spec.env_id,
                            "env_name": spec.env_name,
                            **query_result,
                        }
                    )

                df = env.get_df()
                if df.empty:
                    raise ValueError(
                        f"environment produced an empty dataframe after {n_obs} observations"
                    )
                out_parquet = data_dir / f"{spec.env_id}.parquet"
                final_out_parquet = stage_dir / "data" / f"{spec.env_id}.parquet"

                split = split_visible_hidden(
                    df,
                    spec,
                    split_policy=split_policy,
                    seed=seed_for_env(seed, f"{spec.env_id}:split"),
                    visible_ratio=visible_ratio,
                )
                df_visible = split.visible
                df_hidden = split.hidden

                df_visible.to_parquet(out_parquet, index=False)

                hidden_ref: str | None = None
                if len(df_hidden) > 0:
                    hidden_dir = data_dir / "hidden"
                    hidden_dir.mkdir(parents=True, exist_ok=True)
                    hidden_path = hidden_dir / f"{spec.env_id}.parquet"
                    df_hidden.to_parquet(hidden_path, index=False)
                    final_hidden_path = stage_dir / "data" / "hidden" / f"{spec.env_id}.parquet"
                    hidden_ref = str(final_hidden_path.relative_to(paths.run_dir))

                target = "y" if isinstance(spec, FourFamilySpec) else spec.io_schema.target_name
                if target in df_visible.columns:
                    numeric = pd.to_numeric(df_visible[target], errors="coerce").dropna().tolist()
                    values = [float(v) for v in numeric]
                else:
                    values = []

                summary = summarize_numeric(values)
                summary.update(
                    {
                        "env_id": spec.env_id,
                        "env_name": spec.env_name,
                        "target_name": target,
                    }
                )
                observations.extend(env_observations)
                summaries.append(summary)
                data_refs.append(
                    {
                        "env_id": spec.env_id,
                        "env_name": spec.env_name,
                        "data_ref": str(final_out_parquet.relative_to(paths.run_dir)),
                        "hidden_data_ref": hidden_ref,
                        "split_policy": split.policy_name,
                    }
                )
                split_policy_counts[split.policy_name] = (
                    split_policy_counts.get(split.policy_name, 0) + 1
                )
            except SplitPolicyIneligibleError:
                raise
            except Exception as exc:
                skipped_envs.append(
                    {
                        "env_id": spec.env_id,
                        "env_name": spec.env_name,
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                    }
                )

        if specs and not summaries:
            raise RuntimeError(
                "stage C could not materialize any generated environments; "
                f"first skipped env: {skipped_envs[0] if skipped_envs else 'unknown'}"
            )

        write_jsonl(attempt_dir / "observations.jsonl", observations)
        write_jsonl(attempt_dir / "summaries.jsonl", summaries)
        write_jsonl(attempt_dir / "data_refs.jsonl", data_refs)
        write_jsonl(attempt_dir / "skipped_envs.jsonl", skipped_envs)

        manifest = {
            "status": "success",
            "status_detail": "success_with_skips" if skipped_envs else "success",
            "resumable": not skipped_envs,
            "env_count": len(specs),
            "materialized_env_count": len(summaries),
            "skipped_env_count": len(skipped_envs),
            "data_dir_fresh": True,
            "observation_count": len(observations),
            "summary_count": len(summaries),
            "split_policy_counts": split_policy_counts,
        }
        write_json(attempt_dir / "manifest.json", manifest)
        attempt_dir.replace(stage_dir)
    except Exception:
        _remove_path(attempt_dir)
        _remove_path(stage_dir)
        raise

    return StageResult(stage=stage, status="success", outputs=manifest)

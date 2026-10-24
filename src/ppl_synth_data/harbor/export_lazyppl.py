"""Export validated LazyPPL pipeline outputs into Harbor task directories."""

from __future__ import annotations

import logging
import shutil
from importlib import resources
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import pandas as pd

from ppl_synth_data.harbor.export_shared import (
    _SEQ_RE,
    _completion_runtime_name,
    _load_jinja_env,
    _read_data_refs,
    _resolve_run_relative_path,
    _sha256_text,
    _strip_markdown_fences,
)
from ppl_synth_data.pipeline.contracts import STAGE_DIR_NAMES
from ppl_synth_data.pipeline.io import atomic_write_text, read_jsonl
from ppl_synth_data.pipeline.validation.semantic_evidence import (
    lazyppl_semantic_evidence_failure,
)
from ppl_synth_data.runtime_registry import build_harbor_task_id, get_runtime_spec
from ppl_synth_data.runtimes.lazyppl_runtime import static_safety_grep_pattern
from ppl_synth_data.specs.io import read_envspecs
from ppl_synth_data.specs.models import CompletionRecord, EnvSpec, FourFamilySpec
from ppl_synth_data.specs.record_contracts import normalize_completion_record

log = logging.getLogger(__name__)

_LAZYPPL_DOCKER_APP_FILES = (
    "Main.hs",
    "LazyPPLCompat.hs",
    "score.py",
    "check.sh",
    "REFERENCE.md",
)

_LAZYPPL_DEFAULT_N_DRAWS = 2000

# map seed filename stem → short human-readable description
_LAZYPPL_PROMPT_HINTS: dict[str, str] = {
    "simple_normal": "Estimate the mean and variance of normally distributed observations.",
    "linear_regression": "Fit a linear regression with Gaussian noise.",
    "logistic_regression": "Fit a logistic regression for binary outcomes.",
    "beta_bernoulli": "Estimate the bias of a coin from Bernoulli trial data.",
    "beta_binomial": "Estimate success probability from binomial count data.",
    "changepoint_single": "Detect a single changepoint in a time series.",
    "dp_clustering": "Cluster observations using a Dirichlet process mixture.",
    "exponential_survival": (
        "Estimate the rate parameter of exponentially distributed survival times."
    ),
    "gamma_rate_estimation": "Estimate shape and scale of gamma-distributed observations.",
    "gp_regression": "Fit a Gaussian process regression to noisy observations.",
    "hierarchical_group": (
        "Estimate group-level and observation-level parameters in a hierarchical model."
    ),
    "mixture_two_normals": "Fit a two-component Gaussian mixture model.",
    "multivariate_regression": ("Fit a multivariate linear regression with multiple predictors."),
    "nonlinear_regression": "Fit a nonlinear regression model to the data.",
    "poisson_rate": "Estimate the rate of a Poisson process from count data.",
    "poisson_regression": "Fit a Poisson regression relating predictors to counts.",
}


def _build_lazyppl_task_id(seed_name: str) -> str:
    return build_harbor_task_id("lazyppl", seed_name, sequence=1)


def _build_lazyppl_run_task_id(env_id: str, completion_id: str) -> str:
    seq_match = _SEQ_RE.search(completion_id)
    seq = seq_match.group() if seq_match else "0"
    return build_harbor_task_id("lazyppl", env_id, sequence=seq)


def _lazyppl_prompt_text(seed_name: str) -> str:
    return _LAZYPPL_PROMPT_HINTS.get(
        seed_name,
        f"Write a probabilistic model for the {seed_name.replace('_', ' ')} task.",
    )


def _python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _finite_numeric_values(values: list[Any]) -> list[float] | None:
    parsed: list[float] = []
    for value in values:
        scalar = _python_scalar(value)
        if scalar is None:
            continue
        if isinstance(scalar, bool):
            return None
        try:
            numeric = float(scalar)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        parsed.append(numeric)
    return parsed


def _all_integer_like(values: list[float]) -> bool:
    return all(value.is_integer() for value in values)


def _hs_type_from_rows(
    *,
    field_name: str,
    values: list[Any],
    fallback: str,
) -> str:
    observed = [_python_scalar(value) for value in values if _python_scalar(value) is not None]
    if not observed:
        return fallback
    if all(isinstance(value, bool) for value in observed):
        return "Bool"
    if all(isinstance(value, Integral) and not isinstance(value, bool) for value in observed):
        if field_name == "count" or field_name.endswith("_count"):
            return "Integer"
        return "Int"
    if all(isinstance(value, Real) and not isinstance(value, bool) for value in observed):
        return "Double"
    numeric = _finite_numeric_values(observed)
    if numeric is not None and numeric:
        if field_name == "count" or field_name.endswith("_count"):
            return "Integer" if _all_integer_like(numeric) else "Double"
        if fallback in {"Int", "Integer"} and _all_integer_like(numeric):
            return fallback
        return "Double"
    return fallback


def _lazyppl_columns_for_rows(
    seed_source: str,
    *,
    rows: list[dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    from ppl_synth_data.harbor.taskdata_gen import parse_seed_row_fields

    columns = parse_seed_row_fields(seed_source)
    if not columns:
        raise ValueError("LazyPPL Harbor export requires at least one Row field access")

    all_rows = [*rows, *hidden_rows]
    if not all_rows:
        raise ValueError("LazyPPL Harbor export requires at least one visible or hidden row")

    inferred: list[dict[str, str]] = []
    for col in columns:
        name = col["name"]
        missing = [index for index, row in enumerate(all_rows) if name not in row]
        if missing:
            raise ValueError(
                f"data rows are missing LazyPPL Row field {name!r}; "
                f"first missing row index {missing[0]}"
            )
        inferred.append(
            {
                **col,
                "hs_type": _hs_type_from_rows(
                    field_name=name,
                    values=[row[name] for row in all_rows],
                    fallback=col["hs_type"],
                ),
            }
        )
    return inferred


def export_lazyppl_harbor_tasks(
    seeds_dir: Path,
    out_dir: Path,
    *,
    docker_app_dir: Path | None = None,
    docker_file: Path | None = None,
    lazyppl_local_dir: Path | None = None,
    data_rows: dict[str, list[dict]] | None = None,
    hidden_data_rows: dict[str, list[dict]] | None = None,
) -> dict[str, Any]:
    """Export LazyPPL seed programs as Harbor task directories.

    This seed-level helper is used by tests and one-off task packaging. The
    public run-based operator path is :func:`export_lazyppl_harbor_tasks_from_run`.
    It deliberately refuses to invent placeholder observations: callers must
    pass visible rows for every seed being exported, and hidden rows when
    holdout evaluation is required.

    For each ``.hs`` file in *seeds_dir*:

    1. Parse seed to extract Row fields (via ``taskdata_gen``).
    2. Generate ``TaskData.hs`` with embedded data.
    3. Build Harbor task directory structure.

    Returns a summary dict matching the shape of :func:`export_harbor_tasks`.
    """
    from ppl_synth_data.harbor.taskdata_gen import (
        generate_hidden_taskdata_hs,
        generate_taskdata_hs,
    )

    project_root = Path(__file__).resolve().parents[3]  # src/../../../
    if docker_app_dir is None:
        docker_app_dir = project_root / "docker" / "lazyppl-harbor" / "app"
    if docker_file is None:
        docker_file = project_root / "docker" / "lazyppl-harbor" / "Dockerfile"
    if lazyppl_local_dir is None:
        lazyppl_local_dir = project_root / "docker" / "lazyppl-harbor" / "lazyppl-local"

    seeds = sorted(seeds_dir.glob("*.hs"))
    if not seeds:
        return {
            "tasks_created": 0,
            "tasks_skipped": 0,
            "skipped_details": [],
        }

    jinja_env = _load_jinja_env()
    task_toml_tmpl = jinja_env.get_template("task_lazyppl.toml.j2")
    instruction_tmpl = jinja_env.get_template("instruction_lazyppl.md.j2")
    test_sh_tmpl = jinja_env.get_template("test_lazyppl.sh.j2")
    lazyppl_runtime_spec = get_runtime_spec("lazyppl")

    created = 0
    skipped: list[dict[str, str]] = []
    created_details: list[dict[str, Any]] = []

    for seed_path in seeds:
        seed_name = seed_path.stem
        task_id = _build_lazyppl_task_id(seed_name)
        task_dir = out_dir / task_id

        try:
            seed_source = seed_path.read_text(encoding="utf-8")
            if data_rows is None or seed_name not in data_rows:
                raise ValueError(f"missing visible data_rows for seed {seed_name!r}")

            rows = data_rows[seed_name]
            if not rows:
                raise ValueError(f"visible data_rows for seed {seed_name!r} must not be empty")
            hidden_rows = (
                hidden_data_rows.get(seed_name, []) if hidden_data_rows is not None else []
            )
            columns = _lazyppl_columns_for_rows(
                seed_source,
                rows=rows,
                hidden_rows=hidden_rows,
            )
            col_names = [c["name"] for c in columns]

            taskdata_hs = generate_taskdata_hs(columns, rows)
            hidden_taskdata_hs = generate_hidden_taskdata_hs(columns, hidden_rows)

            _write_lazyppl_task_dir(
                task_dir=task_dir,
                seed_name=seed_name,
                seed_source=seed_source,
                taskdata_hs=taskdata_hs,
                hidden_taskdata_hs=hidden_taskdata_hs,
                col_names=col_names,
                docker_app_dir=docker_app_dir,
                docker_file=docker_file,
                lazyppl_local_dir=lazyppl_local_dir,
                task_toml_tmpl=task_toml_tmpl,
                instruction_tmpl=instruction_tmpl,
                test_sh_tmpl=test_sh_tmpl,
                runtime_spec=lazyppl_runtime_spec,
            )
        except (OSError, jinja2.TemplateError, ValueError, KeyError, TypeError) as exc:
            if task_dir.exists():
                shutil.rmtree(task_dir)
            skipped.append({"seed": seed_name, "reason": f"export failed: {exc}"})
            log.warning("skipped %s: %s", seed_name, exc)
            continue

        created_details.append(
            {
                "seed": seed_name,
                "task_id": task_id,
                "task_dir": str(task_dir),
                "visible_row_count": len(rows),
                "hidden_row_count": len(hidden_rows),
                "columns": col_names,
                "taskdata_sha256": _sha256_text(taskdata_hs),
                "hidden_taskdata_sha256": _sha256_text(hidden_taskdata_hs),
                "taskdata_source": "provided_visible_rows",
                "hidden_taskdata_source": "provided_hidden_rows",
            }
        )
        created += 1

    return {
        "tasks_created": created,
        "tasks_skipped": len(skipped),
        "skipped_details": skipped,
        "created_details": created_details,
    }


def _read_data_rows(parquet_path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    return [{str(key): value for key, value in row.items()} for row in df.to_dict(orient="records")]


def _complexity_value(spec: EnvSpec | FourFamilySpec | None) -> str:
    if spec is None:
        return "medium"
    # complexity is an EnvSpec field; a FourFamilySpec has none, and reading it
    # raises exactly this error rather than guessing a default.
    if isinstance(spec, FourFamilySpec):
        raise AttributeError(
            f"'{type(spec).__name__}' object has no attribute 'complexity'"
        )
    value = spec.complexity
    return str(getattr(value, "value", value))


def export_lazyppl_harbor_tasks_from_run(
    run_dir: Path,
    out_dir: Path,
    run_id: str,
    *,
    docker_app_dir: Path | None = None,
    docker_file: Path | None = None,
    lazyppl_local_dir: Path | None = None,
    require_hidden: bool = True,
    limit: int | None = None,
    min_exported: int | None = None,
) -> dict[str, Any]:
    """Export validated LazyPPL completions from a pipeline run to Harbor tasks."""
    from ppl_synth_data.harbor.taskdata_gen import (
        generate_hidden_taskdata_hs,
        generate_taskdata_hs,
    )

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if min_exported is not None and min_exported <= 0:
        raise ValueError("min_exported must be positive when provided")

    project_root = Path(__file__).resolve().parents[3]
    if docker_app_dir is None:
        docker_app_dir = project_root / "docker" / "lazyppl-harbor" / "app"
    if docker_file is None:
        docker_file = project_root / "docker" / "lazyppl-harbor" / "Dockerfile"
    if lazyppl_local_dir is None:
        lazyppl_local_dir = project_root / "docker" / "lazyppl-harbor" / "lazyppl-local"

    stage_dirs = {stage: run_dir / dirname for stage, dirname in STAGE_DIR_NAMES.items()}
    specs = read_envspecs(stage_dirs["A"] / "envspecs.jsonl")
    spec_by_id: dict[str, EnvSpec | FourFamilySpec] = {spec.env_id: spec for spec in specs}
    data_refs = _read_data_refs(run_dir)
    validated_raw = read_jsonl(stage_dirs["F"] / "validated.jsonl")
    validated = [
        normalize_completion_record(CompletionRecord.model_validate(row)) for row in validated_raw
    ]

    jinja_env = _load_jinja_env()
    task_toml_tmpl = jinja_env.get_template("task_lazyppl.toml.j2")
    instruction_tmpl = jinja_env.get_template("instruction_lazyppl.md.j2")
    test_sh_tmpl = jinja_env.get_template("test_lazyppl.sh.j2")
    lazyppl_runtime_spec = get_runtime_spec("lazyppl")

    created = 0
    skipped: list[dict[str, str]] = []
    created_details: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()

    for rec in validated:
        if limit is not None and created >= limit:
            break
        env_id = rec.env_id
        if not env_id:
            skipped.append({"id": rec.id, "reason": "missing env_id"})
            continue
        if _completion_runtime_name(rec) != "lazyppl":
            skipped.append({"id": rec.id, "reason": "unsupported runtime for LazyPPL export"})
            continue
        stale_reason = lazyppl_semantic_evidence_failure(rec)
        if stale_reason is not None:
            skipped.append(
                {
                    "id": rec.id,
                    "reason": f"missing LazyPPL semantic evidence: {stale_reason}",
                }
            )
            continue
        code = rec.parsed_program or _strip_markdown_fences(rec.raw_completion)
        if not code.strip():
            skipped.append({"id": rec.id, "reason": "empty completion"})
            continue
        ref = data_refs.get(env_id)
        if ref is None:
            skipped.append({"id": rec.id, "reason": f"missing Stage C data ref for {env_id}"})
            continue
        visible_path = _resolve_run_relative_path(run_dir, ref.get("data_ref"))
        if visible_path is None or not visible_path.exists():
            skipped.append(
                {
                    "id": rec.id,
                    "reason": f"missing visible data parquet for {env_id}",
                }
            )
            continue
        hidden_path = _resolve_run_relative_path(run_dir, ref.get("hidden_data_ref"))
        if require_hidden and (hidden_path is None or not hidden_path.exists()):
            skipped.append(
                {
                    "id": rec.id,
                    "reason": f"missing hidden data parquet for {env_id}",
                }
            )
            continue

        task_id = _build_lazyppl_run_task_id(env_id, rec.id)
        if task_id in seen_task_ids:
            skipped.append({"id": rec.id, "reason": f"duplicate task_id: {task_id}"})
            continue
        seen_task_ids.add(task_id)
        task_dir = out_dir / task_id

        try:
            rows = _read_data_rows(visible_path)
            if not rows:
                raise ValueError(f"visible data rows for {env_id} must not be empty")
            hidden_rows = _read_data_rows(hidden_path) if hidden_path is not None else []
            columns = _lazyppl_columns_for_rows(
                code,
                rows=rows,
                hidden_rows=hidden_rows,
            )
            taskdata_hs = generate_taskdata_hs(columns, rows)
            hidden_taskdata_hs = generate_hidden_taskdata_hs(columns, hidden_rows)
            spec = spec_by_id.get(env_id)
            _write_lazyppl_task_dir(
                task_dir=task_dir,
                seed_name=env_id,
                seed_source=code,
                taskdata_hs=taskdata_hs,
                hidden_taskdata_hs=hidden_taskdata_hs,
                col_names=[column["name"] for column in columns],
                docker_app_dir=docker_app_dir,
                docker_file=docker_file,
                lazyppl_local_dir=lazyppl_local_dir,
                task_toml_tmpl=task_toml_tmpl,
                instruction_tmpl=instruction_tmpl,
                test_sh_tmpl=test_sh_tmpl,
                runtime_spec=lazyppl_runtime_spec,
                env_name=spec.env_name if spec is not None else env_id,
                prompt_text=rec.prompt_text,
                domain=spec.domain if spec is not None else "probabilistic-programming",
                complexity=_complexity_value(spec),
            )
        except (OSError, jinja2.TemplateError, ValueError, KeyError, TypeError) as exc:
            if task_dir.exists():
                shutil.rmtree(task_dir)
            skipped.append({"id": rec.id, "reason": f"export failed: {exc}"})
            log.warning("skipped %s: %s", rec.id, exc)
            continue
        created_details.append(
            {
                "id": rec.id,
                "env_id": env_id,
                "task_id": task_id,
                "task_dir": str(task_dir),
                "visible_data_ref": str(visible_path),
                "hidden_data_ref": str(hidden_path) if hidden_path is not None else None,
                "visible_row_count": len(rows),
                "hidden_row_count": len(hidden_rows),
                "columns": [column["name"] for column in columns],
                "taskdata_sha256": _sha256_text(taskdata_hs),
                "hidden_taskdata_sha256": _sha256_text(hidden_taskdata_hs),
                "taskdata_source": "stage_c_visible_parquet",
                "hidden_taskdata_source": (
                    "stage_c_hidden_parquet" if hidden_path is not None else "empty_hidden_rows"
                ),
            }
        )
        created += 1

    if min_exported is not None and created < min_exported:
        raise ValueError(
            f"LazyPPL Harbor export produced {created} task(s), "
            f"below required minimum {min_exported}"
        )

    return {
        "run_id": run_id,
        "tasks_created": created,
        "tasks_skipped": len(skipped),
        "skipped_details": skipped,
        "created_details": created_details,
    }


def _write_lazyppl_task_dir(
    *,
    task_dir: Path,
    seed_name: str,
    seed_source: str,
    taskdata_hs: str,
    hidden_taskdata_hs: str,
    col_names: list[str],
    docker_app_dir: Path,
    docker_file: Path,
    lazyppl_local_dir: Path,
    task_toml_tmpl: jinja2.Template,
    instruction_tmpl: jinja2.Template,
    test_sh_tmpl: jinja2.Template,
    runtime_spec: Any,
    env_name: str | None = None,
    prompt_text: str | None = None,
    domain: str = "probabilistic-programming",
    complexity: str = "medium",
) -> None:
    env_dir = task_dir / "environment"
    workspace = env_dir / "workspace"
    app_dir = env_dir / "app"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"

    for d in (workspace, app_dir, tests_dir, solution_dir):
        d.mkdir(parents=True, exist_ok=True)

    resolved_env_name = env_name or seed_name.replace("_", " ").title()
    resolved_prompt_text = prompt_text or _lazyppl_prompt_text(seed_name)

    atomic_write_text(
        task_dir / "task.toml",
        task_toml_tmpl.render(
            domain=domain,
            complexity=complexity,
            verifier_timeout_sec=float(runtime_spec.harbor_verifier_timeout_sec),
            agent_timeout_sec=float(runtime_spec.harbor_agent_timeout_sec),
        ),
    )
    atomic_write_text(
        task_dir / "instruction.md",
        instruction_tmpl.render(
            env_name=resolved_env_name,
            prompt_text=resolved_prompt_text,
            data_columns=col_names,
            n_draws=_LAZYPPL_DEFAULT_N_DRAWS,
        ),
    )

    if not docker_file.is_file():
        raise FileNotFoundError(f"Dockerfile not found: {docker_file}")
    if not lazyppl_local_dir.is_dir():
        raise FileNotFoundError(f"required lazyppl-local dir not found: {lazyppl_local_dir}")

    base_dockerfile = docker_file.read_text(encoding="utf-8").strip()
    atomic_write_text(
        env_dir / "Dockerfile",
        base_dockerfile + "\n\nCOPY workspace/ /app/\n",
    )
    jinja_env = _load_jinja_env()
    atomic_write_text(
        tests_dir / "Dockerfile",
        jinja_env.get_template("verifier_lazyppl.Dockerfile.j2").render(
            base_dockerfile=base_dockerfile,
        ),
    )
    atomic_write_text(
        tests_dir / "docker-compose.yaml",
        jinja_env.get_template("verifier.docker-compose.yaml.j2").render(),
    )
    for context_dir in (env_dir, tests_dir):
        (context_dir / "app").mkdir(exist_ok=True)
        task_library = context_dir / "lazyppl-local"
        if task_library.exists():
            shutil.rmtree(task_library)
        shutil.copytree(lazyppl_local_dir, task_library)
        for name in ("requirements.lock", "cabal.project"):
            pin_file = docker_file.parent / name
            if pin_file.is_file():
                shutil.copy2(pin_file, context_dir / name)
            elif name in base_dockerfile:
                raise FileNotFoundError(f"required runtime pin file not found: {pin_file}")

    for fname in _LAZYPPL_DOCKER_APP_FILES:
        src = docker_app_dir / fname
        if src.exists():
            shutil.copy2(src, app_dir / fname)
            shutil.copy2(src, workspace / fname)
            shutil.copy2(src, tests_dir / "app" / fname)
        else:
            raise FileNotFoundError(f"required app file not found: {src}")

    # The held-out module goes ONLY under tests/, which Harbor mounts read-only
    # at verify time. environment/workspace/ is COPYed into the agent-visible
    # image by the Dockerfile, so nothing hidden may be written there (same
    # clause as the PyMC lane's holdout.parquet and elpd_reference.json).
    atomic_write_text(workspace / "TaskData.hs", taskdata_hs)
    atomic_write_text(tests_dir / "TaskData.hs", taskdata_hs)
    atomic_write_text(tests_dir / "HiddenTaskData.hs", hidden_taskdata_hs)

    test_sh_content = test_sh_tmpl.render(static_safety_ere=static_safety_grep_pattern())
    atomic_write_text(tests_dir / "test.sh", test_sh_content)
    (tests_dir / "test.sh").chmod(0o755)

    atomic_write_text(solution_dir / "Submission.hs", seed_source)

    templates_pkg = resources.files("ppl_synth_data").joinpath("harbor/templates")
    with resources.as_file(templates_pkg.joinpath("solve_lazyppl.sh")) as src:
        shutil.copy2(src, solution_dir / "solve.sh")
    (solution_dir / "solve.sh").chmod(0o755)

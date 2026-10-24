"""Bundle exported Harbor task packages into a self-contained SkyRL bundle."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any

from ppl_synth_data.harbor.export_shared import _project_root, _sha256_file, _sha256_tree
from ppl_synth_data.pipeline.io import atomic_write_text
from ppl_synth_data.runtime_registry import default_runtime_name, normalize_runtime_name

_SKYRL_BUNDLE_SCHEMA_VERSION = "ppl_synth_data.harbor.skyrl_bundle.v1"
_SKYRL_BUNDLE_AGENT_MODULE = "ppl_synth_data_harbor_agent"
_SKYRL_BUNDLE_AGENT_IMPORT_PATH = f"{_SKYRL_BUNDLE_AGENT_MODULE}:SynthStatsInstalledAgent"

# `harbor/agent.py` is decomposed into these sibling concern modules (see their
# own docstrings). Inside the installed `ppl_synth_data` package they are
# imported by dotted path (`from ppl_synth_data.harbor.agent_env import ...`);
# the bundle vendors all of them into a single flat, package-free `agent/`
# directory (Harbor's `agent_import_path` loads them with no `ppl_synth_data`
# installed), so their cross-imports must be flattened to bare top-level names.
_AGENT_SUPPORT_MODULES = (
    "agent_shared",
    "agent_env",
    "agent_extraction",
    "agent_openai",
    "agent_tokenizer",
    "agent_metadata",
    "agent_eos",
    "agent_atif",
)
_AGENT_DOTTED_IMPORT_RE = re.compile(r"from ppl_synth_data\.harbor\.(agent_\w+) import")


def _flatten_bundled_agent_imports(source: str) -> str:
    """Rewrite intra-package agent imports for the flat, unpackaged bundle copy.

    The bundle's ``agent/`` directory holds the entry module plus its support
    modules as flat top-level files with no ``ppl_synth_data`` package on
    ``sys.path``, so ``from ppl_synth_data.harbor.agent_env import ...``
    becomes ``from agent_env import ...``. Only ``agent_*`` sibling imports are
    rewritten; nothing else in the source is touched.
    """

    return _AGENT_DOTTED_IMPORT_RE.sub(r"from \1 import", source)


def _producer_git_commit() -> str | None:
    project_root = _project_root()
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


def _task_dirs(task_root: Path) -> list[Path]:
    if not task_root.is_dir():
        raise FileNotFoundError(f"Harbor task root does not exist: {task_root}")
    dirs = [path for path in sorted(task_root.iterdir()) if path.is_dir()]
    if not dirs:
        raise ValueError(f"Harbor task root contains no task directories: {task_root}")
    for task_dir in dirs:
        for rel in ("task.toml", "environment/Dockerfile", "tests/test.sh"):
            required = task_dir / rel
            if not required.is_file():
                raise ValueError(f"Harbor task directory {task_dir} is missing {rel}")
    return dirs


def _copy_task_dirs(task_dirs: list[Path], bundle_tasks_dir: Path) -> list[dict[str, str]]:
    bundle_tasks_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()
    for source_task_dir in task_dirs:
        task_id = source_task_dir.name
        if "/" in task_id or "\\" in task_id or task_id in {"", ".", ".."}:
            raise ValueError(f"unsafe Harbor task directory name: {task_id!r}")
        if task_id in seen:
            raise ValueError(f"duplicate Harbor task directory name: {task_id}")
        seen.add(task_id)
        destination = bundle_tasks_dir / task_id
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_task_dir, destination, symlinks=False)
        digest = _sha256_tree(destination)
        tasks.append(
            {
                "task_id": task_id,
                "relative_path": f"tasks/{task_id}",
                "digest_sha256": digest,
                "digest_scope": "recursive_file_sha256_v1",
            }
        )
    return tasks


def _harbor_task_row(
    *,
    task_id: str,
    relative_task_path: str,
    runtime: str,
    env_name: str,
    max_steps: int,
    digest_sha256: str,
) -> dict[str, Any]:
    return {
        "prompt": [
            {
                "role": "system",
                "content": (
                    "You are a probabilistic-program synthesis agent. "
                    "Solve the Harbor task using the structured task metadata "
                    "and submit the final probabilistic program."
                ),
            },
            {
                "role": "user",
                "content": (f"Solve {task_id} from Harbor task package {relative_task_path}."),
            },
        ],
        "env_class": "synthstats-harbor-v0",
        "task_name": task_id,
        "task_path": relative_task_path,
        "task_config": {
            "runtime": runtime,
            "env_name": env_name,
            "max_steps": max_steps,
        },
        "source": "ppl-synth-data.harbor.skyrl_bundle",
        "task_package_digest": digest_sha256,
        "task_package_digest_scope": "recursive_file_sha256_v1",
    }


def _solution_for_task(task_dir: Path) -> tuple[str, str]:
    candidates = (
        (task_dir / "solution" / "model.py", "python"),
        (task_dir / "solution" / "Submission.hs", "haskell"),
    )
    for path, fence in candidates:
        if path.is_file():
            solution = path.read_text(encoding="utf-8").strip()
            if solution:
                return solution, fence
    raise ValueError(f"Harbor task bundle SFT rows require a gold solution in {task_dir}")


def _sft_rows_for_tasks(
    *,
    bundle_tasks_dir: Path,
    tasks: list[dict[str, str]],
    runtime: str,
    env_name: str,
    max_steps: int,
) -> list[dict[str, Any]]:
    from ppl_synth_data.harbor.agent import build_messages

    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["task_id"]
        relative_task_path = task["relative_path"]
        task_dir = bundle_tasks_dir / task_id
        solution, fence = _solution_for_task(task_dir)
        completion = f"```{fence}\n{solution}\n```"
        task_configs: list[dict[str, Any]] = [
            {"runtime": runtime, "env_name": env_name, "max_steps": max_steps},
            {"runtime": runtime, "max_steps": max_steps},
            {"runtime": runtime},
        ]
        instruction = f"Solve {task_id} from Harbor task package {relative_task_path}."
        for index, task_config in enumerate(task_configs):
            rows.append(
                {
                    "prompt": json.dumps(
                        build_messages(instruction, task_config),
                        sort_keys=True,
                    ),
                    "completion": completion,
                    "program": solution,
                    "runtime": runtime,
                    "task_name": task_id,
                    "task_path": relative_task_path,
                    "source": (
                        "ppl-synth-data.harbor.skyrl_bundle.gold_solution_variant_"
                        f"{task_id}_{index}"
                    ),
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def export_harbor_skyrl_bundle(
    *,
    task_root: Path,
    out_dir: Path,
    runtime: str,
    env_name: str,
    max_steps: int = 20,
    run_id: str | None = None,
    min_tasks: int | None = None,
    include_sft_rows: bool = True,
) -> dict[str, Any]:
    """Export Harbor tasks as a self-contained SynthStats/SkyRL bundle.

    The bundle is the producer/trainer interface for live online-train runs. It
    carries copied task directories, a standalone Harbor agent module, relative
    SkyRL task rows, optional gold-solution SFT rows, and a manifest with task
    digests. Consumers should mount the bundle and expand relative paths to the
    mounted bundle root; they should not install or import ``ppl-synth-data``.
    """

    normalized_runtime = normalize_runtime_name(runtime, default=default_runtime_name())
    if not normalized_runtime:
        raise ValueError(f"unsupported runtime: {runtime!r}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if min_tasks is not None and min_tasks <= 0:
        raise ValueError("min_tasks must be positive when provided")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty Harbor bundle directory: {out_dir}")

    task_dirs = _task_dirs(task_root)
    if min_tasks is not None and len(task_dirs) < min_tasks:
        raise ValueError(
            f"Harbor task root contains {len(task_dirs)} task(s), below required "
            f"minimum {min_tasks}"
        )

    bundle_tasks_dir = out_dir / "tasks"
    bundle_agent_dir = out_dir / "agent"
    bundle_agent_dir.mkdir(parents=True, exist_ok=True)
    tasks = _copy_task_dirs(task_dirs, bundle_tasks_dir)

    agent_src = resources.files("ppl_synth_data").joinpath("harbor/agent.py")
    with resources.as_file(agent_src) as src:
        entry_source = _flatten_bundled_agent_imports(src.read_text(encoding="utf-8"))
    atomic_write_text(bundle_agent_dir / f"{_SKYRL_BUNDLE_AGENT_MODULE}.py", entry_source)

    for support_module in _AGENT_SUPPORT_MODULES:
        support_src = resources.files("ppl_synth_data").joinpath(f"harbor/{support_module}.py")
        with resources.as_file(support_src) as src:
            support_source = _flatten_bundled_agent_imports(src.read_text(encoding="utf-8"))
        atomic_write_text(bundle_agent_dir / f"{support_module}.py", support_source)

    atomic_write_text(bundle_agent_dir / "__init__.py", "")

    task_rows = [
        _harbor_task_row(
            task_id=task["task_id"],
            relative_task_path=task["relative_path"],
            runtime=normalized_runtime,
            env_name=env_name,
            max_steps=max_steps,
            digest_sha256=task["digest_sha256"],
        )
        for task in tasks
    ]
    _write_jsonl(out_dir / "task_rows.jsonl", task_rows)

    sft_rows_path: str | None = None
    if include_sft_rows:
        sft_rows = _sft_rows_for_tasks(
            bundle_tasks_dir=bundle_tasks_dir,
            tasks=tasks,
            runtime=normalized_runtime,
            env_name=env_name,
            max_steps=max_steps,
        )
        _write_jsonl(out_dir / "sft_rows.jsonl", sft_rows)
        sft_rows_path = "sft_rows.jsonl"

    manifest = {
        "schema_version": _SKYRL_BUNDLE_SCHEMA_VERSION,
        "producer": "ppl-synth-data",
        "bundle_type": "harbor_skyrl_online_train",
        "run_id": run_id,
        "producer_git_commit": _producer_git_commit(),
        "task_count": len(tasks),
        "task_rows_path": "task_rows.jsonl",
        "task_path_root": "tasks",
        "task_path_semantics": "relative_to_bundle_root",
        "sft_rows_path": sft_rows_path,
        "agent_import_path": _SKYRL_BUNDLE_AGENT_IMPORT_PATH,
        "agent_module_path": f"agent/{_SKYRL_BUNDLE_AGENT_MODULE}.py",
        "agent_module_sha256": _sha256_file(bundle_agent_dir / f"{_SKYRL_BUNDLE_AGENT_MODULE}.py"),
        "agent_support_module_paths": [
            f"agent/{support_module}.py" for support_module in _AGENT_SUPPORT_MODULES
        ],
        "agent_bundle_sha256": _sha256_tree(bundle_agent_dir),
        "task_config_defaults": {
            "runtime": normalized_runtime,
            "env_name": env_name,
            "max_steps": max_steps,
        },
        "tasks": tasks,
    }
    atomic_write_text(
        out_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return {
        "schema_version": _SKYRL_BUNDLE_SCHEMA_VERSION,
        "bundle_dir": str(out_dir),
        "manifest_path": str(out_dir / "manifest.json"),
        "task_rows_path": str(out_dir / "task_rows.jsonl"),
        "sft_rows_path": str(out_dir / "sft_rows.jsonl") if sft_rows_path else None,
        "agent_import_path": _SKYRL_BUNDLE_AGENT_IMPORT_PATH,
        "task_count": len(tasks),
        "task_digest_scope": "recursive_file_sha256_v1",
    }

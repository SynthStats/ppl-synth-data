"""An exported task carries its own verifier environment: no image tag from another machine."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

from ppl_synth_data.harbor.export_pymc import export_harbor_tasks
from ppl_synth_data.runtime_registry import get_runtime_spec
from tests.test_harbor_export import _single_exported_task_dir, run_dir  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def _exported_task(run_dir: Path, tmp_path: Path) -> Path:  # noqa: F811
    out = tmp_path / "harbor_out"
    export_harbor_tasks(
        run_dir,
        out,
        run_id="test-run",
        l6_draws=15,
        l6_tune=15,
        allow_unlocked_reference_runtime=True,
        attempt_id="test-attempt",
    )
    return _single_exported_task_dir(out)


def test_verifier_dockerfile_is_self_contained(run_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    tests_dir = _exported_task(run_dir, tmp_path) / "tests"
    dockerfile = (tests_dir / "Dockerfile").read_text(encoding="utf-8")

    from_lines = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert len(from_lines) == 1
    assert re.fullmatch(r"FROM python:[\w.\-]+@sha256:[0-9a-f]{64}", from_lines[0])
    assert "local-build" not in dockerfile
    assert "pymc-harbor" not in dockerfile
    # The legacy Docker builder must be able to build it.
    assert "--mount=" not in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--no-deps" in dockerfile
    lock = ROOT / "docker" / "pymc-harbor" / "requirements.lock"
    assert (tests_dir / "requirements.lock").read_bytes() == lock.read_bytes()
    for name in ("test.sh", "validator.py", "holdout.parquet", "elpd_reference.json"):
        assert name in dockerfile
        assert (tests_dir / name).is_file()


def test_verifier_runs_separately_and_offline(run_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    task_dir = _exported_task(run_dir, tmp_path)
    task = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))

    assert task["verifier"]["environment_mode"] == "separate"
    assert task["verifier"]["network_mode"] == "no-network"
    assert task["environment"]["network_mode"] == "no-network"
    assert "local-build" not in (task_dir / "task.toml").read_text(encoding="utf-8")
    visible = task_dir / "environment" / "workspace" / "data.parquet"
    assert (task_dir / "tests" / "data.parquet").read_bytes() == visible.read_bytes()
    compose_text = (task_dir / "tests" / "docker-compose.yaml").read_text(encoding="utf-8")
    main = yaml.safe_load(compose_text)["services"]["main"]
    assert "SYS_ADMIN" in main["cap_add"]
    assert "apparmor=unconfined" in main["security_opt"]


def test_runtime_registry_names_no_local_image_tag() -> None:
    assert "local-build" not in str(get_runtime_spec("pymc").harbor_docker_image)

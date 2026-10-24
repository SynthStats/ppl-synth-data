"""A reference is computed by the runtime that scores submissions.

Measured on one task: the same gold program gave a held-out ELPD of -31.0546 when the export ran
in the development environment and -31.3349 in the scorer image, so the gold did not score at
the top of its own scale. The reference fit therefore runs with the export's own interpreter,
and an export refuses to run in an environment whose installed distributions differ from the
lock it ships, unless the caller allows it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pytest

from ppl_synth_data.harbor import export_pymc

ROOT = Path(__file__).resolve().parents[1]
GOLD = (
    "def model(data):\n"
    "    y = np.asarray(data['y'])\n"
    "    with pm.Model() as m:\n"
    "        pm.Normal('obs', 0, 1, observed=y)\n"
    "    return m\n"
)
ROW_ID = "aaaa000000000001"
IMAGE = "sha256:" + "ab" * 32


def _rows() -> list[dict]:
    return [
        {
            "row_id": ROW_ID,
            "runtime": "pymc",
            "prompt": "Scenario",
            "structural_class": "mixture",
            "visible_data": {"y": [0.5, 1.5, 2.25]},
            "heldout_data": {"y": [3.0, -1.0]},
        }
    ]


def _lock(pins: dict[str, str]) -> str:
    return "".join(
        f"{name}=={version} \\\n    --hash=sha256:{'0' * 64}\n" for name, version in pins.items()
    )


@pytest.fixture
def validator(tmp_path):
    path = tmp_path / "validator-pinned.py"
    path.write_text("# pinned validator bytes\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def recorded_fits(monkeypatch):
    calls = []

    def fake_reference(
        code,
        visible,
        hidden,
        *,
        env_id,
        target_name,
        blanks,
        expected_validator_sha256,
        attempt_id,
        l6_draws=None,
        l6_tune=None,
    ):
        calls.append(env_id)
        return {
            "gold_elpd": -1.0,
            "baseline_elpd": -2.0,
            "timing_s": {"gold": 0.1, "baseline": 0.1},
            "attempt_id": attempt_id,
        }

    monkeypatch.setattr(export_pymc, "_compute_elpd_reference", fake_reference)
    return calls


@pytest.fixture
def project_with_lock(tmp_path, monkeypatch):
    """A project root whose scorer recipe carries the given lock, so a test controls the
    comparison.
    """

    def install(pins: dict[str, str]) -> str:
        recipe = tmp_path / "project" / "docker" / "pymc-harbor"
        recipe.mkdir(parents=True)
        text = _lock(pins)
        (recipe / "requirements.lock").write_text(text)
        (recipe / "Dockerfile").write_text(
            (ROOT / "docker" / "pymc-harbor" / "Dockerfile").read_text()
        )
        monkeypatch.setattr(export_pymc, "_project_root", lambda: tmp_path / "project")
        return text

    return install


def _fixture_role_manifest(tmp_path, rows):
    """A fixture manifest for these synthetic rows."""
    import hashlib

    packages = []
    for row in rows:
        visible = row["visible_data"]
        columns = []
        for name in sorted(set(visible) | {"n"}):
            if name == "y":
                columns.append(
                    {
                        "column": name,
                        "role": "TARGET",
                        "treatment": "preserve",
                        "reducer_type": None,
                    }
                )
            elif name == "n":
                columns.append(
                    {
                        "column": name,
                        "role": "UNUSED",
                        "treatment": "preserve (scalar row count, never replaced)",
                        "reducer_type": None,
                    }
                )
            else:
                columns.append(
                    {
                        "column": name,
                        "role": "PREDICTOR",
                        "treatment": "replace",
                        "reducer_type": "quantitative",
                    }
                )
        packages.append(
            {"package_id": row["row_id"], "target": "y", "columns": columns, "status": (
                'complete'
            )}
        )
    manifest = {
        "packages": packages,
        "package_count": len(packages),
        "complete": len(packages),
        "unresolved": 0,
    }
    path = tmp_path / "ROLE-MANIFEST.json"
    path.write_text(json.dumps(manifest))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _supply_a_fixture_manifest(tmp_path, monkeypatch):
    """Every call here gets a fixture manifest when it omits one."""
    real = export_pymc.export_harbor_tasks_from_rows

    def wrapped(
        rows, golds, out_dir, *, role_manifest_path=None, role_manifest_sha256=None, **kwargs
    ):
        if role_manifest_path is None:
            role_manifest_path, role_manifest_sha256 = _fixture_role_manifest(tmp_path, rows)
        return real(
            rows,
            golds,
            out_dir,
            role_manifest_path=role_manifest_path,
            role_manifest_sha256=role_manifest_sha256,
            **kwargs,
        )

    monkeypatch.setattr(export_pymc, "export_harbor_tasks_from_rows", wrapped)


def _export(tmp_path, validator, **kwargs):
    path, sha = validator
    return export_pymc.export_harbor_tasks_from_rows(
        _rows(),
        {ROW_ID: GOLD},
        tmp_path / "tasks",
        validator_path=path,
        validator_sha256=sha,
        excluded_rows=(),
        **kwargs,
        attempt_id="test-attempt",
    )


def test_reference_subprocess_uses_the_exports_own_interpreter(monkeypatch):
    seen = {}

    def stop(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        raise RuntimeError("stopped before the fit")

    monkeypatch.setattr(
        "ppl_synth_data.validation.subprocess_utils.run_with_process_group_timeout", stop
    )
    with pytest.raises(RuntimeError, match="stopped before the fit"):
        export_pymc._run_gold_subprocess({"code": GOLD, "data": {"y": [0.0]}}, timeout=5)

    assert seen["cmd"] == [sys.executable, "-m", (
                'ppl_synth_data.harbor._validate_gold_subproce'
                'ss'
            )]


def test_export_refuses_a_runtime_that_differs_from_the_lock(
    tmp_path, validator, recorded_fits, project_with_lock
):
    project_with_lock({"numpy": importlib.metadata.version("numpy"), "pymc": "0.0.1"})

    with pytest.raises(RuntimeError) as refusal:
        _export(tmp_path, validator)

    message = str(refusal.value)
    assert "requirements.lock" in message
    assert "pymc" in message
    assert "0.0.1" in message
    assert importlib.metadata.version("pymc") in message
    assert "numpy" not in message
    assert recorded_fits == []
    assert not list((tmp_path / "tasks").glob("*/task.toml"))


def test_an_allowed_unlocked_runtime_is_recorded_as_unlocked(
    tmp_path, validator, recorded_fits, project_with_lock
):
    lock = project_with_lock(
        {
            "not-a-real-distribution": "1.0",
            "numpy": importlib.metadata.version("numpy"),
            "pymc": "0.0.1",
        }
    )
    _, sha = validator

    summary = _export(
        tmp_path, validator, allow_unlocked_reference_runtime=True, scorer_image=IMAGE
    )

    assert summary == json.loads((tmp_path / "tasks" / "export_summary.json").read_text())
    assert summary["tasks_created"] == 1
    assert summary["validator_sha256"] == sha
    assert summary["lock_sha256"] == hashlib.sha256(lock.encode()).hexdigest()
    assert summary["scorer_image"] == IMAGE
    assert summary["reference_runtime"] == {
        "python": platform.python_version(),
        "matches_lock": False,
        "mismatches": [
            {"name": "not-a-real-distribution", "locked": "1.0", "installed": None},
            {"name": "pymc", "locked": "0.0.1", "installed": importlib.metadata.version("pymc")},
        ],
    }


def test_a_runtime_equal_to_the_lock_needs_no_allowance(
    tmp_path, validator, recorded_fits, project_with_lock
):
    # The lock spells a name its own way; the comparison is on normalized names.
    project_with_lock(
        {
            "numpy": importlib.metadata.version("numpy"),
            "PyYAML": importlib.metadata.version("pyyaml"),
            "pydantic_core": importlib.metadata.version("pydantic-core"),
        }
    )

    summary = _export(tmp_path, validator)

    assert summary["tasks_created"] == 1
    assert summary["scorer_image"] is None
    assert summary["reference_runtime"]["matches_lock"] is True
    assert summary["reference_runtime"]["mismatches"] == []
    assert recorded_fits == [ROW_ID]

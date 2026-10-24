"""Re-exporting an existing task package: same text, data and gold; current verifier and a fresh
reference.

An existing package already fixes what the task is. A re-export keeps its instruction, its
visible and held-out tables and its gold program byte for byte, takes the target column from the
package's own reference file, and replaces only what the exporter owns: the verifier, the
reference, the task configuration and the two image recipes. The reference fit is replaced by a
recorder in these unit laws.
"""

from __future__ import annotations

import hashlib
import hashlib as _h
import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from ppl_synth_data.harbor import export_pymc

TASK_ID = "pymc-api-latency-regression-00045"
TARGET = "response_time_ms"
GOLD = (
    "def model(data):\n"
    "    y = np.asarray(data['response_time_ms'])\n"
    "    with pm.Model() as m:\n"
    "        pm.Normal('obs', 0, 1, observed=y)\n"
    "    return m\n"
)
INSTRUCTION = "# Task: api latency\n\nThe table holds 3 requests. Mean latency is 2.0 ms.\n"
OLD_TOML = (
    'version = "1.0"\n\n[metadata]\ndifficulty = "simple"\ncategory = "probabilistic-programming"\n'
    'tags = ["pymc", "probabilistic-programming", "technology"]\n\n[environment]\n'
    'docker_image = "pymc-harbor:local-build-v1"\n'
)
FILES = {
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "environment/requirements.lock",
    "environment/workspace/data.parquet",
    "environment/workspace/submission.py",
    "tests/Dockerfile",
    "tests/docker-compose.yaml",
    "tests/requirements.lock",
    "tests/data.parquet",
    "tests/holdout.parquet",
    "tests/elpd_reference.json",
    "tests/test.sh",
    "tests/validator.py",
    "solution/model.py",
    "solution/solve.sh",
}


def _package(root: Path, task_id: str = TASK_ID, *, reference: dict | None = None) -> Path:
    task = root / task_id
    (task / "environment" / "workspace").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "instruction.md").write_text(INSTRUCTION)
    (task / "task.toml").write_text(OLD_TOML)
    pd.DataFrame({TARGET: [1.0, 2.0, 3.0], "load": [0.1, 0.2, 0.3]}).to_parquet(
        task / "environment" / "workspace" / "data.parquet", index=False
    )
    pd.DataFrame({TARGET: [4.0, 5.0], "load": [0.4, 0.5]}).to_parquet(
        task / "tests" / "holdout.parquet", index=False
    )
    if reference is None:
        reference = {"gold_elpd": -9.0, "baseline_elpd": -12.0, "target_name": TARGET}
    (task / "tests" / "elpd_reference.json").write_text(json.dumps(reference))
    (task / "tests" / "validator.py").write_text("# an older verifier\n")
    (task / "solution" / "model.py").write_text(GOLD)
    return task


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
        expected_validator_sha256=None,
        attempt_id=None,
        l6_draws=None,
        l6_tune=None,
    ):
        calls.append(
            {"env_id": env_id, "target_name": target_name, "blanks": sorted(blanks), "code": code}
        )
        return {
            "gold_elpd": -1.0,
            "baseline_elpd": -2.0,
            "timing_s": {"gold": 0.1, "baseline": 0.1},
        }

    monkeypatch.setattr(export_pymc, "_compute_elpd_reference", fake_reference)
    return calls




def _fixture_role_manifest(tmp_path, sources):
    """A fixture manifest for these synthetic packages.

    The columns are read from each source's own visible split, so the manifest cannot miss a column
    the completeness check would then reject. The target is preserved, the row-count key is
    preserved under the sealed wording, and every other column is a replaced quantitative predictor.
    """
    import pandas as _pd

    packages = []
    for source in sources:
        source = Path(source)
        visible = _pd.read_parquet(source / "environment" / "workspace" / "data.parquet")
        # The re-export path reads the target from the package's own reference file, so the fixture
        # manifest must agree with it rather than assume "y".
        reference = json.loads((source / "tests" / "elpd_reference.json").read_text())
        target = reference.get("target_name") or "y"
        columns = []
        # The export injects the row-count key into the data it hands the fit, so the
        # fixture manifest
        # must list it even where the parquet does not carry it. The sealed manifest
        # preserves it under
        # the same wording.
        names = sorted(set(visible.columns) | {"n"})
        for name in names:
            if name == target:
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
            {"package_id": source.name, "target": target, "columns": columns, "status": "complete"}
        )
    manifest = {
        "packages": packages,
        "package_count": len(packages),
        "complete": len(packages),
        "unresolved": 0,
    }
    path = tmp_path / "ROLE-MANIFEST.json"
    path.write_text(json.dumps(manifest))
    return path, _h.sha256(path.read_bytes()).hexdigest()


def _reexport(tmp_path, sources, validator, excluded=()):
    path, sha = validator
    manifest_path, manifest_sha = _fixture_role_manifest(tmp_path, sources)
    return export_pymc.reexport_harbor_tasks(
        sources,
        tmp_path / "new",
        validator_path=path,
        validator_sha256=sha,
        excluded=excluded,
        allow_unlocked_reference_runtime=True,
        role_manifest_path=manifest_path,
        role_manifest_sha256=manifest_sha,
        attempt_id="test-attempt",
    )


def test_text_data_and_gold_are_kept_byte_for_byte(tmp_path, validator, recorded_fits):
    source = _package(tmp_path / "old")
    _reexport(tmp_path, [source], validator)
    task = tmp_path / "new" / TASK_ID

    assert {str(p.relative_to(task)) for p in task.rglob("*") if p.is_file()} == FILES
    for name in (
        "instruction.md",
        "environment/workspace/data.parquet",
        "tests/holdout.parquet",
        "solution/model.py",
    ):
        assert (task / name).read_bytes() == (source / name).read_bytes()
    assert (task / "tests" / "data.parquet").read_bytes() == (
        source / "environment" / "workspace" / "data.parquet"
    ).read_bytes()


def test_verifier_and_reference_are_replaced(tmp_path, validator, recorded_fits):
    source = _package(tmp_path / "old")
    path, sha = validator
    summary = _reexport(tmp_path, [source], validator)
    task = tmp_path / "new" / TASK_ID

    assert (task / "tests" / "validator.py").read_bytes() == path.read_bytes()
    reference = json.loads((task / "tests" / "elpd_reference.json").read_text())
    assert reference["target_name"] == TARGET
    assert (reference["gold_elpd"], reference["baseline_elpd"]) == (-1.0, -2.0)
    assert recorded_fits == [
        {"env_id": TASK_ID, "target_name": TARGET, "blanks": ["load"], "code": GOLD}
    ]
    assert summary["tasks_created"] == 1
    assert summary["validator_sha256"] == sha
    assert summary["references"] == {
        TASK_ID: {
            "source": str(source),
            "previous": {
                "gold_elpd": -9.0,
                "baseline_elpd": -12.0,
                "validator_sha256": hashlib.sha256(b"# an older verifier\n").hexdigest(),
            },
            "current": {"gold_elpd": -1.0, "baseline_elpd": -2.0},
        }
    }
    assert summary == json.loads((tmp_path / "new" / "export_summary.json").read_text())


def test_task_configuration_keeps_the_metadata_and_names_no_image(
    tmp_path, validator, recorded_fits
):
    _reexport(tmp_path, [_package(tmp_path / "old")], validator)
    config = tomllib.loads((tmp_path / "new" / TASK_ID / "task.toml").read_text())

    assert config["metadata"]["difficulty"] == "simple"
    assert config["metadata"]["tags"] == ["pymc", "probabilistic-programming", "technology"]
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["network_mode"] == "no-network"
    assert "docker_image" not in config["environment"]
    assert "docker_image" not in config["verifier"]["environment"]


def test_held_out_rows_stay_under_tests(tmp_path, validator, recorded_fits):
    _reexport(tmp_path, [_package(tmp_path / "old")], validator)
    task = tmp_path / "new" / TASK_ID

    assert not list((task / "environment").rglob("holdout*"))
    assert not list((task / "environment").rglob("elpd_reference*"))
    assert len(pd.read_parquet(task / "environment" / "workspace" / "data.parquet")) == 3


def test_a_package_without_a_target_name_is_refused(tmp_path, validator, recorded_fits):
    source = _package(tmp_path / "old", reference={"gold_elpd": -9.0, "baseline_elpd": -12.0})

    with pytest.raises(ValueError, match=TASK_ID) as refusal:
        _reexport(tmp_path, [source], validator)

    assert "target_name" in str(refusal.value)
    assert recorded_fits == []


def test_an_excluded_task_is_listed_and_not_exported(tmp_path, validator, recorded_fits):
    kept = _package(tmp_path / "old")
    held = _package(tmp_path / "old", "pymc-held-task-00001")

    summary = _reexport(tmp_path, [kept, held], validator, excluded={"pymc-held-task-00001"})

    assert summary["excluded"] == ["pymc-held-task-00001"]
    assert summary["task_dirs"] == [TASK_ID]
    assert not (tmp_path / "new" / "pymc-held-task-00001").exists()
    assert [call["env_id"] for call in recorded_fits] == [TASK_ID]


def test_two_sources_with_one_task_name_are_refused(tmp_path, validator, recorded_fits):
    first = _package(tmp_path / "a")
    second = _package(tmp_path / "b")

    with pytest.raises(ValueError, match=TASK_ID):
        _reexport(tmp_path, [first, second], validator)

    assert recorded_fits == []

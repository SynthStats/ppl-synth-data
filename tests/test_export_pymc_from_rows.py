"""Harbor PyMC tasks from gate rows, for guarded re-scoring.

The bootstrap rows have no stage-A EnvSpec, so export_harbor_tasks cannot read them.
export_harbor_tasks_from_rows reuses
the exporter's own task writer and _compute_elpd_reference, invents no EnvSpec-only field, takes
target and features
from the data keys, pins validator.py by hash, keeps held-out data under tests/ only, and raises on
a row without
held-out data. These are narrow unit laws: the reference fit is replaced by a recorder here; the
acceptance law (the gold
scores under the pinned validator) is marked heavy_lane and runs on a Linux host.
"""

from __future__ import annotations

import hashlib
import hashlib as _hashlib
import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ppl_synth_data.harbor import export_pymc

GOLD = (
    "```python\ndef model(data):\n    y = np.asarray(data['y']"
    ")\n    with pm.Model() as m:\n        pm.Normal('obs', 0,"
    " 1, observed=y)\n    return m\n```"
)


def _row(row_id, visible, heldout, structural_class="mixture"):
    return {
        "row_id": row_id,
        "runtime": "pymc",
        "prompt": f"Scenario for {row_id}",
        "visible_data": visible,
        "heldout_data": heldout,
        "structural_class": structural_class,
    }


@pytest.fixture
def validator(tmp_path):
    path = tmp_path / "validator-pinned.py"
    path.write_text("# pinned validator bytes\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()




@pytest.fixture(autouse=True)
def _supply_a_fixture_manifest(tmp_path, monkeypatch):
    """Every call in this file gets a fixture manifest when it does not pass one.

    Production still requires it: the wrapper only fills the two arguments in when a test omits
    them,
    so a test that wants to prove the refusal can pass them explicitly.
    """
    real = export_pymc.export_harbor_tasks_from_rows
    counter = {"n": 0}

    def wrapped(
        rows, golds, out_dir, *, role_manifest_path=None, role_manifest_sha256=None, **kwargs
    ):
        if role_manifest_path is None:
            manifest_path, manifest_sha = _fixture_role_manifest(tmp_path, rows)
            role_manifest_path, role_manifest_sha256 = manifest_path, manifest_sha
        counter["n"] += 1
        return real(
            rows,
            golds,
            out_dir,
            role_manifest_path=role_manifest_path,
            role_manifest_sha256=role_manifest_sha256,
            **kwargs,
        )

    monkeypatch.setattr(export_pymc, "export_harbor_tasks_from_rows", wrapped)


def _fixture_role_manifest(tmp_path, rows):
    """A fixture manifest for these synthetic rows.

    Every non-target column of every row is a replaced quantitative predictor except the row-count
    key n and the structural column group, which the manifest preserves. Keeping group preserved is
    what lets these suites catch a regression to "blank every non-target column". Written to a
    file and passed with its own sha256, so
    the loader's check is exercised rather than bypassed.
    """
    packages = []
    for row in rows:
        visible = row["visible_data"]
        columns = []
        for name in sorted(visible):
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
            elif name == "group":
                # A fixture with a preserved predictor detects a change
                # to "blank every non-target column". One structural column stays preserved.
                columns.append(
                    {"column": name, "role": "GROUP", "treatment": "preserve", "reducer_type": None}
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
    return path, _hashlib.sha256(path.read_bytes()).hexdigest()


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


def _export(tmp_path, rows, validator, golds=None, excluded=()):
    path, sha = validator
    manifest_path, manifest_sha = _fixture_role_manifest(tmp_path, rows)
    return export_pymc.export_harbor_tasks_from_rows(
        rows,
        golds or {r["row_id"]: GOLD for r in rows},
        tmp_path / "tasks",
        role_manifest_path=manifest_path,
        role_manifest_sha256=manifest_sha,
        validator_path=path,
        validator_sha256=sha,
        excluded_rows=excluded,
        allow_unlocked_reference_runtime=True,
        attempt_id="test-attempt",
    )


def _read_parquet(path: Path) -> dict:
    return {k: list(v) for k, v in pd.read_parquet(path).to_dict(orient="list").items()}


def test_the_task_id_is_the_row_id_and_the_data_round_trips(tmp_path, validator, recorded_fits):
    rows = [_row("aaaa000000000001", {"y": [0.5, 1.5, 2.25]}, {"y": [3.0, -1.0]})]
    _export(tmp_path, rows, validator)
    task = tmp_path / "tasks" / "aaaa000000000001"
    assert _read_parquet(task / "environment" / "workspace" / "data.parquet") == {
        "y": [0.5, 1.5, 2.25]
    }
    assert _read_parquet(task / "tests" / "holdout.parquet") == {"y": [3.0, -1.0]}
    assert (task / "tests/data.parquet").read_bytes() == (
        task / "environment/workspace/data.parquet"
    ).read_bytes()
    recipe = Path(__file__).resolve().parents[1] / "docker/pymc-harbor"
    dockerfile = (task / "tests/Dockerfile").read_text()
    assert dockerfile.startswith((recipe / "Dockerfile").read_text().strip())
    assert "--mount=" not in dockerfile
    assert "COPY test.sh validator.py holdout.parquet elpd_reference.json /tests/" in dockerfile
    assert "COPY data.parquet /workspace/data.parquet" in dockerfile
    assert (task / "tests/requirements.lock").read_bytes() == (
        recipe / "requirements.lock"
    ).read_bytes()
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["artifacts"] == ["/workspace/submission.py"]
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["network_mode"] == "no-network"
    assert config["agent"]["network_mode"] == "no-network"
    assert (
        config["environment"]
        == config["verifier"]["environment"]
        == {
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "network_mode": "no-network",
        }
    )
    assert yaml.safe_load((task / "tests/docker-compose.yaml").read_text()) == {
        "services": {
            "main": {
                "cap_add": ["SYS_ADMIN"],
                "security_opt": ["apparmor=unconfined"],
            }
        },
    }


def test_held_out_data_lives_only_under_tests(tmp_path, validator, recorded_fits):
    rows = [
        _row(
            "aaaa000000000002",
            {"x": [1.0, 2.0], "y": [10.0, 20.0]},
            {"x": [3.0], "y": [987654.321]},
        )
    ]
    _export(tmp_path, rows, validator)
    env = tmp_path / "tasks" / "aaaa000000000002" / "environment"
    for f in env.rglob("*"):
        if f.is_file() and f.suffix == ".parquet":
            assert 987654.321 not in _read_parquet(f).get("y", [])
    assert not any("holdout" in f.name or "elpd_reference" in f.name for f in env.rglob("*"))


def test_validator_is_the_pinned_file(tmp_path, validator, recorded_fits):
    rows = [_row("aaaa000000000003", {"y": [1.0, 2.0]}, {"y": [3.0]})]
    _export(tmp_path, rows, validator)
    written = (tmp_path / "tasks" / "aaaa000000000003" / "tests" / "validator.py").read_bytes()
    assert written == validator[0].read_bytes()


def test_a_validator_with_another_hash_is_refused(tmp_path, validator, recorded_fits):
    path, _ = validator
    with pytest.raises(ValueError, match="sha256"):
        export_pymc.export_harbor_tasks_from_rows(
            [_row("aaaa000000000004", {"y": [1.0]}, {"y": [2.0]})],
            {"aaaa000000000004": GOLD},
            tmp_path / "tasks",
            validator_path=path,
            validator_sha256="0" * 64,
            excluded_rows=(),
            allow_unlocked_reference_runtime=True,
            attempt_id="test-attempt",
        )


def test_a_row_without_held_out_data_raises(tmp_path, validator, recorded_fits):
    with pytest.raises(ValueError, match="aaaa000000000005.*held-out"):
        _export(tmp_path, [_row("aaaa000000000005", {"y": [1.0]}, None)], validator)


def test_target_and_features_come_from_the_data_keys(tmp_path, validator, recorded_fits):
    rows = [
        _row("aaaa000000000006", {"y": [1.0, 2.0]}, {"y": [3.0]}),
        _row("aaaa000000000007", {"x": [1.0, 2.0], "y": [5.0, 7.0]}, {"x": [3.0], "y": [9.0]}),
    ]
    _export(tmp_path, rows, validator)
    by_id = {c["env_id"]: c for c in recorded_fits}
    assert (by_id["aaaa000000000006"]["target_name"], by_id["aaaa000000000006"]["blanks"]) == (
        "y",
        [],
    )
    assert (by_id["aaaa000000000007"]["target_name"], by_id["aaaa000000000007"]["blanks"]) == (
        "y",
        ["x"],
    )
    reference = json.loads(
        (tmp_path / "tasks" / "aaaa000000000007" / "tests" / "elpd_reference.json").read_text()
    )
    assert reference["target_name"] == "y"


def test_a_row_without_a_y_column_raises_instead_of_guessing_a_target(
    tmp_path, validator, recorded_fits
):
    with pytest.raises(ValueError, match="aaaa000000000008.*'y'"):
        _export(tmp_path, [_row("aaaa000000000008", {"z": [1.0]}, {"z": [2.0]})], validator)


def test_an_excluded_row_is_not_exported(tmp_path, validator, recorded_fits):
    rows = [
        _row("aaaa000000000009", {"y": [1.0, 2.0]}, {"y": [3.0]}),
        _row("60e3d7cfe3d9b1c3", {"y": [1.0]}, {"y": [2.0]}),
    ]
    summary = _export(tmp_path, rows, validator, excluded=("60e3d7cfe3d9b1c3",))
    assert not (tmp_path / "tasks" / "60e3d7cfe3d9b1c3").exists()
    assert summary["excluded"] == ["60e3d7cfe3d9b1c3"]
    assert [c["env_id"] for c in recorded_fits] == ["aaaa000000000009"]


def test_task_toml_invents_no_envspec_field(tmp_path, validator, recorded_fits):
    _export(
        tmp_path,
        [
            _row(
                "aaaa000000000010",
                {"y": [1.0, 2.0]},
                {"y": [3.0]},
                structural_class="zero_inflated",
            )
        ],
        validator,
    )
    toml = (tmp_path / "tasks" / "aaaa000000000010" / "task.toml").read_text()
    assert "difficulty" not in toml
    assert '"zero_inflated"' in toml


def test_a_feature_identical_to_the_target_is_refused(tmp_path, validator, recorded_fits):
    with pytest.raises(RuntimeError, match="bit-identical"):
        _export(
            tmp_path,
            [
                _row(
                    "aaaa000000000011", {"x": [1.0, 2.0], "y": [1.0, 2.0]}, {"x": [3.0], "y": [3.0]}
                )
            ],
            validator,
        )


@pytest.mark.heavy_lane
def test_a_real_gold_scores_under_the_pinned_validator(tmp_path):
    """Acceptance (Linux host): set SYNTHSTATS_ROWS, SYNTHSTATS_GOLDS and SYNTHSTATS_VALIDATOR to
    run it on real rows.
    """
    import os

    rows = [
        json.loads(line) for line in Path(os.environ["SYNTHSTATS_ROWS"]).read_text().splitlines()
    ][:2]
    golds = {
        g["row_id"]: g["text"]
        for g in map(json.loads, Path(os.environ["SYNTHSTATS_GOLDS"]).read_text().splitlines())
    }
    path = Path(os.environ["SYNTHSTATS_VALIDATOR"])
    summary = export_pymc.export_harbor_tasks_from_rows(
        rows,
        golds,
        tmp_path / "tasks",
        validator_path=path,
        validator_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        excluded_rows=(),
        allow_unlocked_reference_runtime=True,
        attempt_id="test-attempt",
    )
    assert summary["tasks_created"] == len(rows)
    assert summary["reference_failures"] == []


def _deterministic_reference(
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
    """Module level, so that a spawned reference worker can unpickle it by name."""
    return {
        "gold_elpd": -1.0 - len(visible[target_name]),
        "baseline_elpd": -2.0 - len(hidden[target_name]),
        "timing_s": {"gold": 0.1, "baseline": 0.2},
    }


def _task_files_without_seconds(root: Path) -> dict:
    """Return every package file as raw bytes, including the three-key reference payload."""
    return {
        str(f.relative_to(root)): f.read_bytes()
        for f in sorted(
            p for p in root.rglob("*") if p.is_file() and p.name != "export_summary.json"
        )
    }


def test_parallel_reference_workers_write_the_same_tasks_as_one_worker(
    tmp_path, validator, monkeypatch
):
    """Parallel reference fitting writes the same tasks as one worker."""
    monkeypatch.setattr(export_pymc, "_compute_elpd_reference", _deterministic_reference)
    rows = [
        _row(f"bbbb00000000000{i}", {"y": [float(i), 2.0, 3.0]}, {"y": [4.0] * i})
        for i in range(1, 5)
    ]
    path, sha = validator
    golds = {r["row_id"]: GOLD for r in rows}
    one = export_pymc.export_harbor_tasks_from_rows(
        rows,
        golds,
        tmp_path / "one",
        validator_path=path,
        validator_sha256=sha,
        excluded_rows=(),
        allow_unlocked_reference_runtime=True,
        attempt_id="test-attempt",
    )
    three = export_pymc.export_harbor_tasks_from_rows(
        rows,
        golds,
        tmp_path / "three",
        validator_path=path,
        validator_sha256=sha,
        excluded_rows=(),
        reference_workers=3,
        allow_unlocked_reference_runtime=True,
        attempt_id="test-attempt",
    )
    assert _task_files_without_seconds(tmp_path / "one") == _task_files_without_seconds(
        tmp_path / "three"
    )
    assert one["task_dirs"] == three["task_dirs"] == [r["row_id"] for r in rows]
    assert [f["workers"] for f in three["reference_fits"]] == [3] * len(rows)
    assert (one["reference_workers"], three["reference_workers"]) == (1, 3)


def test_zero_reference_workers_is_refused(tmp_path, validator, recorded_fits):
    path, sha = validator
    with pytest.raises(ValueError, match="reference_workers"):
        export_pymc.export_harbor_tasks_from_rows(
            [_row("aaaa000000000013", {"y": [1.0]}, {"y": [2.0]})],
            {"aaaa000000000013": GOLD},
            tmp_path / "tasks",
            validator_path=path,
            validator_sha256=sha,
            excluded_rows=(),
            reference_workers=0,
            allow_unlocked_reference_runtime=True,
            attempt_id="test-attempt",
        )


def test_every_reference_fit_records_its_host_and_worker_count(tmp_path, validator, recorded_fits):
    """Fit times are wall-clock and vary with load."""
    import socket

    _export(tmp_path, [_row("aaaa000000000012", {"y": [1.0, 2.0]}, {"y": [3.0]})], validator)
    summary = json.loads((tmp_path / "tasks" / "export_summary.json").read_text())
    [fit] = summary["reference_fits"]
    assert fit["row_id"] == "aaaa000000000012"
    assert fit["host"] == socket.gethostname()
    assert fit["workers"] == 1
    # The full record keeps its timings under timing_s, which is what export_summary.json writes.
    assert fit["timing_s"]["gold"] == 0.1 and fit["timing_s"]["baseline"] == 0.1

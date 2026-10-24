from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

try:
    from hypothesis import example, given, settings
    from hypothesis import strategies as st
except ModuleNotFoundError:
    example = given = settings = None
    st = None

from ppl_synth_data.pipeline import split_policy as split_policy_module
from ppl_synth_data.pipeline.artifacts import resolve_run_paths
from ppl_synth_data.pipeline.split_policy import SplitPolicy, split_visible_hidden
from ppl_synth_data.pipeline.stages.stage_c_data import run_stage_c
from ppl_synth_data.specs.models import EnvSpec
from tests.pipeline.helpers import minimal_envspec_record, write_jsonl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env_module(path: Path, class_name: str, rows: list[dict]) -> None:
    """Write a minimal compiled-env Python module to disk."""
    rows_repr = repr(rows)
    path.write_text(
        f"""\
import pandas as _pd

class {class_name}:
    def __init__(self, seed=0):
        self._rows = {rows_repr}
        self._calls = []

    def run_experiment(self, query):
        self._calls.append(query)
        if self._rows:
            return self._rows[0].get("y", 1.0), True
        return None, False

    def get_df(self):
        return _pd.DataFrame(self._rows if self._rows else [{{"y": 0.0}}])
""",
        encoding="utf-8",
    )


def _setup_stage_a(paths, env_id: str, env_name: str, n_obs: int = 2) -> None:
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    spec = minimal_envspec_record(env_id=env_id, env_name=env_name)
    spec["data_plan"]["n_observations"] = n_obs
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])


def _setup_stage_a_with_spec(paths, spec: dict[str, object]) -> None:
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])


def _setup_stage_b_env(paths, env_name: str, rows: list[dict]) -> None:
    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    class_name = "".join(p.capitalize() for p in env_name.split("_")) + "Env"
    _make_env_module(envs_dir / f"{env_name}.py", class_name, rows)


def _setup_two_all_zero_row_envs(paths) -> tuple[list[dict], Path]:
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    specs = [
        minimal_envspec_record(env_id="env-zero-a", env_name="zero_row_env_a"),
        minimal_envspec_record(env_id="env-zero-b", env_name="zero_row_env_b"),
    ]
    for spec in specs:
        spec["data_plan"]["n_observations"] = 2
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", specs)

    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    for env_name in ("zero_row_env_a", "zero_row_env_b"):
        class_name = "".join(part.capitalize() for part in env_name.split("_")) + "Env"
        envs_dir.joinpath(f"{env_name}.py").write_text(
            f"""\
import pandas as pd

class {class_name}:
    def __init__(self, seed=0):
        self._query_count = 0

    def run_experiment(self, query):
        self._query_count += 1
        return 0.0, True

    def get_df(self):
        assert self._query_count == 2
        return pd.DataFrame({{"y": pd.Series([], dtype="float64")}})
""",
            encoding="utf-8",
        )
    return specs, envs_dir


def _preseed_complete_stage_c_tree(stage_dir: Path) -> None:
    sentinels = {
        "observations.jsonl": b"stale-observations\n",
        "summaries.jsonl": b"stale-summaries\n",
        "data_refs.jsonl": b"stale-data-refs\n",
        "skipped_envs.jsonl": b"stale-skips\n",
        "manifest.json": b'{"status":"stale-success"}\n',
        "data/stale-visible.parquet": b"stale-visible",
        "data/hidden/stale-hidden.parquet": b"stale-hidden",
        "data/nested/sentinel.bin": b"stale-nested",
    }
    for relative_path, payload in sentinels.items():
        path = stage_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (stage_dir / "data" / "manifest-link").symlink_to("../manifest.json")


def _snapshot_tree(root: Path) -> dict[str, tuple[object, ...]]:
    snapshot: dict[str, tuple[object, ...]] = {}

    def visit(path: Path, relative_path: str) -> None:
        metadata = path.lstat()
        common = (stat.S_IMODE(metadata.st_mode), metadata.st_size, metadata.st_mtime_ns)
        if stat.S_ISREG(metadata.st_mode):
            snapshot[relative_path] = ("file", *common, path.read_bytes())
            return
        if stat.S_ISLNK(metadata.st_mode):
            snapshot[relative_path] = ("symlink", *common, os.readlink(path))
            return
        if stat.S_ISDIR(metadata.st_mode):
            snapshot[relative_path] = ("directory", *common)
            for child in sorted(path.iterdir(), key=lambda entry: entry.name):
                child_relative = (
                    child.name if relative_path == "." else f"{relative_path}/{child.name}"
                )
                visit(child, child_relative)
            return
        snapshot[relative_path] = ("other", *common)

    visit(root, ".")
    return snapshot


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_run_stage_c_returns_success(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_01")
    _setup_stage_a(paths, "env-1", "test_env", n_obs=2)
    _setup_stage_b_env(paths, "test_env", [{"group": "a", "y": 1.0}, {"group": "b", "y": 2.0}])

    result = run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    assert result.stage == "C"
    assert result.status == "success"


def test_run_stage_c_writes_all_artifacts(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_02")
    _setup_stage_a(paths, "env-2", "test_env2", n_obs=3)
    _setup_stage_b_env(paths, "test_env2", [{"group": "x", "y": 5.0}, {"group": "x", "y": 6.0}])

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    stage_dir = paths.stage_dirs["C"]
    assert (stage_dir / "observations.jsonl").exists()
    assert (stage_dir / "summaries.jsonl").exists()
    assert (stage_dir / "data_refs.jsonl").exists()
    assert (stage_dir / "manifest.json").exists()


def test_run_stage_c_clears_stale_data_files(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_clears_stale")
    _setup_stage_a(paths, "env-stale", "stale_env", n_obs=2)
    _setup_stage_b_env(paths, "stale_env", [{"group": "x", "y": 5.0}, {"group": "x", "y": 6.0}])
    stale = paths.stage_dirs["C"] / "data" / "stale.parquet"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("old", encoding="utf-8")

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    assert not stale.exists()
    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_dir_fresh"] is True


def test_run_stage_c_manifest_counts(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_03")
    _setup_stage_a(paths, "env-3", "counting_env", n_obs=4)
    _setup_stage_b_env(paths, "counting_env", [{"y": float(i)} for i in range(4)])

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["env_count"] == 1
    assert manifest["observation_count"] == 4
    assert manifest["summary_count"] == 1


def test_run_stage_c_writes_parquet_per_env(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_04")
    _setup_stage_a(paths, "env-4", "parquet_env", n_obs=2)
    _setup_stage_b_env(paths, "parquet_env", [{"y": 3.0}, {"y": 7.0}])

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    parquet_file = paths.stage_dirs["C"] / "data" / "env-4.parquet"
    assert parquet_file.exists()
    df = pd.read_parquet(parquet_file)
    assert "y" in df.columns
    assert len(df) > 0


def test_run_stage_c_data_refs_contain_env_id(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_05")
    _setup_stage_a(paths, "env-5", "refs_env", n_obs=1)
    _setup_stage_b_env(paths, "refs_env", [{"y": 9.0}, {"y": 10.0}])

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    lines = (
        (paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines) == 1
    ref = json.loads(lines[0])
    assert ref["env_id"] == "env-5"
    assert ref["env_name"] == "refs_env"
    assert "data_ref" in ref
    assert "split_policy" in ref


def test_run_stage_c_summaries_contain_target_stats(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_06")
    _setup_stage_a(paths, "env-6", "stats_env", n_obs=2)
    _setup_stage_b_env(paths, "stats_env", [{"y": 2.0}, {"y": 4.0}])

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    lines = (
        (paths.stage_dirs["C"] / "summaries.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["env_id"] == "env-6"
    assert summary["target_name"] == "y"
    assert summary["n"] > 0


def test_run_stage_c_observations_include_query_and_result(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_07")
    _setup_stage_a(paths, "env-7", "obs_env", n_obs=2)
    _setup_stage_b_env(paths, "obs_env", [{"y": 1.5}, {"y": 2.5}])

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    lines = (
        (paths.stage_dirs["C"] / "observations.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(lines) == 2  # n_obs=2
    for line in lines:
        row = json.loads(line)
        assert "env_id" in row
        assert "query" in row
        assert "result" in row
        assert "success" in row


def test_run_stage_c_rejects_bounded_json_query_when_declared_query_fails(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_retry_json_bounds")
    spec = minimal_envspec_record(env_id="env-json-bounds", env_name="json_bounds_env")
    spec["io_schema"]["input_type"] = "json"
    spec["io_schema"]["feature_names"] = ["a", "b", "c"]
    spec["io_schema"]["feature_types"] = {"a": "continuous", "b": "continuous", "c": "continuous"}
    # Domain is deliberately entirely above the fake env's declared bound
    # (-5.0, 5.0) below, so every draw is guaranteed out of bounds and the
    # "declared query failed" assertion stays deterministic regardless of seed.
    spec["io_schema"]["feature_domains"] = {"a": [6.0, 10.0], "b": [6.0, 10.0], "c": [6.0, 10.0]}
    spec["data_plan"]["covariates"] = ["a", "b", "c"]
    spec["data_plan"]["n_observations"] = 3
    _setup_stage_a_with_spec(paths, spec)

    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    envs_dir.joinpath("json_bounds_env.py").write_text(
        """\
import json
import pandas as _pd

class JsonBoundsEnvEnv:
    def __init__(self, seed=0):
        self._feature_names = ["a", "b", "c"]
        self._lower_bound = -5.0
        self._upper_bound = 5.0
        self.observed_data = []

    def get_ordered_features(self):
        return list(self._feature_names)

    def run_experiment(self, query):
        payload = json.loads(query)
        values = [float(payload[name]) for name in self._feature_names]
        if any(value < self._lower_bound or value > self._upper_bound for value in values):
            return "out of bounds", False
        y = sum(values)
        self.observed_data.append((*values, y))
        return y, True

    def get_df(self):
        return _pd.DataFrame(self.observed_data, columns=["a", "b", "c", "y"])
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="declared query failed"):
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)


def test_run_stage_c_rejects_scalar_env_for_json_spec(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_retry_scalar")
    spec = minimal_envspec_record(env_id="env-scalar", env_name="scalar_env")
    spec["io_schema"]["input_type"] = "json"
    spec["io_schema"]["feature_names"] = []
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = []
    spec["data_plan"]["n_observations"] = 2
    _setup_stage_a_with_spec(paths, spec)

    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    envs_dir.joinpath("scalar_env.py").write_text(
        """\
import pandas as _pd

class ScalarEnvEnv:
    def __init__(self, seed=0):
        self._feature_names = ["x"]
        self._lower_bound = 0.0
        self._upper_bound = 10.0
        self.observed_data = []

    def get_ordered_features(self):
        return list(self._feature_names)

    def run_experiment(self, query):
        try:
            x = float(query.strip())
        except Exception:
            return "must be scalar", False
        if x < self._lower_bound or x > self._upper_bound:
            return "out of bounds", False
        y = x + 1.0
        self.observed_data.append((x, y))
        return y, True

    def get_df(self):
        return _pd.DataFrame(self.observed_data, columns=["x", "y"])
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="declared query failed"):
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)


def test_run_stage_c_rejects_input_validation_exception(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_retry_input_validation_exception")
    spec = minimal_envspec_record(env_id="env-validation-fallback", env_name="validation_env")
    spec["io_schema"]["input_type"] = "json"
    spec["io_schema"]["feature_names"] = ["x"]
    spec["io_schema"]["feature_types"] = {"x": "continuous"}
    spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
    }
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["x"]
    spec["data_plan"]["n_observations"] = 2
    _setup_stage_a_with_spec(paths, spec)

    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    envs_dir.joinpath("validation_env.py").write_text(
        """\
import pandas as _pd
from ppl_synth_data.envs.base import InputValidationError

class ValidationEnvEnv:
    def __init__(self, seed=0):
        self._feature_names = ["x"]
        self._lower_bound = 0.0
        self._upper_bound = 10.0
        self.observed_data = []

    def get_ordered_features(self):
        return list(self._feature_names)

    def run_experiment(self, query):
        try:
            x = float(query.strip())
        except Exception as exc:
            raise InputValidationError("Error: input must be a single float.") from exc
        y = x + 2.0
        self.observed_data.append((x, y))
        return y, True

    def get_df(self):
        return _pd.DataFrame(self.observed_data, columns=["x", "y"])
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="declared query rejected"):
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)


# ---------------------------------------------------------------------------
# limit parameter
# ---------------------------------------------------------------------------


def test_run_stage_c_limit_restricts_processed_envs(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_limit")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    specs = [minimal_envspec_record(env_id=f"env-{i}", env_name=f"limit_env_{i}") for i in range(3)]
    for s in specs:
        s["data_plan"]["n_observations"] = 1
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", specs)
    for i in range(3):
        _setup_stage_b_env(paths, f"limit_env_{i}", [{"y": float(i)}, {"y": float(i) + 10.0}])

    run_stage_c(paths, seed=0, limit=2, split_policy="auto", visible_ratio=0.8)

    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["env_count"] == 2


def test_run_stage_c_skips_generated_env_runtime_failure(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_skip_bad_env")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    specs = [
        minimal_envspec_record(env_id="env-good", env_name="good_env"),
        minimal_envspec_record(env_id="env-bad", env_name="bad_env"),
    ]
    for spec in specs:
        spec["data_plan"]["n_observations"] = 1
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", specs)
    _setup_stage_b_env(paths, "good_env", [{"y": 1.0}, {"y": 2.0}])
    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.joinpath("bad_env.py").write_text(
        """\
import pandas as _pd

class BadEnvEnv:
    def __init__(self, seed=0):
        pass
    def run_experiment(self, query):
        raise ValueError("bad generated query contract")
    def get_df(self):
        return _pd.DataFrame([{"y": 1.0}])
""",
        encoding="utf-8",
    )

    result = run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    assert result.status == "success"
    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["env_count"] == 2
    assert manifest["materialized_env_count"] == 1
    assert manifest["skipped_env_count"] == 1
    skipped = json.loads((paths.stage_dirs["C"] / "skipped_envs.jsonl").read_text(encoding="utf-8"))
    assert skipped["env_id"] == "env-bad"


def test_run_stage_c_skips_empty_generated_dataframe(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_skip_empty_df")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    specs = [
        minimal_envspec_record(env_id="env-good", env_name="good_env"),
        minimal_envspec_record(env_id="env-empty", env_name="empty_env"),
    ]
    for spec in specs:
        spec["data_plan"]["n_observations"] = 2
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", specs)
    _setup_stage_b_env(paths, "good_env", [{"y": 1.0}, {"y": 2.0}])
    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.joinpath("empty_env.py").write_text(
        """\
import pandas as _pd

class EmptyEnvEnv:
    def __init__(self, seed=0):
        pass
    def run_experiment(self, query):
        return None, True
    def get_df(self):
        return _pd.DataFrame(columns=["y"])
""",
        encoding="utf-8",
    )

    result = run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    assert result.status == "success"
    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["materialized_env_count"] == 1
    assert manifest["skipped_env_count"] == 1
    skipped = json.loads((paths.stage_dirs["C"] / "skipped_envs.jsonl").read_text(encoding="utf-8"))
    assert skipped["env_id"] == "env-empty"
    assert "empty dataframe" in skipped["error"]


def test_run_stage_c_skips_zero_row_environment_before_split(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_skip_zero_rows")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    materialized_spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    materialized_spec["data_plan"]["n_observations"] = 1
    materialized_spec["data_plan"]["grouping"] = []
    materialized_spec["data_plan"]["covariates"] = ["time"]
    materialized_spec["io_schema"]["feature_names"] = ["time"]
    materialized_spec["io_schema"]["feature_types"] = {"time": "continuous"}
    materialized_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in materialized_spec["io_schema"]["feature_names"]
    }
    materialized_spec["generative_family"] = "time_series_level_seasonality"
    zero_row_spec = minimal_envspec_record(env_id="env-zero", env_name="zero_row_env")
    zero_row_spec["data_plan"]["n_observations"] = 2
    write_jsonl(
        paths.stage_dirs["A"] / "envspecs.jsonl",
        [materialized_spec, zero_row_spec],
    )
    _setup_stage_b_env(
        paths,
        "time_env",
        [{"time": idx, "y": float(idx)} for idx in range(10)],
    )
    zero_row_module_path = paths.stage_dirs["B"] / "envs" / "zero_row_env.py"
    zero_row_module_path.write_text(
        """\
import pandas as pd

class ZeroRowEnvEnv:
    def __init__(self, seed=0):
        self._query_count = 0

    def run_experiment(self, query):
        self._query_count += 1
        return 0.0, True

    def get_df(self):
        assert self._query_count == 2
        return pd.DataFrame({"y": pd.Series([], dtype="float64")})
""",
        encoding="utf-8",
    )

    fixture_spec = importlib.util.spec_from_file_location(
        "zero_row_env_fixture",
        zero_row_module_path,
    )
    assert fixture_spec is not None
    assert fixture_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    fixture_env = fixture_module.ZeroRowEnvEnv(seed=0)
    assert fixture_env.run_experiment("") == (0.0, True)
    assert fixture_env.run_experiment("") == (0.0, True)
    fixture_frame = fixture_env.get_df()
    assert fixture_frame.shape == (0, 1)
    assert fixture_frame.columns.tolist() == ["y"]
    assert fixture_frame["y"].dtype == "float64"
    assert fixture_frame.index.equals(pd.RangeIndex(0, 0, 1))
    assert fixture_frame.index.name is None

    result = run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert result.status == "success"
    assert result.outputs == manifest
    assert manifest["status_detail"] == "success_with_skips"
    assert manifest["materialized_env_count"] == 1
    assert manifest["skipped_env_count"] == 1

    data_ref = json.loads((paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8"))
    assert data_ref == {
        "env_id": "env-time",
        "env_name": "time_env",
        "data_ref": "stage_C_data/data/env-time.parquet",
        "hidden_data_ref": "stage_C_data/data/hidden/env-time.parquet",
        "split_policy": "ordered_tail",
    }
    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / "env-time.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-time.parquet")
    pd.testing.assert_frame_equal(
        visible,
        pd.DataFrame({"time": list(range(8)), "y": [float(idx) for idx in range(8)]}),
    )
    pd.testing.assert_frame_equal(
        hidden,
        pd.DataFrame({"time": [8, 9], "y": [8.0, 9.0]}),
    )
    parquet_outputs = sorted(
        path.relative_to(paths.stage_dirs["C"] / "data").as_posix()
        for path in (paths.stage_dirs["C"] / "data").rglob("*.parquet")
    )
    assert parquet_outputs == ["env-time.parquet", "hidden/env-time.parquet"]

    skipped = json.loads((paths.stage_dirs["C"] / "skipped_envs.jsonl").read_text(encoding="utf-8"))
    assert skipped == {
        "env_id": "env-zero",
        "env_name": "zero_row_env",
        "error": "environment produced an empty dataframe after 2 observations",
        "exception_type": "ValueError",
    }


def test_run_stage_c_all_zero_row_environments_raise_aggregate_error(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_all_zero_rows")
    specs, envs_dir = _setup_two_all_zero_row_envs(paths)

    with pytest.raises(RuntimeError) as exc_info:
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    expected_first_skipped = {
        "env_id": "env-zero-a",
        "env_name": "zero_row_env_a",
        "error": "environment produced an empty dataframe after 2 observations",
        "exception_type": "ValueError",
    }
    assert "stage C could not materialize any generated environments" in str(exc_info.value)
    assert f"first skipped env: {expected_first_skipped}" in str(exc_info.value)

    reversed_paths = resolve_run_paths(tmp_path, "run_c_all_zero_rows_reversed")
    reversed_paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    write_jsonl(
        reversed_paths.stage_dirs["A"] / "envspecs.jsonl",
        list(reversed(specs)),
    )
    reversed_envs_dir = reversed_paths.stage_dirs["B"] / "envs"
    reversed_envs_dir.mkdir(parents=True, exist_ok=True)
    for env_name in ("zero_row_env_a", "zero_row_env_b"):
        reversed_envs_dir.joinpath(f"{env_name}.py").write_text(
            envs_dir.joinpath(f"{env_name}.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError) as reversed_exc_info:
        run_stage_c(reversed_paths, seed=0, split_policy="auto", visible_ratio=0.8)

    expected_reversed_first_skipped = {
        "env_id": "env-zero-b",
        "env_name": "zero_row_env_b",
        "error": "environment produced an empty dataframe after 2 observations",
        "exception_type": "ValueError",
    }
    assert "stage C could not materialize any generated environments" in str(
        reversed_exc_info.value
    )
    assert f"first skipped env: {expected_reversed_first_skipped}" in str(reversed_exc_info.value)


def test_run_stage_c_all_zero_row_failure_clears_stale_outputs(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_all_zero_rows_clears_stale")
    _setup_two_all_zero_row_envs(paths)

    _preseed_complete_stage_c_tree(paths.stage_dirs["C"])
    stale_sibling = paths.run_dir / ".stage_C_data.attempt-stale"
    stale_sibling.mkdir(parents=True, exist_ok=True)
    (stale_sibling / "sentinel.bin").write_bytes(b"stale-attempt")

    with pytest.raises(RuntimeError) as exc_info:
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    assert "stage C could not materialize any generated environments" in str(exc_info.value)

    stage_dir = paths.stage_dirs["C"]
    for filename in (
        "observations.jsonl",
        "summaries.jsonl",
        "data_refs.jsonl",
        "skipped_envs.jsonl",
        "manifest.json",
    ):
        assert not (stage_dir / filename).exists()
    assert not (stage_dir / "data").exists()
    assert list(stage_dir.rglob("*.parquet")) == []
    assert list(paths.run_dir.glob(".stage_C_data.attempt-*")) == []


def test_run_stage_c_rejects_zero_hidden_rows(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_rejects_zero_hidden_rows")
    _setup_stage_a(paths, "env-one", "one_row_env", n_obs=1)
    _setup_stage_b_env(paths, "one_row_env", [{"y": 1.0}])
    with pytest.raises(split_policy_module.SplitPolicyIneligibleError):
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    stage_dir = paths.stage_dirs["C"]
    assert not (stage_dir / "manifest.json").exists()
    assert not (stage_dir / "data_refs.jsonl").exists()
    assert not list((stage_dir / "data").rglob("*.parquet"))


def test_run_stage_c_rejects_zero_hidden_rows_for_different_fixture(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_rejects_zero_hidden_rows_companion")
    _setup_stage_a(paths, "env-single", "single_row_env", n_obs=1)
    _setup_stage_b_env(paths, "single_row_env", [{"y": 3.0}])
    with pytest.raises(split_policy_module.SplitPolicyIneligibleError):
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.5)

    stage_dir = paths.stage_dirs["C"]
    assert not (stage_dir / "manifest.json").exists()
    assert not (stage_dir / "data_refs.jsonl").exists()
    assert not list((stage_dir / "data").rglob("*.parquet"))


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_run_stage_c_empty_envspecs_produces_zero_counts(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_empty")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text("", encoding="utf-8")
    paths.stage_dirs["B"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["B"] / "envs").mkdir(parents=True, exist_ok=True)

    result = run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    assert result.status == "success"
    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["env_count"] == 0
    assert manifest["observation_count"] == 0
    assert manifest["summary_count"] == 0


# ---------------------------------------------------------------------------
# Missing target column
# ---------------------------------------------------------------------------


def test_run_stage_c_missing_target_column_yields_empty_summary(tmp_path: Path) -> None:
    """When the dataframe has no target column, summary should record n=0."""
    paths = resolve_run_paths(tmp_path, "run_c_no_target")
    _setup_stage_a(paths, "env-nt", "no_target_env", n_obs=1)

    # Write env that returns a df without 'y'
    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    envs_dir.joinpath("no_target_env.py").write_text(
        """\
import pandas as _pd

class NoTargetEnvEnv:
    def __init__(self, seed=0):
        pass
    def run_experiment(self, query):
        return None, True
    def get_df(self):
        return _pd.DataFrame([{"x": 1.0, "z": 2.0}, {"x": 3.0, "z": 4.0}])
""",
        encoding="utf-8",
    )

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    summary_line = (paths.stage_dirs["C"] / "summaries.jsonl").read_text(encoding="utf-8").strip()
    summary = json.loads(summary_line)
    assert summary["n"] == 0


# ---------------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------------


def test_run_stage_c_same_seed_produces_same_parquet(tmp_path: Path) -> None:
    """Two runs with the same seed should produce identical parquet files."""
    rows = [{"group": f"g{i}", "y": float(i)} for i in range(5)]

    paths_a = resolve_run_paths(tmp_path, "run_c_seed_a")
    _setup_stage_a(paths_a, "env-s", "seed_env", n_obs=3)
    _setup_stage_b_env(paths_a, "seed_env", rows)
    run_stage_c(paths_a, seed=42, split_policy="auto", visible_ratio=0.8)

    paths_b = resolve_run_paths(tmp_path, "run_c_seed_b")
    _setup_stage_a(paths_b, "env-s", "seed_env", n_obs=3)
    _setup_stage_b_env(paths_b, "seed_env", rows)
    run_stage_c(paths_b, seed=42, split_policy="auto", visible_ratio=0.8)

    df_a = pd.read_parquet(paths_a.stage_dirs["C"] / "data" / "env-s.parquet")
    df_b = pd.read_parquet(paths_b.stage_dirs["C"] / "data" / "env-s.parquet")
    assert df_a.equals(df_b)


def test_run_stage_c_grouped_split_holds_out_whole_groups(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_group_holdout")
    spec = minimal_envspec_record(env_id="env-group", env_name="group_env")
    spec["data_plan"]["n_observations"] = 1
    _setup_stage_a_with_spec(paths, spec)
    _setup_stage_b_env(
        paths,
        "group_env",
        [
            {"group": "a", "y": 1.0},
            {"group": "a", "y": 2.0},
            {"group": "b", "y": 3.0},
            {"group": "b", "y": 4.0},
            {"group": "c", "y": 5.0},
            {"group": "c", "y": 6.0},
        ],
    )

    run_stage_c(paths, seed=7, split_policy="auto", visible_ratio=0.5)

    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / "env-group.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-group.parquet")
    assert set(visible["group"]).isdisjoint(set(hidden["group"]))

    ref = json.loads(
        (paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8").strip()
    )
    assert ref["split_policy"] == "group_holdout"
    assert ref["hidden_data_ref"] == "stage_C_data/data/hidden/env-group.parquet"


def test_run_stage_c_forced_group_holdout_matches_auto_membership(
    tmp_path: Path,
) -> None:
    rows = [
        {"source_row_id": "a1", "group": "a", "y": 1.0},
        {"source_row_id": "a2", "group": "a", "y": 2.0},
        {"source_row_id": "b1", "group": "b", "y": 3.0},
        {"source_row_id": "b2", "group": "b", "y": 4.0},
        {"source_row_id": "c1", "group": "c", "y": 5.0},
        {"source_row_id": "c2", "group": "c", "y": 6.0},
    ]

    auto_paths = resolve_run_paths(tmp_path, "run_c_forced_group_auto")
    auto_spec = minimal_envspec_record(env_id="env-group", env_name="group_env")
    auto_spec["data_plan"]["n_observations"] = 1
    _setup_stage_a_with_spec(auto_paths, auto_spec)
    _setup_stage_b_env(auto_paths, "group_env", rows)

    auto_result = run_stage_c(
        auto_paths,
        seed=7,
        split_policy="auto",
        visible_ratio=0.5,
    )

    assert auto_result.outputs["split_policy_counts"] == {"group_holdout": 1}
    auto_ref = json.loads(
        (auto_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    assert auto_ref["split_policy"] == "group_holdout"
    auto_visible = pd.read_parquet(auto_paths.stage_dirs["C"] / "data" / "env-group.parquet")
    auto_hidden = pd.read_parquet(
        auto_paths.stage_dirs["C"] / "data" / "hidden" / "env-group.parquet"
    )
    assert auto_visible["source_row_id"].tolist() == ["a1", "a2", "c1", "c2"]
    assert auto_hidden["source_row_id"].tolist() == ["b1", "b2"]

    forced_paths = resolve_run_paths(tmp_path, "run_c_forced_group_literal")
    forced_spec = minimal_envspec_record(env_id="env-group", env_name="group_env")
    forced_spec["data_plan"]["n_observations"] = 1
    _setup_stage_a_with_spec(forced_paths, forced_spec)
    _setup_stage_b_env(forced_paths, "group_env", rows)

    forced_result = run_stage_c(
        forced_paths,
        seed=7,
        split_policy="group_holdout",
        visible_ratio=0.5,
    )

    assert forced_result.outputs["split_policy_counts"] == {"group_holdout": 1}
    forced_ref = json.loads(
        (forced_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    assert forced_ref["split_policy"] == "group_holdout"
    forced_visible = pd.read_parquet(forced_paths.stage_dirs["C"] / "data" / "env-group.parquet")
    forced_hidden = pd.read_parquet(
        forced_paths.stage_dirs["C"] / "data" / "hidden" / "env-group.parquet"
    )
    assert forced_visible["source_row_id"].tolist() == ["a1", "a2", "c1", "c2"]
    assert forced_hidden["source_row_id"].tolist() == ["b1", "b2"]


def test_run_stage_c_forced_spatial_region_ignores_grouping(tmp_path: Path) -> None:
    rows = [
        {"source_row_id": "r0", "group": "a", "x": 0, "y": 0, "intensity": 0},
        {"source_row_id": "r1", "group": "b", "x": 0, "y": 1, "intensity": 1},
        {"source_row_id": "r2", "group": "a", "x": 0, "y": 2, "intensity": 2},
        {"source_row_id": "r3", "group": "b", "x": 10, "y": 10, "intensity": 3},
        {"source_row_id": "r4", "group": "a", "x": 10, "y": 11, "intensity": 4},
        {"source_row_id": "r5", "group": "b", "x": 10, "y": 12, "intensity": 5},
    ]
    grouped_paths = resolve_run_paths(tmp_path, "run_c_forced_spatial_grouped")
    grouped_spec = minimal_envspec_record(
        env_id="env-space-forced",
        env_name="space_fixture",
    )
    grouped_spec["data_plan"]["n_observations"] = 1
    grouped_spec["data_plan"]["grouping"] = ["group"]
    grouped_spec["data_plan"]["covariates"] = ["x", "y"]
    grouped_spec["io_schema"]["feature_names"] = ["group", "x", "y"]
    grouped_spec["io_schema"]["feature_types"] = {
        "group": "continuous",
        "x": "continuous",
        "y": "continuous",
    }
    grouped_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in grouped_spec["io_schema"]["feature_names"]
    }
    grouped_spec["io_schema"]["target_name"] = "intensity"
    grouped_spec["generative_family"] = "spatial_signal_source"
    _setup_stage_a_with_spec(grouped_paths, grouped_spec)
    _setup_stage_b_env(grouped_paths, "space_fixture", rows)

    ungrouped_paths = resolve_run_paths(tmp_path, "run_c_forced_spatial_ungrouped")
    ungrouped_spec = json.loads(json.dumps(grouped_spec))
    ungrouped_spec["data_plan"]["grouping"] = []
    _setup_stage_a_with_spec(ungrouped_paths, ungrouped_spec)
    _setup_stage_b_env(ungrouped_paths, "space_fixture", rows)

    auto_grouped_result = run_stage_c(
        grouped_paths,
        seed=0,
        split_policy="auto",
        visible_ratio=0.5,
    )
    auto_ungrouped_result = run_stage_c(
        ungrouped_paths,
        seed=0,
        split_policy="auto",
        visible_ratio=0.5,
    )

    auto_grouped_ref = json.loads(
        (grouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    auto_grouped_visible = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "env-space-forced.parquet"
    )
    auto_grouped_hidden = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "hidden" / "env-space-forced.parquet"
    )
    assert auto_grouped_result.outputs["split_policy_counts"] == {"group_holdout": 1}
    assert auto_grouped_ref["split_policy"] == "group_holdout"
    assert auto_grouped_visible["source_row_id"].tolist() == ["r1", "r3", "r5"]
    assert auto_grouped_hidden["source_row_id"].tolist() == ["r0", "r2", "r4"]

    auto_ungrouped_ref = json.loads(
        (ungrouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    auto_ungrouped_visible = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "env-space-forced.parquet"
    )
    auto_ungrouped_hidden = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "hidden" / "env-space-forced.parquet"
    )
    assert auto_ungrouped_result.outputs["split_policy_counts"] == {"spatial_region": 1}
    assert auto_ungrouped_ref["split_policy"] == "spatial_region"
    assert auto_ungrouped_visible["source_row_id"].tolist() == ["r0", "r1", "r2"]
    assert auto_ungrouped_hidden["source_row_id"].tolist() == ["r3", "r4", "r5"]

    forced_grouped_result = run_stage_c(
        grouped_paths,
        seed=0,
        split_policy="spatial_region",
        visible_ratio=0.5,
    )
    forced_ungrouped_result = run_stage_c(
        ungrouped_paths,
        seed=0,
        split_policy="spatial_region",
        visible_ratio=0.5,
    )

    forced_grouped_ref = json.loads(
        (grouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    forced_grouped_visible = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "env-space-forced.parquet"
    )
    forced_grouped_hidden = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "hidden" / "env-space-forced.parquet"
    )
    forced_ungrouped_ref = json.loads(
        (ungrouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    forced_ungrouped_visible = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "env-space-forced.parquet"
    )
    forced_ungrouped_hidden = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "hidden" / "env-space-forced.parquet"
    )

    assert forced_grouped_result.outputs["split_policy_counts"] == {"spatial_region": 1}
    assert forced_ungrouped_result.outputs["split_policy_counts"] == {"spatial_region": 1}
    assert forced_grouped_ref["split_policy"] == "spatial_region"
    assert forced_ungrouped_ref["split_policy"] == "spatial_region"
    assert forced_grouped_visible["source_row_id"].tolist() == ["r0", "r1", "r2"]
    assert forced_grouped_hidden["source_row_id"].tolist() == ["r3", "r4", "r5"]
    assert forced_ungrouped_visible["source_row_id"].tolist() == ["r0", "r1", "r2"]
    assert forced_ungrouped_hidden["source_row_id"].tolist() == ["r3", "r4", "r5"]
    assert (
        len(forced_grouped_visible) + len(forced_grouped_hidden),
        len(forced_grouped_visible),
        len(forced_grouped_hidden),
    ) == (6, 3, 3)
    assert (
        len(forced_ungrouped_visible) + len(forced_ungrouped_hidden),
        len(forced_ungrouped_visible),
        len(forced_ungrouped_hidden),
    ) == (6, 3, 3)
    assert forced_grouped_visible.columns.tolist() == [
        "source_row_id",
        "group",
        "x",
        "y",
        "intensity",
    ]
    assert forced_grouped_visible.columns.tolist() == forced_ungrouped_visible.columns.tolist()
    assert forced_grouped_hidden.columns.tolist() == forced_ungrouped_hidden.columns.tolist()
    assert forced_grouped_visible.to_dict(orient="records") == forced_ungrouped_visible.to_dict(
        orient="records"
    )
    assert forced_grouped_hidden.to_dict(orient="records") == forced_ungrouped_hidden.to_dict(
        orient="records"
    )


def test_run_stage_c_forced_ordered_tail_ignores_grouping(tmp_path: Path) -> None:
    rows = [
        {"source_row_id": "r0", "group": "a", "time": 0, "value": 0},
        {"source_row_id": "r1", "group": "b", "time": 1, "value": 1},
        {"source_row_id": "r2", "group": "a", "time": 2, "value": 2},
        {"source_row_id": "r3", "group": "b", "time": 3, "value": 3},
        {"source_row_id": "r4", "group": "a", "time": 4, "value": 4},
        {"source_row_id": "r5", "group": "b", "time": 5, "value": 5},
    ]
    grouped_paths = resolve_run_paths(tmp_path, "run_c_forced_ordered_grouped")
    grouped_spec = minimal_envspec_record(
        env_id="env-time-forced",
        env_name="time_fixture",
    )
    grouped_spec["data_plan"]["n_observations"] = 1
    grouped_spec["data_plan"]["grouping"] = ["group"]
    grouped_spec["data_plan"]["covariates"] = ["time"]
    grouped_spec["io_schema"]["feature_names"] = ["group", "time"]
    grouped_spec["io_schema"]["feature_types"] = {"group": "continuous", "time": "continuous"}
    grouped_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in grouped_spec["io_schema"]["feature_names"]
    }
    grouped_spec["io_schema"]["target_name"] = "value"
    grouped_spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(grouped_paths, grouped_spec)
    _setup_stage_b_env(grouped_paths, "time_fixture", rows)

    ungrouped_paths = resolve_run_paths(tmp_path, "run_c_forced_ordered_ungrouped")
    ungrouped_spec = json.loads(json.dumps(grouped_spec))
    ungrouped_spec["data_plan"]["grouping"] = []
    _setup_stage_a_with_spec(ungrouped_paths, ungrouped_spec)
    _setup_stage_b_env(ungrouped_paths, "time_fixture", rows)

    auto_grouped_result = run_stage_c(
        grouped_paths,
        seed=0,
        split_policy="auto",
        visible_ratio=0.5,
    )
    auto_ungrouped_result = run_stage_c(
        ungrouped_paths,
        seed=0,
        split_policy="auto",
        visible_ratio=0.5,
    )

    auto_grouped_ref = json.loads(
        (grouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    auto_grouped_visible = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "env-time-forced.parquet"
    )
    auto_grouped_hidden = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "hidden" / "env-time-forced.parquet"
    )
    assert auto_grouped_result.outputs["split_policy_counts"] == {"group_holdout": 1}
    assert auto_grouped_ref["split_policy"] == "group_holdout"
    assert auto_grouped_visible["source_row_id"].tolist() == ["r1", "r3", "r5"]
    assert auto_grouped_hidden["source_row_id"].tolist() == ["r0", "r2", "r4"]

    auto_ungrouped_ref = json.loads(
        (ungrouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    auto_ungrouped_visible = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "env-time-forced.parquet"
    )
    auto_ungrouped_hidden = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "hidden" / "env-time-forced.parquet"
    )
    assert auto_ungrouped_result.outputs["split_policy_counts"] == {"ordered_tail": 1}
    assert auto_ungrouped_ref["split_policy"] == "ordered_tail"
    assert auto_ungrouped_visible["source_row_id"].tolist() == ["r0", "r1", "r2"]
    assert auto_ungrouped_hidden["source_row_id"].tolist() == ["r3", "r4", "r5"]

    forced_grouped_result = run_stage_c(
        grouped_paths,
        seed=0,
        split_policy="ordered_tail",
        visible_ratio=0.5,
    )
    forced_ungrouped_result = run_stage_c(
        ungrouped_paths,
        seed=0,
        split_policy="ordered_tail",
        visible_ratio=0.5,
    )

    forced_grouped_ref = json.loads(
        (grouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    forced_grouped_visible = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "env-time-forced.parquet"
    )
    forced_grouped_hidden = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "hidden" / "env-time-forced.parquet"
    )
    forced_ungrouped_ref = json.loads(
        (ungrouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    forced_ungrouped_visible = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "env-time-forced.parquet"
    )
    forced_ungrouped_hidden = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "hidden" / "env-time-forced.parquet"
    )

    assert forced_grouped_result.outputs["split_policy_counts"] == {"ordered_tail": 1}
    assert forced_ungrouped_result.outputs["split_policy_counts"] == {"ordered_tail": 1}
    assert forced_grouped_ref["split_policy"] == "ordered_tail"
    assert forced_ungrouped_ref["split_policy"] == "ordered_tail"
    assert forced_grouped_visible["source_row_id"].tolist() == ["r0", "r1", "r2"]
    assert forced_grouped_hidden["source_row_id"].tolist() == ["r3", "r4", "r5"]
    assert forced_ungrouped_visible["source_row_id"].tolist() == ["r0", "r1", "r2"]
    assert forced_ungrouped_hidden["source_row_id"].tolist() == ["r3", "r4", "r5"]
    assert (
        len(forced_grouped_visible) + len(forced_grouped_hidden),
        len(forced_grouped_visible),
        len(forced_grouped_hidden),
    ) == (6, 3, 3)
    assert (
        len(forced_ungrouped_visible) + len(forced_ungrouped_hidden),
        len(forced_ungrouped_visible),
        len(forced_ungrouped_hidden),
    ) == (6, 3, 3)
    assert forced_grouped_visible.columns.tolist() == [
        "source_row_id",
        "group",
        "time",
        "value",
    ]
    assert forced_grouped_visible.columns.tolist() == forced_ungrouped_visible.columns.tolist()
    assert forced_grouped_hidden.columns.tolist() == forced_ungrouped_hidden.columns.tolist()
    assert forced_grouped_visible.to_dict(orient="records") == forced_ungrouped_visible.to_dict(
        orient="records"
    )
    assert forced_grouped_hidden.to_dict(orient="records") == forced_ungrouped_hidden.to_dict(
        orient="records"
    )


def test_run_stage_c_forced_random_row_ignores_grouping(tmp_path: Path) -> None:
    rows = [
        {"source_row_id": "r0", "group": "A", "feature": 0, "value": 0},
        {"source_row_id": "r1", "group": "A", "feature": 1, "value": 1},
        {"source_row_id": "r2", "group": "A", "feature": 2, "value": 2},
        {"source_row_id": "r3", "group": "B", "feature": 3, "value": 3},
        {"source_row_id": "r4", "group": "B", "feature": 4, "value": 4},
        {"source_row_id": "r5", "group": "B", "feature": 5, "value": 5},
    ]
    grouped_paths = resolve_run_paths(tmp_path, "run_c_forced_random_grouped")
    grouped_spec = minimal_envspec_record(
        env_id="hash-random",
        env_name="random_fixture",
    )
    grouped_spec["data_plan"]["n_observations"] = 1
    grouped_spec["data_plan"]["grouping"] = ["group"]
    grouped_spec["data_plan"]["covariates"] = ["feature"]
    grouped_spec["io_schema"]["feature_names"] = ["feature"]
    grouped_spec["io_schema"]["feature_types"] = {"feature": "continuous"}
    grouped_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in grouped_spec["io_schema"]["feature_names"]
    }
    grouped_spec["io_schema"]["target_name"] = "value"
    grouped_spec["generative_family"] = "linear_regression"
    _setup_stage_a_with_spec(grouped_paths, grouped_spec)
    _setup_stage_b_env(grouped_paths, "random_fixture", rows)

    ungrouped_paths = resolve_run_paths(tmp_path, "run_c_forced_random_ungrouped")
    ungrouped_spec = json.loads(json.dumps(grouped_spec))
    ungrouped_spec["data_plan"]["grouping"] = []
    _setup_stage_a_with_spec(ungrouped_paths, ungrouped_spec)
    _setup_stage_b_env(ungrouped_paths, "random_fixture", rows)

    auto_grouped_result = run_stage_c(
        grouped_paths,
        seed=2,
        split_policy="auto",
        visible_ratio=0.5,
    )
    auto_ungrouped_result = run_stage_c(
        ungrouped_paths,
        seed=2,
        split_policy="auto",
        visible_ratio=0.5,
    )

    auto_grouped_ref = json.loads(
        (grouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    auto_grouped_visible = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "hash-random.parquet"
    )
    auto_grouped_hidden = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "hidden" / "hash-random.parquet"
    )
    assert auto_grouped_result.outputs["split_policy_counts"] == {"group_holdout": 1}
    assert auto_grouped_ref["split_policy"] == "group_holdout"
    assert auto_grouped_visible["source_row_id"].tolist() == ["r3", "r4", "r5"]
    assert auto_grouped_hidden["source_row_id"].tolist() == ["r0", "r1", "r2"]

    auto_ungrouped_ref = json.loads(
        (ungrouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    auto_ungrouped_visible = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "hash-random.parquet"
    )
    auto_ungrouped_hidden = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "hidden" / "hash-random.parquet"
    )
    assert auto_ungrouped_result.outputs["split_policy_counts"] == {"random_row": 1}
    assert auto_ungrouped_ref["split_policy"] == "random_row"
    assert auto_ungrouped_visible["source_row_id"].tolist() == ["r0", "r1", "r3"]
    assert auto_ungrouped_hidden["source_row_id"].tolist() == ["r2", "r4", "r5"]

    forced_grouped_result = run_stage_c(
        grouped_paths,
        seed=2,
        split_policy="random_row",
        visible_ratio=0.5,
    )
    forced_ungrouped_result = run_stage_c(
        ungrouped_paths,
        seed=2,
        split_policy="random_row",
        visible_ratio=0.5,
    )

    forced_grouped_ref = json.loads(
        (grouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    forced_grouped_visible = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "hash-random.parquet"
    )
    forced_grouped_hidden = pd.read_parquet(
        grouped_paths.stage_dirs["C"] / "data" / "hidden" / "hash-random.parquet"
    )
    forced_ungrouped_ref = json.loads(
        (ungrouped_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    forced_ungrouped_visible = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "hash-random.parquet"
    )
    forced_ungrouped_hidden = pd.read_parquet(
        ungrouped_paths.stage_dirs["C"] / "data" / "hidden" / "hash-random.parquet"
    )

    assert forced_grouped_result.outputs["split_policy_counts"] == {"random_row": 1}
    assert forced_ungrouped_result.outputs["split_policy_counts"] == {"random_row": 1}
    assert forced_grouped_ref["split_policy"] == "random_row"
    assert forced_ungrouped_ref["split_policy"] == "random_row"
    assert forced_grouped_visible["source_row_id"].tolist() == ["r0", "r1", "r3"]
    assert forced_grouped_hidden["source_row_id"].tolist() == ["r2", "r4", "r5"]
    assert forced_ungrouped_visible["source_row_id"].tolist() == ["r0", "r1", "r3"]
    assert forced_ungrouped_hidden["source_row_id"].tolist() == ["r2", "r4", "r5"]
    assert (
        len(forced_grouped_visible) + len(forced_grouped_hidden),
        len(forced_grouped_visible),
        len(forced_grouped_hidden),
    ) == (6, 3, 3)
    assert (
        len(forced_ungrouped_visible) + len(forced_ungrouped_hidden),
        len(forced_ungrouped_visible),
        len(forced_ungrouped_hidden),
    ) == (6, 3, 3)
    assert forced_grouped_visible.columns.tolist() == [
        "source_row_id",
        "group",
        "feature",
        "value",
    ]
    assert forced_grouped_visible.columns.tolist() == forced_ungrouped_visible.columns.tolist()
    assert forced_grouped_hidden.columns.tolist() == forced_ungrouped_hidden.columns.tolist()
    assert forced_grouped_visible.to_dict(orient="records") == forced_ungrouped_visible.to_dict(
        orient="records"
    )
    assert forced_grouped_hidden.to_dict(orient="records") == forced_ungrouped_hidden.to_dict(
        orient="records"
    )


def test_run_stage_c_forced_group_holdout_ineligible_does_not_fall_through(
    tmp_path: Path,
) -> None:
    time_rows = [
        {"source_row_id": f"r{index}", "group": "a", "time": index, "y": index}
        for index in range(4)
    ]
    time_auto_paths = resolve_run_paths(tmp_path, "run_c_group_ineligible_time_auto")
    time_auto_spec = minimal_envspec_record(
        env_id="env-g-time",
        env_name="g_time_env",
    )
    time_auto_spec["data_plan"]["n_observations"] = 1
    time_auto_spec["data_plan"]["grouping"] = ["group"]
    time_auto_spec["data_plan"]["covariates"] = ["time"]
    time_auto_spec["io_schema"]["feature_names"] = ["group", "time"]
    time_auto_spec["io_schema"]["feature_types"] = {"group": "continuous", "time": "continuous"}
    time_auto_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in time_auto_spec["io_schema"]["feature_names"]
    }
    time_auto_spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(time_auto_paths, time_auto_spec)
    _setup_stage_b_env(time_auto_paths, "g_time_env", time_rows)

    random_rows = [
        {
            "source_row_id": f"r{index}",
            "group": "a",
            "value": index,
            "y": index,
        }
        for index in range(4)
    ]
    random_auto_paths = resolve_run_paths(tmp_path, "run_c_group_ineligible_random_auto")
    random_auto_spec = minimal_envspec_record(
        env_id="env-g-random",
        env_name="g_random_env",
    )
    random_auto_spec["data_plan"]["n_observations"] = 1
    random_auto_spec["data_plan"]["grouping"] = ["group"]
    random_auto_spec["data_plan"]["covariates"] = []
    random_auto_spec["io_schema"]["feature_names"] = ["group", "value"]
    random_auto_spec["io_schema"]["feature_types"] = {"group": "continuous", "value": "continuous"}
    random_auto_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in random_auto_spec["io_schema"]["feature_names"]
    }
    random_auto_spec["generative_family"] = "hierarchical_grouped"
    _setup_stage_a_with_spec(random_auto_paths, random_auto_spec)
    _setup_stage_b_env(random_auto_paths, "g_random_env", random_rows)

    time_auto_result = run_stage_c(
        time_auto_paths,
        seed=0,
        split_policy=SplitPolicy.AUTO,
        visible_ratio=0.5,
    )
    random_auto_result = run_stage_c(
        random_auto_paths,
        seed=0,
        split_policy=SplitPolicy.AUTO,
        visible_ratio=0.5,
    )

    time_auto_ref = json.loads(
        (time_auto_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    time_auto_visible = pd.read_parquet(
        time_auto_paths.stage_dirs["C"] / "data" / "env-g-time.parquet"
    )
    time_auto_hidden = pd.read_parquet(
        time_auto_paths.stage_dirs["C"] / "data" / "hidden" / "env-g-time.parquet"
    )
    assert time_auto_result.outputs["split_policy_counts"] == {"ordered_tail": 1}
    assert time_auto_ref["split_policy"] == "ordered_tail"
    assert time_auto_visible["source_row_id"].tolist() == ["r0", "r1"]
    assert time_auto_hidden["source_row_id"].tolist() == ["r2", "r3"]

    random_auto_ref = json.loads(
        (random_auto_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    random_auto_visible = pd.read_parquet(
        random_auto_paths.stage_dirs["C"] / "data" / "env-g-random.parquet"
    )
    random_auto_hidden = pd.read_parquet(
        random_auto_paths.stage_dirs["C"] / "data" / "hidden" / "env-g-random.parquet"
    )
    assert random_auto_result.outputs["split_policy_counts"] == {"random_row": 1}
    assert random_auto_ref["split_policy"] == "random_row"
    assert random_auto_visible["source_row_id"].tolist() == ["r2", "r3"]
    assert random_auto_hidden["source_row_id"].tolist() == ["r0", "r1"]

    time_forced_paths = resolve_run_paths(tmp_path, "run_c_group_ineligible_time_forced")
    _setup_stage_a_with_spec(
        time_forced_paths,
        json.loads(json.dumps(time_auto_spec)),
    )
    _setup_stage_b_env(time_forced_paths, "g_time_env", time_rows)

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as time_error:
        run_stage_c(
            time_forced_paths,
            seed=0,
            split_policy=SplitPolicy.GROUP_HOLDOUT,
            visible_ratio=0.5,
        )

    assert type(time_error.value) is split_policy_module.SplitPolicyIneligibleError
    for filename in (
        "observations.jsonl",
        "summaries.jsonl",
        "data_refs.jsonl",
        "skipped_envs.jsonl",
        "manifest.json",
    ):
        assert not (time_forced_paths.stage_dirs["C"] / filename).exists()
    assert list(time_forced_paths.stage_dirs["C"].rglob("*.parquet")) == []

    random_forced_paths = resolve_run_paths(tmp_path, "run_c_group_ineligible_random_forced")
    _setup_stage_a_with_spec(
        random_forced_paths,
        json.loads(json.dumps(random_auto_spec)),
    )
    _setup_stage_b_env(random_forced_paths, "g_random_env", random_rows)

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as random_error:
        run_stage_c(
            random_forced_paths,
            seed=0,
            split_policy=SplitPolicy.GROUP_HOLDOUT,
            visible_ratio=0.5,
        )

    assert type(random_error.value) is split_policy_module.SplitPolicyIneligibleError
    assert random_error.value.args == time_error.value.args
    assert str(random_error.value) == str(time_error.value)
    for filename in (
        "observations.jsonl",
        "summaries.jsonl",
        "data_refs.jsonl",
        "skipped_envs.jsonl",
        "manifest.json",
    ):
        assert not (random_forced_paths.stage_dirs["C"] / filename).exists()
    assert list(random_forced_paths.stage_dirs["C"].rglob("*.parquet")) == []


def test_run_stage_c_forced_spatial_region_ineligible_does_not_fall_through(
    tmp_path: Path,
) -> None:
    time_rows = [{"source_row_id": f"r{index}", "time": index, "y": index} for index in range(4)]
    time_auto_paths = resolve_run_paths(tmp_path, "run_c_spatial_ineligible_time_auto")
    time_auto_spec = minimal_envspec_record(
        env_id="env-s-time",
        env_name="s_time_env",
    )
    time_auto_spec["data_plan"]["n_observations"] = 1
    time_auto_spec["data_plan"]["grouping"] = []
    time_auto_spec["data_plan"]["covariates"] = ["time"]
    time_auto_spec["io_schema"]["feature_names"] = ["time"]
    time_auto_spec["io_schema"]["feature_types"] = {"time": "continuous"}
    time_auto_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in time_auto_spec["io_schema"]["feature_names"]
    }
    time_auto_spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(time_auto_paths, time_auto_spec)
    _setup_stage_b_env(time_auto_paths, "s_time_env", time_rows)

    random_rows = [{"source_row_id": f"r{index}", "value": index, "y": index} for index in range(4)]
    random_auto_paths = resolve_run_paths(tmp_path, "run_c_spatial_ineligible_random_auto")
    random_auto_spec = minimal_envspec_record(
        env_id="env-s-random",
        env_name="s_random_env",
    )
    random_auto_spec["data_plan"]["n_observations"] = 1
    random_auto_spec["data_plan"]["grouping"] = []
    random_auto_spec["data_plan"]["covariates"] = []
    random_auto_spec["io_schema"]["feature_names"] = ["value"]
    random_auto_spec["io_schema"]["feature_types"] = {"value": "continuous"}
    random_auto_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in random_auto_spec["io_schema"]["feature_names"]
    }
    random_auto_spec["generative_family"] = "regression_continuous"
    _setup_stage_a_with_spec(random_auto_paths, random_auto_spec)
    _setup_stage_b_env(random_auto_paths, "s_random_env", random_rows)

    time_auto_result = run_stage_c(
        time_auto_paths,
        seed=0,
        split_policy=SplitPolicy.AUTO,
        visible_ratio=0.5,
    )
    random_auto_result = run_stage_c(
        random_auto_paths,
        seed=0,
        split_policy=SplitPolicy.AUTO,
        visible_ratio=0.5,
    )

    time_auto_ref = json.loads(
        (time_auto_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    time_auto_visible = pd.read_parquet(
        time_auto_paths.stage_dirs["C"] / "data" / "env-s-time.parquet"
    )
    time_auto_hidden = pd.read_parquet(
        time_auto_paths.stage_dirs["C"] / "data" / "hidden" / "env-s-time.parquet"
    )
    assert time_auto_result.outputs["split_policy_counts"] == {"ordered_tail": 1}
    assert time_auto_ref["split_policy"] == "ordered_tail"
    assert time_auto_visible["source_row_id"].tolist() == ["r0", "r1"]
    assert time_auto_hidden["source_row_id"].tolist() == ["r2", "r3"]

    random_auto_ref = json.loads(
        (random_auto_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    random_auto_visible = pd.read_parquet(
        random_auto_paths.stage_dirs["C"] / "data" / "env-s-random.parquet"
    )
    random_auto_hidden = pd.read_parquet(
        random_auto_paths.stage_dirs["C"] / "data" / "hidden" / "env-s-random.parquet"
    )
    assert random_auto_result.outputs["split_policy_counts"] == {"random_row": 1}
    assert random_auto_ref["split_policy"] == "random_row"
    assert random_auto_visible["source_row_id"].tolist() == ["r2", "r3"]
    assert random_auto_hidden["source_row_id"].tolist() == ["r0", "r1"]

    time_forced_paths = resolve_run_paths(tmp_path, "run_c_spatial_ineligible_time_forced")
    _setup_stage_a_with_spec(
        time_forced_paths,
        json.loads(json.dumps(time_auto_spec)),
    )
    _setup_stage_b_env(time_forced_paths, "s_time_env", time_rows)

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as time_error:
        run_stage_c(
            time_forced_paths,
            seed=0,
            split_policy=SplitPolicy.SPATIAL_REGION,
            visible_ratio=0.5,
        )

    assert type(time_error.value) is split_policy_module.SplitPolicyIneligibleError
    for filename in (
        "observations.jsonl",
        "summaries.jsonl",
        "data_refs.jsonl",
        "skipped_envs.jsonl",
        "manifest.json",
    ):
        assert not (time_forced_paths.stage_dirs["C"] / filename).exists()
    assert list(time_forced_paths.stage_dirs["C"].rglob("*.parquet")) == []

    random_forced_paths = resolve_run_paths(tmp_path, "run_c_spatial_ineligible_random_forced")
    _setup_stage_a_with_spec(
        random_forced_paths,
        json.loads(json.dumps(random_auto_spec)),
    )
    _setup_stage_b_env(random_forced_paths, "s_random_env", random_rows)

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as random_error:
        run_stage_c(
            random_forced_paths,
            seed=0,
            split_policy=SplitPolicy.SPATIAL_REGION,
            visible_ratio=0.5,
        )

    assert type(random_error.value) is split_policy_module.SplitPolicyIneligibleError
    assert random_error.value.args == time_error.value.args
    assert str(random_error.value) == str(time_error.value)
    for filename in (
        "observations.jsonl",
        "summaries.jsonl",
        "data_refs.jsonl",
        "skipped_envs.jsonl",
        "manifest.json",
    ):
        assert not (random_forced_paths.stage_dirs["C"] / filename).exists()
    assert list(random_forced_paths.stage_dirs["C"].rglob("*.parquet")) == []


def test_run_stage_c_forced_ordered_tail_ineligible_does_not_fall_through(
    tmp_path: Path,
) -> None:
    random_rows = [{"source_row_id": f"r{index}", "value": index, "y": index} for index in range(4)]
    random_auto_paths = resolve_run_paths(tmp_path, "run_c_ordered_ineligible_random_auto")
    random_auto_spec = minimal_envspec_record(
        env_id="env-o-random",
        env_name="o_random_env",
    )
    random_auto_spec["data_plan"]["n_observations"] = 1
    random_auto_spec["data_plan"]["grouping"] = []
    random_auto_spec["data_plan"]["covariates"] = []
    random_auto_spec["io_schema"]["feature_names"] = ["value"]
    random_auto_spec["io_schema"]["feature_types"] = {"value": "continuous"}
    random_auto_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in random_auto_spec["io_schema"]["feature_names"]
    }
    random_auto_spec["generative_family"] = "regression_continuous"
    _setup_stage_a_with_spec(random_auto_paths, random_auto_spec)
    _setup_stage_b_env(random_auto_paths, "o_random_env", random_rows)

    group_rows = [
        {
            "source_row_id": f"r{index}",
            "group": "a" if index < 2 else "b",
            "value": index,
            "y": index,
        }
        for index in range(4)
    ]
    group_auto_paths = resolve_run_paths(tmp_path, "run_c_ordered_ineligible_group_auto")
    group_auto_spec = minimal_envspec_record(
        env_id="env-o-group",
        env_name="o_group_env",
    )
    group_auto_spec["data_plan"]["n_observations"] = 1
    group_auto_spec["data_plan"]["grouping"] = ["group"]
    group_auto_spec["data_plan"]["covariates"] = []
    group_auto_spec["io_schema"]["feature_names"] = ["group", "value"]
    group_auto_spec["io_schema"]["feature_types"] = {"group": "continuous", "value": "continuous"}
    group_auto_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in group_auto_spec["io_schema"]["feature_names"]
    }
    group_auto_spec["generative_family"] = "hierarchical_grouped"
    _setup_stage_a_with_spec(group_auto_paths, group_auto_spec)
    _setup_stage_b_env(group_auto_paths, "o_group_env", group_rows)

    random_auto_result = run_stage_c(
        random_auto_paths,
        seed=0,
        split_policy=SplitPolicy.AUTO,
        visible_ratio=0.5,
    )
    group_auto_result = run_stage_c(
        group_auto_paths,
        seed=0,
        split_policy=SplitPolicy.AUTO,
        visible_ratio=0.5,
    )

    random_auto_ref = json.loads(
        (random_auto_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    random_auto_visible = pd.read_parquet(
        random_auto_paths.stage_dirs["C"] / "data" / "env-o-random.parquet"
    )
    random_auto_hidden = pd.read_parquet(
        random_auto_paths.stage_dirs["C"] / "data" / "hidden" / "env-o-random.parquet"
    )
    assert random_auto_result.outputs["split_policy_counts"] == {"random_row": 1}
    assert random_auto_ref["split_policy"] == "random_row"
    assert random_auto_visible["source_row_id"].tolist() == ["r0", "r2"]
    assert random_auto_hidden["source_row_id"].tolist() == ["r1", "r3"]

    group_auto_ref = json.loads(
        (group_auto_paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8")
    )
    group_auto_visible = pd.read_parquet(
        group_auto_paths.stage_dirs["C"] / "data" / "env-o-group.parquet"
    )
    group_auto_hidden = pd.read_parquet(
        group_auto_paths.stage_dirs["C"] / "data" / "hidden" / "env-o-group.parquet"
    )
    assert group_auto_result.outputs["split_policy_counts"] == {"group_holdout": 1}
    assert group_auto_ref["split_policy"] == "group_holdout"
    assert group_auto_visible["source_row_id"].tolist() == ["r2", "r3"]
    assert group_auto_hidden["source_row_id"].tolist() == ["r0", "r1"]

    random_forced_paths = resolve_run_paths(tmp_path, "run_c_ordered_ineligible_random_forced")
    _setup_stage_a_with_spec(
        random_forced_paths,
        json.loads(json.dumps(random_auto_spec)),
    )
    _setup_stage_b_env(random_forced_paths, "o_random_env", random_rows)

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as random_error:
        run_stage_c(
            random_forced_paths,
            seed=0,
            split_policy=SplitPolicy.ORDERED_TAIL,
            visible_ratio=0.5,
        )

    assert type(random_error.value) is split_policy_module.SplitPolicyIneligibleError
    for filename in (
        "observations.jsonl",
        "summaries.jsonl",
        "data_refs.jsonl",
        "skipped_envs.jsonl",
        "manifest.json",
    ):
        assert not (random_forced_paths.stage_dirs["C"] / filename).exists()
    assert list(random_forced_paths.stage_dirs["C"].rglob("*.parquet")) == []

    group_forced_paths = resolve_run_paths(tmp_path, "run_c_ordered_ineligible_group_forced")
    _setup_stage_a_with_spec(
        group_forced_paths,
        json.loads(json.dumps(group_auto_spec)),
    )
    _setup_stage_b_env(group_forced_paths, "o_group_env", group_rows)

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as group_error:
        run_stage_c(
            group_forced_paths,
            seed=0,
            split_policy=SplitPolicy.ORDERED_TAIL,
            visible_ratio=0.5,
        )

    assert type(group_error.value) is split_policy_module.SplitPolicyIneligibleError
    assert group_error.value.args == random_error.value.args
    assert str(group_error.value) == str(random_error.value)
    for filename in (
        "observations.jsonl",
        "summaries.jsonl",
        "data_refs.jsonl",
        "skipped_envs.jsonl",
        "manifest.json",
    ):
        assert not (group_forced_paths.stage_dirs["C"] / filename).exists()
    assert list(group_forced_paths.stage_dirs["C"].rglob("*.parquet")) == []


def test_run_stage_c_forced_policy_failure_clears_stale_outputs(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_forced_policy_failure_clears_stale")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    time_spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    time_spec["data_plan"]["n_observations"] = 1
    time_spec["data_plan"]["grouping"] = []
    time_spec["data_plan"]["covariates"] = ["time"]
    time_spec["io_schema"]["feature_names"] = ["time"]
    time_spec["io_schema"]["feature_types"] = {"time": "continuous"}
    time_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in time_spec["io_schema"]["feature_names"]
    }
    time_spec["generative_family"] = "time_series_level_seasonality"
    random_spec = minimal_envspec_record(env_id="env-random", env_name="random_env")
    random_spec["data_plan"]["n_observations"] = 1
    random_spec["data_plan"]["grouping"] = []
    random_spec["data_plan"]["covariates"] = []
    random_spec["io_schema"]["feature_names"] = ["value"]
    random_spec["io_schema"]["feature_types"] = {"value": "continuous"}
    random_spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in random_spec["io_schema"]["feature_names"]
    }
    random_spec["generative_family"] = "regression_continuous"
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [time_spec, random_spec])
    _setup_stage_b_env(
        paths,
        "time_env",
        [{"time": index, "y": float(index)} for index in range(4)],
    )
    _setup_stage_b_env(
        paths,
        "random_env",
        [{"value": index, "y": float(index)} for index in range(4)],
    )
    _preseed_complete_stage_c_tree(paths.stage_dirs["C"])
    stale_sibling = paths.run_dir / ".stage_C_data.attempt-stale"
    stale_sibling.mkdir(parents=True, exist_ok=True)
    (stale_sibling / "sentinel.bin").write_bytes(b"stale-attempt")

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as exc_info:
        run_stage_c(
            paths,
            seed=0,
            split_policy=SplitPolicy.ORDERED_TAIL,
            visible_ratio=0.5,
        )

    assert type(exc_info.value) is split_policy_module.SplitPolicyIneligibleError

    stage_dir = paths.stage_dirs["C"]
    for filename in (
        "observations.jsonl",
        "summaries.jsonl",
        "data_refs.jsonl",
        "skipped_envs.jsonl",
        "manifest.json",
    ):
        assert not (stage_dir / filename).exists()
    assert not (stage_dir / "data").exists()
    assert list(stage_dir.rglob("*.parquet")) == []
    assert list(paths.run_dir.glob(".stage_C_data.attempt-*")) == []


def test_split_visible_hidden_returns_group_holdout_policy_enum() -> None:
    spec = EnvSpec.model_validate(minimal_envspec_record(env_id="env-group", env_name="group_env"))
    frame = pd.DataFrame(
        [
            {"source_row_id": "a1", "group": "a", "y": 1.0},
            {"source_row_id": "a2", "group": "a", "y": 2.0},
            {"source_row_id": "b1", "group": "b", "y": 3.0},
            {"source_row_id": "b2", "group": "b", "y": 4.0},
            {"source_row_id": "c1", "group": "c", "y": 5.0},
            {"source_row_id": "c2", "group": "c", "y": 6.0},
        ]
    )

    result = split_visible_hidden(
        frame,
        spec,
        split_policy=SplitPolicy.GROUP_HOLDOUT,
        seed=7,
        visible_ratio=0.5,
    )

    assert type(result.policy_name) is SplitPolicy
    assert result.policy_name is SplitPolicy.GROUP_HOLDOUT


def test_split_visible_hidden_returns_spatial_region_policy_enum() -> None:
    spec_record = minimal_envspec_record(env_id="env-space-forced", env_name="space_fixture")
    spec_record["data_plan"]["grouping"] = []
    spec_record["data_plan"]["covariates"] = ["x", "y"]
    spec_record["io_schema"]["feature_names"] = ["x", "y"]
    spec_record["io_schema"]["feature_types"] = {"x": "continuous", "y": "continuous"}
    spec_record["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec_record["io_schema"]["feature_names"]
    }
    spec_record["io_schema"]["target_name"] = "intensity"
    spec_record["generative_family"] = "spatial_signal_source"
    spec = EnvSpec.model_validate(spec_record)
    frame = pd.DataFrame(
        [
            {"source_row_id": "r0", "x": 0, "y": 0, "intensity": 0},
            {"source_row_id": "r1", "x": 0, "y": 1, "intensity": 1},
            {"source_row_id": "r2", "x": 0, "y": 2, "intensity": 2},
            {"source_row_id": "r3", "x": 10, "y": 10, "intensity": 3},
            {"source_row_id": "r4", "x": 10, "y": 11, "intensity": 4},
            {"source_row_id": "r5", "x": 10, "y": 12, "intensity": 5},
        ]
    )

    result = split_visible_hidden(
        frame,
        spec,
        split_policy=SplitPolicy.SPATIAL_REGION,
        seed=0,
        visible_ratio=0.5,
    )

    assert type(result.policy_name) is SplitPolicy
    assert result.policy_name is SplitPolicy.SPATIAL_REGION


def test_split_visible_hidden_returns_ordered_tail_policy_enum() -> None:
    spec_record = minimal_envspec_record(env_id="env-time-forced", env_name="time_fixture")
    spec_record["data_plan"]["grouping"] = []
    spec_record["data_plan"]["covariates"] = ["time"]
    spec_record["io_schema"]["feature_names"] = ["time"]
    spec_record["io_schema"]["feature_types"] = {"time": "continuous"}
    spec_record["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec_record["io_schema"]["feature_names"]
    }
    spec_record["io_schema"]["target_name"] = "value"
    spec_record["generative_family"] = "time_series_level_seasonality"
    spec = EnvSpec.model_validate(spec_record)
    frame = pd.DataFrame(
        [
            {"source_row_id": "r0", "time": 0, "value": 0},
            {"source_row_id": "r1", "time": 1, "value": 1},
            {"source_row_id": "r2", "time": 2, "value": 2},
            {"source_row_id": "r3", "time": 3, "value": 3},
            {"source_row_id": "r4", "time": 4, "value": 4},
            {"source_row_id": "r5", "time": 5, "value": 5},
        ]
    )

    result = split_visible_hidden(
        frame,
        spec,
        split_policy=SplitPolicy.ORDERED_TAIL,
        seed=0,
        visible_ratio=0.5,
    )

    assert type(result.policy_name) is SplitPolicy
    assert result.policy_name is SplitPolicy.ORDERED_TAIL


def _forced_no_cascade_frame_and_spec(
    case_key: str, n: int, env_id: str
) -> tuple[pd.DataFrame, EnvSpec]:
    """Build a frame and EnvSpec where the forced resolver is ineligible, a
    second resolver is eligible, and the third is ineligible.

    AUTO therefore resolves through the eligible resolver, which proves the
    data is genuinely eligible for that resolver and is not silently falling
    back to RANDOM_ROW. The forced named policy must still raise instead of
    falling through to that eligible resolver.
    """
    record = minimal_envspec_record(env_id=env_id, env_name=env_id)
    if case_key == "spatial_ineligible_group_eligible":
        record["data_plan"]["grouping"] = ["group"]
        record["data_plan"]["covariates"] = ["value"]
        record["io_schema"]["feature_names"] = ["group", "value"]
        record["io_schema"]["feature_types"] = {"group": "continuous", "value": "continuous"}
        record["io_schema"]["feature_domains"] = {
            name: [0.0, 10.0] for name in record["io_schema"]["feature_names"]
        }
        record["io_schema"]["target_name"] = "outcome"
        record["generative_family"] = "hierarchical_grouped"
        rows = [
            {
                "source_row_id": f"r{i}",
                "group": "a" if i % 2 == 0 else "b",
                "value": float(i),
                "outcome": float(i),
            }
            for i in range(n)
        ]
    else:
        # Single group makes the group resolver ineligible, and no temporal
        # column makes the ordered resolver ineligible, so the spatial
        # resolver is the only eligible one.
        record["data_plan"]["grouping"] = ["group"]
        record["data_plan"]["covariates"] = ["x", "y"]
        record["io_schema"]["feature_names"] = ["group", "x", "y"]
        record["io_schema"]["feature_types"] = {
            "group": "continuous",
            "x": "continuous",
            "y": "continuous",
        }
        record["io_schema"]["feature_domains"] = {
            name: [0.0, 10.0] for name in record["io_schema"]["feature_names"]
        }
        record["io_schema"]["target_name"] = "intensity"
        record["generative_family"] = "regression_continuous"
        rows = [
            {
                "source_row_id": f"r{i}",
                "group": "a",
                "x": float(i),
                "y": float(2 * i),
                "intensity": float(i),
            }
            for i in range(n)
        ]
    return pd.DataFrame(rows), EnvSpec.model_validate(record)


@pytest.mark.parametrize(
    (
        "case_key",
        "forced_policy",
        "eligible_policy",
        "n",
        "visible_ratio",
        "seed",
        "env_id",
    ),
    [
        pytest.param(
            "group_ineligible_spatial_eligible",
            SplitPolicy.GROUP_HOLDOUT,
            SplitPolicy.SPATIAL_REGION,
            8,
            3 / 8,
            11,
            "env-group-ineligible-spatial",
            id="group-ineligible-spatial-eligible",
        ),
        pytest.param(
            "spatial_ineligible_group_eligible",
            SplitPolicy.SPATIAL_REGION,
            SplitPolicy.GROUP_HOLDOUT,
            16,
            5 / 16,
            23,
            "env-spatial-ineligible-group",
            id="spatial-ineligible-group-eligible",
        ),
        pytest.param(
            "ordered_ineligible_spatial_eligible",
            SplitPolicy.ORDERED_TAIL,
            SplitPolicy.SPATIAL_REGION,
            32,
            7 / 32,
            37,
            "env-ordered-ineligible-spatial",
            id="ordered-ineligible-spatial-eligible",
        ),
    ],
)
def test_forced_split_policy_raises_even_when_other_resolver_eligible(
    case_key: str,
    forced_policy: SplitPolicy,
    eligible_policy: SplitPolicy,
    n: int,
    visible_ratio: float,
    seed: int,
    env_id: str,
) -> None:
    frame, spec = _forced_no_cascade_frame_and_spec(case_key, n, env_id)

    auto_split = split_visible_hidden(
        frame,
        spec,
        split_policy=SplitPolicy.AUTO,
        seed=seed,
        visible_ratio=visible_ratio,
    )
    assert auto_split.policy_name is eligible_policy

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as error:
        split_visible_hidden(
            frame,
            spec,
            split_policy=forced_policy,
            seed=seed,
            visible_ratio=visible_ratio,
        )
    assert type(error.value) is split_policy_module.SplitPolicyIneligibleError


def test_split_visible_hidden_returns_random_row_policy_enum() -> None:
    spec_record = minimal_envspec_record(env_id="hash-random", env_name="random_fixture")
    spec_record["data_plan"]["grouping"] = []
    spec_record["data_plan"]["covariates"] = ["feature"]
    spec_record["io_schema"]["feature_names"] = ["feature"]
    spec_record["io_schema"]["feature_types"] = {"feature": "continuous"}
    spec_record["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec_record["io_schema"]["feature_names"]
    }
    spec_record["io_schema"]["target_name"] = "value"
    spec_record["generative_family"] = "linear_regression"
    spec = EnvSpec.model_validate(spec_record)
    frame = pd.DataFrame(
        [
            {"source_row_id": "r0", "group": "A", "feature": 0, "value": 0},
            {"source_row_id": "r1", "group": "A", "feature": 1, "value": 1},
            {"source_row_id": "r2", "group": "A", "feature": 2, "value": 2},
            {"source_row_id": "r3", "group": "B", "feature": 3, "value": 3},
            {"source_row_id": "r4", "group": "B", "feature": 4, "value": 4},
            {"source_row_id": "r5", "group": "B", "feature": 5, "value": 5},
        ]
    )

    result = split_visible_hidden(
        frame,
        spec,
        split_policy=SplitPolicy.RANDOM_ROW,
        seed=2,
        visible_ratio=0.5,
    )

    assert type(result.policy_name) is SplitPolicy
    assert result.policy_name is SplitPolicy.RANDOM_ROW


def test_split_visible_hidden_forced_random_row_ignores_all_earlier_eligible_routes() -> None:
    grouped_record = minimal_envspec_record(env_id="hash-random-alt", env_name="random_fixture_alt")
    grouped_record["data_plan"]["n_observations"] = 1
    grouped_record["data_plan"]["grouping"] = ["group"]
    grouped_record["data_plan"]["covariates"] = ["time", "x", "y"]
    grouped_record["io_schema"]["feature_names"] = ["group", "time", "x", "y"]
    grouped_record["io_schema"]["feature_types"] = {
        "group": "continuous",
        "time": "continuous",
        "x": "continuous",
        "y": "continuous",
    }
    grouped_record["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in grouped_record["io_schema"]["feature_names"]
    }
    grouped_record["io_schema"]["target_name"] = "value"
    grouped_record["generative_family"] = "spatial_signal_source"
    grouped_spec = EnvSpec.model_validate(grouped_record)

    ungrouped_record = json.loads(json.dumps(grouped_record))
    ungrouped_record["data_plan"]["grouping"] = []
    ungrouped_spec = EnvSpec.model_validate(ungrouped_record)
    frame = pd.DataFrame(
        [
            {
                "source_row_id": "r0",
                "group": "A",
                "time": 0,
                "x": 0,
                "y": 0,
                "value": 0,
            },
            {
                "source_row_id": "r1",
                "group": "A",
                "time": 1,
                "x": 0,
                "y": 1,
                "value": 1,
            },
            {
                "source_row_id": "r2",
                "group": "A",
                "time": 2,
                "x": 0,
                "y": 2,
                "value": 2,
            },
            {
                "source_row_id": "r3",
                "group": "B",
                "time": 3,
                "x": 10,
                "y": 10,
                "value": 3,
            },
            {
                "source_row_id": "r4",
                "group": "B",
                "time": 4,
                "x": 10,
                "y": 11,
                "value": 4,
            },
            {
                "source_row_id": "r5",
                "group": "B",
                "time": 5,
                "x": 10,
                "y": 12,
                "value": 5,
            },
        ]
    )

    grouped_auto = split_visible_hidden(
        frame,
        grouped_spec,
        split_policy=SplitPolicy.AUTO,
        seed=17,
        visible_ratio=0.5,
    )
    ungrouped_auto = split_visible_hidden(
        frame,
        ungrouped_spec,
        split_policy=SplitPolicy.AUTO,
        seed=17,
        visible_ratio=0.5,
    )

    assert type(grouped_auto.policy_name) is SplitPolicy
    assert grouped_auto.policy_name is SplitPolicy.GROUP_HOLDOUT
    assert grouped_auto.visible["source_row_id"].tolist() == ["r3", "r4", "r5"]
    assert grouped_auto.hidden["source_row_id"].tolist() == ["r0", "r1", "r2"]
    assert type(ungrouped_auto.policy_name) is SplitPolicy
    assert ungrouped_auto.policy_name is SplitPolicy.SPATIAL_REGION
    assert ungrouped_auto.visible["source_row_id"].tolist() == ["r0", "r1", "r2"]
    assert ungrouped_auto.hidden["source_row_id"].tolist() == ["r3", "r4", "r5"]

    grouped_random = split_visible_hidden(
        frame,
        grouped_spec,
        split_policy=SplitPolicy.RANDOM_ROW,
        seed=17,
        visible_ratio=0.5,
    )
    ungrouped_random = split_visible_hidden(
        frame,
        ungrouped_spec,
        split_policy=SplitPolicy.RANDOM_ROW,
        seed=17,
        visible_ratio=0.5,
    )

    columns = ["source_row_id", "group", "time", "x", "y", "value"]
    visible_records = [
        {"source_row_id": "r2", "group": "A", "time": 2, "x": 0, "y": 2, "value": 2},
        {
            "source_row_id": "r3",
            "group": "B",
            "time": 3,
            "x": 10,
            "y": 10,
            "value": 3,
        },
        {
            "source_row_id": "r5",
            "group": "B",
            "time": 5,
            "x": 10,
            "y": 12,
            "value": 5,
        },
    ]
    hidden_records = [
        {"source_row_id": "r0", "group": "A", "time": 0, "x": 0, "y": 0, "value": 0},
        {"source_row_id": "r1", "group": "A", "time": 1, "x": 0, "y": 1, "value": 1},
        {
            "source_row_id": "r4",
            "group": "B",
            "time": 4,
            "x": 10,
            "y": 11,
            "value": 4,
        },
    ]
    for result in (grouped_random, ungrouped_random):
        assert type(result.policy_name) is SplitPolicy
        assert result.policy_name is SplitPolicy.RANDOM_ROW
        assert result.visible.columns.tolist() == columns
        assert result.hidden.columns.tolist() == columns
        assert result.visible.to_dict(orient="records") == visible_records
        assert result.hidden.to_dict(orient="records") == hidden_records
    assert grouped_random.visible.to_dict(orient="records") == ungrouped_random.visible.to_dict(
        orient="records"
    )
    assert grouped_random.hidden.to_dict(orient="records") == ungrouped_random.hidden.to_dict(
        orient="records"
    )


def test_run_stage_c_temporal_split_holds_out_tail_rows(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_time_holdout")
    spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["time"]
    spec["io_schema"]["feature_names"] = ["time"]
    spec["io_schema"]["feature_types"] = {"time": "continuous"}
    spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
    }
    spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(paths, spec)
    _setup_stage_b_env(
        paths,
        "time_env",
        [{"time": idx, "y": float(idx)} for idx in range(10)],
    )

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.7)

    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / "env-time.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-time.parquet")
    assert visible["time"].tolist() == list(range(7))
    assert hidden["time"].tolist() == [7, 8, 9]

    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_policy_counts"] == {"ordered_tail": 1}


def test_run_stage_c_spatial_split_holds_out_local_region(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_spatial_holdout")
    spec = minimal_envspec_record(env_id="env-space", env_name="space_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["x", "y"]
    spec["io_schema"]["feature_names"] = ["x", "y"]
    spec["io_schema"]["feature_types"] = {"x": "continuous", "y": "continuous"}
    spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
    }
    spec["io_schema"]["target_name"] = "intensity"
    spec["generative_family"] = "spatial_signal_source"
    _setup_stage_a_with_spec(paths, spec)
    _setup_stage_b_env(
        paths,
        "space_env",
        [
            {"x": 0.0, "y": 0.0, "intensity": 0.0},
            {"x": 0.0, "y": 1.0, "intensity": 1.0},
            {"x": 0.0, "y": 2.0, "intensity": 2.0},
            {"x": 10.0, "y": 10.0, "intensity": 3.0},
            {"x": 10.0, "y": 11.0, "intensity": 4.0},
            {"x": 10.0, "y": 12.0, "intensity": 5.0},
        ],
    )

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.5)

    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-space.parquet")
    assert hidden["x"].nunique() == 1

    ref = json.loads(
        (paths.stage_dirs["C"] / "data_refs.jsonl").read_text(encoding="utf-8").strip()
    )
    assert ref["split_policy"] == "spatial_region"


# ---------------------------------------------------------------------------
# Missing compiled env module
# ---------------------------------------------------------------------------


def test_run_stage_c_skips_when_env_module_missing(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_missing_mod")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    specs = [
        minimal_envspec_record(env_id="env-good", env_name="good_env"),
        minimal_envspec_record(env_id="env-m", env_name="missing_env"),
    ]
    for spec in specs:
        spec["data_plan"]["n_observations"] = 1
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", specs)
    _setup_stage_b_env(paths, "good_env", [{"y": 1.0}, {"y": 2.0}])

    result = run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.8)

    assert result.status == "success"
    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["materialized_env_count"] == 1
    assert manifest["skipped_env_count"] == 1
    skipped = [
        json.loads(line)
        for line in (paths.stage_dirs["C"] / "skipped_envs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert skipped[0]["env_id"] == "env-m"


def test_run_stage_c_requires_split_policy(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_requires_split_policy")
    spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["time"]
    spec["io_schema"]["feature_names"] = ["time"]
    spec["io_schema"]["feature_types"] = {"time": "continuous"}
    spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
    }
    spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(paths, spec)
    _setup_stage_b_env(
        paths,
        "time_env",
        [
            {"time": 0, "y": 0.0},
            {"time": 1, "y": 1.0},
            {"time": 2, "y": 2.0},
            {"time": 3, "y": 3.0},
        ],
    )
    _preseed_complete_stage_c_tree(paths.stage_dirs["C"])
    prior_stage_c = _snapshot_tree(paths.stage_dirs["C"])

    with pytest.raises(TypeError, match="split_policy"):
        run_stage_c(paths, seed=0, visible_ratio=0.5)
    assert _snapshot_tree(paths.stage_dirs["C"]) == prior_stage_c

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.5)

    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / "env-time.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-time.parquet")
    assert visible["time"].tolist() == [0, 1]
    assert hidden["time"].tolist() == [2, 3]

    data_refs = [
        json.loads(line)
        for line in (paths.stage_dirs["C"] / "data_refs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(data_refs) == 1
    assert data_refs[0]["split_policy"] == "ordered_tail"

    manifest = json.loads((paths.stage_dirs["C"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_policy_counts"] == {"ordered_tail": 1}


def test_run_stage_c_requires_visible_ratio(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_requires_visible_ratio")
    spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["time"]
    spec["io_schema"]["feature_names"] = ["time"]
    spec["io_schema"]["feature_types"] = {"time": "continuous"}
    spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
    }
    spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(paths, spec)
    _setup_stage_b_env(
        paths,
        "time_env",
        [
            {"time": 0, "y": 0.0},
            {"time": 1, "y": 1.0},
            {"time": 2, "y": 2.0},
            {"time": 3, "y": 3.0},
        ],
    )
    _preseed_complete_stage_c_tree(paths.stage_dirs["C"])
    prior_stage_c = _snapshot_tree(paths.stage_dirs["C"])

    with pytest.raises(TypeError, match="visible_ratio"):
        run_stage_c(paths, seed=0, split_policy="auto")
    assert _snapshot_tree(paths.stage_dirs["C"]) == prior_stage_c

    run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.5)

    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / "env-time.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-time.parquet")
    assert visible["time"].tolist() == [0, 1]
    assert hidden["time"].tolist() == [2, 3]


@pytest.mark.parametrize(
    ("visible_ratio", "expected_visible", "expected_hidden"),
    [
        (0.5, [0, 1, 2, 3], [4, 5, 6, 7]),
        (0.625, [0, 1, 2, 3, 4], [5, 6, 7]),
    ],
)
def test_run_stage_c_preserves_each_visible_ratio_value(
    tmp_path: Path,
    visible_ratio: float,
    expected_visible: list[int],
    expected_hidden: list[int],
) -> None:
    ratio_token = str(visible_ratio).replace(".", "-")
    paths = resolve_run_paths(tmp_path, f"run_c_visible_ratio_{ratio_token}")
    spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["time"]
    spec["io_schema"]["feature_names"] = ["time"]
    spec["io_schema"]["feature_types"] = {"time": "continuous"}
    spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
    }
    spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(paths, spec)
    _setup_stage_b_env(
        paths,
        "time_env",
        [{"time": index, "y": float(index)} for index in range(8)],
    )

    run_stage_c(
        paths,
        seed=0,
        split_policy="auto",
        visible_ratio=visible_ratio,
    )

    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / "env-time.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-time.parquet")
    assert visible["time"].tolist() == expected_visible
    assert hidden["time"].tolist() == expected_hidden


if st is not None:
    _BINARY_EXACT_EXPONENT = st.shared(
        st.integers(min_value=1, max_value=6),
        key="stage_c_binary_exact_exponent",
    )
else:
    _BINARY_EXACT_EXPONENT = None


if st is not None and settings is not None and example is not None and given is not None:

    @settings(max_examples=32, deadline=None, derandomize=True, database=None)
    @example(n=8, k=3)
    @example(n=128, k=3)
    @given(
        n=_BINARY_EXACT_EXPONENT.map(lambda exponent: 1 << exponent),
        k=_BINARY_EXACT_EXPONENT.flatmap(
            lambda exponent: st.integers(min_value=1, max_value=(1 << exponent) - 1)
        ),
    )
    def test_run_stage_c_preserves_binary_exact_visible_count(n: int, k: int) -> None:
        visible_ratio = k / n
        assert n * visible_ratio == k

        with TemporaryDirectory() as temp_dir:
            paths = resolve_run_paths(Path(temp_dir), "run_c_binary_exact_visible_count")
            spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
            spec["data_plan"]["n_observations"] = 1
            spec["data_plan"]["grouping"] = []
            spec["data_plan"]["covariates"] = ["time"]
            spec["io_schema"]["feature_names"] = ["time"]
            spec["io_schema"]["feature_types"] = {"time": "continuous"}
            spec["io_schema"]["feature_domains"] = {
                name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
            }
            spec["generative_family"] = "time_series_level_seasonality"
            _setup_stage_a_with_spec(paths, spec)
            _setup_stage_b_env(
                paths,
                "time_env",
                [
                    {"source_position": position, "time": position, "y": float(position)}
                    for position in range(n)
                ],
            )

            result = run_stage_c(
                paths,
                seed=0,
                split_policy="auto",
                visible_ratio=visible_ratio,
            )

            assert result.status == "success"
            visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / "env-time.parquet")
            hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-time.parquet")
            assert visible["source_position"].tolist() == list(range(k))
            assert hidden["source_position"].tolist() == list(range(k, n))
else:

    @pytest.mark.skip(reason="hypothesis is not installed")
    def test_run_stage_c_preserves_binary_exact_visible_count() -> None:
        pass


@pytest.mark.parametrize(
    "visible_ratio",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative-one"),
        pytest.param(1.0, id="one"),
        pytest.param(2.0, id="two"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_run_stage_c_rejects_invalid_visible_ratio(
    tmp_path: Path,
    visible_ratio: float,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_rejects_invalid_visible_ratio")
    spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["time"]
    spec["io_schema"]["feature_names"] = ["time"]
    spec["io_schema"]["feature_types"] = {"time": "continuous"}
    spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
    }
    spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(paths, spec)
    _setup_stage_b_env(
        paths,
        "time_env",
        [
            {"time": 0, "y": 0.0},
            {"time": 1, "y": 1.0},
            {"time": 2, "y": 2.0},
            {"time": 3, "y": 3.0},
        ],
    )
    _preseed_complete_stage_c_tree(paths.stage_dirs["C"])
    prior_stage_c = _snapshot_tree(paths.stage_dirs["C"])

    with pytest.raises(Exception) as exc_info:
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=visible_ratio)

    error_type = getattr(split_policy_module, "SplitPolicyIneligibleError", None)
    assert error_type is not None
    assert type(exc_info.value) is error_type
    assert _snapshot_tree(paths.stage_dirs["C"]) == prior_stage_c


def test_run_stage_c_rejects_unknown_policy_before_creating_stage_path(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_rejects_unknown_policy_before_stage_path")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [])
    assert not paths.stage_dirs["C"].exists()

    with pytest.raises(ValueError) as exc_info:
        run_stage_c(paths, seed=0, split_policy="not-a-policy", visible_ratio=0.5)

    assert type(exc_info.value) is ValueError
    assert not paths.stage_dirs["C"].exists()


def test_run_stage_c_rejects_invalid_ratio_before_creating_stage_path(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_rejects_invalid_ratio_before_stage_path")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [])
    assert not paths.stage_dirs["C"].exists()

    with pytest.raises(Exception) as exc_info:
        run_stage_c(paths, seed=0, split_policy="auto", visible_ratio=0.0)

    error_type = getattr(split_policy_module, "SplitPolicyIneligibleError", None)
    assert error_type is not None
    assert type(exc_info.value) is error_type
    assert not paths.stage_dirs["C"].exists()


@pytest.mark.parametrize("split_policy", ["not-a-policy", None, "AUTO", " auto "])
def test_run_stage_c_rejects_invalid_split_policy_without_mutating_prior_output(
    tmp_path: Path,
    split_policy: object,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_c_rejects_invalid_split_policy")
    spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["time"]
    spec["io_schema"]["feature_names"] = ["time"]
    spec["io_schema"]["feature_types"] = {"time": "continuous"}
    spec["io_schema"]["feature_domains"] = {
        name: [0.0, 10.0] for name in spec["io_schema"]["feature_names"]
    }
    spec["generative_family"] = "time_series_level_seasonality"
    _setup_stage_a_with_spec(paths, spec)
    _setup_stage_b_env(
        paths,
        "time_env",
        [
            {"time": 0, "y": 0.0},
            {"time": 1, "y": 1.0},
            {"time": 2, "y": 2.0},
            {"time": 3, "y": 3.0},
        ],
    )
    _preseed_complete_stage_c_tree(paths.stage_dirs["C"])
    prior_stage_c = _snapshot_tree(paths.stage_dirs["C"])

    with pytest.raises(ValueError) as exc_info:
        run_stage_c(paths, seed=0, split_policy=split_policy, visible_ratio=0.5)

    assert type(exc_info.value) is ValueError
    assert _snapshot_tree(paths.stage_dirs["C"]) == prior_stage_c


@pytest.mark.parametrize(
    ("env_id", "forced_policy", "alternate_policy"),
    [
        ("env-group-only", SplitPolicy.SPATIAL_REGION, SplitPolicy.GROUP_HOLDOUT),
        ("env-space-only", SplitPolicy.GROUP_HOLDOUT, SplitPolicy.SPATIAL_REGION),
        ("env-space-no-time", SplitPolicy.ORDERED_TAIL, SplitPolicy.SPATIAL_REGION),
    ],
)
@pytest.mark.parametrize(
    ("n", "k", "seed"),
    [
        (8, 3, 0),
        (16, 5, 1),
        (32, 9, 2),
        (64, 17, 3),
        (128, 33, 4),
    ],
)
def test_split_visible_hidden_forced_ineligible_does_not_fall_through(
    env_id: str,
    forced_policy: SplitPolicy,
    alternate_policy: SplitPolicy,
    n: int,
    k: int,
    seed: int,
) -> None:
    visible_ratio = k / n
    assert n * visible_ratio == k

    spec_record = minimal_envspec_record(
        env_id=env_id,
        env_name=f"{env_id.replace('-', '_')}_env",
    )
    if alternate_policy is SplitPolicy.GROUP_HOLDOUT:
        spec_record["data_plan"]["grouping"] = ["group"]
        spec_record["data_plan"]["covariates"] = []
        spec_record["io_schema"]["feature_names"] = ["group"]
        spec_record["io_schema"]["feature_types"] = {"group": "continuous"}
        spec_record["io_schema"]["feature_domains"] = {
            name: [0.0, 10.0] for name in spec_record["io_schema"]["feature_names"]
        }
        spec_record["io_schema"]["target_name"] = "y"
        spec_record["generative_family"] = "hierarchical_grouped"
        frame = pd.DataFrame(
            [
                {
                    "source_row_id": f"r{i}",
                    "group": f"g{i % 3}",
                    "y": float(i),
                }
                for i in range(n)
            ]
        )
    else:
        spec_record["data_plan"]["grouping"] = []
        spec_record["data_plan"]["covariates"] = ["x", "y"]
        spec_record["io_schema"]["feature_names"] = ["x", "y"]
        spec_record["io_schema"]["feature_types"] = {"x": "continuous", "y": "continuous"}
        spec_record["io_schema"]["feature_domains"] = {
            name: [0.0, 10.0] for name in spec_record["io_schema"]["feature_names"]
        }
        spec_record["io_schema"]["target_name"] = "intensity"
        spec_record["generative_family"] = "spatial_signal_source"
        frame = pd.DataFrame(
            [
                {
                    "source_row_id": f"r{i}",
                    "x": float(i % 4),
                    "y": float(i // 4),
                    "intensity": float(i),
                }
                for i in range(n)
            ]
        )
    spec = EnvSpec.model_validate(spec_record)

    auto_result = split_visible_hidden(
        frame,
        spec,
        split_policy=SplitPolicy.AUTO,
        seed=seed,
        visible_ratio=visible_ratio,
    )
    assert type(auto_result.policy_name) is SplitPolicy
    assert auto_result.policy_name is alternate_policy

    with pytest.raises(split_policy_module.SplitPolicyIneligibleError) as exc_info:
        split_visible_hidden(
            frame,
            spec,
            split_policy=forced_policy,
            seed=seed,
            visible_ratio=visible_ratio,
        )
    assert type(exc_info.value) is split_policy_module.SplitPolicyIneligibleError

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import ppl_synth_data.cli as cli_mod
import ppl_synth_data.cli.runs as cli_runs
from ppl_synth_data.cli import app
from ppl_synth_data.pipeline.artifacts import resolve_run_paths, stage_is_resumable
from ppl_synth_data.pipeline.contracts import STAGE_ORDER
from ppl_synth_data.pipeline.runner import run_all
from ppl_synth_data.pipeline.types import StageResult
from tests.pipeline.helpers import minimal_envspec_record, write_jsonl

runner = CliRunner()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPLDF_COMMAND = ["uv", "run", "--project", str(PROJECT_ROOT), "--frozen", "ppldf"]
LIMIT1_RUN = PROJECT_ROOT / "artifacts/runs/example-pymc-limit1"
LIMIT2_RUN = PROJECT_ROOT / "artifacts/runs/example-pymc-limit2"


def _write_temporal_stage_c_input(run_root: Path, run_id: str, *, row_count: int = 4) -> Path:
    run_dir = run_root / run_id
    stage_a = run_dir / "stage_A_envspec"
    stage_b_envs = run_dir / "stage_B_envs" / "envs"
    stage_a.mkdir(parents=True)
    stage_b_envs.mkdir(parents=True)

    spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["time"]
    spec["io_schema"]["feature_names"] = ["time"]
    spec["io_schema"]["feature_types"] = {"time": "continuous"}
    spec["io_schema"]["feature_domains"] = {"time": [0.0, 10.0]}
    spec["generative_family"] = "time_series_level_seasonality"
    write_jsonl(stage_a / "envspecs.jsonl", [spec])
    (stage_b_envs / "time_env.py").write_text(
        """\
import pandas as _pd

class TimeEnvEnv:
    def __init__(self, seed=0):
        pass

    def run_experiment(self, query):
        return 0.0, True

    def get_df(self):
        return _pd.DataFrame(
            {rows!r}
        )
""".format(rows=[{"time": index, "y": float(index)} for index in range(row_count)]),
        encoding="utf-8",
    )
    return run_dir


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


def _run_ppldf(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*PPLDF_COMMAND, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _logical_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()
        ),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative_bytes = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _ordered_record_sha256(frame: pd.DataFrame) -> str:
    payload = json.dumps(
        frame.to_dict(orient="records"),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stage_b_manifest(run_dir: Path) -> tuple[Path, dict[str, object]]:
    path = run_dir / "stage_B_envs" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return path, payload


def _assert_compiled_modules_confined(run_dir: Path) -> tuple[Path, ...]:
    _, manifest = _stage_b_manifest(run_dir)
    compiled_modules = manifest.get("compiled_modules")
    if not isinstance(compiled_modules, list) or not compiled_modules:
        raise ValueError("compiled_modules must be a non-empty list")

    resolved_copy_root = run_dir.resolve(strict=True)
    resolved_modules: list[Path] = []
    for index, raw_path in enumerate(compiled_modules):
        if not isinstance(raw_path, str):
            raise ValueError(f"compiled_modules[{index}] is not a path string")
        module_path = Path(raw_path)
        if not module_path.is_absolute():
            module_path = run_dir / "stage_B_envs" / module_path
        try:
            resolved_module = module_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"compiled_modules[{index}] is missing from copied run") from exc
        if not resolved_module.is_file() or not resolved_module.is_relative_to(resolved_copy_root):
            raise ValueError(f"compiled_modules[{index}] escapes copied run")
        resolved_modules.append(resolved_module)
    return tuple(resolved_modules)


def _relocate_compiled_modules(run_dir: Path) -> tuple[Path, ...]:
    manifest_path, manifest = _stage_b_manifest(run_dir)
    compiled_modules = manifest.get("compiled_modules")
    assert isinstance(compiled_modules, list) and compiled_modules
    manifest["compiled_modules"] = [
        str(run_dir / "stage_B_envs" / "envs" / Path(str(raw_path)).name)
        for raw_path in compiled_modules
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return _assert_compiled_modules_confined(run_dir)


def _assert_all_stages_resumable(run_dir: Path, *, run_root: Path) -> None:
    paths = resolve_run_paths(run_root, run_dir.name)
    assert {
        stage: stage_is_resumable(
            stage,
            paths.stage_dirs[stage],
            requested_runtime="pymc",
        )
        for stage in STAGE_ORDER
    } == {stage: True for stage in STAGE_ORDER}


def test_run_all_command_is_registered() -> None:
    assert cli_runs.run_all_cmd.__name__ == "run_all_cmd"
    result = runner.invoke(app, ["run", "all", "--help"])
    assert result.exit_code == 0
    assert "run-id" in result.stdout


def test_run_stage_g_command_is_registered() -> None:
    result = runner.invoke(app, ["stage", "g", "--help"])
    assert result.exit_code == 0
    assert "repo-id" in result.stdout


def test_stage_c_cli_requires_split_policy_without_mutating_prior_output(tmp_path: Path) -> None:
    run_id = "stage-c-requires-split-policy"
    run_dir = _write_temporal_stage_c_input(tmp_path, run_id)
    stage_c = run_dir / "stage_C_data"
    _preseed_complete_stage_c_tree(stage_c)
    prior_stage_c = _snapshot_tree(stage_c)

    result = _run_ppldf(
        "stage",
        "c",
        "--run-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--seed",
        "0",
        "--limit",
        "1",
        "--visible-ratio",
        "0.8",
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "Missing option '--split-policy'" in result.stdout + result.stderr
    assert _snapshot_tree(stage_c) == prior_stage_c


def test_stage_c_cli_rejects_unknown_split_policy_as_invalid_value(tmp_path: Path) -> None:
    run_id = "stage-c-invalid-split-policy"
    run_dir = _write_temporal_stage_c_input(tmp_path, run_id)
    stage_c = run_dir / "stage_C_data"
    _preseed_complete_stage_c_tree(stage_c)
    prior_stage_c = _snapshot_tree(stage_c)

    result = _run_ppldf(
        "stage",
        "c",
        "--run-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--seed",
        "0",
        "--limit",
        "1",
        "--split-policy",
        "not-a-policy",
        "--visible-ratio",
        "0.8",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "Invalid value for '--split-policy'" in output
    assert "No such option" not in output
    assert _snapshot_tree(stage_c) == prior_stage_c


def test_stage_c_cli_requires_visible_ratio_without_mutating_prior_output(tmp_path: Path) -> None:
    run_id = "stage-c-requires-visible-ratio"
    run_dir = _write_temporal_stage_c_input(tmp_path, run_id)
    stage_c = run_dir / "stage_C_data"
    _preseed_complete_stage_c_tree(stage_c)
    prior_stage_c = _snapshot_tree(stage_c)

    result = _run_ppldf(
        "stage",
        "c",
        "--run-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--seed",
        "0",
        "--limit",
        "1",
        "--split-policy",
        "auto",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "Missing option '--visible-ratio'" in output
    assert _snapshot_tree(stage_c) == prior_stage_c


@pytest.mark.parametrize(
    "visible_ratio",
    [
        pytest.param("0.0", id="zero"),
        pytest.param("-1.0", id="negative-one"),
        pytest.param("1.0", id="one"),
        pytest.param("2.0", id="two"),
        pytest.param("nan", id="nan"),
        pytest.param("+inf", id="positive-infinity"),
        pytest.param("-inf", id="negative-infinity"),
    ],
)
def test_stage_c_cli_rejects_invalid_visible_ratio_without_mutating_prior_output(
    tmp_path: Path,
    visible_ratio: str,
) -> None:
    run_id = "stage-c-invalid-visible-ratio"
    run_dir = _write_temporal_stage_c_input(tmp_path, run_id)
    stage_c = run_dir / "stage_C_data"
    _preseed_complete_stage_c_tree(stage_c)
    prior_run = _snapshot_tree(run_dir)

    result = _run_ppldf(
        "stage",
        "c",
        "--run-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--seed",
        "0",
        "--limit",
        "1",
        "--split-policy",
        "auto",
        "--visible-ratio",
        visible_ratio,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "SplitPolicyIneligibleError" in output
    assert result.args[-2:] == ["--visible-ratio", visible_ratio]
    assert _snapshot_tree(run_dir) == prior_run


def test_stage_c_cli_explicit_visible_ratio_uses_exact_temporal_membership(tmp_path: Path) -> None:
    run_id = "stage-c-explicit-visible-ratio"
    run_dir = _write_temporal_stage_c_input(tmp_path, run_id)

    result = _run_ppldf(
        "stage",
        "c",
        "--run-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--seed",
        "0",
        "--limit",
        "1",
        "--split-policy",
        "auto",
        "--visible-ratio",
        "0.5",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert result.args[-4:] == ["--split-policy", "auto", "--visible-ratio", "0.5"]
    stage_c = run_dir / "stage_C_data"
    visible = pd.read_parquet(stage_c / "data" / "env-time.parquet")
    hidden = pd.read_parquet(stage_c / "data" / "hidden" / "env-time.parquet")
    assert visible["time"].tolist() == [0, 1]
    assert hidden["time"].tolist() == [2, 3]


@pytest.mark.parametrize(
    ("visible_ratio", "expected_visible", "expected_hidden"),
    [
        ("0.5", [0, 1, 2, 3], [4, 5, 6, 7]),
        ("0.625", [0, 1, 2, 3, 4], [5, 6, 7]),
    ],
)
def test_stage_c_cli_preserves_each_visible_ratio_value(
    tmp_path: Path,
    visible_ratio: str,
    expected_visible: list[int],
    expected_hidden: list[int],
) -> None:
    run_id = f"stage-c-visible-ratio-{visible_ratio.replace('.', '-')}"
    run_dir = _write_temporal_stage_c_input(tmp_path, run_id, row_count=8)

    result = _run_ppldf(
        "stage",
        "c",
        "--run-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--seed",
        "0",
        "--limit",
        "1",
        "--split-policy",
        "auto",
        "--visible-ratio",
        visible_ratio,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    stage_c = run_dir / "stage_C_data"
    visible = pd.read_parquet(stage_c / "data" / "env-time.parquet")
    hidden = pd.read_parquet(stage_c / "data" / "hidden" / "env-time.parquet")
    assert visible["time"].tolist() == expected_visible
    assert hidden["time"].tolist() == expected_hidden


def test_stage_c_cli_explicit_auto_uses_ordered_tail_with_ratio_0_8(tmp_path: Path) -> None:
    run_id = "stage-c-explicit-auto"
    run_dir = _write_temporal_stage_c_input(tmp_path, run_id)

    result = _run_ppldf(
        "stage",
        "c",
        "--run-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--seed",
        "0",
        "--limit",
        "1",
        "--split-policy",
        "auto",
        "--visible-ratio",
        "0.8",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    stage_c = run_dir / "stage_C_data"
    visible = pd.read_parquet(stage_c / "data" / "env-time.parquet")
    hidden = pd.read_parquet(stage_c / "data" / "hidden" / "env-time.parquet")
    assert visible["time"].tolist() == [0, 1, 2]
    assert hidden["time"].tolist() == [3]
    data_ref = json.loads((stage_c / "data_refs.jsonl").read_text(encoding="utf-8"))
    assert data_ref["split_policy"] == "ordered_tail"


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
@pytest.mark.parametrize(
    ("entry_index", "escape_kind"),
    [(0, "source-path"), (0, "symlink"), (1, "source-path"), (1, "symlink")],
)
def test_copied_run_confinement_rejects_every_limit2_module_escape(
    entry_index: int,
    escape_kind: str,
) -> None:
    source_snapshot = _snapshot_tree(LIMIT2_RUN)
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    copied_run_dir = temp_root / LIMIT2_RUN.name
    try:
        shutil.copytree(LIMIT2_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)
        assert len(relocated) == 2
        manifest_path, manifest = _stage_b_manifest(copied_run_dir)
        compiled_modules = manifest["compiled_modules"]
        assert isinstance(compiled_modules, list) and len(compiled_modules) == 2

        if escape_kind == "source-path":
            _, source_manifest = _stage_b_manifest(LIMIT2_RUN)
            source_modules = source_manifest["compiled_modules"]
            assert isinstance(source_modules, list) and len(source_modules) == 2
            compiled_modules[entry_index] = source_modules[entry_index]
        else:
            outside_module = temp_root / f"outside-{entry_index}.py"
            outside_module.write_bytes(relocated[entry_index].read_bytes())
            link_path = copied_run_dir / "stage_B_envs" / "envs" / f"outside-link-{entry_index}.py"
            link_path.symlink_to(outside_module)
            compiled_modules[entry_index] = str(link_path)

        honest_index = 1 - entry_index
        honest_module = Path(str(compiled_modules[honest_index])).resolve(strict=True)
        assert honest_module == relocated[honest_index]
        assert honest_module.is_file()
        assert honest_module.is_relative_to(copied_run_dir.resolve(strict=True))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(
            ValueError,
            match=rf"compiled_modules\[{entry_index}\] escapes copied run",
        ):
            _assert_compiled_modules_confined(copied_run_dir)
    finally:
        try:
            assert _snapshot_tree(LIMIT2_RUN) == source_snapshot
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
def test_copied_run_confinement_accepts_both_limit2_modules() -> None:
    source_snapshot = _snapshot_tree(LIMIT2_RUN)
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    copied_run_dir = temp_root / LIMIT2_RUN.name
    try:
        shutil.copytree(LIMIT2_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)

        assert len(relocated) == 2
        assert all(path.is_relative_to(copied_run_dir.resolve(strict=True)) for path in relocated)
    finally:
        try:
            assert _snapshot_tree(LIMIT2_RUN) == source_snapshot
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
def test_run_all_requires_runner_mapping_on_resumable_copied_run() -> None:
    source_snapshot = _snapshot_tree(LIMIT1_RUN)
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    run_id = LIMIT1_RUN.name
    copied_run_dir = temp_root / run_id
    try:
        shutil.copytree(LIMIT1_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)
        assert len(relocated) == 1
        _assert_all_stages_resumable(copied_run_dir, run_root=temp_root)
        paths = resolve_run_paths(temp_root, run_id)

        with pytest.raises(TypeError, match="runners"):
            run_all(paths, resume=True, runtime_name="pymc")
    finally:
        try:
            assert _snapshot_tree(LIMIT1_RUN) == source_snapshot
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
def test_run_all_cli_requires_split_policy_without_mutating_copied_run() -> None:
    source_snapshot = _snapshot_tree(LIMIT1_RUN)
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    run_id = LIMIT1_RUN.name
    copied_run_dir = temp_root / run_id
    try:
        shutil.copytree(LIMIT1_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)
        assert len(relocated) == 1
        _assert_all_stages_resumable(copied_run_dir, run_root=temp_root)
        caller_input = _snapshot_tree(copied_run_dir)

        result = _run_ppldf(
            "run",
            "all",
            "--run-root",
            str(temp_root),
            "--run-id",
            run_id,
            "--seed",
            "0",
            "--limit",
            "1",
            "--resume",
            "--skip-upload",
            "--runtime",
            "pymc",
            "--visible-ratio",
            "0.8",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 2, output
        assert "Missing option '--split-policy'" in output
        assert result.args[-2:] == ["--visible-ratio", "0.8"]
        assert _snapshot_tree(copied_run_dir) == caller_input
    finally:
        try:
            assert _snapshot_tree(LIMIT1_RUN) == source_snapshot
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
def test_run_all_cli_requires_visible_ratio_without_mutating_copied_run() -> None:
    source_snapshot = _snapshot_tree(LIMIT1_RUN)
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    run_id = LIMIT1_RUN.name
    copied_run_dir = temp_root / run_id
    try:
        shutil.copytree(LIMIT1_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)
        assert len(relocated) == 1
        _assert_all_stages_resumable(copied_run_dir, run_root=temp_root)
        caller_input = _snapshot_tree(copied_run_dir)

        result = _run_ppldf(
            "run",
            "all",
            "--run-root",
            str(temp_root),
            "--run-id",
            run_id,
            "--seed",
            "0",
            "--limit",
            "1",
            "--resume",
            "--skip-upload",
            "--runtime",
            "pymc",
            "--split-policy",
            "auto",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 2, output
        assert "Missing option '--visible-ratio'" in output
        assert _snapshot_tree(copied_run_dir) == caller_input
    finally:
        try:
            assert _snapshot_tree(LIMIT1_RUN) == source_snapshot
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
def test_run_all_cli_rejects_unknown_split_policy_without_mutating_copied_run() -> None:
    source_snapshot = _snapshot_tree(LIMIT1_RUN)
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    run_id = LIMIT1_RUN.name
    copied_run_dir = temp_root / run_id
    try:
        shutil.copytree(LIMIT1_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)
        assert len(relocated) == 1
        _assert_all_stages_resumable(copied_run_dir, run_root=temp_root)
        caller_input = _snapshot_tree(copied_run_dir)

        result = _run_ppldf(
            "run",
            "all",
            "--run-root",
            str(temp_root),
            "--run-id",
            run_id,
            "--seed",
            "0",
            "--limit",
            "1",
            "--resume",
            "--skip-upload",
            "--runtime",
            "pymc",
            "--split-policy",
            "not-a-policy",
            "--visible-ratio",
            "0.8",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 2, output
        assert "Invalid value for '--split-policy'" in output
        assert "No such option" not in output
        assert result.args[-2:] == ["--visible-ratio", "0.8"]
        assert _snapshot_tree(copied_run_dir) == caller_input
    finally:
        try:
            assert _snapshot_tree(LIMIT1_RUN) == source_snapshot
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
@pytest.mark.parametrize(
    "visible_ratio",
    [
        pytest.param("0.0", id="zero"),
        pytest.param("-1.0", id="negative-one"),
        pytest.param("1.0", id="one"),
        pytest.param("2.0", id="two"),
        pytest.param("nan", id="nan"),
        pytest.param("+inf", id="positive-infinity"),
        pytest.param("-inf", id="negative-infinity"),
    ],
)
def test_run_all_cli_rejects_invalid_visible_ratio_without_mutating_copied_run(
    visible_ratio: str,
) -> None:
    source_snapshot = _snapshot_tree(LIMIT1_RUN)
    source_digest = _logical_tree_sha256(LIMIT1_RUN)
    assert source_digest == "95a0eb09ad3e49a781a49296af36541685197bbc14d425f5e7f2c7cd8edc9888"
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    run_id = LIMIT1_RUN.name
    copied_run_dir = temp_root / run_id
    try:
        shutil.copytree(LIMIT1_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)
        assert len(relocated) == 1
        _assert_all_stages_resumable(copied_run_dir, run_root=temp_root)
        caller_input = _snapshot_tree(copied_run_dir)

        result = _run_ppldf(
            "run",
            "all",
            "--run-root",
            str(temp_root),
            "--run-id",
            run_id,
            "--seed",
            "0",
            "--limit",
            "1",
            "--resume",
            "--skip-upload",
            "--runtime",
            "pymc",
            "--split-policy",
            "auto",
            "--visible-ratio",
            visible_ratio,
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0, output
        assert "SplitPolicyIneligibleError" in output
        assert "Missing option" not in output
        assert "No such option" not in output
        assert result.args[-2:] == ["--visible-ratio", visible_ratio]
        assert _snapshot_tree(copied_run_dir) == caller_input
    finally:
        try:
            assert _snapshot_tree(LIMIT1_RUN) == source_snapshot
            assert _logical_tree_sha256(LIMIT1_RUN) == source_digest
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
def test_run_all_cli_explicit_auto_rebuilds_only_stage_c_on_copied_run() -> None:
    source_snapshot = _snapshot_tree(LIMIT1_RUN)
    source_digest = _logical_tree_sha256(LIMIT1_RUN)
    assert source_digest == "95a0eb09ad3e49a781a49296af36541685197bbc14d425f5e7f2c7cd8edc9888"
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    run_id = LIMIT1_RUN.name
    copied_run_dir = temp_root / run_id
    try:
        shutil.copytree(LIMIT1_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)
        assert len(relocated) == 1
        shutil.rmtree(copied_run_dir / "stage_C_data")
        (copied_run_dir / "run_manifest.json").unlink()

        paths = resolve_run_paths(temp_root, run_id)
        resumable = {
            stage: stage_is_resumable(
                stage,
                paths.stage_dirs[stage],
                requested_runtime="pymc",
            )
            for stage in STAGE_ORDER
        }
        assert resumable == {
            "A": True,
            "B": True,
            "C": False,
            "D": True,
            "E": True,
            "F": True,
            "G": True,
        }

        result = _run_ppldf(
            "run",
            "all",
            "--run-root",
            str(temp_root),
            "--run-id",
            run_id,
            "--seed",
            "0",
            "--limit",
            "1",
            "--resume",
            "--skip-upload",
            "--runtime",
            "pymc",
            "--split-policy",
            "auto",
            "--visible-ratio",
            "0.8",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert result.args[-4:] == ["--split-policy", "auto", "--visible-ratio", "0.8"]
        expected_statuses = {
            "A": "skipped",
            "B": "skipped",
            "C": "success",
            "D": "skipped",
            "E": "skipped",
            "F": "skipped",
            "G": "skipped",
        }
        assert json.loads(result.stdout) == expected_statuses
        run_manifest = json.loads(
            (copied_run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert run_manifest["status_by_stage"] == expected_statuses

        data_ref = json.loads(
            (copied_run_dir / "stage_C_data" / "data_refs.jsonl").read_text(encoding="utf-8")
        )
        assert data_ref["split_policy"] == "random_row"
        assert "split_policy_request" not in data_ref
        visible = pd.read_parquet(copied_run_dir / data_ref["data_ref"])
        hidden = pd.read_parquet(copied_run_dir / data_ref["hidden_data_ref"])
        assert len(visible) == 96
        assert len(hidden) == 24
        assert _ordered_record_sha256(visible) == (
            "4a73266622fd0acbc9f5c2c646f0eb2bd1bd4a1b67a49f1dc8a8d48ccdeefd63"
        )
        assert _ordered_record_sha256(hidden) == (
            "08b43caf4ebb73e53ec1e23fc66a544a7bed9f7f6bb940cddfb7eaaa19a9ac8f"
        )
    finally:
        try:
            assert _snapshot_tree(LIMIT1_RUN) == source_snapshot
            assert _logical_tree_sha256(LIMIT1_RUN) == source_digest
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
@pytest.mark.acceptance
@pytest.mark.parametrize(
    ("visible_ratio", "visible_count", "hidden_count", "visible_sha256", "hidden_sha256"),
    [
        (
            "0.5",
            60,
            60,
            "0cfe200d0c0985323dd7a3a0c6247696540ed43310977b92ce7fc72d3e1b0faf",
            "88c619f5040349cc850d9419bf926861ea6d3cab5e02a038a242ca48c5f8672e",
        ),
        (
            "0.625",
            75,
            45,
            "c02f5a309f0bd3350cf455d95767678bf39b516a78001b56f24bb82efc23c5f5",
            "48d62cb2a17ec3f39b45a44e87e1954a4934ae93c975c91eb1baaf671b9a014d",
        ),
    ],
)
def test_run_all_cli_preserves_each_visible_ratio_on_copied_run(
    visible_ratio: str,
    visible_count: int,
    hidden_count: int,
    visible_sha256: str,
    hidden_sha256: str,
) -> None:
    source_snapshot = _snapshot_tree(LIMIT1_RUN)
    source_digest = _logical_tree_sha256(LIMIT1_RUN)
    assert source_digest == "95a0eb09ad3e49a781a49296af36541685197bbc14d425f5e7f2c7cd8edc9888"
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    run_id = LIMIT1_RUN.name
    copied_run_dir = temp_root / run_id
    try:
        shutil.copytree(LIMIT1_RUN, copied_run_dir)
        relocated = _relocate_compiled_modules(copied_run_dir)
        assert len(relocated) == 1
        shutil.rmtree(copied_run_dir / "stage_C_data")
        (copied_run_dir / "run_manifest.json").unlink()

        paths = resolve_run_paths(temp_root, run_id)
        assert {
            stage: stage_is_resumable(
                stage,
                paths.stage_dirs[stage],
                requested_runtime="pymc",
            )
            for stage in STAGE_ORDER
        } == {
            "A": True,
            "B": True,
            "C": False,
            "D": True,
            "E": True,
            "F": True,
            "G": True,
        }

        result = _run_ppldf(
            "run",
            "all",
            "--run-root",
            str(temp_root),
            "--run-id",
            run_id,
            "--seed",
            "0",
            "--limit",
            "1",
            "--resume",
            "--skip-upload",
            "--runtime",
            "pymc",
            "--split-policy",
            "auto",
            "--visible-ratio",
            visible_ratio,
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert result.args[-4:] == ["--split-policy", "auto", "--visible-ratio", visible_ratio]
        expected_statuses = {
            "A": "skipped",
            "B": "skipped",
            "C": "success",
            "D": "skipped",
            "E": "skipped",
            "F": "skipped",
            "G": "skipped",
        }
        assert json.loads(result.stdout) == expected_statuses
        run_manifest = json.loads(
            (copied_run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert run_manifest["status_by_stage"] == expected_statuses

        data_ref = json.loads(
            (copied_run_dir / "stage_C_data" / "data_refs.jsonl").read_text(encoding="utf-8")
        )
        assert data_ref["split_policy"] == "random_row"
        visible = pd.read_parquet(copied_run_dir / data_ref["data_ref"])
        hidden = pd.read_parquet(copied_run_dir / data_ref["hidden_data_ref"])
        assert len(visible) == visible_count
        assert len(hidden) == hidden_count
        assert _ordered_record_sha256(visible) == visible_sha256
        assert _ordered_record_sha256(hidden) == hidden_sha256
    finally:
        try:
            assert _snapshot_tree(LIMIT1_RUN) == source_snapshot
            assert _logical_tree_sha256(LIMIT1_RUN) == source_digest
        finally:
            temp_dir.cleanup()
            assert not temp_root.exists()


def test_run_stage_g_requires_repo_id_unless_skip_upload() -> None:
    result = runner.invoke(app, ["stage", "g", "--run-id", "r1"])
    assert result.exit_code != 0
    msg = result.output or str(result.exception)
    assert "--repo-id is required" in msg


def test_run_all_requires_repo_id_unless_skip_upload() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "all",
            "--run-id",
            "r1",
            "--split-policy",
            "auto",
            "--visible-ratio",
            "0.8",
        ],
    )
    assert result.exit_code != 0
    msg = result.output or str(result.exception)
    assert "--repo-id is required" in msg


def test_run_all_rejects_invalid_run_id() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "all",
            "--run-id",
            "../bad",
            "--skip-upload",
            "--split-policy",
            "auto",
            "--visible-ratio",
            "0.8",
        ],
    )
    assert result.exit_code != 0
    msg = result.output or str(result.exception)
    assert "run_id must match" in msg


@pytest.mark.narrow_unit_not_acceptance
def test_resolve_paths_uses_yaml_run_root_when_not_overridden(monkeypatch, tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "default.yaml").write_text(
        "run:\n  root: custom_runs_from_yaml\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUN_ROOT", raising=False)

    _, paths = cli_mod._resolve_paths("run_yaml_root", None)
    assert paths.run_root == Path("custom_runs_from_yaml")


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_skip_upload_does_not_require_hf_token(monkeypatch) -> None:
    def _deny_token(self):  # pragma: no cover - asserted by behavior
        raise RuntimeError("should not request token")

    def _fake_default_stage_runners(_cfg):
        def _runner(_paths):
            return StageResult(stage="A", status="success", outputs={})

        return {stage: _runner for stage in ("A", "B", "C", "D", "E", "F", "G")}

    def _fake_run_all(paths, *, runners, resume, runtime_name=None):
        _ = (paths, runners, resume, runtime_name)
        return {
            "A": StageResult(stage="A", status="success", outputs={}),
            "B": StageResult(stage="B", status="success", outputs={}),
            "C": StageResult(stage="C", status="success", outputs={}),
            "D": StageResult(stage="D", status="success", outputs={}),
            "E": StageResult(stage="E", status="success", outputs={}),
            "F": StageResult(stage="F", status="success", outputs={}),
            "G": StageResult(stage="G", status="success", outputs={}),
        }

    monkeypatch.setattr(cli_mod.Settings, "require_hf_token", _deny_token)
    monkeypatch.setattr(cli_mod, "default_stage_runners", _fake_default_stage_runners)
    monkeypatch.setattr(cli_mod, "run_all", _fake_run_all)

    result = runner.invoke(
        app,
        [
            "run",
            "all",
            "--run-id",
            "r1",
            "--skip-upload",
            "--split-policy",
            "auto",
            "--visible-ratio",
            "0.8",
            "--backend",
            "litellm",
            "--sync",
        ],
    )

    assert result.exit_code == 0
    assert '"A": "success"' in result.stdout


def test_run_all_runtime_flag_is_registered() -> None:
    result = runner.invoke(app, ["run", "all", "--help"])
    assert result.exit_code == 0
    assert "--runtime" in result.stdout


def test_stage_d_runtime_flag_is_registered() -> None:
    result = runner.invoke(app, ["stage", "d", "--help"])
    assert result.exit_code == 0
    assert "--runtime" in result.stdout


def test_stage_f_runtime_flag_is_registered() -> None:
    result = runner.invoke(app, ["stage", "f", "--help"])
    assert result.exit_code == 0
    assert "--runtime" in result.stdout


def test_stage_g_runtime_flag_is_registered() -> None:
    result = runner.invoke(app, ["stage", "g", "--help"])
    assert result.exit_code == 0
    assert "--runtime" in result.stdout


@pytest.mark.narrow_unit_not_acceptance
@pytest.mark.parametrize(
    ("stage_name", "patch_target", "stage_letter", "extra_args"),
    [
        ("d", "ppl_synth_data.pipeline.stages.stage_d_prompts.run_stage_d", "D", ["--sync"]),
        ("e", "ppl_synth_data.pipeline.stages.stage_e_completions.run_stage_e", "E", ["--sync"]),
        ("f", "ppl_synth_data.pipeline.stages.stage_f_validate.run_stage_f", "F", []),
        ("g", "ppl_synth_data.pipeline.stages.stage_g_dataset.run_stage_g", "G", ["--skip-upload"]),
    ],
)
def test_single_stage_commands_default_runtime_to_pymc_when_unset(
    monkeypatch,
    tmp_path: Path,
    stage_name: str,
    patch_target: str,
    stage_letter: str,
    extra_args: list[str],
) -> None:
    captured: dict[str, str] = {}

    def _fake_run(_paths, **kwargs):
        captured["runtime"] = kwargs["runtime"].name
        return StageResult(stage=stage_letter, status="success", outputs={})

    monkeypatch.setattr(patch_target, _fake_run)
    result = runner.invoke(
        app,
        [
            "stage",
            stage_name,
            "--run-id",
            f"{stage_name}-default-runtime",
            "--run-root",
            str(tmp_path),
            *extra_args,
        ],
    )

    assert result.exit_code == 0
    assert captured["runtime"] == "pymc"


@pytest.mark.narrow_unit_not_acceptance
def test_single_stage_commands_infer_runtime_from_existing_manifests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "infer-runtime"
    (run_dir / "stage_D_prompts").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_D_prompts" / "manifest.json").write_text(
        json.dumps({"status": "success", "runtime": "lazyppl"}),
        encoding="utf-8",
    )
    (run_dir / "stage_D_prompts" / "prompts.jsonl").write_text("", encoding="utf-8")

    captured: dict[str, str] = {}

    def _fake_run(_paths, **kwargs):
        captured["runtime"] = kwargs["runtime"].name
        return StageResult(stage="E", status="success", outputs={})

    monkeypatch.setattr("ppl_synth_data.pipeline.stages.stage_e_completions.run_stage_e", _fake_run)
    result = runner.invoke(
        app,
        [
            "stage",
            "e",
            "--run-id",
            "infer-runtime",
            "--run-root",
            str(tmp_path),
            "--sync",
        ],
    )

    assert result.exit_code == 0
    assert captured["runtime"] == "lazyppl"


def test_runtime_resolution_rejects_conflicting_existing_manifests(tmp_path: Path) -> None:
    run_dir = tmp_path / "conflicting-runtime"
    (run_dir / "stage_D_prompts").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_E_completions").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_D_prompts" / "manifest.json").write_text(
        json.dumps({"status": "success", "runtime": "pymc"}),
        encoding="utf-8",
    )
    (run_dir / "stage_D_prompts" / "prompts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "stage_E_completions" / "manifest.json").write_text(
        json.dumps({"status": "success", "runtime": "lazyppl"}),
        encoding="utf-8",
    )
    (run_dir / "stage_E_completions" / "completions.jsonl").write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "stage",
            "f",
            "--run-id",
            "conflicting-runtime",
            "--run-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    msg = result.output or str(result.exception)
    assert "Conflicting runtime metadata exists in this run" in msg


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_maps_cli_args_to_config(monkeypatch, tmp_path: Path) -> None:
    """Verify CLI flags map to PipelineRunConfig fields without drift."""
    captured_cfg: dict = {}

    def _capture_runners(cfg):
        captured_cfg.update(
            {
                "limit": cfg.limit,
                "backend_name": cfg.backend_name,
                "use_batch": cfg.use_batch,
                "envspec_model": cfg.envspec_model,
                "prompt_model": cfg.prompt_model,
                "completion_model": cfg.completion_model,
                "repair_model": cfg.repair_model,
                "judge_model": cfg.judge_model,
                "judge_backend_name": cfg.judge_backend_name,
                "target_language": cfg.target_language,
                "split_policy": cfg.split_policy.value,
                "visible_ratio": cfg.visible_ratio,
                "seed": cfg.seed,
                "skip_upload": cfg.skip_upload,
            }
        )

        def _ok(paths):
            return StageResult(stage="A", status="success", outputs={})

        return {s: _ok for s in "ABCDEFG"}

    def _fake_run_all(paths, *, runners, resume, runtime_name=None):
        _ = runtime_name
        return {s: runners[s](paths) for s in "ABCDEFG"}

    monkeypatch.setattr(cli_mod, "default_stage_runners", _capture_runners)
    monkeypatch.setattr(cli_mod, "run_all", _fake_run_all)

    result = runner.invoke(
        app,
        [
            "run",
            "all",
            "--run-id",
            "test-mapping",
            "--skip-upload",
            "--split-policy",
            "auto",
            "--visible-ratio",
            "0.8",
            "--limit",
            "10",
            "--backend",
            "litellm",
            "--sync",
            "--envspec-model",
            "gpt-4.1",
            "--prompt-model",
            "gpt-4.1-mini",
            "--completion-model",
            "o3",
            "--repair-model",
            "gpt-4.1",
            "--judge-model",
            "gpt-4.1",
            "--judge-backend",
            "openai",
            "--runtime",
            "lazyppl",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert captured_cfg["limit"] == 10
    assert captured_cfg["backend_name"] == "litellm"
    assert captured_cfg["use_batch"] is False
    assert captured_cfg["envspec_model"] == "gpt-4.1"
    assert captured_cfg["prompt_model"] == "gpt-4.1-mini"
    assert captured_cfg["completion_model"] == "o3"
    assert captured_cfg["repair_model"] == "gpt-4.1"
    assert captured_cfg["judge_model"] == "gpt-4.1"
    assert captured_cfg["judge_backend_name"] == "openai"
    assert captured_cfg["target_language"] == "lazyppl"
    assert captured_cfg["split_policy"] == "auto"
    assert captured_cfg["visible_ratio"] == 0.8
    assert captured_cfg["seed"] == 42
    assert captured_cfg["skip_upload"] is True


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_defaults_runtime_to_pymc(monkeypatch) -> None:
    captured_cfg: dict[str, str] = {}

    def _capture_runners(cfg):
        captured_cfg["target_language"] = cfg.target_language

        def _ok(_paths):
            return StageResult(stage="A", status="success", outputs={})

        return {s: _ok for s in "ABCDEFG"}

    def _fake_run_all(paths, *, runners, resume, runtime_name=None):
        _ = (paths, resume, runtime_name)
        return {s: runners[s](paths) for s in "ABCDEFG"}

    monkeypatch.setattr(cli_mod, "default_stage_runners", _capture_runners)
    monkeypatch.setattr(cli_mod, "run_all", _fake_run_all)

    result = runner.invoke(
        app,
        [
            "run",
            "all",
            "--run-id",
            "default-runtime",
            "--skip-upload",
            "--split-policy",
            "auto",
            "--visible-ratio",
            "0.8",
        ],
    )

    assert result.exit_code == 0
    assert captured_cfg["target_language"] == "pymc"


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_infers_runtime_from_existing_manifests(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run-all-infer"
    (run_dir / "stage_F_validation").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_F_validation" / "manifest.json").write_text(
        json.dumps({"status": "success", "runtime": "lazyppl"}),
        encoding="utf-8",
    )
    (run_dir / "stage_F_validation" / "validated.jsonl").write_text("", encoding="utf-8")
    (run_dir / "stage_F_validation" / "rejected.jsonl").write_text("", encoding="utf-8")

    captured_cfg: dict[str, str] = {}

    def _capture_runners(cfg):
        captured_cfg["target_language"] = cfg.target_language

        def _ok(_paths):
            return StageResult(stage="A", status="success", outputs={})

        return {s: _ok for s in "ABCDEFG"}

    def _fake_run_all(paths, *, runners, resume, runtime_name=None):
        _ = (paths, runners, resume, runtime_name)
        return {s: StageResult(stage=s, status="success", outputs={}) for s in "ABCDEFG"}

    monkeypatch.setattr(cli_mod, "default_stage_runners", _capture_runners)
    monkeypatch.setattr(cli_mod, "run_all", _fake_run_all)

    result = runner.invoke(
        app,
        [
            "run",
            "all",
            "--run-id",
            "run-all-infer",
            "--run-root",
            str(tmp_path),
            "--skip-upload",
            "--split-policy",
            "auto",
            "--visible-ratio",
            "0.8",
        ],
    )

    assert result.exit_code == 0
    assert captured_cfg["target_language"] == "lazyppl"


@pytest.mark.narrow_unit_not_acceptance
def test_run_stage_g_skip_upload_does_not_require_hf_token(monkeypatch, tmp_path) -> None:
    run_id = "stage_g_skip_upload"
    run_root = tmp_path
    run_dir = run_root / run_id
    (run_dir / "stage_A_envspec").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_C_data").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_F_validation").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_A_envspec" / "envspecs.jsonl").write_text(
        json.dumps(
            {
                "env_id": "known-env",
                "env_name": "known_env",
                "domain": "healthcare",
                "complexity": "medium",
                "model_type": "predictive",
                "key_concepts": ["hierarchical_model"],
                "io_schema": {
                    "input_type": "json",
                    "input_description": "JSON",
                    "output_type": "float",
                    "target_name": "y",
                    "feature_names": ["group"],
                },
                "data_plan": {
                    "n_observations": 3,
                    "grouping": ["group"],
                    "covariates": [],
                },
                "generative_family": "hierarchical_grouped",
                "generative_params": {},
                "prompt_guidance": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "stage_C_data" / "data_refs.jsonl").write_text(
        json.dumps(
            {
                "env_id": "known-env",
                "data_ref": "stage_C_data/data/known-env.parquet",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "stage_F_validation" / "validated.jsonl").write_text("", encoding="utf-8")
    (run_dir / "stage_F_validation" / "rejected.jsonl").write_text("", encoding="utf-8")

    def _deny_token(self):  # pragma: no cover - asserted by behavior
        raise RuntimeError("should not request token")

    monkeypatch.setattr(cli_mod.Settings, "require_hf_token", _deny_token)

    result = runner.invoke(
        app,
        [
            "stage",
            "g",
            "--run-id",
            run_id,
            "--run-root",
            str(run_root),
            "--skip-upload",
            "--runtime",
            "pymc",
        ],
    )

    assert result.exit_code == 0

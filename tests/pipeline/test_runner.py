from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ppl_synth_data.pipeline.artifacts import resolve_run_paths, stage_is_resumable
from ppl_synth_data.pipeline.contracts import STAGE_ORDER
from ppl_synth_data.pipeline.runner import run_all
from ppl_synth_data.pipeline.stages.stage_b_compile import run_stage_b
from ppl_synth_data.pipeline.types import StageResult
from tests.pipeline.helpers import minimal_envspec_record, write_jsonl


def test_resolve_run_paths_uses_stage_layout(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_001")

    assert paths.run_dir == tmp_path / "run_001"
    assert paths.stage_dirs["A"] == tmp_path / "run_001" / "stage_A_envspec"
    assert paths.stage_dirs["G"] == tmp_path / "run_001" / "stage_G_dataset"


def test_resolve_run_paths_rejects_path_traversal(tmp_path: Path) -> None:
    for bad in ["../escape", "run/child", "run child", ""]:
        try:
            resolve_run_paths(tmp_path, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_run_all_preserves_real_stage_b_refusals_and_retries(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_refused_response_law")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "manifest.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )
    good = minimal_envspec_record(env_id="good-law", env_name="good_law")
    good["generative_family"] = "regression_continuous"
    good["generative_params"] = {
        "intercept": 120.0,
        "noise_sigma": 8.5,
        "coefficients": {"group": 12.0},
    }
    bad = {**good, "env_id": "refused-law", "env_name": "refused_law"}
    bad["generative_params"] = {"intercept": 120.0, "noise_sigma": 8.5}
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [good, bad])

    for _ in range(2):
        out = run_all(paths, runners={"B": run_stage_b}, resume=True)

        assert out["A"].status == "skipped"
        assert out["B"].status == "completed_with_refusals"
        assert out["B"].outputs["skipped_env_count"] == 1
        assert "coefficients" in out["B"].outputs["skipped_envs"][0]["error"]
        assert not stage_is_resumable("B", paths.stage_dirs["B"])
        stage_manifest = json.loads((paths.stage_dirs["B"] / "manifest.json").read_text())
        assert stage_manifest == out["B"].outputs
        run_manifest = json.loads((paths.run_dir / "run_manifest.json").read_text())
        assert run_manifest["status_by_stage"]["B"] == "completed_with_refusals"
        # Only Stage B is supplied, so the next stage reports its missing runner.
        assert out["C"].status == "failed"
        assert out["C"].outputs["exception_type"] == "KeyError"


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_resume_skips_completed_stage(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_002")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "manifest.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text("", encoding="utf-8")

    called: list[str] = []

    def runner(stage: str):
        def inner(_paths):
            called.append(stage)
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    runners = {stage: runner(stage) for stage in STAGE_ORDER}
    out = run_all(paths, runners=runners, resume=True)

    assert "A" not in called
    assert called[0] == "B"
    assert out["A"].status == "skipped"
    assert out["G"].status == "success"


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_resume_requires_stage_artifacts(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_003")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "manifest.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )

    called: list[str] = []

    def runner(stage: str):
        def inner(_paths):
            called.append(stage)
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    runners = {stage: runner(stage) for stage in STAGE_ORDER}
    out = run_all(paths, runners=runners, resume=True)

    assert "A" in called
    assert out["A"].status == "success"


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_resume_stage_b_requires_compiled_modules(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_003b")
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "manifest.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text("", encoding="utf-8")

    paths.stage_dirs["B"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["B"] / "manifest.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )
    (paths.stage_dirs["B"] / "envs").mkdir(parents=True, exist_ok=True)

    called: list[str] = []

    def runner(stage: str):
        def inner(_paths):
            called.append(stage)
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    runners = {stage: runner(stage) for stage in STAGE_ORDER}
    out = run_all(paths, runners=runners, resume=True)

    assert out["A"].status == "skipped"
    assert out["B"].status == "success"
    assert "B" in called


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_resume_rebuilds_runtime_sensitive_stage_on_runtime_mismatch(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_runtime_mismatch")

    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "manifest.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text("", encoding="utf-8")

    paths.stage_dirs["B"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["B"] / "envs").mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["B"] / "envs" / "x.py").write_text("class X: pass\n", encoding="utf-8")
    (paths.stage_dirs["B"] / "manifest.json").write_text(
        json.dumps({"status": "success", "compiled_modules": ["envs/x.py"]}),
        encoding="utf-8",
    )

    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["C"] / "manifest.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "observations.jsonl").write_text("", encoding="utf-8")
    (paths.stage_dirs["C"] / "summaries.jsonl").write_text("", encoding="utf-8")
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text("", encoding="utf-8")

    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["D"] / "manifest.json").write_text(
        json.dumps({"status": "success", "runtime": "pymc"}),
        encoding="utf-8",
    )
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text("", encoding="utf-8")

    called: list[str] = []

    def runner(stage: str):
        def inner(_paths):
            called.append(stage)
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    runners = {stage: runner(stage) for stage in STAGE_ORDER}
    out = run_all(paths, runners=runners, resume=True, runtime_name="lazyppl")

    assert out["A"].status == "skipped"
    assert out["B"].status == "skipped"
    assert out["C"].status == "skipped"
    assert called[0] == "D"
    assert out["D"].status == "success"


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_resume_rebuilds_runtime_sensitive_stage_without_runtime_name(
    tmp_path: Path,
) -> None:
    paths = resolve_run_paths(tmp_path, "run_runtime_unspecified")

    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["A"] / "manifest.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    (paths.stage_dirs["A"] / "envspecs.jsonl").write_text("", encoding="utf-8")

    paths.stage_dirs["B"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["B"] / "envs").mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["B"] / "envs" / "x.py").write_text("class X: pass\n", encoding="utf-8")
    (paths.stage_dirs["B"] / "manifest.json").write_text(
        json.dumps({"status": "success", "compiled_modules": ["envs/x.py"]}),
        encoding="utf-8",
    )

    paths.stage_dirs["C"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["C"] / "manifest.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    (paths.stage_dirs["C"] / "observations.jsonl").write_text("", encoding="utf-8")
    (paths.stage_dirs["C"] / "summaries.jsonl").write_text("", encoding="utf-8")
    (paths.stage_dirs["C"] / "data_refs.jsonl").write_text("", encoding="utf-8")

    paths.stage_dirs["D"].mkdir(parents=True, exist_ok=True)
    (paths.stage_dirs["D"] / "manifest.json").write_text(
        json.dumps({"status": "success", "runtime": "pymc"}),
        encoding="utf-8",
    )
    (paths.stage_dirs["D"] / "prompts.jsonl").write_text("", encoding="utf-8")

    called: list[str] = []

    def runner(stage: str):
        def inner(_paths):
            called.append(stage)
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    runners = {stage: runner(stage) for stage in STAGE_ORDER}
    out = run_all(paths, runners=runners, resume=True)

    assert out["A"].status == "skipped"
    assert out["B"].status == "skipped"
    assert out["C"].status == "skipped"
    assert called[0] == "D"
    assert out["D"].status == "success"


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_catches_stage_exceptions_and_writes_manifest(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_004")

    def ok(stage: str):
        def inner(_paths):
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    def boom(_paths):
        raise RuntimeError("boom")

    runners = {stage: ok(stage) for stage in STAGE_ORDER}
    runners["C"] = boom

    out = run_all(paths, runners=runners, resume=False)

    assert out["C"].status == "failed"
    assert out["D"].status == "skipped"
    run_manifest = json.loads((paths.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["status_by_stage"]["C"] == "failed"


def test_run_all_with_empty_runner_mapping_does_not_fall_back_to_defaults(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_empty_runners")

    out = run_all(paths, runners={}, resume=False)

    assert out["A"].status == "failed"
    assert out["A"].outputs["exception_type"] == "KeyError"
    assert out["B"].status == "skipped"


def test_stage_b_resumable_accepts_paths_relative_to_stage_dir(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run_rel" / "stage_B_envs"
    (stage_dir / "envs").mkdir(parents=True, exist_ok=True)
    (stage_dir / "envs" / "x.py").write_text("class X: pass\n", encoding="utf-8")
    (stage_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "compiled_modules": ["envs/x.py"]}),
        encoding="utf-8",
    )

    assert stage_is_resumable("B", stage_dir)


def test_stage_b_resumable_rejects_partial_compile_success(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run_partial_b" / "stage_B_envs"
    (stage_dir / "envs").mkdir(parents=True, exist_ok=True)
    (stage_dir / "envs" / "x.py").write_text("class X: pass\n", encoding="utf-8")
    (stage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "compiled_modules": ["envs/x.py"],
                "skipped_env_count": 1,
            }
        ),
        encoding="utf-8",
    )

    assert stage_is_resumable("B", stage_dir) is False


def test_stage_c_resumable_rejects_partial_materialization_success(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run_partial_c" / "stage_C_data"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name in ("observations.jsonl", "summaries.jsonl", "data_refs.jsonl"):
        (stage_dir / name).write_text("", encoding="utf-8")
    (stage_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "skipped_env_count": 1}),
        encoding="utf-8",
    )

    assert stage_is_resumable("C", stage_dir) is False


def test_stage_g_resumable_rejects_dropped_or_empty_dataset(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run_partial_g" / "stage_G_dataset"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "dropped.jsonl").write_text("", encoding="utf-8")
    (stage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "runtime": "pymc",
                "count_written": 1,
                "count_dropped": 1,
            }
        ),
        encoding="utf-8",
    )
    assert stage_is_resumable("G", stage_dir, requested_runtime="pymc") is False

    (stage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "runtime": "pymc",
                "count_written": 0,
                "count_dropped": 0,
            }
        ),
        encoding="utf-8",
    )
    assert stage_is_resumable("G", stage_dir, requested_runtime="pymc") is False


@pytest.mark.narrow_unit_not_acceptance
def test_stage_b_resumable_ignores_cwd_collisions_for_relative_module_paths(
    tmp_path: Path, monkeypatch
) -> None:
    stage_dir = tmp_path / "run_rel_collision" / "stage_B_envs"
    (stage_dir / "envs").mkdir(parents=True, exist_ok=True)
    (stage_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "compiled_modules": ["envs/x.py"]}),
        encoding="utf-8",
    )

    foreign_cwd = tmp_path / "foreign"
    (foreign_cwd / "envs").mkdir(parents=True, exist_ok=True)
    (foreign_cwd / "envs" / "x.py").write_text("class NotRelevant: pass\n", encoding="utf-8")
    monkeypatch.chdir(foreign_cwd)

    assert stage_is_resumable("B", stage_dir) is False


def test_stage_is_resumable_returns_false_for_malformed_manifest(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run_bad" / "stage_A_envspec"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "manifest.json").write_text("{not-json", encoding="utf-8")

    assert stage_is_resumable("A", stage_dir) is False


def test_stage_is_resumable_requires_file_artifact_for_stage_a(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run_bad_type" / "stage_A_envspec"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "manifest.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (stage_dir / "envspecs.jsonl").mkdir(parents=True, exist_ok=True)

    assert stage_is_resumable("A", stage_dir) is False


def test_pipeline_package_exports_orchestration_api() -> None:
    import ppl_synth_data.pipeline as pkg

    assert hasattr(pkg, "run_all")
    assert hasattr(pkg, "STAGE_ORDER")
    assert hasattr(pkg, "resolve_run_paths")
    assert hasattr(pkg, "default_stage_runners")
    assert not hasattr(pkg, "run_stage_a")


def test_stages_package_lazy_loads_stage_functions() -> None:
    import ppl_synth_data.pipeline.stages as stages_pkg

    assert hasattr(stages_pkg, "run_stage_b")
    from ppl_synth_data.pipeline.stages import run_stage_b

    assert callable(run_stage_b)


def test_stages_package_raises_for_unknown_attr() -> None:
    import ppl_synth_data.pipeline.stages as stages_pkg

    try:
        _ = stages_pkg.no_such_attr  # type: ignore[attr-defined]
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


def test_default_stage_runners_returns_all_stages() -> None:
    from ppl_synth_data.pipeline.runner import default_stage_runners
    from ppl_synth_data.pipeline.types import PipelineRunConfig

    runners = default_stage_runners(
        PipelineRunConfig(split_policy="auto", visible_ratio=0.8, skip_upload=True)
    )
    assert set(runners.keys()) == set(STAGE_ORDER)


def test_pipeline_run_config_requires_split_policy() -> None:
    from ppl_synth_data.pipeline.types import PipelineRunConfig

    with pytest.raises(TypeError, match="split_policy"):
        PipelineRunConfig(visible_ratio=0.8)


def test_pipeline_run_config_requires_visible_ratio() -> None:
    from ppl_synth_data.pipeline.types import PipelineRunConfig

    with pytest.raises(TypeError, match="visible_ratio"):
        PipelineRunConfig(split_policy="auto")


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
def test_pipeline_run_config_rejects_invalid_visible_ratio(
    visible_ratio: float,
) -> None:
    from ppl_synth_data.pipeline import split_policy as split_policy_module
    from ppl_synth_data.pipeline.types import PipelineRunConfig

    with pytest.raises(Exception) as exc_info:
        PipelineRunConfig(split_policy="auto", visible_ratio=visible_ratio)

    error_type = getattr(split_policy_module, "SplitPolicyIneligibleError", None)
    assert error_type is not None
    assert type(exc_info.value) is error_type


@pytest.mark.parametrize("split_policy", ["not-a-policy", None, "AUTO", " auto "])
def test_pipeline_run_config_rejects_invalid_split_policy(split_policy: object) -> None:
    from ppl_synth_data.pipeline.types import PipelineRunConfig

    with pytest.raises(ValueError) as exc_info:
        PipelineRunConfig(split_policy=split_policy, visible_ratio=0.8)

    assert type(exc_info.value) is ValueError
    if split_policy == "not-a-policy":
        assert "not-a-policy" in str(exc_info.value)


def test_pipeline_run_config_coerces_auto_split_policy() -> None:
    from ppl_synth_data.pipeline.types import PipelineRunConfig

    cfg = PipelineRunConfig(split_policy="auto", visible_ratio=0.8)

    assert type(cfg.split_policy) is not str
    assert cfg.split_policy.name == "AUTO"
    assert cfg.split_policy.value == "auto"


@pytest.mark.parametrize(
    ("visible_ratio", "expected_visible", "expected_hidden"),
    [
        (0.5, [0, 1, 2, 3], [4, 5, 6, 7]),
        (0.625, [0, 1, 2, 3, 4], [5, 6, 7]),
    ],
)
def test_default_stage_runner_preserves_each_configured_visible_ratio(
    tmp_path: Path,
    visible_ratio: float,
    expected_visible: list[int],
    expected_hidden: list[int],
) -> None:
    from ppl_synth_data.pipeline.runner import default_stage_runners
    from ppl_synth_data.pipeline.types import PipelineRunConfig

    ratio_token = str(visible_ratio).replace(".", "-")
    paths = resolve_run_paths(tmp_path, f"runner_visible_ratio_{ratio_token}")
    spec = minimal_envspec_record(env_id="env-time", env_name="time_env")
    spec["data_plan"]["n_observations"] = 1
    spec["data_plan"]["grouping"] = []
    spec["data_plan"]["covariates"] = ["time"]
    spec["io_schema"]["feature_names"] = ["time"]
    spec["io_schema"]["feature_types"] = {"time": "continuous"}
    spec["io_schema"]["feature_domains"] = {"time": [0.0, 10.0]}
    spec["generative_family"] = "time_series_level_seasonality"
    paths.stage_dirs["A"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.stage_dirs["A"] / "envspecs.jsonl", [spec])
    envs_dir = paths.stage_dirs["B"] / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"time": index, "y": float(index)} for index in range(8)]
    (envs_dir / "time_env.py").write_text(
        f"""\
import pandas as _pd

class TimeEnvEnv:
    def __init__(self, seed=0):
        pass

    def run_experiment(self, query):
        return 0.0, True

    def get_df(self):
        return _pd.DataFrame({rows!r})
""",
        encoding="utf-8",
    )
    cfg = PipelineRunConfig(
        split_policy="auto",
        visible_ratio=visible_ratio,
        limit=1,
        skip_upload=True,
    )

    result = default_stage_runners(cfg)["C"](paths)

    assert result.status == "success"
    assert cfg.visible_ratio == visible_ratio
    visible = pd.read_parquet(paths.stage_dirs["C"] / "data" / "env-time.parquet")
    hidden = pd.read_parquet(paths.stage_dirs["C"] / "data" / "hidden" / "env-time.parquet")
    assert visible["time"].tolist() == expected_visible
    assert hidden["time"].tolist() == expected_hidden


def test_default_stage_runners_requires_config() -> None:
    from ppl_synth_data.pipeline.runner import default_stage_runners

    with pytest.raises(TypeError, match="cfg"):
        default_stage_runners()


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_invalid_stage_status_becomes_failed(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_invalid_status")

    def bad_status(_paths):
        return StageResult(stage="A", status="bogus", outputs={})

    def ok(stage: str):
        def inner(_paths):
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    runners = {stage: ok(stage) for stage in STAGE_ORDER}
    runners["A"] = bad_status

    out = run_all(paths, runners=runners, resume=False)
    assert out["A"].status == "failed"
    assert "invalid stage status" in (out["A"].message or "")
    assert out["B"].status == "skipped"


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_resume_check_failure_marks_failed_and_skips_rest(tmp_path: Path) -> None:
    from ppl_synth_data.pipeline.artifacts import resolve_run_paths

    paths = resolve_run_paths(tmp_path, "run_resume_err")
    stage_a_dir = paths.stage_dirs["A"]
    stage_a_dir.mkdir(parents=True, exist_ok=True)
    (stage_a_dir / "manifest.json").write_text("{{invalid json", encoding="utf-8")
    (stage_a_dir / "envspecs.jsonl").write_text("", encoding="utf-8")

    called: list[str] = []

    def runner(stage: str):
        def inner(_paths):
            called.append(stage)
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    runners = {stage: runner(stage) for stage in STAGE_ORDER}
    run_all(paths, runners=runners, resume=True)

    # stage_is_resumable returns False for malformed manifest, so A runs
    assert "A" in called


@pytest.mark.narrow_unit_not_acceptance
def test_run_all_no_resume_runs_all_stages(tmp_path: Path) -> None:
    paths = resolve_run_paths(tmp_path, "run_no_resume")

    called: list[str] = []

    def runner(stage: str):
        def inner(_paths):
            called.append(stage)
            return StageResult(stage=stage, status="success", outputs={})

        return inner

    runners = {stage: runner(stage) for stage in STAGE_ORDER}
    out = run_all(paths, runners=runners, resume=False)

    assert called == list(STAGE_ORDER)
    assert all(out[s].status == "success" for s in STAGE_ORDER)
    assert (paths.run_dir / "run_manifest.json").exists()

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.narrow_unit_not_acceptance

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_PPLDF_PREFIX = f"uv run --project {ROOT} --frozen ppldf"


def _make_dry_run(*args: str) -> subprocess.CompletedProcess[str]:
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is not available")
    return subprocess.run(
        [make, "-n", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _pipeline_check_dry_run(*assignments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/usr/bin/make",
            "-n",
            "-C",
            str(ROOT),
            "pipeline-check",
            "RUN_ID=probe",
            "RUN_LIMIT=1",
            *assignments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return function


def _parameter_default(function: ast.FunctionDef, name: str) -> ast.expr | None:
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(function.args.defaults)
    ) + list(function.args.defaults)
    defaults = {
        argument.arg: default
        for argument, default in zip(positional, positional_defaults, strict=True)
    }
    defaults.update(
        {
            argument.arg: default
            for argument, default in zip(
                function.args.kwonlyargs,
                function.args.kw_defaults,
                strict=True,
            )
        }
    )
    if name not in defaults:
        raise KeyError(name)
    return defaults[name]


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _is_name(expression: ast.expr | None, name: str) -> bool:
    return isinstance(expression, ast.Name) and expression.id == name


def _is_cfg_visible_ratio(expression: ast.expr | None) -> bool:
    return (
        isinstance(expression, ast.Attribute)
        and expression.attr == "visible_ratio"
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "cfg"
    )


def test_make_check_target_runs_quality_gates() -> None:
    result = _make_dry_run("check")

    assert result.returncode == 0, result.stderr
    assert "uv run pytest -q" in result.stdout
    assert "uv run ruff check ." in result.stdout
    assert "uv run mypy src" in result.stdout


def test_make_pipeline_check_target_exposes_run_id_and_limit() -> None:
    result = _make_dry_run(
        "pipeline-check",
        "RUN_ID=dry-run",
        "RUN_LIMIT=2",
        "SPLIT_POLICY=auto",
        "VISIBLE_RATIO=0.8",
    )

    assert result.returncode == 0, result.stderr
    assert (
        f'{CANONICAL_PPLDF_PREFIX} run all --run-id "dry-run" --limit "2" '
        '--split-policy "auto" --visible-ratio "0.8" --skip-upload'
    ) in result.stdout


def test_make_pipeline_check_requires_split_policy_before_rendering_command() -> None:
    result = _pipeline_check_dry_run()
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "SPLIT_POLICY is required" in output
    assert not ("uv run" in output and "ppldf" in output)


def test_make_pipeline_check_requires_visible_ratio_before_rendering_command() -> None:
    result = _pipeline_check_dry_run("SPLIT_POLICY=auto")
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "VISIBLE_RATIO is required" in output
    assert not ("uv run" in output and "ppldf" in output)


def test_make_pipeline_check_project_follows_the_makefile_location(tmp_path: Path) -> None:
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is not available")
    checkout = tmp_path / "relocated-checkout"
    checkout.mkdir()
    shutil.copy2(ROOT / "Makefile", checkout / "Makefile")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = subprocess.run(
        [
            make,
            "-n",
            "-f",
            str(checkout / "Makefile"),
            "pipeline-check",
            "RUN_ID=probe",
            "RUN_LIMIT=1",
            "SPLIT_POLICY=auto",
            "VISIBLE_RATIO=0.5",
        ],
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"uv run --project {checkout.resolve()} --frozen ppldf run all" in result.stdout


def test_makefile_names_no_home_directory() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for home in ("/Users/", "/home/", "/root/"):
        assert home not in makefile


@pytest.mark.parametrize(
    ("split_policy", "visible_ratio"),
    [("policy_probe_a", "0.375"), ("policy_probe_b", "0.625")],
)
def test_make_pipeline_check_forwards_exact_split_policy_and_visible_ratio(
    split_policy: str,
    visible_ratio: str,
) -> None:
    result = _pipeline_check_dry_run(
        f"SPLIT_POLICY={split_policy}",
        f"VISIBLE_RATIO={visible_ratio}",
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert f'--split-policy "{split_policy}"' in result.stdout
    assert f'--visible-ratio "{visible_ratio}"' in result.stdout
    assert CANONICAL_PPLDF_PREFIX in result.stdout


def test_visible_ratio_is_required_and_forwarded_by_named_caller_expressions() -> None:
    stage_c_path = ROOT / "src/ppl_synth_data/pipeline/stages/stage_c_data.py"
    types_path = ROOT / "src/ppl_synth_data/pipeline/types.py"
    runner_path = ROOT / "src/ppl_synth_data/pipeline/runner.py"
    stage_cli_path = ROOT / "src/ppl_synth_data/cli/stages.py"
    run_cli_path = ROOT / "src/ppl_synth_data/cli/runs.py"

    stage_c_tree = ast.parse(stage_c_path.read_text(encoding="utf-8"))
    types_tree = ast.parse(types_path.read_text(encoding="utf-8"))
    runner_tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    stage_cli_tree = ast.parse(stage_cli_path.read_text(encoding="utf-8"))
    run_cli_tree = ast.parse(run_cli_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    run_stage_c = _function(stage_c_tree, "run_stage_c")
    try:
        if _parameter_default(run_stage_c, "visible_ratio") is not None:
            failures.append("run_stage_c.visible_ratio has a default")
    except KeyError:
        failures.append("run_stage_c.visible_ratio is absent")
    split_calls = _calls(run_stage_c, "split_visible_hidden")
    if len(split_calls) != 1 or not _is_name(
        _keyword(split_calls[0], "visible_ratio"), "visible_ratio"
    ):
        failures.append("run_stage_c does not pass visible_ratio=visible_ratio to the splitter")

    config_class = next(
        node
        for node in types_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PipelineRunConfig"
    )
    ratio_fields = [
        node
        for node in config_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "visible_ratio"
    ]
    if len(ratio_fields) != 1:
        failures.append("PipelineRunConfig.visible_ratio is absent")
    elif ratio_fields[0].value is not None:
        failures.append("PipelineRunConfig.visible_ratio has a default")

    default_runners = _function(runner_tree, "default_stage_runners")
    runner_stage_c_calls = _calls(default_runners, "run_stage_c")
    if len(runner_stage_c_calls) != 1 or not _is_cfg_visible_ratio(
        _keyword(runner_stage_c_calls[0], "visible_ratio")
    ):
        failures.append("default_stage_runners does not pass visible_ratio=cfg.visible_ratio")

    stage_c_cmd = _function(stage_cli_tree, "run_stage_c_cmd")
    try:
        if _parameter_default(stage_c_cmd, "visible_ratio") is not None:
            failures.append("the Stage C CLI visible ratio has a default")
    except KeyError:
        failures.append("the Stage C CLI visible ratio option is absent")
    stage_cli_calls = _calls(stage_c_cmd, "run_stage_c")
    if len(stage_cli_calls) != 1 or not _is_name(
        _keyword(stage_cli_calls[0], "visible_ratio"), "visible_ratio"
    ):
        failures.append("run_stage_c_cmd does not pass visible_ratio=visible_ratio")

    run_all_cmd = _function(run_cli_tree, "run_all_cmd")
    try:
        if _parameter_default(run_all_cmd, "visible_ratio") is not None:
            failures.append("the run-all CLI visible ratio has a default")
    except KeyError:
        failures.append("the run-all CLI visible ratio option is absent")
    config_assignments = [
        node
        for node in ast.walk(run_all_cmd)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "PipelineRunConfig"
    ]
    config_name: str | None = None
    if len(config_assignments) != 1:
        failures.append("run_all_cmd does not assign one PipelineRunConfig")
    else:
        assignment = config_assignments[0]
        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            config_name = targets[0].id
        else:
            failures.append("run_all_cmd does not name its PipelineRunConfig")
        if not _is_name(_keyword(assignment.value, "visible_ratio"), "visible_ratio"):
            failures.append("run_all_cmd does not pass visible_ratio=visible_ratio into config")
    default_runner_calls = [
        node
        for node in ast.walk(run_all_cmd)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "default_stage_runners"
    ]
    if (
        len(default_runner_calls) != 1
        or config_name is None
        or len(default_runner_calls[0].args) != 1
        or not _is_name(default_runner_calls[0].args[0], config_name)
    ):
        failures.append("run_all_cmd does not pass the named config to default_stage_runners")

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    policy_guard = "$(if $(strip $(SPLIT_POLICY)),,$(error SPLIT_POLICY is required))"
    ratio_guard = "$(if $(strip $(VISIBLE_RATIO)),,$(error VISIBLE_RATIO is required))"
    if ratio_guard not in makefile:
        failures.append("pipeline-check does not require VISIBLE_RATIO")
    elif policy_guard not in makefile or makefile.index(policy_guard) > makefile.index(ratio_guard):
        failures.append("pipeline-check does not validate SPLIT_POLICY before VISIBLE_RATIO")
    result = _pipeline_check_dry_run("SPLIT_POLICY=auto", "VISIBLE_RATIO=0.8")
    if result.returncode != 0 or CANONICAL_PPLDF_PREFIX not in result.stdout:
        failures.append("pipeline-check does not use the canonical ppldf prefix")
    if '--visible-ratio "$(VISIBLE_RATIO)"' not in makefile:
        failures.append("pipeline-check does not pass $(VISIBLE_RATIO) to --visible-ratio")

    assert failures == []


def test_make_harbor_e2e_target_exposes_run_variables() -> None:
    result = _make_dry_run(
        "harbor-e2e",
        "RUN_ID=harbor-validation",
        "RUN_ROOT=artifacts/custom-runs",
    )

    assert result.returncode == 0, result.stderr
    assert 'bash scripts/harbor_e2e.sh "harbor-validation" "artifacts/custom-runs"' in result.stdout

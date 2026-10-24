"""Tests for CLI commands: compile-envspec and scan-leakage with custom banned terms."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ppl_synth_data.cli.commands as cli_commands
from ppl_synth_data.cli import app
from ppl_synth_data.pipeline.types import RunPaths

runner = CliRunner()


def _minimal_envspec_json() -> dict:
    return {
        "env_id": "cli_compile_test_v1",
        "env_name": "cli_compile_test_v1",
        "domain": "healthcare",
        "complexity": "simple",
        "model_type": "predictive",
        "key_concepts": ["test"],
        "io_schema": {
            "input_type": "float",
            "input_description": "numeric",
            "output_type": "float",
            "target_name": "y",
            "feature_names": ["x"],
            "feature_domains": {"x": [0.0, 1.0]},
        },
        "data_plan": {
            "n_observations": 5,
            "grouping": [],
            "covariates": [],
        },
        "generative_family": "regression_continuous",
        "generative_params": {
            "intercept": 0.5,
            "noise_sigma": 0.1,
            "coefficients": {"x": 1.0},
        },
        "prompt_guidance": {},
    }


def _valid_completion_record_json() -> str:
    return json.dumps(
        {
            "id": "c1",
            "prompt_id": "p1",
            "prompt_text": "Prompt text",
            "raw_completion": "```python\\nprint(1)\\n```",
            "metadata": {"source": "test"},
        }
    )


# compile-envspec.


def test_compile_envspec_produces_py_file(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(_minimal_envspec_json()), encoding="utf-8")
    out_dir = tmp_path / "compiled"
    out_dir.mkdir()

    result = runner.invoke(app, ["compile-envspec", str(spec_file), str(out_dir)])

    assert result.exit_code == 0, result.output
    output_path = Path(result.stdout.strip())
    assert output_path.exists()
    assert output_path.suffix == ".py"


def test_compile_envspec_fails_on_bad_json(tmp_path: Path) -> None:
    spec_file = tmp_path / "bad.json"
    spec_file.write_text('{"env_id": "x"}', encoding="utf-8")
    out_dir = tmp_path / "compiled"
    out_dir.mkdir()

    result = runner.invoke(app, ["compile-envspec", str(spec_file), str(out_dir)])

    assert result.exit_code != 0


# scan-leakage with custom banned terms file.


def test_scan_leakage_with_custom_banned_terms(tmp_path: Path) -> None:
    banned = tmp_path / "banned.txt"
    banned.write_text("secret_word\nanother_term\n", encoding="utf-8")

    text_file = tmp_path / "prompt.txt"
    text_file.write_text("This prompt contains secret word in it.", encoding="utf-8")

    result = runner.invoke(
        app,
        ["scan-leakage", str(text_file), "--banned-terms-file", str(banned)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "secret word" in payload["hits"]
    assert payload["count"] >= 1


def test_scan_leakage_no_hits(tmp_path: Path) -> None:
    banned = tmp_path / "banned.txt"
    banned.write_text("forbidden\n", encoding="utf-8")

    text_file = tmp_path / "prompt.txt"
    text_file.write_text("A perfectly clean prompt.", encoding="utf-8")

    result = runner.invoke(
        app,
        ["scan-leakage", str(text_file), "--banned-terms-file", str(banned)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["hits"] == []
    assert payload["count"] == 0


def test_scan_leakage_comments_in_banned_file_ignored(tmp_path: Path) -> None:
    banned = tmp_path / "banned.txt"
    banned.write_text("# this is a comment\nforbidden\n", encoding="utf-8")

    # text contains "this is a comment"; if comment lines were treated
    # as banned terms, this would produce a false hit
    text_file = tmp_path / "prompt.txt"
    text_file.write_text("This is a comment about a forbidden topic.", encoding="utf-8")

    result = runner.invoke(
        app,
        ["scan-leakage", str(text_file), "--banned-terms-file", str(banned)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "forbidden" in payload["hits"]
    assert payload["count"] == 1  # only "forbidden", not the comment line


# export-harbor coverage branches.


@pytest.mark.narrow_unit_not_acceptance
@pytest.mark.parametrize(
    "reference_args,defer_references,reference_workers",
    [
        ([], False, 1),
        (["--defer-references", "--reference-workers", "2"], True, 2),
    ],
)
def test_export_harbor_uses_resolved_run_dir_and_validate_gold_flag(
    monkeypatch,
    tmp_path: Path,
    reference_args: list[str],
    defer_references: bool,
    reference_workers: int,
) -> None:
    fake_paths = RunPaths(
        run_root=tmp_path,
        run_id="demo-run",
        run_dir=tmp_path / "demo-run",
        stage_dirs={},
    )
    captured: dict[str, object] = {}

    def _fake_resolve_paths(run_id: str, run_root: Path | None):
        captured["run_id"] = run_id
        captured["run_root"] = run_root
        return object(), fake_paths

    def _fake_export_harbor_tasks(
        *,
        run_dir: Path,
        out_dir: Path,
        run_id: str,
        attempt_id: str,
        validate_gold: bool,
        defer_references: bool,
        reference_workers: int,
        allow_unlocked_reference_runtime: bool,
        scorer_image: str | None,
    ):
        captured["export_call"] = {
            "run_dir": run_dir,
            "out_dir": out_dir,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "validate_gold": validate_gold,
            "defer_references": defer_references,
            "reference_workers": reference_workers,
            "allow_unlocked_reference_runtime": allow_unlocked_reference_runtime,
            "scorer_image": scorer_image,
        }
        return {"status": "ok", "tasks_written": 2}

    monkeypatch.setattr(cli_commands, "_resolve_paths", _fake_resolve_paths)
    monkeypatch.setattr("ppl_synth_data.harbor.export_harbor_tasks", _fake_export_harbor_tasks)

    out_dir = tmp_path / "harbor-out"
    result = runner.invoke(
        app,
        [
            "export-harbor",
            "--run-id",
            "demo-run",
            "--run-root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--attempt-id",
            "demo-attempt",
            "--validate-gold",
            *reference_args,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["run_id"] == "demo-run"
    assert captured["run_root"] == tmp_path
    assert captured["export_call"] == {
        "run_dir": fake_paths.run_dir,
        "out_dir": out_dir,
        "run_id": "demo-run",
        "attempt_id": "demo-attempt",
        "validate_gold": True,
        "defer_references": defer_references,
        "reference_workers": reference_workers,
        "allow_unlocked_reference_runtime": False,
        "scorer_image": None,
    }
    assert json.loads(result.stdout) == {"status": "ok", "tasks_written": 2}


@pytest.mark.narrow_unit_not_acceptance
def test_export_harbor_lazyppl_uses_resolved_run_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_paths = RunPaths(
        run_root=tmp_path,
        run_id="demo-run",
        run_dir=tmp_path / "demo-run",
        stage_dirs={},
    )
    captured: dict[str, object] = {}

    def _fake_resolve_paths(run_id: str, run_root: Path | None):
        captured["run_id"] = run_id
        captured["run_root"] = run_root
        return object(), fake_paths

    def _fake_export_lazyppl_harbor_tasks_from_run(
        *,
        run_dir: Path,
        out_dir: Path,
        run_id: str,
        limit: int | None,
        min_exported: int | None,
    ):
        captured["export_call"] = {
            "run_dir": run_dir,
            "out_dir": out_dir,
            "run_id": run_id,
            "limit": limit,
            "min_exported": min_exported,
        }
        return {"status": "ok", "tasks_created": 2}

    monkeypatch.setattr(cli_commands, "_resolve_paths", _fake_resolve_paths)
    monkeypatch.setattr(
        "ppl_synth_data.harbor.export_lazyppl_harbor_tasks_from_run",
        _fake_export_lazyppl_harbor_tasks_from_run,
    )

    out_dir = tmp_path / "harbor-out"
    result = runner.invoke(
        app,
        [
            "export-harbor-lazyppl",
            "--run-id",
            "demo-run",
            "--run-root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--limit",
            "2",
            "--min-exported",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["run_id"] == "demo-run"
    assert captured["run_root"] == tmp_path
    assert captured["export_call"] == {
        "run_dir": fake_paths.run_dir,
        "out_dir": out_dir,
        "run_id": "demo-run",
        "limit": 2,
        "min_exported": 1,
    }
    assert json.loads(result.stdout) == {"status": "ok", "tasks_created": 2}


@pytest.mark.narrow_unit_not_acceptance
def test_export_harbor_skyrl_bundle_invokes_producer_export(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_export_harbor_skyrl_bundle(**kwargs: object):
        captured.update(kwargs)
        return {"status": "ok", "task_count": 2}

    monkeypatch.setattr(
        "ppl_synth_data.harbor.export_harbor_skyrl_bundle",
        _fake_export_harbor_skyrl_bundle,
    )

    task_root = tmp_path / "tasks"
    out_dir = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "export-harbor-skyrl-bundle",
            "--task-root",
            str(task_root),
            "--out",
            str(out_dir),
            "--runtime",
            "pymc",
            "--env-name",
            "PyMC online-train",
            "--max-steps",
            "9",
            "--run-id",
            "demo-run",
            "--min-tasks",
            "2",
            "--no-include-sft-rows",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "task_root": task_root,
        "out_dir": out_dir,
        "runtime": "pymc",
        "env_name": "PyMC online-train",
        "max_steps": 9,
        "run_id": "demo-run",
        "min_tasks": 2,
        "include_sft_rows": False,
    }
    assert json.loads(result.stdout) == {"status": "ok", "task_count": 2}


@pytest.mark.narrow_unit_not_acceptance
def test_compose_live_rich_reward_command_invokes_producer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_compose_live_rich_reward(**kwargs: object):
        captured.update(kwargs)
        return {"status": "success", "reward_source": "heuristic.composite_reward"}

    monkeypatch.setattr(
        "ppl_synth_data.dataset.rich_reward_export.compose_live_rich_reward",
        _fake_compose_live_rich_reward,
    )

    trial_dir = tmp_path / "trial"
    trajectory = trial_dir / "artifacts" / "trajectory.json"
    reward = trial_dir / "artifacts" / "reward.json"
    result = runner.invoke(
        app,
        [
            "compose-live-rich-reward",
            "--trial-dir",
            str(trial_dir),
            "--trajectory",
            str(trajectory),
            "--reward",
            str(reward),
            "--rubric-model",
            "gpt-5.5",
            "--heuristic-weight",
            "0.2",
            "--process-weight",
            "0.05",
            "--max-judge-chars",
            "1000",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["trial_dir"] == trial_dir
    assert captured["trajectory_path"] == trajectory
    assert captured["reward_path"] == reward
    assert captured["rubric_model"] == "gpt-5.5"
    assert captured["heuristic_weight"] == 0.2
    assert captured["process_weight"] == 0.05
    assert captured["max_judge_chars"] == 1000
    assert json.loads(result.stdout)["reward_source"] == "heuristic.composite_reward"

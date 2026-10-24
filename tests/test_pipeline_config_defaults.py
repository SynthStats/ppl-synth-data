from __future__ import annotations

from pathlib import Path

from ppl_synth_data.pipeline.config_defaults import (
    effective_reasoning_effort,
    load_default_pipeline_defaults,
    load_pipeline_defaults,
    resolve_default_config_layers,
)


def test_load_pipeline_defaults_reads_yaml_files(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    (configs / "stages").mkdir(parents=True, exist_ok=True)

    (configs / "default.yaml").write_text(
        (
            "run:\n"
            "  root: artifacts/custom_runs\n"
            "providers:\n"
            "  default_backend: litellm\n"
            "  openai:\n"
            "    model_envspec: model-envspec\n"
            "    model_prompt: model-prompt\n"
            "    model_completion: model-completion\n"
            "    model_repair: model-repair\n"
        ),
        encoding="utf-8",
    )
    (configs / "stages" / "envspec.yaml").write_text(
        "stage: envspec\nmodel: envspec-stage-model\n",
        encoding="utf-8",
    )
    (configs / "stages" / "prompts.yaml").write_text(
        "stage: prompts\nmodel: prompts-stage-model\nreasoning_effort: low\n",
        encoding="utf-8",
    )
    (configs / "stages" / "completions.yaml").write_text(
        ("stage: completions\nmodel: completions-stage-model\nreasoning_effort: medium\n"),
        encoding="utf-8",
    )

    defaults = load_pipeline_defaults(configs)

    assert defaults.run_root == "artifacts/custom_runs"
    assert defaults.default_backend == "litellm"
    assert defaults.models.envspec == "model-envspec"
    assert defaults.models.prompt == "model-prompt"
    assert defaults.models.completion == "model-completion"
    assert defaults.models.repair == "model-repair"

    assert defaults.stages.envspec.model == "envspec-stage-model"
    assert defaults.stages.prompts.model == "prompts-stage-model"
    assert defaults.stages.prompts.reasoning_effort == "low"
    assert defaults.stages.completions.model == "completions-stage-model"
    assert defaults.stages.completions.reasoning_effort == "medium"
    assert defaults.stages.completions.max_output_tokens is None


def test_load_pipeline_defaults_ignores_boolean_max_output_tokens(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    (configs / "stages").mkdir(parents=True, exist_ok=True)

    (configs / "stages" / "completions.yaml").write_text(
        (
            "stage: completions\n"
            "model: completions-stage-model\n"
            "reasoning_effort: medium\n"
            "max_output_tokens: true\n"
        ),
        encoding="utf-8",
    )

    defaults = load_pipeline_defaults(configs)

    assert defaults.stages.completions.max_output_tokens is None


def test_reasoning_effort_none_is_sent_to_current_openai_models() -> None:
    assert effective_reasoning_effort("none") == "none"
    assert effective_reasoning_effort("omit") is None


def test_resolve_default_config_layers_ignores_empty_local_configs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True)
    local_cwd = tmp_path / "cwd"
    (local_cwd / "configs").mkdir(parents=True)

    base_dir, overlay_dir = resolve_default_config_layers(cwd=local_cwd, repo_root=repo_root)

    assert base_dir == repo_root / "configs"
    assert overlay_dir is None


def test_load_default_pipeline_defaults_uses_repo_default_with_local_stage_overrides(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_configs = repo_root / "configs"
    (repo_configs / "stages").mkdir(parents=True, exist_ok=True)
    (repo_configs / "default.yaml").write_text(
        "run:\n  root: repo/runs\nproviders:\n  default_backend: openai\n",
        encoding="utf-8",
    )
    (repo_configs / "stages" / "prompts.yaml").write_text(
        "model: repo-prompt-model\n",
        encoding="utf-8",
    )

    local_cwd = tmp_path / "cwd"
    local_configs = local_cwd / "configs"
    (local_configs / "stages").mkdir(parents=True, exist_ok=True)
    (local_configs / "stages" / "prompts.yaml").write_text(
        "model: local-prompt-model\nreasoning_effort: low\n",
        encoding="utf-8",
    )

    defaults = load_default_pipeline_defaults(cwd=local_cwd, repo_root=repo_root)

    assert defaults.run_root == "repo/runs"
    assert defaults.default_backend == "openai"
    assert defaults.stages.prompts.model == "local-prompt-model"
    assert defaults.stages.prompts.reasoning_effort == "low"


def test_load_default_pipeline_defaults_ignores_malformed_local_default_when_repo_exists(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_configs = repo_root / "configs"
    repo_configs.mkdir(parents=True, exist_ok=True)
    (repo_configs / "default.yaml").write_text(
        "run:\n  root: repo/runs\nproviders:\n  default_backend: openai\n",
        encoding="utf-8",
    )

    local_cwd = tmp_path / "cwd"
    local_configs = local_cwd / "configs"
    local_configs.mkdir(parents=True, exist_ok=True)
    (local_configs / "default.yaml").write_text("run: [broken\n", encoding="utf-8")

    defaults = load_default_pipeline_defaults(cwd=local_cwd, repo_root=repo_root)

    assert defaults.run_root == "repo/runs"
    assert defaults.default_backend == "openai"

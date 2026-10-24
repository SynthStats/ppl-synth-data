from __future__ import annotations

from pathlib import Path

from ppl_synth_data.pipeline.config_defaults import (
    PipelineDefaults,
    _as_opt_int,
    _as_opt_str,
    _as_str,
    load_pipeline_defaults,
)


def test_as_str_valid() -> None:
    assert _as_str("hello", "default") == "hello"


def test_as_str_empty() -> None:
    assert _as_str("", "default") == "default"
    assert _as_str("   ", "default") == "default"


def test_as_str_non_string() -> None:
    assert _as_str(None, "default") == "default"
    assert _as_str(123, "default") == "default"


def test_as_opt_str_valid() -> None:
    assert _as_opt_str("value") == "value"


def test_as_opt_str_empty() -> None:
    assert _as_opt_str("") is None
    assert _as_opt_str(None) is None


def test_as_opt_int_valid() -> None:
    assert _as_opt_int(42) == 42
    assert _as_opt_int("100") == 100


def test_as_opt_int_none() -> None:
    assert _as_opt_int(None) is None
    assert _as_opt_int(None, fallback=10) == 10


def test_as_opt_int_bool_returns_fallback() -> None:
    assert _as_opt_int(True) is None
    assert _as_opt_int(False, fallback=5) == 5


def test_as_opt_int_invalid_string() -> None:
    assert _as_opt_int("abc") is None


def test_load_pipeline_defaults_missing_dir(tmp_path: Path) -> None:
    defaults = load_pipeline_defaults(tmp_path / "nonexistent")
    assert isinstance(defaults, PipelineDefaults)
    assert defaults.default_backend == "openai"
    assert defaults.run_root == "artifacts/runs"


def test_load_pipeline_defaults_with_config(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "default.yaml").write_text(
        "run:\n  root: my/runs\nproviders:\n  default_backend: litellm\n"
    )
    stages = configs / "stages"
    stages.mkdir()
    (stages / "envspec.yaml").write_text("model: custom-model\n")
    (stages / "prompts.yaml").write_text("")
    (stages / "completions.yaml").write_text("")

    defaults = load_pipeline_defaults(configs)
    assert defaults.run_root == "my/runs"
    assert defaults.default_backend == "litellm"
    assert defaults.stages.envspec.model == "custom-model"

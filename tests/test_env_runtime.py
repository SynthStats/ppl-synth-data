from __future__ import annotations

from pathlib import Path

from ppl_synth_data.validation.env_runtime import validate_compiled_env


def test_validate_compiled_env_rejects_invalid_class_name(tmp_path: Path) -> None:
    module = tmp_path / "env.py"
    module.write_text("class Okay:\n    pass\n", encoding="utf-8")
    result = validate_compiled_env(module, "not valid")
    assert result.ok is False
    assert "invalid class name" in result.message


def test_validate_compiled_env_rejects_non_python_path(tmp_path: Path) -> None:
    module = tmp_path / "env.txt"
    module.write_text("placeholder", encoding="utf-8")
    result = validate_compiled_env(module, "Env")
    assert result.ok is False
    assert ".py file" in result.message

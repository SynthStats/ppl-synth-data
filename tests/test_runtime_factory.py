from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ppl_synth_data.envs.runtime_factory import load_env_class


def test_load_env_class_valid(tmp_path: Path) -> None:
    mod_file = tmp_path / "mymod.py"
    mod_file.write_text(
        textwrap.dedent("""\
        class MyEnv:
            def __init__(self):
                self.value = 42
        """)
    )
    cls = load_env_class(mod_file, "MyEnv")
    assert cls.__name__ == "MyEnv"
    instance = cls()
    assert instance.value == 42


def test_load_env_class_missing_class(tmp_path: Path) -> None:
    mod_file = tmp_path / "empty.py"
    mod_file.write_text("x = 1\n")
    with pytest.raises(AttributeError):
        load_env_class(mod_file, "DoesNotExist")


def test_load_env_class_bad_path(tmp_path: Path) -> None:
    bad = tmp_path / "nonexistent.py"
    with pytest.raises((RuntimeError, FileNotFoundError, ImportError)):
        load_env_class(bad, "Anything")

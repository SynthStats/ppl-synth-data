from __future__ import annotations

import importlib.util
from pathlib import Path

from ppl_synth_data.envs.base import BaseEnv


def load_env_class(module_path: Path, class_name: str) -> type[BaseEnv]:
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import env module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

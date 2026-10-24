from __future__ import annotations

import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

from ppl_synth_data.runtimes.base import ValidationResult
from ppl_synth_data.validation.subprocess_utils import (
    build_limited_env,
    resolve_python_cmd,
    run_with_process_group_timeout,
    set_resource_limits,
)

_HARNESS = """
import importlib.util
import inspect
import json
import sys

module_path = sys.argv[1]
class_name = sys.argv[2]

spec = importlib.util.spec_from_file_location('generated_env', module_path)
if spec is None or spec.loader is None:
    print(json.dumps({"ok": False, "msg": "cannot import module"}))
    raise SystemExit(0)

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cls = getattr(mod, class_name, None)
if cls is None:
    print(json.dumps({"ok": False, "msg": "class not found"}))
    raise SystemExit(0)

try:
    env = cls(seed=0)
except Exception as exc:
    print(json.dumps({"ok": False, "msg": f"constructor failed: {exc}"}))
    raise SystemExit(0)

required = [
    'reset',
    'step',
    'validate_input',
    'run_experiment',
    'get_data',
    'get_df',
    'generate_system_message',
]
for name in required:
    attr = getattr(env, name, None)
    if attr is None or not callable(attr):
        print(json.dumps({"ok": False, "msg": f"missing callable {name}"}))
        raise SystemExit(0)

try:
    sig = inspect.signature(env.step)
    if len(sig.parameters) != 1:
        print(json.dumps({"ok": False, "msg": "step must accept one query parameter"}))
        raise SystemExit(0)
except Exception as exc:
    print(json.dumps({"ok": False, "msg": f"signature check failed: {exc}"}))
    raise SystemExit(0)

print(json.dumps({"ok": True, "msg": "ok"}))
"""


def validate_compiled_env(module_path: Path, class_name: str) -> ValidationResult:
    if not class_name.isidentifier():
        return ValidationResult(ok=False, message=f"invalid class name: {class_name}")
    module_resolved = module_path.resolve()
    if module_resolved.suffix != ".py":
        return ValidationResult(
            ok=False,
            message=f"module path must be a .py file: {module_resolved}",
        )
    if not module_resolved.is_file():
        return ValidationResult(
            ok=False,
            message=f"module file not found: {module_resolved}",
        )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        script = td_path / "env_check.py"
        script.write_text(_HARNESS, encoding="utf-8")
        run_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 20,
            "check": False,
            "cwd": str(td_path),
            "env": build_limited_env(home_dir=td_path / "home"),
        }
        if platform.system() != "Windows":
            run_kwargs["preexec_fn"] = set_resource_limits
        try:
            proc = run_with_process_group_timeout(
                [*resolve_python_cmd(), str(script), str(module_resolved), class_name],
                **run_kwargs,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(ok=False, message="env validation timed out (20s)")
        except OSError as exc:
            return ValidationResult(
                ok=False,
                message=f"failed to launch subprocess: {exc}",
            )
    if proc.returncode != 0:
        return ValidationResult(
            ok=False,
            message=cast(str, proc.stderr).strip() or cast(str, proc.stdout).strip(),
        )
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return ValidationResult(
            ok=False,
            message=f"invalid validator output: {cast(str, proc.stdout).strip()}",
        )
    return ValidationResult(
        ok=bool(result.get("ok")),
        message=str(result.get("msg", "")),
    )

from __future__ import annotations

import platform
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ppl_synth_data.validation.sample_data import serialize_sample_data
from ppl_synth_data.validation.subprocess_utils import (
    build_limited_env,
    resolve_python_cmd,
    run_with_process_group_timeout,
    set_resource_limits,
)

if TYPE_CHECKING:
    from ppl_synth_data.runtimes.base import ValidationResult


def _completed_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def runtime_compile_check(
    code: str,
    # Wall-clock hang backstop only: RLIMIT_CPU (set_resource_limits) bounds
    # actual compute at 30 CPU-seconds. On a loaded host the `uv run python`
    # spawn + pymc import alone can take minutes of wall time at low CPU, so a
    # tight wall budget misclassifies good code as "timeout" and triggers
    # spurious LLM repair calls. 120s matches pymc_selection.py.
    timeout_sec: int = 120,
    sample_data: dict[str, Any] | None = None,
) -> ValidationResult:
    from ppl_synth_data.runtimes.base import ValidationResult

    if sample_data is None:
        return ValidationResult(
            ok=False,
            message=(
                "sample_data is required for PyMC runtime validation; "
                "Stage C visible data parquet must be present"
            ),
        )
    data_repr = serialize_sample_data(sample_data)
    harness = f"""
import pymc as pm
import numpy as np

{code}

data = {data_repr}
m = model(data)
assert isinstance(m, pm.Model)
print("OK")
"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        path = td_path / "check.py"
        path.write_text(harness, encoding="utf-8")
        run_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": timeout_sec,
            "check": False,
            "cwd": str(td_path),
            "env": build_limited_env(
                extra={"PYTHONNOUSERSITE": "1"},
                home_dir=td_path / "home",
            ),
        }
        if platform.system() != "Windows":
            run_kwargs["preexec_fn"] = set_resource_limits
        try:
            proc = run_with_process_group_timeout(
                [*resolve_python_cmd(), str(path)],
                **run_kwargs,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(ok=False, message="timeout")
    if proc.returncode != 0:
        stderr = _completed_text(proc.stderr).strip()
        stdout = _completed_text(proc.stdout).strip()
        return ValidationResult(
            ok=False,
            message=stderr or stdout,
        )
    return ValidationResult(ok=True, message=_completed_text(proc.stdout).strip())

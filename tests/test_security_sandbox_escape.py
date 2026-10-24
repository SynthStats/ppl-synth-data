"""Real subprocess escape tests for the validation sandbox primitives.

These exercise `build_limited_env`, `run_with_process_group_timeout`, and
`set_resource_limits` (ppl_synth_data.validation.subprocess_utils) the same
way `env_runtime.py` and `pymc_runtime.py` use them to run LLM-generated code:
a real child Python process, real OS resource limits, real environment
filtering. No mocks, no monkeypatched subprocess calls; every test here
spawns an actual `python -c ...` subprocess.

This sandbox has NO filesystem chroot/container and NO network
namespace. It defends via (1) environment variable allowlisting, (2) an
isolated HOME so `~`/`$HOME`-convention credential lookups miss, (3) POSIX
resource limits (CPU, process count, and on Linux, address space), and (4)
process-group timeout/kill. It does NOT stop code that hardcodes an absolute
path to the real filesystem, and it does NOT block network egress. Those need
Docker (or equivalent containerization), a separate, acceptance-grade lane.
Tests below prove both what IS enforced and what is NOT, rather than
asserting a stronger guarantee than the code actually provides.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from ppl_synth_data.validation.subprocess_utils import (
    build_limited_env,
    run_with_process_group_timeout,
    set_resource_limits,
)

pytestmark = [pytest.mark.slow, pytest.mark.real_boundary]

_CANARY_MARKER = "CANARY-SECRET-DO-NOT-LEAK-0f3a9c2b"


def _run(
    script: Path,
    *args: str,
    env: dict[str, str],
    cwd: Path,
    timeout: float = 10.0,
    preexec_fn: Any = None,
) -> subprocess.CompletedProcess[str]:
    run_kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "check": False,
        "cwd": str(cwd),
        "env": env,
    }
    if preexec_fn is not None:
        run_kwargs["preexec_fn"] = preexec_fn
    return run_with_process_group_timeout([sys.executable, str(script), *args], **run_kwargs)


def test_home_isolation_blocks_expanduser_credential_lookup(tmp_path: Path) -> None:
    """The env-var-based (`~`, `$HOME`) credential lookup path is closed."""
    fake_real_home = tmp_path / "fake_real_home"
    (fake_real_home / ".aws").mkdir(parents=True)
    (fake_real_home / ".aws" / "credentials").write_text(_CANARY_MARKER, encoding="utf-8")

    isolated_home = tmp_path / "isolated_home"

    script = tmp_path / "read_via_expanduser.py"
    script.write_text(
        "import json, os\n"
        "path = os.path.expanduser('~/.aws/credentials')\n"
        "try:\n"
        "    with open(path, encoding='utf-8') as fh:\n"
        "        content = fh.read()\n"
        "except OSError:\n"
        "    content = None\n"
        "print(json.dumps({'home': os.environ.get('HOME'), 'content': content}))\n",
        encoding="utf-8",
    )

    proc = _run(
        script,
        env=build_limited_env(home_dir=isolated_home),
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["home"] == str(isolated_home)
    assert result["content"] is None
    assert _CANARY_MARKER not in proc.stdout


def test_home_isolation_does_not_block_hardcoded_absolute_path_read(tmp_path: Path) -> None:
    """Documented gap: no chroot/container means a hardcoded real path still reads.

    This is NOT a passing security guarantee; it is the honest boundary of
    what env-var-based HOME isolation can do without Docker. Full filesystem
    isolation is the Docker lane's job (see module docstring).
    """
    fake_real_home = tmp_path / "fake_real_home"
    (fake_real_home / ".aws").mkdir(parents=True)
    canary_path = fake_real_home / ".aws" / "credentials"
    canary_path.write_text(_CANARY_MARKER, encoding="utf-8")

    isolated_home = tmp_path / "isolated_home"

    script = tmp_path / "read_via_hardcoded_path.py"
    script.write_text(
        "import json, sys\n"
        "with open(sys.argv[1], encoding='utf-8') as fh:\n"
        "    content = fh.read()\n"
        "print(json.dumps({'content': content}))\n",
        encoding="utf-8",
    )

    proc = _run(
        script,
        str(canary_path),
        env=build_limited_env(home_dir=isolated_home),
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    # Documents the gap: a hardcoded absolute path still reaches the real
    # canary despite HOME isolation, because there is no chroot/container.
    assert result["content"] == _CANARY_MARKER


def test_secret_env_vars_never_reach_child_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret_names = [
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "TINKER_API_KEY",
        "DAYTONA_API_KEY",
        "HF_TOKEN",
        "ANTHROPIC_API_KEY",
    ]
    for name in secret_names:
        monkeypatch.setenv(name, f"secret-value-for-{name}")

    script = tmp_path / "read_secrets.py"
    script.write_text(
        "import json, os, sys\n"
        "names = sys.argv[1:]\n"
        "print(json.dumps({n: os.environ.get(n) for n in names}))\n",
        encoding="utf-8",
    )

    proc = _run(
        script,
        *secret_names,
        env=build_limited_env(home_dir=tmp_path / "home"),
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    for name in secret_names:
        assert result[name] is None, f"{name} leaked into validation subprocess env"


def test_cpu_rlimit_kills_busy_loop_quickly(tmp_path: Path) -> None:
    """A pure-CPU busy loop gets killed by RLIMIT_CPU well before the outer timeout."""
    script = tmp_path / "busy_loop.py"
    script.write_text("while True:\n    pass\n", encoding="utf-8")

    def _limits() -> None:
        set_resource_limits(cpu_seconds=1, memory_gb=1)

    started = time.monotonic()
    proc = _run(
        script,
        env=build_limited_env(home_dir=tmp_path / "home"),
        cwd=tmp_path,
        timeout=10.0,
        preexec_fn=_limits,
    )
    elapsed = time.monotonic() - started

    # Killed by SIGXCPU (or a subsequent signal), not by our outer timeout:
    # a POSIX process killed by a signal reports a negative returncode.
    assert proc.returncode != 0
    assert proc.returncode < 0, (
        f"expected termination by signal (negative returncode), got {proc.returncode}"
    )
    assert elapsed < 8.0, f"rlimit did not fire quickly enough: {elapsed:.2f}s"


def test_resource_limits_do_not_block_legitimate_interpreter_spawn(
    tmp_path: Path,
) -> None:
    """Resource limits allow the harness to launch a child and grandchild.

    RLIMIT_NPROC counts processes across an effective UID, so it remains unset.
    A process-tree limit requires a separate container policy.
    """
    script = tmp_path / "legitimate_spawn.py"
    script.write_text(
        "import json, subprocess, sys\n"
        "proc = subprocess.run([sys.executable, '-c', 'print(1)'], "
        "capture_output=True, text=True)\n"
        "print(json.dumps({'rc': proc.returncode, 'out': proc.stdout.strip()}))\n",
        encoding="utf-8",
    )

    def _limits() -> None:
        set_resource_limits(cpu_seconds=30, memory_gb=2)

    proc = _run(
        script,
        env=build_limited_env(home_dir=tmp_path / "home"),
        cwd=tmp_path,
        timeout=30.0,
        preexec_fn=_limits,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0, "legitimate grandchild spawn failed under resource limits"
    assert result["out"] == "1"


@pytest.mark.skip(
    reason=(
        "Network egress isolation is NOT implemented without Docker (no "
        "network namespace/firewall applied here). Asserting a blocked "
        "connection would claim a guarantee this sandbox does not provide. "
        "Tracked for the Docker/container acceptance-grade isolation lane."
    )
)
def test_network_egress_isolation_documented_gap() -> None:
    raise AssertionError("not reached: see skip reason")

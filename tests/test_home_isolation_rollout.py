"""Exercise isolated HOME and working-directory configuration through real child processes.

score_pymc_semantics and _validate_gold_code create a per-call temporary directory and pass its
home subdirectory to build_limited_env. Their syntax gates reject untrusted OS access
separately. These tests execute child Python processes that read their own HOME, working
directory and an isolated credential-file control.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from ppl_synth_data.validation.subprocess_utils import (
    build_limited_env,
    resolve_python_cmd,
    run_with_process_group_timeout,
)

pytestmark = [pytest.mark.slow, pytest.mark.real_boundary]

_CANARY_MARKER = "CANARY-SECRET-HOMEDIR-ROLLOUT-7c1d4e9a"


def _tempdir_wired_env_and_cwd(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Reproduce the exact tempdir/home_dir/cwd shape now used by
    `score_pymc_semantics` and `_validate_gold_code`: a fresh per-invocation
    tempdir, HOME isolated to `<tempdir>/home`, cwd pinned to the tempdir
    itself. Both functions build this with `tempfile.TemporaryDirectory()`;
    here `tmp_path` (pytest's own per-test tempdir) plays that role.
    """
    home_dir = tmp_path / "home"
    env = build_limited_env(home_dir=home_dir)
    return env, tmp_path


def test_score_pymc_semantics_wiring_isolates_home_and_pins_cwd(tmp_path: Path) -> None:
    """`score_pymc_semantics`'s subprocess env/cwd shape, exercised directly.

    Mirrors validation/pymc_semantics.py's `run_with_process_group_timeout`
    call: `env=build_limited_env(..., home_dir=temp_path / "home")` and
    `cwd=str(temp_path)`. Proves a real child sees the isolated HOME, not the
    real one, and runs from the isolated tempdir, not the caller's cwd.
    """
    env, cwd = _tempdir_wired_env_and_cwd(tmp_path)

    script = tmp_path / "report_home_and_cwd.py"
    script.write_text(
        "import json, os\n"
        "print(json.dumps({'home': os.environ.get('HOME'), 'cwd': os.getcwd()}))\n",
        encoding="utf-8",
    )

    proc = run_with_process_group_timeout(
        [*resolve_python_cmd(), str(script)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        cwd=str(cwd),
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["home"] == str(tmp_path / "home")
    assert result["home"] != os.environ.get("HOME")
    assert Path(result["cwd"]).resolve() == cwd.resolve()


def test_validate_gold_code_wiring_isolates_home_and_pins_cwd(tmp_path: Path) -> None:
    """Exercise _validate_gold_code's isolated HOME and working directory with stdin payload input.

    """
    env, cwd = _tempdir_wired_env_and_cwd(tmp_path)

    proc = subprocess.run(
        [
            *resolve_python_cmd(),
            "-c",
            "import json, os, sys\n"
            "sys.stdin.read()\n"
            "print(json.dumps({'home': os.environ.get('HOME'), 'cwd': os.getcwd()}))\n",
        ],
        input="unused-stdin-payload",
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        start_new_session=True,
        cwd=str(cwd),
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["home"] == str(tmp_path / "home")
    assert result["home"] != os.environ.get("HOME")
    assert Path(result["cwd"]).resolve() == cwd.resolve()


def test_tempdir_home_dir_blocks_expanduser_credential_lookup(tmp_path: Path) -> None:
    """Combined cwd+home_dir wiring still closes the `~`/`$HOME` credential path.

    `test_security_sandbox_escape.py` already proves this for `home_dir` in
    isolation; this test additionally sets `cwd` to the same tempdir (the
    delta this rollout added to `pymc_semantics.py`/`export_pymc.py`), so the
    combination is exercised the way those two production call sites now use
    it, not `home_dir` on its own.
    """
    fake_real_home = tmp_path / "fake_real_home"
    (fake_real_home / ".aws").mkdir(parents=True)
    (fake_real_home / ".aws" / "credentials").write_text(_CANARY_MARKER, encoding="utf-8")

    isolated_root = tmp_path / "isolated_root"
    isolated_root.mkdir()
    env = build_limited_env(home_dir=isolated_root / "home")

    script = isolated_root / "read_via_expanduser.py"
    script.write_text(
        "import json, os\n"
        "path = os.path.expanduser('~/.aws/credentials')\n"
        "try:\n"
        "    with open(path, encoding='utf-8') as fh:\n"
        "        content = fh.read()\n"
        "except OSError:\n"
        "    content = None\n"
        "print(json.dumps({'content': content}))\n",
        encoding="utf-8",
    )

    proc = run_with_process_group_timeout(
        [*resolve_python_cmd(), str(script)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        cwd=str(isolated_root),
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["content"] is None
    assert _CANARY_MARKER not in proc.stdout


def test_build_limited_env_requires_home_dir() -> None:
    """Calling build_limited_env without its required home_dir argument raises TypeError."""
    with pytest.raises(TypeError):
        build_limited_env()  # type: ignore[call-arg]

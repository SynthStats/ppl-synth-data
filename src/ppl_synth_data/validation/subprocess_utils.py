from __future__ import annotations

import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_BYTES_PER_GIB = 1024 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)


def _source_project_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def resolve_python_cmd() -> list[str]:
    if platform.system() == "Windows":
        # Windows Job Objects are assigned after process creation in the
        # subprocess path. Avoid an extra uv wrapper that could spawn the
        # validator child before that assignment completes.
        return [sys.executable]
    uv = shutil.which("uv")
    if uv:
        project_root = _source_project_root()
        if project_root is not None:
            return [uv, "run", "--project", str(project_root), "python"]
    return [sys.executable]


def set_resource_limits(*, cpu_seconds: int = 30, memory_gb: int = 4) -> None:
    """Set CPU and memory limits for subprocess (Unix only).

    Intended for the POSIX child hook passed through run_with_process_group_timeout(). Sets:
    - RLIMIT_CPU: configurable CPU time (also bounds a fork bomb: runaway
      children burn the shared CPU budget and the whole group is killed on
      timeout via the process-group runner)
    - RLIMIT_AS: configurable virtual memory on Linux only; macOS maps shared
      libraries into virtual address space, making strict AS limits
      too restrictive for Python processes)

    RLIMIT_NPROC is deliberately NOT set. It is a per-effective-uid limit that
    counts every process the invoking user owns, not just this sandbox's
    process tree, so any value low enough to stop a fork bomb also blocks the
    interpreter launch itself (the verifier spawns `uv run python`, which forks)
    and fails with EAGAIN on a normally-loaded host. Per-process-tree limiting
    belongs to the Docker/cgroups acceptance lane, alongside network isolation.
    """
    if platform.system() == "Windows":
        return
    import resource

    cpu_seconds = max(1, int(cpu_seconds))
    memory_gb = max(1, int(memory_gb))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    if platform.system() == "Linux":
        memory_bytes = memory_gb * _BYTES_PER_GIB
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def run_with_process_group_timeout(
    args: Sequence[str],
    *,
    timeout: float,
    kill_timeout: float = 5.0,
    capture_output: bool = False,
    check: bool = False,
    input: str | bytes | None = None,  # noqa: A002 - mirrors subprocess.run
    **kwargs: Any,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """Run a command and terminate its whole process group on timeout.

    `resolve_python_cmd()` may return `uv run --project ... python`, where `uv`
    is only a wrapper around the actual Python interpreter. `subprocess.run(...,
    timeout=...)` kills the wrapper but can leave the child Python process
    orphaned. Stage validators run untrusted probabilistic programs, so timeout
    cleanup must target the process group.
    """
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr may not be used with capture_output")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input may not both be used")
        kwargs["stdin"] = subprocess.PIPE

    # Always create a new process group/session so timeout cleanup can include
    # wrapper-spawned descendants such as `uv run python`.
    kwargs.pop("start_new_session", None)
    if platform.system() == "Windows":
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(  # noqa: S603 - callers pass explicit argv only
        list(args),
        **kwargs,
    )
    windows_job = _create_windows_job_for_process(proc)
    communicate_kwargs: dict[str, Any] = {"timeout": timeout}
    if input is not None:
        communicate_kwargs["input"] = input
    try:
        stdout, stderr = proc.communicate(**communicate_kwargs)
    except subprocess.TimeoutExpired as exc:
        _signal_process_group(proc, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc, windows_job=windows_job)
            stdout, stderr = proc.communicate()
        else:
            if platform.system() == "Windows" and windows_job is not None:
                _kill_process_group(proc, windows_job=windows_job)
        raise subprocess.TimeoutExpired(
            cmd=list(args),
            timeout=timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    finally:
        if windows_job is not None:
            _close_windows_handle(windows_job)

    result = subprocess.CompletedProcess(list(args), proc.returncode, stdout, stderr)
    if check and result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            list(args),
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _signal_process_group(proc: subprocess.Popen[Any], sig: int) -> None:
    if platform.system() == "Windows":
        if proc.poll() is not None:
            return
        if sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        # The child moved to a process group we may not signal as a group
        # (observed on macOS under load). Fall back to the direct child.
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            return


def _kill_process_group(
    proc: subprocess.Popen[Any],
    *,
    windows_job: int | None = None,
) -> None:
    if platform.system() == "Windows":
        if windows_job is not None and _terminate_windows_job(windows_job):
            return
        if proc.poll() is not None:
            return
        proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        # killpg denied (macOS, under load): kill the direct child at least,
        # rather than letting the timeout handler crash.
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError):
            return


def _load_windows_kernel32() -> Any | None:
    if platform.system() != "Windows":
        return None
    import ctypes
    from ctypes import wintypes

    windll = getattr(ctypes, "WinDLL", None)
    if windll is None:
        return None
    kernel32 = windll("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _windows_handle_value(handle: Any) -> int | None:
    value = getattr(handle, "value", handle)
    if value is None:
        return None
    return int(value)


def _create_windows_job_for_process(proc: subprocess.Popen[Any]) -> int | None:
    kernel32 = _load_windows_kernel32()
    if kernel32 is None:
        if platform.system() == "Windows":
            _LOGGER.warning(
                "Windows Job Object APIs are unavailable; validator timeout "
                "cleanup will only target the direct subprocess"
            )
        return None
    job = kernel32.CreateJobObjectW(None, None)
    job_handle = _windows_handle_value(job)
    if job_handle is None:
        _LOGGER.warning(
            "CreateJobObjectW returned an invalid handle%s; validator timeout "
            "cleanup will only target the direct subprocess",
            _windows_last_error_message(),
        )
        return None
    process_handle = _windows_handle_value(getattr(proc, "_handle", None))
    if process_handle is None:
        _close_windows_handle(job_handle)
        _LOGGER.warning(
            "subprocess handle is unavailable for Windows Job Object assignment; "
            "validator timeout cleanup will only target the direct subprocess"
        )
        return None
    if not kernel32.AssignProcessToJobObject(job_handle, process_handle):
        error_message = _windows_last_error_message()
        _close_windows_handle(job_handle)
        _LOGGER.warning(
            "AssignProcessToJobObject failed%s; validator timeout cleanup will "
            "only target the direct subprocess",
            error_message,
        )
        return None
    return job_handle


def _terminate_windows_job(job_handle: int) -> bool:
    kernel32 = _load_windows_kernel32()
    if kernel32 is None:
        if platform.system() == "Windows":
            _LOGGER.warning(
                "Windows Job Object APIs are unavailable while terminating a "
                "validator job; descendant cleanup may have failed"
            )
        return False
    terminated = bool(kernel32.TerminateJobObject(job_handle, 1))
    if not terminated:
        _LOGGER.warning(
            "TerminateJobObject failed%s; descendant cleanup may have failed",
            _windows_last_error_message(),
        )
    return terminated


def _windows_last_error_message() -> str:
    if platform.system() != "Windows":
        return ""
    import ctypes

    get_last_error = getattr(ctypes, "get_last_error", None)
    if get_last_error is None:
        return ""
    error_code = int(get_last_error())
    if not error_code:
        return ""
    format_error = getattr(ctypes, "FormatError", None)
    if format_error is None:
        return f" (GetLastError={error_code})"
    return f" (GetLastError={error_code}: {format_error(error_code).strip()})"


def _close_windows_handle(handle: int) -> None:
    kernel32 = _load_windows_kernel32()
    if kernel32 is not None:
        if not kernel32.CloseHandle(handle):
            _LOGGER.warning(
                "CloseHandle failed for Windows Job Object handle%s",
                _windows_last_error_message(),
            )


_SAFE_ENV_KEYS = frozenset(
    {
        "CURL_CA_BUNDLE",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SHELL",
        "SYSTEMROOT",
        "TMP",
        "TMPDIR",
        "TEMP",
        "USER",
        "UV_CACHE_DIR",
        "UV_CONFIG_FILE",
        "UV_OFFLINE",
        "UV_PYTHON",
        "UV_PYTHON_INSTALL_DIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)
_SECRET_TOKENS = (
    "ACCESS_KEY",
    "API_BASE",
    "API_KEY",
    "BEARER",
    "CREDENTIAL",
    "CREDENTIALS",
    "DATABASE_URL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)


def _looks_secret_key(name: str) -> bool:
    upper = name.upper()
    return any(token in upper for token in _SECRET_TOKENS)


def build_limited_env(
    *,
    home_dir: str | Path,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build an allowlisted environment for a generated-code validator subprocess.

    `home_dir` overrides HOME (created if missing) instead of passing the real
    HOME through. This closes the "read credentials via `~` / `$HOME`
    convention" path (os.path.expanduser, `$HOME/.aws/credentials`,
    `$HOME/.netrc`, ...) for untrusted code running in the subprocess. It is
    NOT filesystem sandboxing: code that hardcodes an absolute path to the
    real home directory can still read it, because there is no chroot/
    container here. Full filesystem isolation needs Docker.

    `home_dir` is required, not optional: a security-isolation parameter that
    silently falls back to the real HOME on omission is exactly the kind of
    silent default this project bans. Every caller must supply an explicit
    per-invocation tempdir (see `env_runtime.py`, `pymc_runtime.py`,
    `pymc_semantics.py`, `harbor/export_pymc.py` for the shared pattern).
    """
    # Pass only runtime plumbing into generated-code validators. This is an
    # allowlist because these subprocesses execute untrusted completions.
    filtered_env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _SAFE_ENV_KEYS and not _looks_secret_key(key)
    }
    env = {
        **filtered_env,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    home_path = Path(home_dir)
    home_path.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home_path)
    if extra:
        leaked_keys = sorted(key for key in extra if _looks_secret_key(key))
        if leaked_keys:
            raise ValueError(
                "refusing to pass secret-like environment keys to validation "
                f"subprocess: {', '.join(leaked_keys)}"
            )
        env.update(extra)
    return env

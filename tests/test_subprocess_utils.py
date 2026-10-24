from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

import ppl_synth_data.validation.subprocess_utils as subprocess_utils
from ppl_synth_data.validation.subprocess_utils import (
    build_limited_env,
    resolve_python_cmd,
    run_with_process_group_timeout,
    set_resource_limits,
)


@pytest.mark.narrow_unit_not_acceptance
def test_resolve_python_cmd_prefers_uv(monkeypatch) -> None:
    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        "ppl_synth_data.validation.subprocess_utils.shutil.which",
        lambda name: "/tmp/uv",
    )
    cmd = resolve_python_cmd()

    assert cmd == [
        "/tmp/uv",
        "run",
        "--project",
        str(Path(__file__).resolve().parents[1]),
        "python",
    ]


@pytest.mark.narrow_unit_not_acceptance
def test_resolve_python_cmd_pins_project_from_meta_folder(monkeypatch) -> None:
    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        "ppl_synth_data.validation.subprocess_utils.shutil.which",
        lambda name: "/tmp/uv",
    )
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    cmd = resolve_python_cmd()

    assert cmd == [
        "/tmp/uv",
        "run",
        "--project",
        str(Path(__file__).resolve().parents[1]),
        "python",
    ]


@pytest.mark.narrow_unit_not_acceptance
def test_resolve_python_cmd_falls_back_to_current_python(monkeypatch) -> None:
    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        "ppl_synth_data.validation.subprocess_utils.shutil.which",
        lambda name: None,
    )
    assert resolve_python_cmd() == [sys.executable]


@pytest.mark.narrow_unit_not_acceptance
def test_resolve_python_cmd_uses_current_python_on_windows_to_avoid_uv_wrapper_race(
    monkeypatch,
) -> None:
    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        "ppl_synth_data.validation.subprocess_utils.shutil.which",
        lambda name: "/tmp/uv",
    )

    assert resolve_python_cmd() == [sys.executable]


@pytest.mark.narrow_unit_not_acceptance
def test_build_limited_env_sets_thread_limits_and_merges_extra(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOME_PARENT_ENV", "yes")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-key")
    monkeypatch.setenv("DATABASE_URL", "postgres://secret")
    monkeypatch.setenv("PRIVATE_KEY", "private-key")
    out = build_limited_env(extra={"PYTHONNOUSERSITE": "1"}, home_dir=tmp_path / "home")

    assert out["OMP_NUM_THREADS"] == "1"
    assert out["MKL_NUM_THREADS"] == "1"
    assert out["OPENBLAS_NUM_THREADS"] == "1"
    assert out["NUMEXPR_NUM_THREADS"] == "1"
    assert out["SystemRoot"] == r"C:\Windows"
    assert out["PYTHONNOUSERSITE"] == "1"
    assert "SOME_PARENT_ENV" not in out
    assert "AWS_ACCESS_KEY_ID" not in out
    assert "AWS_SECRET_ACCESS_KEY" not in out
    assert "DATABASE_URL" not in out
    assert "PRIVATE_KEY" not in out


def test_build_limited_env_rejects_secret_like_extra(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_limited_env(extra={"OPENAI_API_KEY": "secret"}, home_dir=tmp_path / "home")


@pytest.mark.narrow_unit_not_acceptance
def test_set_resource_limits_uses_configured_cpu_and_memory(monkeypatch) -> None:
    resource = pytest.importorskip("resource")
    calls: list[tuple[int, tuple[int, int]]] = []

    monkeypatch.setattr(
        "ppl_synth_data.validation.subprocess_utils.platform.system",
        lambda: "Linux",
    )
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda key, limits: calls.append((key, limits)),
    )

    set_resource_limits(cpu_seconds=75, memory_gb=6)

    assert (resource.RLIMIT_CPU, (75, 75)) in calls
    assert (resource.RLIMIT_AS, (6 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024)) in calls


@pytest.mark.narrow_unit_not_acceptance
def test_run_with_process_group_timeout_kills_group(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    class FakeProc:
        pid = 12345
        returncode = None

        def __init__(self) -> None:
            self.communicate_count = 0

        def communicate(self, timeout=None):  # noqa: ANN001
            self.communicate_count += 1
            if self.communicate_count == 1:
                raise subprocess.TimeoutExpired(cmd=["cmd"], timeout=timeout)
            self.returncode = -15
            return "", ""

        def poll(self):  # noqa: ANN201
            return self.returncode

    def fake_popen(*args, **kwargs):  # noqa: ANN001
        _ = args
        assert kwargs["start_new_session"] is True
        assert "creationflags" not in kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "ppl_synth_data.validation.subprocess_utils.os.killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_with_process_group_timeout(["cmd"], timeout=0.01)

    assert calls
    assert calls[0][0] == 12345


@pytest.mark.narrow_unit_not_acceptance
def test_run_with_process_group_timeout_escalates_to_sigkill(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    class FakeProc:
        pid = 12345
        returncode = None

        def __init__(self) -> None:
            self.communicate_count = 0

        def communicate(self, timeout=None):  # noqa: ANN001
            self.communicate_count += 1
            if self.communicate_count <= 2:
                raise subprocess.TimeoutExpired(cmd=["cmd"], timeout=timeout)
            self.returncode = -9
            return "", ""

        def poll(self):  # noqa: ANN201
            return self.returncode

    def fake_popen(*args, **kwargs):  # noqa: ANN001
        _ = args
        assert kwargs["start_new_session"] is True
        assert "creationflags" not in kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        subprocess_utils.os,
        "killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_with_process_group_timeout(["cmd"], timeout=0.01, kill_timeout=0.01)

    assert calls == [
        (12345, subprocess_utils.signal.SIGTERM),
        (12345, subprocess_utils.signal.SIGKILL),
    ]


@pytest.mark.narrow_unit_not_acceptance
def test_run_with_process_group_timeout_signals_group_after_wrapper_exit(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeProc:
        pid = 12345
        returncode = 0

        def __init__(self) -> None:
            self.communicate_count = 0

        def communicate(self, timeout=None):  # noqa: ANN001
            self.communicate_count += 1
            if self.communicate_count == 1:
                raise subprocess.TimeoutExpired(cmd=["cmd"], timeout=timeout)
            return "", ""

        def poll(self):  # noqa: ANN201
            return self.returncode

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Linux")

    def fake_popen(*args, **kwargs):  # noqa: ANN001
        _ = args
        assert kwargs["start_new_session"] is True
        return FakeProc()

    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        subprocess_utils.os,
        "killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_with_process_group_timeout(["cmd"], timeout=0.01)

    assert calls == [(12345, subprocess_utils.signal.SIGTERM)]


@pytest.mark.narrow_unit_not_acceptance
def test_create_windows_job_warns_when_job_api_unavailable(monkeypatch, caplog) -> None:
    class FakeProc:
        pass

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess_utils, "_load_windows_kernel32", lambda: None)

    with caplog.at_level(logging.WARNING, logger=subprocess_utils.__name__):
        assert subprocess_utils._create_windows_job_for_process(FakeProc()) is None

    assert "Windows Job Object APIs are unavailable" in caplog.text


@pytest.mark.narrow_unit_not_acceptance
def test_create_windows_job_warns_with_last_error_for_invalid_job_handle(
    monkeypatch,
    caplog,
) -> None:
    class FakeKernel32:
        def CreateJobObjectW(self, security_attrs, name):  # noqa: ANN001, N802
            _ = security_attrs, name
            return None

    class FakeProc:
        _handle = 54321

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess_utils, "_load_windows_kernel32", lambda: FakeKernel32())
    monkeypatch.setattr(
        subprocess_utils,
        "_windows_last_error_message",
        lambda: " (GetLastError=8)",
    )

    with caplog.at_level(logging.WARNING, logger=subprocess_utils.__name__):
        assert subprocess_utils._create_windows_job_for_process(FakeProc()) is None

    assert "CreateJobObjectW returned an invalid handle (GetLastError=8)" in caplog.text


@pytest.mark.narrow_unit_not_acceptance
def test_create_windows_job_assigns_process_handle(monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []

    class FakeKernel32:
        def CreateJobObjectW(self, security_attrs, name):  # noqa: ANN001, N802
            assert security_attrs is None
            assert name is None
            return 9876

        def AssignProcessToJobObject(self, job_handle, process_handle):  # noqa: ANN001, N802
            calls.append(("assign", job_handle, process_handle))
            return True

    class FakeProc:
        _handle = 54321

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess_utils, "_load_windows_kernel32", lambda: FakeKernel32())

    assert subprocess_utils._create_windows_job_for_process(FakeProc()) == 9876
    assert calls == [("assign", 9876, 54321)]


@pytest.mark.narrow_unit_not_acceptance
def test_create_windows_job_warns_and_closes_without_process_handle(
    monkeypatch,
    caplog,
) -> None:
    calls: list[str] = []

    class FakeKernel32:
        def CreateJobObjectW(self, security_attrs, name):  # noqa: ANN001, N802
            _ = security_attrs, name
            return 9876

    class FakeProc:
        pass

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess_utils, "_load_windows_kernel32", lambda: FakeKernel32())
    monkeypatch.setattr(
        subprocess_utils,
        "_close_windows_handle",
        lambda handle: calls.append(f"close:{handle}"),
    )

    with caplog.at_level(logging.WARNING, logger=subprocess_utils.__name__):
        assert subprocess_utils._create_windows_job_for_process(FakeProc()) is None

    assert calls == ["close:9876"]
    assert "subprocess handle is unavailable" in caplog.text


@pytest.mark.narrow_unit_not_acceptance
def test_create_windows_job_warns_and_closes_when_assignment_fails(
    monkeypatch,
    caplog,
) -> None:
    calls: list[str] = []

    class FakeKernel32:
        def CreateJobObjectW(self, security_attrs, name):  # noqa: ANN001, N802
            _ = security_attrs, name
            return 9876

        def AssignProcessToJobObject(self, job_handle, process_handle):  # noqa: ANN001, N802
            calls.append(f"assign:{job_handle}:{process_handle}")
            return False

    class FakeProc:
        _handle = 54321

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess_utils, "_load_windows_kernel32", lambda: FakeKernel32())
    monkeypatch.setattr(
        subprocess_utils,
        "_close_windows_handle",
        lambda handle: calls.append(f"close:{handle}"),
    )
    monkeypatch.setattr(
        subprocess_utils,
        "_windows_last_error_message",
        lambda: " (GetLastError=5)",
    )

    with caplog.at_level(logging.WARNING, logger=subprocess_utils.__name__):
        assert subprocess_utils._create_windows_job_for_process(FakeProc()) is None

    assert calls == ["assign:9876:54321", "close:9876"]
    assert "AssignProcessToJobObject failed (GetLastError=5)" in caplog.text


@pytest.mark.narrow_unit_not_acceptance
def test_terminate_windows_job_warns_when_termination_fails(monkeypatch, caplog) -> None:
    class FakeKernel32:
        def TerminateJobObject(self, job_handle, exit_code):  # noqa: ANN001, N802
            assert job_handle == 9876
            assert exit_code == 1
            return False

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess_utils, "_load_windows_kernel32", lambda: FakeKernel32())
    monkeypatch.setattr(
        subprocess_utils,
        "_windows_last_error_message",
        lambda: " (GetLastError=5)",
    )

    with caplog.at_level(logging.WARNING, logger=subprocess_utils.__name__):
        assert subprocess_utils._terminate_windows_job(9876) is False

    assert "TerminateJobObject failed (GetLastError=5)" in caplog.text


@pytest.mark.narrow_unit_not_acceptance
def test_close_windows_handle_warns_when_close_fails(monkeypatch, caplog) -> None:
    class FakeKernel32:
        def CloseHandle(self, handle):  # noqa: ANN001, N802
            assert handle == 9876
            return False

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess_utils, "_load_windows_kernel32", lambda: FakeKernel32())
    monkeypatch.setattr(
        subprocess_utils,
        "_windows_last_error_message",
        lambda: " (GetLastError=6)",
    )

    with caplog.at_level(logging.WARNING, logger=subprocess_utils.__name__):
        subprocess_utils._close_windows_handle(9876)

    assert "CloseHandle failed for Windows Job Object handle (GetLastError=6)" in caplog.text


@pytest.mark.narrow_unit_not_acceptance
def test_run_with_process_group_timeout_uses_windows_process_controls(
    monkeypatch,
) -> None:
    calls: list[str] = []
    creation_flag = 0x200
    job_handle = 9876

    class FakeProc:
        pid = 12345
        returncode = None
        _handle = 54321

        def __init__(self) -> None:
            self.communicate_count = 0

        def communicate(self, timeout=None):  # noqa: ANN001
            self.communicate_count += 1
            if self.communicate_count <= 2:
                raise subprocess.TimeoutExpired(cmd=["cmd"], timeout=timeout)
            self.returncode = -9
            return "", ""

        def poll(self):  # noqa: ANN201
            return self.returncode

        def terminate(self) -> None:
            calls.append("terminate")

        def kill(self) -> None:
            calls.append("kill")

    def fake_terminate_job(handle: int) -> bool:
        calls.append(f"terminate_job:{handle}")
        return True

    def fake_popen(*args, **kwargs):  # noqa: ANN001
        _ = args
        assert kwargs["creationflags"] & creation_flag
        assert "start_new_session" not in kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        subprocess_utils.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        creation_flag,
        raising=False,
    )
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        subprocess_utils.os,
        "killpg",
        lambda *args, **kwargs: pytest.fail("Windows path must not call os.killpg"),
    )
    monkeypatch.setattr(
        subprocess_utils,
        "_create_windows_job_for_process",
        lambda proc: job_handle,
    )
    monkeypatch.setattr(
        subprocess_utils,
        "_terminate_windows_job",
        fake_terminate_job,
    )
    monkeypatch.setattr(
        subprocess_utils,
        "_close_windows_handle",
        lambda handle: calls.append(f"close_job:{handle}"),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_with_process_group_timeout(["cmd"], timeout=0.01, kill_timeout=0.01)

    assert calls == [
        "terminate",
        f"terminate_job:{job_handle}",
        f"close_job:{job_handle}",
    ]


@pytest.mark.narrow_unit_not_acceptance
def test_run_with_process_group_timeout_falls_back_when_windows_job_unavailable(
    monkeypatch,
) -> None:
    calls: list[str] = []
    creation_flag = 0x200

    class FakeProc:
        pid = 12345
        returncode = None

        def __init__(self) -> None:
            self.communicate_count = 0

        def communicate(self, timeout=None):  # noqa: ANN001
            self.communicate_count += 1
            if self.communicate_count <= 2:
                raise subprocess.TimeoutExpired(cmd=["cmd"], timeout=timeout)
            self.returncode = -9
            return "", ""

        def poll(self):  # noqa: ANN201
            return self.returncode

        def terminate(self) -> None:
            calls.append("terminate")

        def kill(self) -> None:
            calls.append("kill")

    def fake_popen(*args, **kwargs):  # noqa: ANN001
        _ = args
        assert kwargs["creationflags"] & creation_flag
        return FakeProc()

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        subprocess_utils.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        creation_flag,
        raising=False,
    )
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        subprocess_utils,
        "_create_windows_job_for_process",
        lambda proc: None,
    )
    monkeypatch.setattr(
        subprocess_utils,
        "_terminate_windows_job",
        lambda handle: pytest.fail("no Windows job should be terminated"),
    )
    monkeypatch.setattr(
        subprocess_utils.os,
        "killpg",
        lambda *args, **kwargs: pytest.fail("Windows path must not call os.killpg"),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_with_process_group_timeout(["cmd"], timeout=0.01, kill_timeout=0.01)

    assert calls == ["terminate", "kill"]


@pytest.mark.narrow_unit_not_acceptance
def test_run_with_process_group_timeout_terminates_windows_job_after_wrapper_exit(
    monkeypatch,
) -> None:
    calls: list[str] = []
    creation_flag = 0x200
    job_handle = 6789

    class FakeProc:
        pid = 12345
        returncode = None

        def __init__(self) -> None:
            self.communicate_count = 0

        def communicate(self, timeout=None):  # noqa: ANN001
            self.communicate_count += 1
            if self.communicate_count == 1:
                self.returncode = 0
                raise subprocess.TimeoutExpired(cmd=["cmd"], timeout=timeout)
            return "", ""

        def poll(self):  # noqa: ANN201
            return self.returncode

        def terminate(self) -> None:
            calls.append("terminate")

        def kill(self) -> None:
            calls.append("kill")

    def fake_popen(*args, **kwargs):  # noqa: ANN001
        _ = args
        assert kwargs["creationflags"] & creation_flag
        return FakeProc()

    def fake_terminate_job(handle: int) -> bool:
        calls.append(f"terminate_job:{handle}")
        return True

    monkeypatch.setattr(subprocess_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        subprocess_utils.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        creation_flag,
        raising=False,
    )
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        subprocess_utils.os,
        "killpg",
        lambda *args, **kwargs: pytest.fail("Windows path must not call os.killpg"),
    )
    monkeypatch.setattr(
        subprocess_utils,
        "_create_windows_job_for_process",
        lambda proc: job_handle,
    )
    monkeypatch.setattr(
        subprocess_utils,
        "_terminate_windows_job",
        fake_terminate_job,
    )
    monkeypatch.setattr(
        subprocess_utils,
        "_close_windows_handle",
        lambda handle: calls.append(f"close_job:{handle}"),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_with_process_group_timeout(["cmd"], timeout=0.01, kill_timeout=0.01)

    assert calls == [
        f"terminate_job:{job_handle}",
        f"close_job:{job_handle}",
    ]

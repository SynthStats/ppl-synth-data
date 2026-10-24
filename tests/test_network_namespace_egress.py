"""Egress isolation: prove `unshare -n` denies network access to untrusted code.

Two kinds of test:
- Content-pin tests (portable, run everywhere incl. macOS CI): export the real
  Harbor tasks and assert the generated test.sh / test_lazyppl.sh wrap the
  untrusted validator/compiler/run invocations in `unshare -n`.
- Real egress-denial test (Linux + root gated): actually runs `unshare -n`
  around a raw-TCP + DNS probe subprocess and asserts 0 endpoints reachable,
  with a positive control (no unshare) asserting > 0 reachable. Skips with an
  explicit reason everywhere else (this is not a silent pass).

Context: Daytona's own `network_block_all` / `network_allow_list` only enforce
at the DNS layer (raw TCP to an IP still connects); `unshare -n` is verified to
deny egress at the kernel level. The positive control checks connectivity without a network
namespace.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ppl_synth_data.harbor.export_pymc import export_harbor_tasks

# Reuse the existing export fixtures/helpers rather than duplicating them;
# neither test module has a shared conftest.py in this repo, so pytest
# fixtures are imported by name from their defining test module.
from tests.test_harbor_export import run_dir  # noqa: F401
from tests.test_harbor_lazyppl_export import (  # noqa: F401
    _run_export,
    docker_app_dir,
    docker_file,
    lazyppl_local_dir,
    seeds_dir,
)


def _single_exported_task_dir(out: Path) -> Path:
    entries = set(out.iterdir())
    summary = out / "export_summary.json"
    assert summary in entries
    assert summary.is_file()
    assert len(entries) == 2
    (task_dir,) = entries - {summary}
    assert task_dir.is_dir()
    return task_dir


def test_exported_scripts_wrap_untrusted_in_netns(
    run_dir: Path,  # noqa: F811
    tmp_path: Path,
    seeds_dir: Path,  # noqa: F811
    docker_app_dir: Path,  # noqa: F811
    docker_file: Path,  # noqa: F811
    lazyppl_local_dir: Path,  # noqa: F811
) -> None:
    # PyMC lane: test.sh.
    pymc_out = tmp_path / "pymc_harbor_out"
    export_harbor_tasks(
        run_dir,
        pymc_out,
        run_id="test-run",
        attempt_id="network-isolation",
        allow_unlocked_reference_runtime=True,
    )
    pymc_task_dir = _single_exported_task_dir(pymc_out)
    test_sh = (pymc_task_dir / "tests" / "test.sh").read_text(encoding="utf-8")

    assert "unshare -n true" in test_sh  # preflight capability probe (not just binary presence)
    # Network isolation invariant: the untrusted validator/worker python invocation is wrapped in
    # `unshare -n` (an empty network namespace = kernel egress denial; the netns is
    # inherited by any subprocess the invocation spawns). Pin the invariant, not the
    # exact script path, so a legitimate refactor of the invocation (e.g. a
    # map/reduce split running the validator from a different path) cannot silently
    # drop the wrap without tripping this, while an incidental path change does not.
    assert "unshare -n python" in test_sh
    assert "--submission /workspace/submission.py" in test_sh
    assert "--data /workspace/data.parquet" in test_sh
    assert "--reward-artifact-out /logs/verifier/reward_artifact.json" in test_sh
    assert "--harbor-rewards-out /logs/verifier/reward.json" in test_sh
    assert "set -uo pipefail" in test_sh

    # LazyPPL lane: test.sh (rendered from test_lazyppl.sh.j2).
    lazyppl_out = tmp_path / "lazyppl_harbor_out"
    _run_export(seeds_dir, lazyppl_out, docker_app_dir, docker_file, lazyppl_local_dir)
    lazyppl_task_dir = lazyppl_out / "lazyppl-simple-normal-00001"
    lazyppl_test_sh = (lazyppl_task_dir / "tests" / "test.sh").read_text(encoding="utf-8")

    assert "unshare -n true" in lazyppl_test_sh  # preflight capability probe
    assert "timeout 10s unshare -n ghc -fno-code -i/app /app/Submission.hs" in lazyppl_test_sh
    assert "timeout 30s unshare -n ghc --make -O0" in lazyppl_test_sh
    assert "timeout 120s unshare -n /tmp/run-task > /tmp/log_lik.json" in lazyppl_test_sh
    # score.py is trusted (local-data ELPD scoring); left unwrapped.
    assert "unshare -n python3 /app/score.py" not in lazyppl_test_sh
    assert "timeout 30s python3 /app/score.py" in lazyppl_test_sh


_RAW_TCP_AND_DNS_PROBE = """
import json
import socket

results = {}
for host, port in [("1.1.1.1", 443), ("8.8.8.8", 443)]:
    try:
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
        results[f"tcp_{host}"] = "CONNECTED"
    except OSError as exc:
        results[f"tcp_{host}"] = f"BLOCKED:{type(exc).__name__}"
try:
    socket.getaddrinfo("github.com", 443)
    results["dns_github"] = "CONNECTED"
except socket.gaierror as exc:
    results["dns_github"] = f"BLOCKED:{type(exc).__name__}"

reachable = sum(1 for status in results.values() if status == "CONNECTED")
print(json.dumps({"results": results, "reachable": reachable}))
"""


def test_unshare_denies_egress_real(tmp_path: Path) -> None:
    if platform.system() != "Linux" or os.geteuid() != 0 or shutil.which("unshare") is None:
        pytest.skip(
            "real netns egress proof requires Linux, root, and unshare"
        )
    # unshare binary present and running as root, but a stock-capability container
    # can still deny netns creation (needs CAP_SYS_ADMIN). Probe it so the test
    # SKIPS with a clear reason rather than ERRORing on CalledProcessError below.
    if subprocess.run(["unshare", "-n", "true"], capture_output=True).returncode != 0:
        pytest.skip(
            "unshare present but netns creation not permitted here (needs CAP_SYS_ADMIN); "
            "the real egress proof runs on the Daytona training lane"
        )

    probe_script = tmp_path / "probe.py"
    probe_script.write_text(_RAW_TCP_AND_DNS_PROBE, encoding="utf-8")

    # Positive control: no unshare, egress must be reachable (proves the probe
    # itself detects egress rather than trivially reporting zero everywhere).
    baseline = subprocess.run(
        [sys.executable, str(probe_script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    baseline_result = json.loads(baseline.stdout)
    assert baseline_result["reachable"] > 0, (
        f"positive control expected some egress without unshare: {baseline_result}"
    )

    # Real proof: wrapped in unshare -n, all egress must be denied.
    netns = subprocess.run(
        ["unshare", "-n", sys.executable, str(probe_script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    netns_result = json.loads(netns.stdout)
    assert netns_result["reachable"] == 0, f"unshare -n must deny ALL egress: {netns_result}"

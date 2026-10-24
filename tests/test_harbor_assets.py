from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_harbor_e2e_defaults_to_artifacts_run_root() -> None:
    script = ROOT / "scripts" / "harbor_e2e.sh"
    text = script.read_text(encoding="utf-8")
    assert 'RUN_ROOT="${2:-artifacts/runs}"' in text
    assert 'SCRATCH_ROOT="${PPLDF_HARBOR_E2E_TMPDIR:-$PROJECT_ROOT/artifacts/harbor-e2e}"' in text
    assert 'OUT_DIR=$(mktemp -d "$SCRATCH_ROOT/run.XXXXXX")' in text
    assert 'docker build -t harbor-e2e-test "$TASK_DIR/environment/"' in text
    assert "pymc-harbor:local-build-v1" not in text
    assert "docker image inspect" not in text


def test_harbor_dockerfiles_pin_base_images_by_digest() -> None:
    pymc_dockerfile = (ROOT / "docker" / "pymc-harbor" / "Dockerfile").read_text(encoding="utf-8")
    lazyppl_dockerfile = (ROOT / "docker" / "lazyppl-harbor" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "@sha256:" in pymc_dockerfile.splitlines()[0]
    assert "@sha256:" in lazyppl_dockerfile.splitlines()[0]
    assert (
        "COPY --from=ghcr.io/astral-sh/uv:0.11.21@sha256:"
        "ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946 /uv /uvx /bin/"
    ) in pymc_dockerfile
    assert "UV_LINK_MODE=copy" in pymc_dockerfile
    assert "--mount=" not in pymc_dockerfile
    assert "--no-cache" in pymc_dockerfile
    assert "uv pip install --system" in pymc_dockerfile
    assert "COPY requirements.lock /requirements.lock" in pymc_dockerfile
    assert "--no-deps --require-hashes -r /requirements.lock" in pymc_dockerfile
    pymc_lock = (ROOT / "docker" / "pymc-harbor" / "requirements.lock").read_text(encoding="utf-8")
    assert "arviz==0.23.4 \\\n" in pymc_lock


def test_harbor_docker_contexts_define_dockerignore() -> None:
    assert (ROOT / "docker" / "pymc-harbor" / ".dockerignore").exists()
    assert (ROOT / "docker" / "lazyppl-harbor" / ".dockerignore").exists()

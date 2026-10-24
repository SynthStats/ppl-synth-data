"""A LazyPPL task carries its own verifier environment, and its runtime recipe is pinned.

The LazyPPL scorer image was tested as one build (GHC 9.6.7 with hmatrix 0.20.2, statistics
0.12.0.0 and the other libraries below; Python 3.9 with arviz 0.17.1, numpy 1.26.4, scipy
1.12.0). The recipe left the pip ranges and the Hackage index open, so a rebuild could install
other versions, and the exported task named an image tag that exists on one machine only. These
laws pin the recipe and make the task build its own candidate and verifier environments.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from ppl_synth_data.runtime_registry import get_runtime_spec
from tests.test_harbor_lazyppl_export import (  # noqa: F401
    _run_export,
    lazyppl_local_dir,
)
from tests.test_harbor_lazyppl_export import (
    docker_app_dir as _shared_docker_app_dir,
)
from tests.test_harbor_lazyppl_export import (
    docker_file as _shared_docker_file,
)
from tests.test_harbor_lazyppl_export import (
    seeds_dir as _shared_seeds_dir,
)

seeds_dir = _shared_seeds_dir
del _shared_seeds_dir
docker_app_dir = _shared_docker_app_dir
del _shared_docker_app_dir
docker_file = _shared_docker_file
del _shared_docker_file

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "docker" / "lazyppl-harbor"
HASKELL_PINS = {
    "hmatrix": "0.20.2",
    "log-domain": "0.13.2",
    "math-functions": "0.3.4.4",
    "mwc-random": "0.15.3.0",
    "random": "1.3.1",
    "statistics": "0.12.0.0",
    "vector": "0.13.2.0",
}
PYTHON_PINS = {"arviz": "0.17.1", "numpy": "1.26.4", "scipy": "1.12.0"}


def _task(seeds_dir, tmp_path, docker_app_dir, docker_file, lazyppl_local_dir) -> Path:  # noqa: F811
    out = tmp_path / "harbor_out"
    _run_export(seeds_dir, out, docker_app_dir, docker_file, lazyppl_local_dir)
    return out / "lazyppl-simple-normal-00001"


def _recipe_text() -> str:
    """The Dockerfile and the pin files beside it (not the application or library sources)."""
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(RECIPE.iterdir()) if path.is_file()
    )


def test_task_names_no_image_and_verifies_separately_without_network(
    seeds_dir,
    tmp_path,
    docker_app_dir,
    docker_file,
    lazyppl_local_dir,  # noqa: F811
) -> None:
    task = _task(seeds_dir, tmp_path, docker_app_dir, docker_file, lazyppl_local_dir)
    text = (task / "task.toml").read_text(encoding="utf-8")
    config = tomllib.loads(text)

    assert config["artifacts"] == ["/app/Submission.hs"]
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["network_mode"] == "no-network"
    assert config["agent"]["network_mode"] == "no-network"
    assert config["environment"]["network_mode"] == "no-network"
    assert config["verifier"]["environment"]["network_mode"] == "no-network"
    assert "docker_image" not in config["environment"]
    assert "docker_image" not in config["verifier"]["environment"]
    assert "allow_internet" not in config["environment"]
    assert "local-build" not in text


def test_verifier_recipe_is_the_runtime_recipe_plus_the_test_files(
    seeds_dir,
    tmp_path,
    docker_app_dir,
    docker_file,
    lazyppl_local_dir,  # noqa: F811
) -> None:
    task = _task(seeds_dir, tmp_path, docker_app_dir, docker_file, lazyppl_local_dir)
    tests_dir = task / "tests"
    dockerfile = (tests_dir / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith(docker_file.read_text(encoding="utf-8").strip())
    assert "--mount=" not in dockerfile
    for name in ("test.sh", "TaskData.hs", "HiddenTaskData.hs"):
        assert name in dockerfile
        assert (tests_dir / name).is_file()
    # The runtime recipe copies the application and the patched library from its build context.
    assert (tests_dir / "app" / "Main.hs").read_bytes() == (docker_app_dir / "Main.hs").read_bytes()
    assert (tests_dir / "lazyppl-local" / "lazyppl.cabal").read_bytes() == (
        lazyppl_local_dir / "lazyppl.cabal"
    ).read_bytes()


def test_held_out_rows_stay_out_of_the_candidate_environment(
    seeds_dir,
    tmp_path,
    docker_app_dir,
    docker_file,
    lazyppl_local_dir,  # noqa: F811
) -> None:
    task = _task(seeds_dir, tmp_path, docker_app_dir, docker_file, lazyppl_local_dir)

    assert not list((task / "environment").rglob("HiddenTaskData.hs"))
    assert (task / "tests" / "HiddenTaskData.hs").is_file()


def test_runtime_registry_names_no_local_lazyppl_image_tag() -> None:
    assert "local-build" not in str(get_runtime_spec("lazyppl").harbor_docker_image)


def test_haskell_libraries_are_pinned_to_the_tested_versions() -> None:
    recipe = _recipe_text()

    assert re.search(r"index-state", recipe), "the recipe names no Hackage index state"
    for name, version in HASKELL_PINS.items():
        pinned = re.search(rf"\b{re.escape(name)}(?:-|\s*==\s*){re.escape(version)}\b", recipe)
        assert pinned, f"{name} is not pinned to {version}"


def test_python_packages_come_from_a_hash_locked_file() -> None:
    dockerfile = (RECIPE / "Dockerfile").read_text(encoding="utf-8")
    lock = (RECIPE / "requirements.lock").read_text(encoding="utf-8")

    assert "--require-hashes" in dockerfile
    assert "requirements.lock" in dockerfile
    assert not re.search(r"pip install[^\n]*[<>]=?\s*\d", dockerfile), (
        "a version range is still installed"
    )
    entries = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", lock, re.MULTILINE))
    for name, version in PYTHON_PINS.items():
        assert entries.get(name) == version
    assert lock.count("--hash=sha256:") >= len(entries)

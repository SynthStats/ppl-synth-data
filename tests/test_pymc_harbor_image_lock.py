"""The scorer image installs exactly the locked, hash-pinned distribution set."""

from __future__ import annotations

from pathlib import Path

DOCKER_DIR = Path(__file__).resolve().parents[1] / "docker" / "pymc-harbor"
SCORING_PINS = {
    "pymc": "5.28.5",
    "pytensor": "2.38.3",
    "numpy": "2.4.6",
    "arviz": "0.23.4",
    "pytest": "9.1.1",
}


def _lock_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in (DOCKER_DIR / "requirements.lock").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--hash")):
            continue
        name, _, version = line.rstrip(" \\").partition("==")
        entries[name.lower().replace("_", "-")] = version
    return entries


def test_lock_pins_every_distribution_with_a_hash() -> None:
    text = (DOCKER_DIR / "requirements.lock").read_text(encoding="utf-8")
    entries = _lock_entries()

    assert len(entries) == 52
    assert all(version for version in entries.values())
    for name, version in SCORING_PINS.items():
        assert entries[name] == version
    assert text.count("--hash=sha256:") >= len(entries)


def test_scorer_image_installs_only_from_the_lock() -> None:
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "requirements.lock" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--no-deps" in dockerfile
    for name in SCORING_PINS:
        assert f"{name}==" not in dockerfile

#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = ["uv", "run", "ppldf", *sys.argv[1:]]
    try:
        os.chdir(ROOT)
        os.execvp(cmd[0], cmd)  # nosec B606
    except FileNotFoundError as exc:
        raise SystemExit(
            "uv is required to run scripts/forge.py; install uv or invoke 'ppldf' directly."
        ) from exc


if __name__ == "__main__":
    main()

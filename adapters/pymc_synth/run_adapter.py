"""CLI for the PyMC synthesis Harbor adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import load_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy pre-exported Harbor tasks")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.input_dir, args.output_dir)
    print(json.dumps({"tasks_copied": len(tasks), "task_ids": tasks}))


if __name__ == "__main__":
    main()

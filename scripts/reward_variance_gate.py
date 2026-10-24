"""Command-line entrypoint for the reward-variance gate.

Run with ``uv run python scripts/reward_variance_gate.py --out-dir <dir> <stage> ...``;
the implementation lives in ``ppl_synth_data.gates.reward_variance``.
"""

from __future__ import annotations

import argparse
import sys

from ppl_synth_data.gates.reward_variance import (
    build_parser as _build_parser,
)
from ppl_synth_data.gates.reward_variance import (
    main as _main,
)


def build_parser() -> argparse.ArgumentParser:
    return _build_parser()


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())

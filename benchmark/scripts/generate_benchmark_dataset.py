#!/usr/bin/env python3
"""Run MPC collection, safe APRBS collection, and compaction as one pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: This directory. Sibling scripts are located from here.
BENCH = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # What the collector needs on its path to import `i4b`, which is not this directory.
    parser.add_argument("--i4b-root", type=Path, default=BENCH.parent)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    i4b_root = args.i4b_root.resolve()
    subprocess.run(
        [
            sys.executable,
            str(args.collector.resolve()),
            "--i4b-root",
            str(i4b_root),
            "--dataset",
            str(dataset),
            "--workers",
            str(args.workers),
            "--solver-threads",
            str(args.solver_threads),
            "--horizon",
            str(args.horizon),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(BENCH / "scripts/generate_safe_aprbs.py"),
            "--dataset",
            str(dataset),
            "--workers",
            str(args.workers),
        ],
        cwd=BENCH,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(BENCH / "scripts/finalize_benchmark_dataset.py"),
            str(dataset),
        ],
        cwd=BENCH,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(BENCH / "scripts/finalize_benchmark_tables.py"),
            str(dataset),
            str(args.output.resolve()),
        ],
        cwd=BENCH,
        check=True,
    )


if __name__ == "__main__":
    main()

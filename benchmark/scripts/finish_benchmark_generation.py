#!/usr/bin/env python3
"""Wait for MPC collection, then generate safe episodes and finalize the dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MPC_TRAJECTORIES = 191 * 2 * 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "i4b-benchmark")
    parser.add_argument("--mpc-service", default="i4b-benchmark-generation.service")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _service_active(service: str) -> bool:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "is-active",
            "--quiet",
            service,
        ],
        check=False,
    )
    return result.returncode == 0


def _complete_mpc_trajectories(dataset: Path) -> int:
    count = 0
    for path in (dataset / ".staging").glob("*.json"):
        metadata = json.loads(path.read_text())
        if (
            metadata.get("controller_id", "").startswith("mpc-")
            and metadata.get("row_count") == 35039
        ):
            count += 1
    return count


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    while _service_active(args.mpc_service):
        complete = _complete_mpc_trajectories(dataset)
        print(f"MPC trajectories: {complete}/{EXPECTED_MPC_TRAJECTORIES}", flush=True)
        time.sleep(60)
    complete = _complete_mpc_trajectories(dataset)
    if complete != EXPECTED_MPC_TRAJECTORIES:
        raise RuntimeError(
            f"MPC generation stopped with {complete}/{EXPECTED_MPC_TRAJECTORIES} trajectories"
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_safe_aprbs.py",
            "--dataset",
            str(dataset),
            "--workers",
            str(args.workers),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/finalize_benchmark_dataset.py", str(dataset)],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

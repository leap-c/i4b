#!/usr/bin/env python3
"""Write the benchmark's primary and time-only split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from i4b_bench.generation import make_split_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("i4b-benchmark"))
    args = parser.parse_args()

    trajectories = pd.read_parquet(args.dataset / "trajectories.parquet")
    buildings = pd.read_parquet(args.dataset / "buildings.parquet")
    trajectories = trajectories.merge(
        buildings[["building_id", "building_family_id", "country_code"]],
        on="building_id",
        how="left",
        validate="many_to_one",
    )
    primary, time_only = make_split_manifests(trajectories)
    primary.to_parquet(args.dataset / "split.parquet", index=False)
    ablations = args.dataset / "ablation-splits"
    ablations.mkdir(parents=True, exist_ok=True)
    time_only.to_parquet(ablations / "time.parquet", index=False)


if __name__ == "__main__":
    main()

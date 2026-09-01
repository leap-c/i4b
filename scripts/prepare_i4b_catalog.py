#!/usr/bin/env python3
"""Build the simulation-ready I4B building catalog from acquired Parquet data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from i4b_bench.corpus import make_catalog, validate_buildings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("source-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("i4b-benchmark"))
    parser.add_argument(
        "--config", type=Path, default=Path("scripts/benchmark_source_data.json")
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    locations = {entry["country"]: entry for entry in config["locations"]}
    building_path = args.source_dir / "normalized" / "buildings" / "tabula_sfh.parquet"
    weather_dir = args.source_dir / "normalized" / "weather_reference"
    buildings = pd.read_parquet(building_path)
    weather = {}
    for path in sorted(weather_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        country = str(frame["location_id"].iloc[0]).split("_", 1)[0].upper()
        weather.setdefault(country, []).append(frame)
    ambient = {
        country: pd.concat(frames, ignore_index=True)["temperature_2m_C"]
        for country, frames in weather.items()
    }
    catalog, params_by_id = make_catalog(buildings, locations, ambient)
    if len(catalog) != 191 or catalog["building_family_id"].nunique() != 64:
        raise ValueError("expected exactly 191 buildings and 64 building families")
    validate_buildings(params_by_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(args.output_dir / "buildings.parquet", index=False)
    print(f"wrote {len(catalog)} buildings to {args.output_dir / 'buildings.parquet'}")


if __name__ == "__main__":
    main()

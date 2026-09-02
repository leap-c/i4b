#!/usr/bin/env python3
"""Fold the generated excitation levels into the published benchmark tables.

Additive by design: existing shards and tables are never rewritten, only appended to, so the
2,674 trajectories already published keep their bytes and their provenance. New transitions go
into fresh shards; `trajectories`, `split`, `controllers.json` and `manifest.json` gain rows.

Split assignment is inherited, not invented: each new controller clones the building-to-split
mapping and the time windows of `open-loop-aprbs`, so a building that was held out stays held
out at every excitation level.

The per-controller replay tables under `controllers/` are deliberately not extended. Those exist
so the evaluation can replay a *baseline controller*; these levels are open-loop excitation data
for training, not controllers anything is benchmarked against, and duplicating them there would
add ~1.5 GB of redundancy.

    uv run python scripts/finalize_excitation_levels.py --dataset production
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROWS_PER_SHARD = 4_000_000
BASE_CONTROLLER = "open-loop-aprbs"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("production"))
    parser.add_argument("--staging", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    dataset = args.dataset.resolve()
    staging = args.staging or dataset / ".staging-excitation"
    levels = sorted(p for p in staging.iterdir() if p.is_dir())
    if not levels:
        raise SystemExit(f"no levels under {staging}")

    metadata = [json.loads(p.read_text()) for level in levels for p in sorted(level.glob("*.json"))]
    frame = pd.DataFrame(metadata)
    print(f"{len(frame)} trajectories over {len(levels)} levels: "
          f"{', '.join(sorted(frame['controller_id'].unique()))}")

    trajectories = pd.read_parquet(dataset / "trajectories.parquet")
    already = set(trajectories["trajectory_id"])
    if overlap := already & set(frame["trajectory_id"]):
        raise SystemExit(f"{len(overlap)} trajectories are already published, e.g. {sorted(overlap)[0]}")

    # --- new trajectory rows, matching the published schema ---
    new_rows = pd.DataFrame({
        "trajectory_id": frame["trajectory_id"],
        "building_id": frame["building_id"],
        "controller_id": frame["controller_id"],
        "period_id": frame["period_id"],
        "start_time_utc": pd.to_datetime(frame["start_time_utc"], utc=True),
        "end_time_utc": pd.to_datetime(frame["end_time_utc"], utc=True),
        "timestep_seconds": frame["timestep_seconds"].astype(trajectories["timestep_seconds"].dtype),
        "row_count": frame["row_count"].astype(trajectories["row_count"].dtype),
        "controller_config_hash": frame["controller_config_hash"],
        "solver_fallback_count": 0,
        "solver_fallback_policy": pd.NA,
        "solver_fallback_statuses_json": "{}",
        "scenario_id": frame["building_id"] + "--" + frame["period_id"],
    })[list(trajectories.columns)]

    # --- split: clone the base controller's building assignment and windows ---
    split = pd.read_parquet(dataset / "split.parquet")
    base = split[split["trajectory_id"].str.endswith(f"--{BASE_CONTROLLER}")]
    if base.empty:
        raise SystemExit(f"no {BASE_CONTROLLER} rows in split.parquet to inherit from")
    new_split = pd.concat(
        [
            base.assign(trajectory_id=base["trajectory_id"].str.replace(
                f"--{BASE_CONTROLLER}$", f"--{name}", regex=True))
            for name in sorted(frame["controller_id"].unique())
        ],
        ignore_index=True,
    )
    published = set(new_rows["trajectory_id"])
    missing = set(new_split["trajectory_id"]) - published
    if missing:
        raise SystemExit(f"split references {len(missing)} trajectories that were not generated")

    print(f"  trajectories.parquet: {len(trajectories)} -> {len(trajectories) + len(new_rows)}")
    print(f"  split.parquet:        {len(split)} -> {len(split) + len(new_split)}")
    print(new_split.assign(c=new_split["trajectory_id"].str.split("--").str[2])
          .groupby(["c", "split"]).size().to_string())
    if args.dry_run:
        return

    # --- back up the small tables before touching anything ---
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = dataset / f".backup-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("trajectories.parquet", "split.parquet", "controllers.json", "manifest.json"):
        shutil.copy2(dataset / name, backup / name)
    print(f"\nbacked up published tables to {backup}")

    # --- new transition shards; existing shards are untouched ---
    schema = pq.ParquetFile(sorted((dataset / "transitions").glob("part-*.parquet"))[0]).schema_arrow
    existing = sorted((dataset / "transitions").glob("part-*.parquet"))
    index = len(existing)
    written: list[str] = []
    buffer: list[pd.DataFrame] = []
    buffered = 0

    def flush() -> None:
        nonlocal buffer, buffered, index
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.concat(buffer, ignore_index=True), preserve_index=False)
        table = table.cast(schema)
        name = f"part-{index:05d}.parquet"
        pq.write_table(table, dataset / "transitions" / name)
        written.append(f"transitions/{name}")
        print(f"  wrote {name}: {table.num_rows:,} rows")
        index += 1
        buffer, buffered = [], 0

    for level in levels:
        for path in sorted(level.glob("*.parquet")):
            part = pd.read_parquet(path)
            buffer.append(part)
            buffered += len(part)
            if buffered >= ROWS_PER_SHARD:
                flush()
    flush()

    # --- tables ---
    pd.concat([trajectories, new_rows], ignore_index=True).to_parquet(
        dataset / "trajectories.parquet", index=False)
    pd.concat([split, new_split], ignore_index=True).to_parquet(
        dataset / "split.parquet", index=False)

    controllers = json.loads((dataset / "controllers.json").read_text())
    template = controllers["controllers"][BASE_CONTROLLER]
    for name, group in frame.groupby("controller_id"):
        entry = {k: v for k, v in template.items() if k != "trajectory_configurations"}
        entry.update({
            "implementation": "scripts/generate_excitation_levels.py",
            "residual_amplitude_C": float(group["residual_amplitude_C"].iloc[0]),
            "derived_from": BASE_CONTROLLER,
            "waveform_seed_policy": "shared with open-loop-aprbs (amplitude ladder, one realisation)",
            "trajectory_configurations": {
                row["trajectory_id"]: {
                    "configuration_hash": row["controller_config_hash"],
                    "residual_amplitude_C": row["residual_amplitude_C"],
                    "coverage": row["coverage"],
                    "hold_steps": row["hold_steps"],
                    "room_temperature_range_C": row["room_temperature_range_C"],
                    "waveform_seed_id": row["waveform_seed_id"],
                }
                for _, row in group.iterrows()
            },
        })
        controllers["controllers"][name] = entry
    (dataset / "controllers.json").write_text(json.dumps(controllers, indent=2, sort_keys=True))

    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["shards"] = list(manifest["shards"]) + written
    manifest.setdefault("excitation_levels", {})["open_loop_ladder_C"] = sorted(
        float(a) for a in frame["residual_amplitude_C"].unique())
    # Counted from the tables rather than incremented, so the manifest cannot drift out of step
    # with them across repeated appends.
    manifest["trajectory_count"] = len(trajectories) + len(new_rows)
    manifest["transition_count"] = sum(
        pq.ParquetFile(shard).metadata.num_rows
        for shard in sorted((dataset / "transitions").glob("part-*.parquet"))
    )
    (dataset / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"\nadded {len(new_rows)} trajectories, {len(written)} shards, "
          f"{len(frame['controller_id'].unique())} controllers")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and compact staged I4B benchmark trajectories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from i4b_bench.corpus import TRANSITION_COLUMNS, make_split_manifests

EXPECTED_TRAJECTORIES = 191 * 2 * 7
MPC_CONTROLLERS = {
    "mpc-nominal",
    "mpc-offset-plus-2K",
    "mpc-offset-minus-2K",
    "mpc-aprbs-low",
    "mpc-aprbs-medium",
    "mpc-aprbs-high",
}
EXPECTED_CONTROLLERS = MPC_CONTROLLERS | {"open-loop-aprbs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("i4b-benchmark"))
    parser.add_argument("--rows-per-shard", type=int, default=4_000_000)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _validate(path: Path, metadata: dict) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(TRANSITION_COLUMNS):
        raise ValueError(f"invalid columns in {path}")
    invalid_dtypes = {
        column: str(frame[column].dtype)
        for column in TRANSITION_COLUMNS[2:]
        if frame[column].dtype != np.dtype("float32")
    }
    if invalid_dtypes:
        raise ValueError(f"invalid numeric dtypes in {path}: {invalid_dtypes}")
    if len(frame) != metadata["row_count"]:
        raise ValueError(f"row count mismatch in {path}")
    if frame["trajectory_id"].nunique() != 1:
        raise ValueError(f"multiple trajectory IDs in {path}")
    if frame["trajectory_id"].iloc[0] != metadata["trajectory_id"]:
        raise ValueError(f"trajectory ID mismatch in {path}")
    timestamps = frame["timestamp_utc"]
    if timestamps.dt.tz is None or str(timestamps.dt.tz) != "UTC":
        raise ValueError(f"timestamps are not UTC in {path}")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"timestamps are not unique and increasing in {path}")
    if (
        len(timestamps) > 1
        and not timestamps.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    ):
        raise ValueError(f"timestamps are not 15-minute aligned in {path}")
    values = frame[list(TRANSITION_COLUMNS[2:])].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite transition values in {path}")
    return frame


def _controllers(metadata: list[dict]) -> dict:
    residuals = {
        "mpc-nominal": {"kind": "constant", "amplitude_K": 0.0},
        "mpc-offset-plus-2K": {"kind": "constant", "amplitude_K": 2.0},
        "mpc-offset-minus-2K": {"kind": "constant", "amplitude_K": -2.0},
        "mpc-aprbs-low": {"kind": "aprbs", "amplitude_K": 0.5},
        "mpc-aprbs-medium": {"kind": "aprbs", "amplitude_K": 1.5},
        "mpc-aprbs-high": {"kind": "aprbs", "amplitude_K": 3.0},
    }
    mpc_metadata = [item for item in metadata if item["controller_id"] in MPC_CONTROLLERS]
    horizons = {item.get("horizon_steps") for item in mpc_metadata}
    if len(horizons) != 1 or None in horizons:
        raise ValueError(f"inconsistent MPC horizons: {horizons}")
    forecast_sources = {item.get("forecast_source") for item in mpc_metadata}
    forecast_selections = {item.get("forecast_selection") for item in mpc_metadata}
    if len(forecast_sources) != 1 or len(forecast_selections) != 1:
        raise ValueError("inconsistent MPC forecast configuration")
    forecast_policy_keys = (
        "forecast_availability_delay_hours",
        "temperature_interpolation",
        "irradiance_interpolation",
        "temperature_correction_fraction",
        "temperature_correction_decay_hours",
    )
    for key in forecast_policy_keys:
        values = {item.get(key) for item in mpc_metadata}
        if len(values) != 1 or None in values:
            raise ValueError(f"inconsistent MPC forecast policy for {key}: {values}")
    common = {
        "implementation": "leapc_lab.i4b.planner.I4bPlanner",
        "implementation_revision": {
            "repository_commit": "a11360ee1a24cfdd07a5686af596e66496ac4059",
            "collector_sha256": "1065d09e0c48094f3daab908d28df1f66c29e9966f2e5546a3cbaaba43a3a41b",
            "planner_sha256": "1ae139fb59b782678ec4239f0cd403a753f6c1fb8309268361e4966cb998bc83",
            "ocp_sha256": "ed0b2978213ded225191b005e99c8ac259faadd442230cb12e1807ebedd3af96",
        },
        "horizon_steps": horizons.pop(),
        "forecast_source": forecast_sources.pop(),
        "forecast_selection": forecast_selections.pop(),
        **{
            key: mpc_metadata[0][key]
            for key in forecast_policy_keys
        },
        "timestep_seconds": 900,
        "objective": {
            "stage": "Qth_kW / (COP * 100)",
            "terminal": 0.0,
            "comfort_slack_quadratic_weight": 1.0,
            "room_comfort_band_C": [20.0, 26.0],
        },
        "action_constraints_C": [5.0, 65.0],
        "thermal_power_constraints_kW": [0.0, 26.0],
        "residual_application": "added to first MPC action before environment clipping",
    }
    controllers = {}
    for controller_id, residual in residuals.items():
        configuration_hashes = {
            item["controller_config_hash"]
            for item in mpc_metadata
            if item["controller_id"] == controller_id
        }
        if len(configuration_hashes) != 1:
            raise ValueError(f"inconsistent configuration for {controller_id}")
        controllers[controller_id] = {
            **common,
            "residual": residual,
            "configuration_hash": configuration_hashes.pop(),
        }

    aprbs_configurations = {}
    for item in metadata:
        if item["controller_id"] != "open-loop-aprbs":
            continue
        aprbs_configurations[item["trajectory_id"]] = {
            "baseline": item["baseline"],
            "residual_amplitude_C": item["residual_amplitude_C"],
            "coverage": item["coverage"],
            "room_temperature_range_C": item["room_temperature_range_C"],
            "hold_steps": item["hold_steps"],
            "configuration_hash": item["controller_config_hash"],
        }
    controllers["open-loop-aprbs"] = {
        "implementation": "scripts/generate_safe_aprbs.py",
        "implementation_revision": {
            "repository_commit": "6f86f423efe89cbf3cad34df7f7eb040a0a659eb",
            "source_sha256": "d36e2174983b39613b478b76cde307bca60d1b0b9e2d1d9e90396144e7acf76c",
        },
        "baseline": "matched_nominal_applied_action_replay",
        "residual_amplitude_C": 1.5,
        "hold_steps": [4, 24],
        "trajectory_configurations": aprbs_configurations,
    }
    return {"schema_version": 1, "controllers": controllers}


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    staging = dataset / ".staging"
    metadata_paths = sorted(staging.glob("*.json"))
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    if not args.allow_partial and len(metadata) != EXPECTED_TRAJECTORIES:
        raise ValueError(
            f"expected {EXPECTED_TRAJECTORIES} trajectories, found {len(metadata)}"
        )
    trajectory_ids = [item["trajectory_id"] for item in metadata]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("duplicate trajectory metadata")
    composition = pd.DataFrame(metadata)
    if set(composition["controller_id"]) != EXPECTED_CONTROLLERS:
        raise ValueError("unexpected controller composition")
    grouped = composition.groupby(["building_id", "period_id"])
    invalid_groups = [
        key
        for key, group in grouped
        if set(group["controller_id"]) != EXPECTED_CONTROLLERS
        or len(group) != len(EXPECTED_CONTROLLERS)
        or not group["row_count"].eq(35039).all()
    ]
    if invalid_groups or len(grouped) != 191 * 2:
        raise ValueError(f"invalid building-period composition: {invalid_groups[:5]}")

    temporary_dir = dataset / f".transitions-{os.getpid()}"
    if temporary_dir.exists() or (dataset / "transitions").exists():
        raise FileExistsError(
            "remove the previous transitions output before recompacting"
        )
    temporary_dir.mkdir(parents=True)
    writer = None
    shard_index = 0
    shard_rows = 0
    total_rows = 0
    try:
        for item in metadata:
            frame = _validate(staging / f"{item['trajectory_id']}.parquet", item)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is not None and shard_rows + len(frame) > args.rows_per_shard:
                writer.close()
                writer = None
                shard_index += 1
                shard_rows = 0
            if writer is None:
                shard_path = temporary_dir / f"part-{shard_index:05d}.parquet"
                writer = pq.ParquetWriter(shard_path, table.schema, compression="zstd")
            writer.write_table(table)
            shard_rows += len(frame)
            total_rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    temporary_dir.replace(dataset / "transitions")

    trajectory_columns = [
        "trajectory_id",
        "building_id",
        "controller_id",
        "period_id",
        "start_time_utc",
        "end_time_utc",
        "timestep_seconds",
        "row_count",
        "controller_config_hash",
        "horizon_steps",
        "forecast_source",
        "forecast_selection",
        "forecast_availability_delay_hours",
        "temperature_interpolation",
        "irradiance_interpolation",
        "temperature_correction_fraction",
        "temperature_correction_decay_hours",
        "solver_fallback_count",
        "solver_fallback_policy",
        "solver_fallback_statuses_json",
    ]
    trajectory_metadata = pd.DataFrame(metadata)
    trajectory_metadata["solver_fallback_count"] = (
        trajectory_metadata["solver_fallback_count"].fillna(0).astype("int64")
    )
    trajectory_metadata["solver_fallback_statuses_json"] = trajectory_metadata[
        "solver_fallback_statuses"
    ].apply(
        lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) else "{}"
    )
    trajectories = trajectory_metadata[trajectory_columns]
    trajectories[["start_time_utc", "end_time_utc"]] = trajectories[
        ["start_time_utc", "end_time_utc"]
    ].apply(pd.to_datetime, utc=True)
    trajectories_path = dataset / "trajectories.parquet"
    _atomic_parquet(trajectories, trajectories_path)
    _atomic_json(_controllers(metadata), dataset / "controllers.json")

    buildings_path = dataset / "buildings.parquet"
    buildings = pd.read_parquet(buildings_path)
    split_input = trajectories.merge(
        buildings[["building_id", "building_family_id", "country_code"]],
        on="building_id",
        how="left",
        validate="many_to_one",
    )
    primary, time_only = make_split_manifests(split_input)
    split_path = dataset / "split.parquet"
    time_path = dataset / "ablation-splits" / "time.parquet"
    time_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(primary, split_path)
    _atomic_parquet(time_only, time_path)

    final_shards = sorted((dataset / "transitions").glob("*.parquet"))
    manifest = {
        "schema_version": 1,
        "trajectory_count": len(trajectories),
        "transition_count": total_rows,
        "transition_columns": list(TRANSITION_COLUMNS),
        "shards": [str(path.relative_to(dataset)) for path in final_shards],
    }
    _atomic_json(manifest, dataset / "manifest.json")
    print(
        f"compacted {len(trajectories)} trajectories and {total_rows} transitions "
        f"into {len(final_shards)} shards"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the downstream-oriented I4B Parquet tables from a compact release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


TRANSITION_COLUMNS = [
    "trajectory_id",
    "timestamp_utc",
    "T_room",
    "T_wall",
    "T_hp_ret",
    "T_hp_sup_applied",
    "T_amb",
    "Qdot_gains",
]
CONTROLLER_COLUMNS = [
    "trajectory_id",
    "scenario_id",
    "timestamp_utc",
    "T_room",
    "T_wall",
    "T_hp_ret",
    "T_hp_sup_applied",
]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-data", type=Path, default=Path("source-data"))
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("scripts/benchmark_source_data.json"),
    )
    return parser.parse_args()


def _write_table(writer: pq.ParquetWriter | None, frame: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _read_trajectory(dataset: ds.Dataset, trajectory_id: str) -> pd.DataFrame:
    frame = dataset.to_table(
        filter=ds.field("trajectory_id") == trajectory_id,
        columns=TRANSITION_COLUMNS,
    ).to_pandas()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.sort_values("timestamp_utc", ignore_index=True)


def _locations(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {item["country"]: item for item in payload["locations"]}


def _source_frame(directory: Path, pattern: str, columns: list[str]) -> pd.DataFrame:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no source files match {directory / pattern}")
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)


def _aligned_source(
    source: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    timestamp_column: str,
) -> pd.DataFrame:
    source = source.copy()
    source[timestamp_column] = pd.to_datetime(source[timestamp_column], utc=True)
    source = source.set_index(timestamp_column).sort_index()
    aligned = source.reindex(source.index.union(timestamps)).sort_index()
    if "temperature_2m_C" in aligned:
        aligned["temperature_2m_C"] = aligned["temperature_2m_C"].interpolate(
            method="time"
        )
    held_columns = [column for column in aligned if column != "temperature_2m_C"]
    aligned[held_columns] = aligned[held_columns].ffill()
    return aligned.reindex(timestamps).reset_index(names="timestamp_utc")


def _write_forecasts(source_data: Path, output: Path) -> None:
    forecast_dataset = ds.dataset(
        source_data / "normalized" / "weather_forecasts",
        format="parquet",
    )
    table = forecast_dataset.to_table()
    frame = table.to_pandas()
    frame["initialization_time_utc"] = pd.to_datetime(
        frame["initialization_time_utc"], utc=True
    )
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    frame = frame.sort_values(
        ["location_id", "model", "initialization_time_utc", "valid_time_utc"]
    )
    frame.to_parquet(output, index=False, compression="zstd")


def _write_prices(source_data: Path, output: Path) -> None:
    price_dir = source_data / "normalized" / "electricity_prices"
    paths = sorted(price_dir.glob("DE-LU_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no DE-LU price files found in {price_dir}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["delivery_start_utc"] = pd.to_datetime(frame["delivery_start_utc"], utc=True)
    frame["price_signal_id"] = "de-lu-day-ahead"
    frame["signal_type"] = "common_proxy"
    frame = frame.sort_values("delivery_start_utc", ignore_index=True)
    if frame["delivery_start_utc"].duplicated().any():
        raise ValueError("duplicate DE-LU price timestamps")
    frame.to_parquet(output, index=False, compression="zstd")


def _validate_tables(release: Path, scenarios: pd.DataFrame, trajectories: pd.DataFrame) -> None:
    expected_rows = int(trajectories["row_count"].sum() / trajectories["controller_id"].nunique())
    exogenous_file = pq.ParquetFile(release / "exogenous.parquet")
    if exogenous_file.metadata.num_rows != expected_rows:
        raise ValueError(
            f"unexpected exogenous row count: {exogenous_file.metadata.num_rows}"
        )
    required_exogenous_columns = {
        "scenario_id",
        "timestamp_utc",
        "T_amb",
        "Qdot_gains",
    }
    if not required_exogenous_columns.issubset(exogenous_file.schema.names):
        raise ValueError(f"incomplete exogenous schema: {exogenous_file.schema.names}")

    for controller_id, members in trajectories.groupby("controller_id"):
        path = release / "controllers" / f"{controller_id}.parquet"
        parquet_file = pq.ParquetFile(path)
        if parquet_file.metadata.num_rows != int(members["row_count"].sum()):
            raise ValueError(f"row count mismatch for controller {controller_id}")
        if parquet_file.schema.names != CONTROLLER_COLUMNS:
            raise ValueError(f"invalid controller schema for {controller_id}")

    forecast_columns = pq.read_schema(release / "forecasts.parquet").names
    required_forecast_columns = {
        "location_id",
        "model",
        "initialization_time_utc",
        "valid_time_utc",
        "lead_hours",
    }
    if not required_forecast_columns.issubset(forecast_columns):
        raise ValueError(f"incomplete forecast schema: {forecast_columns}")

    price_columns = pq.read_schema(release / "prices.parquet").names
    required_price_columns = {
        "price_signal_id",
        "delivery_start_utc",
        "price_eur_per_mwh",
        "market_id",
        "source",
        "signal_type",
    }
    if not required_price_columns.issubset(price_columns):
        raise ValueError(f"incomplete price schema: {price_columns}")


def main() -> None:
    args = _args()
    release = args.release.resolve()
    output = args.output.resolve()
    temporary_output = output.with_name(f".{output.name}-{os.getpid()}")
    if output.exists() or temporary_output.exists():
        raise FileExistsError(f"output already exists: {output}")
    temporary_output.mkdir(parents=True)
    source_data = args.source_data.resolve()
    locations = _locations(args.source_config.resolve())
    trajectories = pd.read_parquet(release / "trajectories.parquet")
    buildings = pd.read_parquet(release / "buildings.parquet")
    transitions = ds.dataset(release / "transitions", format="parquet")

    trajectories["scenario_id"] = (
        trajectories["building_id"].astype(str) + "--" + trajectories["period_id"]
    )
    scenario_columns = [
        "scenario_id",
        "building_id",
        "period_id",
        "start_time_utc",
        "end_time_utc",
        "timestep_seconds",
    ]
    scenarios = trajectories[scenario_columns].drop_duplicates("scenario_id")
    scenarios = scenarios.merge(
        buildings[["building_id", "country_code"]],
        on="building_id",
        how="left",
        validate="many_to_one",
    )
    scenarios["location_id"] = scenarios["country_code"].map(
        {country: item["id"] for country, item in locations.items()}
    )
    scenarios["timezone"] = scenarios["country_code"].map(
        {country: item["timezone"] for country, item in locations.items()}
    )
    scenarios["market_id"] = "DE-LU"
    scenarios.to_parquet(
        temporary_output / "scenarios.parquet", index=False, compression="zstd"
    )

    exogenous_dir = temporary_output / f".exogenous-{os.getpid()}"
    exogenous_dir.mkdir()
    exogenous_path = exogenous_dir / "exogenous.parquet"
    exogenous_writer = None
    try:
        for scenario in scenarios.to_dict("records"):
            members = trajectories[trajectories["scenario_id"] == scenario["scenario_id"]]
            reference = members.iloc[0]
            frame = _read_trajectory(transitions, reference["trajectory_id"])
            timestamps = pd.DatetimeIndex(frame["timestamp_utc"])
            location_id = scenario["location_id"]
            period_id = scenario["period_id"]
            weather = _source_frame(
                source_data / "normalized" / "weather_reference",
                f"{location_id}_{period_id}.parquet",
                [
                    "valid_time_utc",
                    "temperature_2m_C",
                    "ghi_W_m2",
                    "dni_W_m2",
                    "dhi_W_m2",
                ],
            )
            weather = _aligned_source(weather, timestamps, "valid_time_utc")
            local_time = timestamps.tz_convert(scenario["timezone"])
            exogenous = pd.DataFrame(
                {
                    "scenario_id": scenario["scenario_id"],
                    "timestamp_utc": timestamps,
                    "local_time": local_time.tz_localize(None),
                    "utc_offset_minutes": [
                        int(timestamp.utcoffset().total_seconds() / 60)
                        for timestamp in local_time
                    ],
                    "is_dst": [bool(timestamp.dst()) for timestamp in local_time],
                    "temperature_2m_C": weather["temperature_2m_C"],
                    "ghi_W_m2": weather["ghi_W_m2"],
                    "dni_W_m2": weather["dni_W_m2"],
                    "dhi_W_m2": weather["dhi_W_m2"],
                    "T_amb": frame["T_amb"].to_numpy(),
                    "Qdot_gains": frame["Qdot_gains"].to_numpy(),
                }
            )
            exogenous_writer = _write_table(
                exogenous_writer,
                exogenous,
                exogenous_path,
            )
    finally:
        if exogenous_writer is not None:
            exogenous_writer.close()
    exogenous_path.replace(temporary_output / "exogenous.parquet")
    exogenous_dir.rmdir()

    controller_dir = temporary_output / "controllers"
    controller_dir.mkdir(exist_ok=True)
    for controller_id, members in trajectories.groupby("controller_id"):
        temporary = controller_dir / f".{controller_id}.{os.getpid()}.parquet"
        writer = None
        try:
            for item in members.to_dict("records"):
                frame = _read_trajectory(transitions, item["trajectory_id"])
                frame["scenario_id"] = item["scenario_id"]
                frame = frame[CONTROLLER_COLUMNS]
                writer = _write_table(writer, frame, temporary)
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(controller_dir / f"{controller_id}.parquet")

    trajectories.to_parquet(
        temporary_output / "trajectories.parquet", index=False, compression="zstd"
    )
    for name in ("buildings.parquet", "controllers.json", "split.parquet"):
        shutil.copy2(release / name, temporary_output / name)
    if (release / "ablation-splits").exists():
        shutil.copytree(
            release / "ablation-splits", temporary_output / "ablation-splits"
        )
    _write_forecasts(source_data, temporary_output / "forecasts.parquet")
    _write_prices(source_data, temporary_output / "prices.parquet")
    _validate_tables(temporary_output, scenarios, trajectories)

    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("shards", None)
    manifest.pop("transition_columns", None)
    transition_count = manifest.pop("transition_count", int(trajectories["row_count"].sum()))
    manifest.update(
        {
            "schema_version": 2,
            "controller_transition_count": transition_count,
            "tables": {
                "scenarios": "scenarios.parquet",
                "exogenous": "exogenous.parquet",
                "trajectories": "trajectories.parquet",
                "controllers": "controllers/*.parquet",
                "forecasts": "forecasts.parquet",
                "prices": "prices.parquet",
            },
            "transition_semantics": "state_t + applied_input_t + exogenous_t -> state_(t+1)",
            "controller_columns": CONTROLLER_COLUMNS,
            "exogenous_columns": [
                "scenario_id",
                "timestamp_utc",
                "local_time",
                "utc_offset_minutes",
                "is_dst",
                "temperature_2m_C",
                "ghi_W_m2",
                "dni_W_m2",
                "dhi_W_m2",
                "T_amb",
                "Qdot_gains",
            ],
            "price_policy": (
                "prices.parquet contains an optional DE-LU common proxy signal; "
                "current controller trajectories are not price-optimized."
            ),
        }
    )
    (temporary_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary_output.replace(output)


if __name__ == "__main__":
    main()

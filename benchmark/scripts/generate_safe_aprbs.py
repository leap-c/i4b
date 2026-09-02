#!/usr/bin/env python3
"""Generate resumable full-year open-loop APRBS trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from i4b_bench.corpus import (
    TRANSITION_COLUMNS,
    load_params,
    prepare_disturbances,
    read_reference_weather,
)
from i4b_bench.generation import aprbs

from i4b.gym_interface.room_env import RoomHeatEnv

_catalog: pd.DataFrame
_dataset: Path
_root: Path
APRBS_AMPLITUDE_C = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("production"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-jobs", type=int)
    return parser.parse_args()


def _init_worker(root: str, dataset: str) -> None:
    global _catalog, _dataset, _root
    _root = Path(root)
    _dataset = Path(dataset)
    _catalog = pd.read_parquet(_dataset / "buildings.parquet")


def _weather_path(country: str, period_id: str) -> Path:
    config = json.loads((_root / "scripts/benchmark_source_data.json").read_text())
    location = next(
        item["id"] for item in config["locations"] if item["country"] == country
    )
    return (
        _root
        / "source-data/normalized/weather_reference"
        / f"{location}_{period_id}.parquet"
    )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _simulate(
    params: dict,
    disturbances: pd.DataFrame,
    trajectory_id: str,
    nominal_actions: np.ndarray,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    steps = len(disturbances) - 1
    waveform = aprbs(
        disturbances.index[:-1],
        trajectory_id,
        low=-1.0,
        high=1.0,
        min_hold_steps=4,
        max_hold_steps=24,
    ).to_numpy()
    levels = nominal_actions + APRBS_AMPLITUDE_C * waveform
    env = RoomHeatEnv(
        hp_model="Heatpump_AW",
        building=None,
        building_params=params,
        disturbances=disturbances,
        method="4R3C",
        mdot_HP=float(params["mdot_hp"]),
        internal_gain_profile="unused",
        backend="legacy",
    )
    observation = env.get_obs()
    values = np.empty((steps, 6), dtype=np.float32)
    room_min = float(observation[0])
    room_max = room_min
    for step in range(steps):
        disturbance = disturbances.iloc[step]
        next_observation, _, _, _, info = env.step(
            env.normalize_action(float(levels[step]))
        )
        values[step] = (
            observation[0],
            observation[1],
            observation[2],
            info["u"],
            disturbance["T_amb"],
            disturbance["Qdot_gains"],
        )
        observation = next_observation
        room_min = min(room_min, float(observation[0]))
        room_max = max(room_max, float(observation[0]))
    frame = pd.DataFrame(values, columns=TRANSITION_COLUMNS[2:])
    frame.insert(0, "timestamp_utc", disturbances.index[:-1])
    frame.insert(0, "trajectory_id", trajectory_id)
    return frame, (room_min, room_max)


def _job(job: tuple[str, str]) -> dict:
    building_id, period_id = job
    row = _catalog.loc[_catalog["building_id"] == building_id].iloc[0]
    params = load_params(_catalog, building_id)
    weather = read_reference_weather(_weather_path(str(row["country_code"]), period_id))
    weather = weather.rename(columns={
        "temperature_2m_C": "T_amb",
        "ghi_W_m2": "ghi",
        "dni_W_m2": "dni",
        "dhi_W_m2": "dhi",
    })
    profile = _root / "i4b_data/profiles/InternalGains/ResidentialDetached.csv"
    disturbances = prepare_disturbances(weather, params, profile)
    nominal_id = f"{building_id}--{period_id}--mpc-nominal"
    nominal_path = _dataset / ".staging" / f"{nominal_id}.parquet"
    if not nominal_path.exists():
        raise FileNotFoundError(f"missing nominal trajectory {nominal_path}")
    nominal = pd.read_parquet(nominal_path, columns=["T_hp_sup_applied"])
    if len(nominal) != len(disturbances) - 1:
        raise ValueError(f"nominal trajectory is incomplete: {nominal_id}")
    staging = _dataset / ".staging"
    trajectory_id = f"{building_id}--{period_id}--open-loop-aprbs"
    path = staging / f"{trajectory_id}.parquet"
    metadata_path = staging / f"{trajectory_id}.json"
    expected_rows = len(disturbances) - 1
    config = {
        "baseline": "matched_nominal_applied_action_replay",
        "residual_amplitude_C": APRBS_AMPLITUDE_C,
        "coverage": "full_period",
        "hold_steps": [4, 24],
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (
        path.exists()
        and metadata_path.exists()
        and len(pd.read_parquet(path)) == expected_rows
    ):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("controller_config_hash") != config_hash:
            metadata.update(config)
            metadata["controller_config_hash"] = config_hash
            _atomic_json(metadata, metadata_path)
        return {"building_id": building_id, "period_id": period_id, "status": "cached"}
    frame, room_temperature_range = _simulate(
        params,
        disturbances,
        trajectory_id,
        nominal["T_hp_sup_applied"].to_numpy(),
    )
    _atomic_parquet(frame, path)
    _atomic_json(
        {
            "trajectory_id": trajectory_id,
            "building_id": building_id,
            "controller_id": "open-loop-aprbs",
            "period_id": period_id,
            "start_time_utc": disturbances.index[0].isoformat(),
            "end_time_utc": disturbances.index[-1].isoformat(),
            "timestep_seconds": 900,
            "row_count": expected_rows,
            "controller_config_hash": config_hash,
            "room_temperature_range_C": list(room_temperature_range),
            **config,
        },
        metadata_path,
    )
    return {"building_id": building_id, "period_id": period_id, "status": "written"}


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = args.dataset.resolve()
    catalog = pd.read_parquet(dataset / "buildings.parquet")
    jobs = [
        (building_id, period_id)
        for building_id in catalog["building_id"].astype(str)
        for period_id in ("period_a", "period_b")
    ]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(str(root), str(dataset)),
    ) as executor:
        futures = [executor.submit(_job, job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()

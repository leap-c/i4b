#!/usr/bin/env python3
"""Generate open-loop APRBS trajectories at additional excitation amplitudes.

The corpus tops out at 3 K of excitation (`mpc-aprbs-high`), and open loop exists at only one
amplitude (`open-loop-aprbs`, 1.5 K). Identifiability of the control's effect rises steeply with
amplitude, so the levels above 3 K are the ones missing for any study of how much excitation a
dynamics model needs.

This follows `generate_safe_aprbs.py` exactly -- same waveform, same `legacy` integrator, same
metadata -- with three differences:

  * the amplitude is a parameter, and each level becomes its own controller id
  * the nominal baseline is read from the published transitions rather than `.staging`, which
    is emptied once a corpus is finalised
  * the disturbances are read from the corpus' own `exogenous.parquet` rather than re-derived
    from raw weather, so an appended level cannot disagree with the corpus it joins

The waveform is seeded on the *base* trajectory id by default, so the levels form a clean
amplitude ladder over one realisation rather than confounding amplitude with a reseed.

    uv run python scripts/generate_excitation_levels.py --amplitude 6 12 --workers 30
"""

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
import pyarrow.dataset as ds
from i4b_bench.corpus import TRANSITION_COLUMNS, load_params
from i4b_bench.generation import aprbs

from i4b.gym_interface.room_env import RoomHeatEnv

BASE_CONTROLLER = "open-loop-aprbs"
NOMINAL_CONTROLLER = "mpc-nominal"
_catalog: pd.DataFrame
_dataset: Path
_root: Path


def controller_id(amplitude: float) -> str:
    return f"{BASE_CONTROLLER}-{amplitude:g}K".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("production"))
    parser.add_argument("--amplitude", type=float, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="seed the waveform per level instead of sharing the base realisation",
    )
    return parser.parse_args()


def _init_worker(root: str, dataset: str) -> None:
    global _catalog, _dataset, _root
    # scalar ODE work: threads buy nothing and oversubscribe badly at this worker count
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    _root, _dataset = Path(root), Path(dataset)
    _catalog = pd.read_parquet(_dataset / "buildings.parquet")


def _published_disturbances(scenario_id: str) -> pd.DataFrame:
    """The disturbances the corpus recorded for this scenario, not a fresh derivation of them.

    Re-deriving from raw weather is what made the first attempt at these levels unusable: the
    corpus was built before `prepare_disturbances` gained interpolated ambient and local-time
    internal gains, so a re-derivation disagreed with `exogenous.parquet` by up to 6.0 K and
    168.6 W, and moved the scored channel by as much as the benchmark's entire MAE range. Reading
    the record instead makes a trajectory consistent with the corpus it is appended to by
    construction, whatever convention that corpus was built under.
    """
    frame = pd.read_parquet(
        _dataset / "exogenous.parquet",
        columns=["timestamp_utc", "T_amb", "Qdot_gains"],
        filters=[("scenario_id", "==", scenario_id)],
    ).sort_values("timestamp_utc")
    if frame.empty:
        raise FileNotFoundError(f"no exogenous rows for {scenario_id}")
    index = pd.DatetimeIndex(frame["timestamp_utc"], name="timestamp_utc")
    published = pd.DataFrame(
        {"T_amb": frame["T_amb"].to_numpy(), "Qdot_gains": frame["Qdot_gains"].to_numpy()},
        index=index,
    )
    # `exogenous` holds one row per *step*; the plant reads one row past the last step, which is
    # never recorded. Repeating the final row leaves every recorded step untouched.
    tail = published.iloc[[-1]].set_axis([index[-1] + (index[1] - index[0])])
    return pd.concat([published, tail]).astype("float32")


def _nominal_actions(building_id: str, period_id: str) -> np.ndarray:
    """The applied actions of the nominal run, from the published transitions."""
    trajectory_id = f"{building_id}--{period_id}--{NOMINAL_CONTROLLER}"
    table = ds.dataset(_dataset / "transitions", format="parquet").to_table(
        filter=ds.field("trajectory_id") == trajectory_id, columns=["T_hp_sup_applied"]
    )
    if table.num_rows == 0:
        raise FileNotFoundError(f"no nominal trajectory {trajectory_id}")
    return table.to_pandas()["T_hp_sup_applied"].to_numpy(float)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _simulate(params, disturbances, seed_id, nominal_actions, amplitude):
    steps = len(disturbances) - 1
    waveform = aprbs(
        disturbances.index[:-1], seed_id, low=-1.0, high=1.0,
        min_hold_steps=4, max_hold_steps=24,
    ).to_numpy()
    levels = nominal_actions + amplitude * waveform
    env = RoomHeatEnv(
        hp_model="Heatpump_AW", building=None, building_params=params,
        disturbances=disturbances, method="4R3C", mdot_HP=float(params["mdot_hp"]),
        internal_gain_profile="unused", backend="legacy",
    )
    observation = env.get_obs()
    values = np.empty((steps, 6), dtype=np.float32)
    room_min = room_max = float(observation[0])
    for step in range(steps):
        disturbance = disturbances.iloc[step]
        next_observation, _, _, _, info = env.step(env.normalize_action(float(levels[step])))
        values[step] = (
            observation[0], observation[1], observation[2],
            info["u"], disturbance["T_amb"], disturbance["Qdot_gains"],
        )
        observation = next_observation
        room_min = min(room_min, float(observation[0]))
        room_max = max(room_max, float(observation[0]))
    frame = pd.DataFrame(values, columns=TRANSITION_COLUMNS[2:])
    frame.insert(0, "timestamp_utc", disturbances.index[:-1])
    frame.insert(0, "trajectory_id", f"{seed_id}")
    return frame, (room_min, room_max)


def _job(job: tuple[str, str, float, bool]) -> dict:
    building_id, period_id, amplitude, reseed = job
    params = load_params(_catalog, building_id)
    disturbances = _published_disturbances(f"{building_id}--{period_id}")
    nominal = _nominal_actions(building_id, period_id)
    expected_rows = len(disturbances) - 1
    if len(nominal) != expected_rows:
        raise ValueError(f"nominal trajectory is incomplete: {building_id}--{period_id}")

    name = controller_id(amplitude)
    trajectory_id = f"{building_id}--{period_id}--{name}"
    base_id = f"{building_id}--{period_id}--{BASE_CONTROLLER}"
    seed_id = trajectory_id if reseed else base_id

    staging = _dataset / ".staging-excitation" / name
    staging.mkdir(parents=True, exist_ok=True)
    path, metadata_path = staging / f"{trajectory_id}.parquet", staging / f"{trajectory_id}.json"
    config = {
        "baseline": "matched_nominal_applied_action_replay",
        "residual_amplitude_C": amplitude,
        "coverage": "full_period",
        "hold_steps": [4, 24],
        "waveform_seed_id": seed_id,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if path.exists() and metadata_path.exists() and len(pd.read_parquet(path)) == expected_rows:
        return {"trajectory_id": trajectory_id, "status": "cached"}

    frame, room_range = _simulate(params, disturbances, seed_id, nominal, amplitude)
    frame["trajectory_id"] = trajectory_id  # seed id may differ from the trajectory's own id
    _atomic_parquet(frame, path)
    _atomic_json(
        {
            "trajectory_id": trajectory_id, "building_id": building_id, "controller_id": name,
            "period_id": period_id, "start_time_utc": disturbances.index[0].isoformat(),
            "end_time_utc": disturbances.index[-1].isoformat(), "timestep_seconds": 900,
            "row_count": expected_rows, "controller_config_hash": config_hash,
            "room_temperature_range_C": list(room_range), **config,
        },
        metadata_path,
    )
    return {"trajectory_id": trajectory_id, "status": "written",
            "room_range_C": [round(v, 1) for v in room_range]}


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = args.dataset.resolve()
    catalog = pd.read_parquet(dataset / "buildings.parquet")
    jobs = [
        (building_id, period_id, amplitude, args.reseed)
        for amplitude in args.amplitude
        for building_id in catalog["building_id"].astype(str)
        for period_id in ("period_a", "period_b")
    ]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    print(f"{len(jobs)} trajectories over amplitudes {args.amplitude} on {args.workers} workers")

    written = failed = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context,
        initializer=_init_worker, initargs=(str(root), str(dataset)),
    ) as executor:
        futures = [executor.submit(_job, job) for job in jobs]
        for future in as_completed(futures):
            try:
                future.result()  # re-raises whatever the worker hit
            except Exception as error:  # keep going; the run is resumable
                failed += 1
                print(f"FAILED: {error!r}", flush=True)
                continue
            written += 1
            if written % 50 == 0:
                print(f"  {written}/{len(jobs)} done", flush=True)
    print(f"\n{written} trajectories, {failed} failed")


if __name__ == "__main__":
    main()

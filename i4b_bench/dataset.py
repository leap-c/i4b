"""I4B benchmark dataset loading and typed container."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .observation import CONTROL_CHANNELS, STATE_CHANNELS

STEP = pd.Timedelta(minutes=15)
PER_DAY = 96

#: How hard each controller drives the supply temperature, from MPC left alone to a pure
#: open-loop campaign. The MPC offsets shift the comfort bound, not the excitation. Ordered, so
#: any frame carrying the label sorts and groups from least to most excited without a caller
#: having to remember an ordering.
EXCITATION = {
    "mpc-nominal": "none",
    "mpc-offset-plus-2K": "none",
    "mpc-offset-minus-2K": "none",
    "mpc-aprbs-low": "low",
    "mpc-aprbs-medium": "medium",
    "mpc-aprbs-high": "high",
    "open-loop-aprbs": "open loop 1.5K",
    "open-loop-aprbs-3K": "open loop 3K",
    "open-loop-aprbs-6K": "open loop 6K",
    "open-loop-aprbs-12K": "open loop 12K",
    "open-loop-aprbs-24K": "open loop 24K",
}
EXCITATION_DTYPE = pd.CategoricalDtype(
    [
        "none",
        "low",
        "medium",
        "high",
        "open loop 1.5K",
        "open loop 3K",
        "open loop 6K",
        "open loop 12K",
        "open loop 24K",
    ],
    ordered=True,
)


_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "production"

_RAW_COLUMN_RENAMES = {
    "ghi_W_m2": "ghi",
    "dni_W_m2": "dni",
    "dhi_W_m2": "dhi",
    "temperature_2m_C": "T_amb",
}
# All disturbance columns available after renaming (exogenous has both
# pre-computed gains and raw irradiance; forecasts only have raw weather).
_ALL_DISTURBANCE_COLS = ("T_amb", "Qdot_gains", "ghi", "dni", "dhi")


class BenchmarkDataset(NamedTuple):
    """Typed container for the I4B benchmark dataset tables."""

    buildings: pd.DataFrame
    scenarios: pd.DataFrame
    trajectories: pd.DataFrame
    exogenous: pd.DataFrame
    split: pd.DataFrame
    forecasts: pd.DataFrame
    controllers_dir: Path


def load_dataset(dataset_dir: str | Path | None = None) -> BenchmarkDataset:
    """Load all benchmark tables from a directory.

    Parameters
    ----------
    dataset_dir : str, Path, or None
        Path to the dataset root. Defaults to ``<repo>/production``.
    """
    d = Path(dataset_dir or _DEFAULT_DIR)
    exogenous = pd.read_parquet(d / "exogenous.parquet").rename(columns=_RAW_COLUMN_RENAMES)
    # Exogenous already has T_amb; drop the duplicate created by the rename.
    exogenous = exogenous.loc[:, ~exogenous.columns.duplicated()]
    forecasts = pd.read_parquet(d / "forecasts.parquet").rename(columns=_RAW_COLUMN_RENAMES)
    return BenchmarkDataset(
        buildings=pd.read_parquet(d / "buildings.parquet"),
        scenarios=pd.read_parquet(d / "scenarios.parquet"),
        trajectories=pd.read_parquet(d / "trajectories.parquet"),
        exogenous=exogenous,
        split=pd.read_parquet(d / "split.parquet"),
        forecasts=forecasts,
        controllers_dir=d / "controllers",
    )


def evaluation_scenarios(
    dataset: BenchmarkDataset, split: str = "test", limit: int | None = None
) -> list[str]:
    """Scenario ids belonging to a split, sorted.

    The corpus already decides which buildings are held out -- ``split.parquet`` assigns whole
    building families, and the test split is period B of families that appear nowhere in
    training. Deriving the evaluation set from it keeps closed-loop results comparable with
    open-loop ones and removes the need for anyone to agree on a hand-picked list.

    ``limit`` takes an evenly spaced subset rather than a prefix. Scenario ids sort by
    country, so the first ten of the test split are three countries out of seven; spacing
    them keeps a short run representative of the whole set.
    """
    # TODO: verify that scenarios from right split are loaded? And format trajectory_id.
    trajectories = dataset.trajectories[["trajectory_id", "scenario_id"]]
    chosen = dataset.split[dataset.split["split"] == split]
    merged = chosen.merge(trajectories, on="trajectory_id", how="left")
    scenarios = sorted(merged["scenario_id"].dropna().unique())
    if limit is None or limit >= len(scenarios):
        return scenarios
    step = len(scenarios) / limit
    return [scenarios[int(i * step)] for i in range(limit)]


def load_controller_data(
    dataset: BenchmarkDataset, controller_id: str, scenario_id: str
) -> pd.DataFrame:
    """Load a single controller's trajectory for a given scenario.

    Joins with exogenous data to include T_amb and Qdot_gains alongside
    the controller's state and action columns.
    """
    path = dataset.controllers_dir / f"{controller_id}.parquet"
    frame = pd.read_parquet(path, filters=[("scenario_id", "==", scenario_id)])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)

    exo = dataset.exogenous[dataset.exogenous["scenario_id"] == scenario_id][
        ["timestamp_utc"] + list(_ALL_DISTURBANCE_COLS)
    ].copy()
    exo["timestamp_utc"] = pd.to_datetime(exo["timestamp_utc"], utc=True)

    frame = frame.merge(exo, on="timestamp_utc", how="left")
    return frame.sort_values("timestamp_utc", ignore_index=True)


def utc(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def aligned(frame: pd.DataFrame, view_disturbances: tuple[str, ...]) -> pd.DataFrame:
    """Move each row's applied action and disturbances onto the state they produced.

    `transitions.parquet` stores ``state_t + applied_input_t -> state_(t+1)``; the observation
    contract pairs a state with what produced it. This is the same one-step shift
    `ScenarioEnv` documents, applied to a corpus read instead of a rolling buffer.
    """
    inputs = [*CONTROL_CHANNELS, *view_disturbances]
    return frame[list(STATE_CHANNELS)].join(frame[inputs].shift(1)).iloc[1:]


@cache
def shard_index(root: Path, cache_path: Path | None = None) -> pd.Series:
    """trajectory_id -> shard file. Scanning one column of every shard is worth caching."""
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_parquet(cache_path).set_index("trajectory_id")["shard"]
    parts = [
        pd.DataFrame(
            {
                "trajectory_id": pq.read_table(shard, columns=["trajectory_id"])[
                    "trajectory_id"
                ].unique(),
                "shard": shard.name,
            }
        )
        for shard in sorted((Path(root) / "transitions").glob("part-*.parquet"))
    ]
    if not parts:
        raise FileNotFoundError(f"no transition shards under {root}")
    index = pd.concat(parts, ignore_index=True).drop_duplicates("trajectory_id")
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        index.to_parquet(cache_path, index=False)
    return index.set_index("trajectory_id")["shard"]


def read_window(
    root: Path,
    trajectory_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """One trajectory between two timestamps, tz dropped -- several model libraries reject it."""
    shard = Path(root) / "transitions" / shard_index(Path(root), cache_path)[trajectory_id]
    table = ds.dataset(shard).to_table(
        filter=(pc.field("trajectory_id") == trajectory_id)
        & (pc.field("timestamp_utc") >= utc(start).to_pydatetime())
        & (pc.field("timestamp_utc") <= utc(end).to_pydatetime())
    )
    frame = table.to_pandas().set_index("timestamp_utc").sort_index()
    frame.index = frame.index.tz_convert(None)
    return frame

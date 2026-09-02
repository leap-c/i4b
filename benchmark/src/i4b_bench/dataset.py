"""I4B benchmark dataset loading and typed container."""

from __future__ import annotations

from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .observation import (
    CONTROL_CHANNELS,
    DISTURBANCE_CHANNELS,
    PLANT_STATE_CHANNELS,
    STATE_CHANNELS,
    ObsView,
)

STEP = pd.Timedelta(minutes=15)
PER_DAY = 96

#: How hard each controller drives the supply temperature. The MPC offsets shift the comfort
#: bound, not the excitation. An ordered dtype, so labelled frames sort least- to most-excited.
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


_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

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
    #: The corpus directory. `transitions/` under it is the one store of trajectory states.
    root: Path


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
        root=d,
    )


def evaluation_scenarios(
    dataset: BenchmarkDataset, split: str = "test", limit: int | None = None
) -> list[str]:
    """Scenario ids belonging to a split, sorted.

    ``split.parquet`` assigns whole building families, so a family held out of training is held
    out in every refurbishment state and weather year.

    Parameters
    ----------
    dataset : BenchmarkDataset
        The corpus to read the split from.
    split : {"train", "validation", "test"}
        Which split to return.
    limit : int, optional
        Return an evenly spaced subset of this size rather than a prefix. Scenario ids sort by
        country, so a prefix would be a few countries and a spaced subset is representative.

    Returns
    -------
    list of str
        Sorted scenario ids.
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
    """Load one controller's recorded trajectory for one scenario.

    Parameters
    ----------
    dataset : BenchmarkDataset
        The corpus to read from.
    controller_id : str
        Recorded controller, e.g. ``"mpc-nominal"``.
    scenario_id : str
        The building, e.g. ``"BG.N.SFH.02.Gen.ReEx.001.001--period_b"``.

    Reads `transitions/`, the one store of trajectory states, so every recorded controller is
    reachable the same way.

    Returns
    -------
    pandas.DataFrame
        The controller's states and actions, timestamp-sorted, joined with the scenario's
        exogenous channels.
    """
    row = dataset.scenarios[dataset.scenarios["scenario_id"] == scenario_id]
    if row.empty:
        raise KeyError(f"unknown scenario {scenario_id!r}")
    frame = read_window(
        dataset.root,
        f"{scenario_id}--{controller_id}",
        row.iloc[0]["start_time_utc"],
        row.iloc[0]["end_time_utc"],
    ).reset_index()
    if frame.empty:
        raise KeyError(f"no trajectory {scenario_id}--{controller_id}")
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.tz_localize("UTC")
    frame["scenario_id"] = scenario_id
    # the shards carry the disturbances too; the exogenous join below is the authority on them
    frame = frame.drop(columns=[c for c in _ALL_DISTURBANCE_COLS if c in frame])

    exo = dataset.exogenous[dataset.exogenous["scenario_id"] == scenario_id][
        ["timestamp_utc"] + list(_ALL_DISTURBANCE_COLS)
    ].copy()
    exo["timestamp_utc"] = pd.to_datetime(exo["timestamp_utc"], utc=True)

    frame = frame.merge(exo, on="timestamp_utc", how="left")
    # One column order for both paths, derived from the channel constants rather than from
    # whichever file the rows came out of.
    order = [
        "trajectory_id",
        "scenario_id",
        "timestamp_utc",
        *PLANT_STATE_CHANNELS,
        *CONTROL_CHANNELS,
        *_ALL_DISTURBANCE_COLS,
    ]
    return frame[order].sort_values("timestamp_utc", ignore_index=True)


def utc(value) -> pd.Timestamp:
    """`value` as a UTC timestamp; naive values are read as UTC rather than local time."""
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def aligned(frame: pd.DataFrame, view: ObsView) -> pd.DataFrame:
    """Move each row's applied action and disturbances onto the state they produced.

    `transitions.parquet` stores ``state_t + applied_input_t -> state_(t+1)``; the observation
    contract pairs a state with what produced it. This is the same one-step shift `ScenarioEnv`
    documents, applied to a corpus read instead of a rolling buffer -- and it takes a view for
    the same reason, so a corpus-built history carries exactly the channels the env would build.

    Parameters
    ----------
    frame : pandas.DataFrame
        Timestamp-indexed transitions. Must already carry the view's channels: a corpus
        transition frame satisfies `perfect` as read, while `realistic` needs the irradiance
        joined on from `exogenous.parquet` first.
    view : {"perfect", "realistic"}
        Which channels the result carries.

    Returns
    -------
    pandas.DataFrame
        One row shorter than `frame`; the first has no preceding input to pair with.
    """
    inputs = [*CONTROL_CHANNELS, *DISTURBANCE_CHANNELS[view]]
    return frame[list(STATE_CHANNELS[view])].join(frame[inputs].shift(1)).iloc[1:]


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
    """One trajectory between two timestamps, inclusive.

    Parameters
    ----------
    root : Path
        Corpus directory holding `transitions/`.
    trajectory_id : str
        ``"<building>--<period>--<controller>"``.
    start, end : pandas.Timestamp
        Bounds, inclusive. Naive values are read as UTC.
    cache_path : Path, optional
        Where to memoise the shard index; see `shard_index`.

    Returns
    -------
    pandas.DataFrame
        Timestamp-indexed, with the timezone dropped -- several model libraries reject it.
    """
    shard = Path(root) / "transitions" / shard_index(Path(root), cache_path)[trajectory_id]
    table = ds.dataset(shard).to_table(
        filter=(pc.field("trajectory_id") == trajectory_id)
        & (pc.field("timestamp_utc") >= utc(start).to_pydatetime())
        & (pc.field("timestamp_utc") <= utc(end).to_pydatetime())
    )
    frame = table.to_pandas().set_index("timestamp_utc").sort_index()
    frame.index = frame.index.tz_convert(None)
    return frame


def step_of(dataset, scenario_id: str, when: date | datetime | str) -> int:
    """The step index of a timestamp within a scenario, so configuration can speak in time.

    A step index depends on where a period happens to start; a timestamp does not, and a reader
    can tell at a glance whether a run lands in the heating season.

    Parameters
    ----------
    dataset : BenchmarkDataset
        The corpus whose clock `when` is resolved against.
    scenario_id : str
        The building whose period defines step zero.
    when : datetime.date, datetime.datetime, or str
        A date is midnight UTC; a datetime may name any time on the corpus' 15-minute grid.
        Naive values are read as UTC.

    Returns
    -------
    int
        Steps from the scenario's start.

    Raises
    ------
    ValueError
        If `when` precedes the scenario, or does not land on a step boundary.
    """
    row = dataset.scenarios[dataset.scenarios["scenario_id"] == scenario_id]
    if row.empty:
        raise KeyError(f"unknown scenario {scenario_id!r}")
    start = pd.Timestamp(row.iloc[0]["start_time_utc"])
    when = utc(when)
    offset = when - start
    if offset % STEP != pd.Timedelta(0):
        raise ValueError(f"{when} is not on the {STEP} grid that {scenario_id} runs on")
    step = int(offset / STEP)
    if step < 0:
        raise ValueError(f"{when} is before {scenario_id} begins ({start})")
    return step


def scenario_metadata(dataset, scenario_id: str) -> dict:
    """What a scenario *is*, so a result row can be read without joining back to the corpus.

    A scenario is one building, in one refurbishment state, under one year of weather. Carrying
    that alongside the metrics is what lets results be grouped by country, construction era or
    insulation level without anyone reloading the building table.
    """
    row = dataset.scenarios[dataset.scenarios["scenario_id"] == scenario_id]
    if row.empty:
        raise KeyError(f"unknown scenario {scenario_id!r}")
    row = row.iloc[0]
    building = dataset.buildings[dataset.buildings["building_id"] == row["building_id"]]
    if building.empty:
        raise KeyError(f"scenario {scenario_id!r} names an unknown building")
    building = building.iloc[0]
    return {
        "building_id": row["building_id"],
        "country": row["country_code"],
        "period_id": row["period_id"],
        # the refurbishment variant, and the transmission it implies
        "variant": int(building["variant_number"]),
        "transmission_W_m2K": float(building["transmission_W_m2K"]),
        "year_start": int(building["year_start"]),
        "year_end": int(building["year_end"]),
        "floor_area_m2": float(building["area_floor"]),
    }

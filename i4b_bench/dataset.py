"""I4B benchmark dataset loading and typed container."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd


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
    exogenous = pd.read_parquet(d / "exogenous.parquet").rename(
        columns=_RAW_COLUMN_RENAMES
    )
    # Exogenous already has T_amb; drop the duplicate created by the rename.
    exogenous = exogenous.loc[:, ~exogenous.columns.duplicated()]
    forecasts = pd.read_parquet(d / "forecasts.parquet").rename(
        columns=_RAW_COLUMN_RENAMES
    )
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
    frame = pd.read_parquet(
        path, filters=[("scenario_id", "==", scenario_id)]
    )
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)

    exo = dataset.exogenous[dataset.exogenous["scenario_id"] == scenario_id][
        ["timestamp_utc"] + list(_ALL_DISTURBANCE_COLS)
    ].copy()
    exo["timestamp_utc"] = pd.to_datetime(exo["timestamp_utc"], utc=True)

    frame = frame.merge(exo, on="timestamp_utc", how="left")
    return frame.sort_values("timestamp_utc", ignore_index=True)

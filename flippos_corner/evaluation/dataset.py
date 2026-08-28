"""I4B benchmark dataset loading and typed container."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd


_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "production"


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
    return BenchmarkDataset(
        buildings=pd.read_parquet(d / "buildings.parquet"),
        scenarios=pd.read_parquet(d / "scenarios.parquet"),
        trajectories=pd.read_parquet(d / "trajectories.parquet"),
        exogenous=pd.read_parquet(d / "exogenous.parquet"),
        split=pd.read_parquet(d / "split.parquet"),
        forecasts=pd.read_parquet(d / "forecasts.parquet"),
        controllers_dir=d / "controllers",
    )


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
        ["timestamp_utc", "T_amb", "Qdot_gains"]
    ].copy()
    exo["timestamp_utc"] = pd.to_datetime(exo["timestamp_utc"], utc=True)

    frame = frame.merge(exo, on="timestamp_utc", how="left")
    return frame.sort_values("timestamp_utc", ignore_index=True)

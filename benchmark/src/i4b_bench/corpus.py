"""Small helpers for building and evaluating the I4B benchmark corpus."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

TRANSITION_COLUMNS = (
    "trajectory_id",
    "timestamp_utc",
    "T_room",
    "T_wall",
    "T_hp_ret",
    "T_hp_sup_applied",
    "T_amb",
    "Qdot_gains",
)


def load_params(catalog: pd.DataFrame, building_id: str) -> dict:
    """Load one nested parameter dictionary from a catalog row."""
    row = catalog.loc[catalog["building_id"] == building_id]
    if len(row) != 1:
        raise KeyError(f"expected one building row for {building_id!r}")
    return json.loads(row.iloc[0]["params_json"])


#: Source weather carries its provider's column names; `prepare_disturbances` works in I4B's.

#: One mapping at the read boundary, so the stricter contract downstream has exactly one place

#: that satisfies it.

REFERENCE_WEATHER_COLUMNS = {
    "temperature_2m_C": "T_amb",
    "ghi_W_m2": "ghi",
    "dni_W_m2": "dni",
    "dhi_W_m2": "dhi",
}


def read_reference_weather(path: str | Path) -> pd.DataFrame:
    """Read a normalized weather reference file into the columns `prepare_disturbances` wants."""
    return pd.read_parquet(path).rename(columns=REFERENCE_WEATHER_COLUMNS)


def prepare_disturbances(
    weather: pd.DataFrame,
    building_params: Mapping,
    internal_gain_profile: str | Path,
    *,
    delta_t: int = 900,
    keep_irradiance: bool = False,
) -> pd.DataFrame:
    """Create 15-minute I4B disturbances from normalized hourly reference weather.

    Returns exactly the two channels the plant takes. `keep_irradiance` additionally returns the
    resampled `ghi`/`dni`/`dhi` the gains were computed from, which the `realistic` view's
    forecasts need -- `RoomHeatEnv` rejects a frame with any other column, so it is off by
    default."""
    required = {"valid_time_utc", "T_amb", "ghi", "dni", "dhi"}
    missing = required - set(weather.columns)
    if missing:
        raise ValueError(f"weather is missing columns: {sorted(missing)}")
    index = pd.DatetimeIndex(weather["valid_time_utc"])
    if index.tz is None:
        raise ValueError("weather valid_time_utc must be timezone-aware")
    if str(index.tz) != "UTC":
        index = index.tz_convert("UTC")
    source = pd.DataFrame(
        {
            "T_amb": weather["T_amb"].to_numpy(),
            "ghi": weather["ghi"].to_numpy(),
            "dni": weather["dni"].to_numpy(),
            "dhi": weather["dhi"].to_numpy(),
        },
        index=index,
    ).sort_index()
    if source.index.duplicated().any() or not source.index.is_monotonic_increasing:
        raise ValueError("weather timestamps must be unique and increasing")
    target_index = pd.date_range(
        source.index[0],
        source.index[-1] + pd.Timedelta(minutes=45),
        freq=f"{delta_t}s",
    )
    temperature = source["T_amb"].reindex(target_index).interpolate(method="time")
    irradiance = source[["ghi", "dni", "dhi"]].reindex(target_index, method="ffill")
    source = pd.concat([temperature, irradiance], axis=1)

    from i4b.disturbances import get_int_gains, get_solar_gains

    gains_weather = source[["T_amb", "ghi", "dni", "dhi"]]
    timezone = building_params.get("position", {}).get("timezone")
    if not timezone:
        raise ValueError("building position must define a timezone")
    local_time = source.index.tz_convert(timezone)
    internal = get_int_gains(local_time, str(internal_gain_profile), building_params["area_floor"])
    internal.index = source.index
    solar = get_solar_gains(gains_weather, building_params)
    result = pd.DataFrame(
        {
            "T_amb": source["T_amb"].astype("float32"),
            "Qdot_gains": (solar + internal["Qdot_tot"]).astype("float32"),
        },
        index=source.index,
    )
    if keep_irradiance:
        for channel in ("ghi", "dni", "dhi"):
            result[channel] = source[channel].astype("float32")
    result.index.name = "timestamp_utc"
    return result


def prepare_forecast_runs(
    forecasts: pd.DataFrame,
    building_params: Mapping,
    internal_gain_profile: str | Path,
    *,
    delta_t: int = 900,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Convert archived weather runs into building-specific MPC disturbances."""
    required = {
        "initialization_time_utc",
        "valid_time_utc",
        "T_amb",
        "ghi",
        "dni",
        "dhi",
    }
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecasts are missing columns: {sorted(missing)}")

    runs = {}
    for initialization, run in forecasts.groupby("initialization_time_utc", sort=True):
        initialization = pd.Timestamp(initialization)
        if initialization.tzinfo is None:
            raise ValueError("forecast initialization times must be timezone-aware")
        runs[initialization.tz_convert("UTC")] = prepare_disturbances(
            run,
            building_params,
            internal_gain_profile,
            delta_t=delta_t,
            keep_irradiance=True,
        )
    return runs


def select_forecast_disturbances(
    runs: Mapping[pd.Timestamp, pd.DataFrame],
    decision_time: pd.Timestamp,
    horizon_steps: int,
    current_disturbance: np.ndarray,
    *,
    channels: Sequence[str] = ("T_amb", "Qdot_gains"),
    delta_t: int = 900,
    availability_delay_hours: float = 6.0,
    correction_fraction: float = 0.5,
    correction_decay_hours: float = 6.0,
) -> np.ndarray:
    """Select the latest as-of run that covers a complete MPC horizon.

    `channels` are the disturbance channels the caller's view declares, and the returned array
    has one column per channel. Assuming the `perfect` view's two here is what made
    `realistic` + `use_forecast` unrunnable.
    """
    channels = list(channels)
    if len(current_disturbance) != len(channels):
        raise ValueError(
            f"current_disturbance has {len(current_disturbance)} entries "
            f"but {len(channels)} channels were requested"
        )
    if horizon_steps < 0:
        raise ValueError("horizon_steps must be non-negative")
    if availability_delay_hours < 0:
        raise ValueError("availability_delay_hours must be non-negative")
    if not 0 <= correction_fraction <= 1:
        raise ValueError("correction_fraction must be between zero and one")
    if correction_decay_hours <= 0:
        raise ValueError("correction_decay_hours must be positive")
    decision_time = pd.Timestamp(decision_time)
    delay = pd.Timedelta(hours=availability_delay_hours)
    eligible = sorted(run for run in runs if run + delay <= decision_time)
    target = pd.date_range(
        decision_time,
        periods=horizon_steps + 1,
        freq=f"{delta_t}s",
    )
    if not eligible:
        return np.repeat(np.asarray(current_disturbance)[None], len(target), axis=0)
    for initialization in reversed(eligible):
        selected = runs[initialization].reindex(target)
        # Archived runs can end before the 24-hour control horizon. Holding the
        # last published forecast is causal; using realized future weather is not.
        selected = selected.ffill()
        if not selected.isna().any().any():
            values = selected[channels].to_numpy(copy=True)
            # Correct the ambient channel by name. It happens to be first in both current
            # views, which is exactly the kind of coincidence that breaks when a view gains a
            # channel or a second zone reorders one.
            if "T_amb" in channels:
                ambient = channels.index("T_amb")
                error = float(current_disturbance[ambient] - values[0, ambient])
                lead_hours = np.arange(len(target)) * delta_t / 3600
                values[:, ambient] += (
                    correction_fraction * error * np.exp(-lead_hours / correction_decay_hours)
                )
            values[0] = current_disturbance
            return values
    # A missing run or coverage gap must not expose future realized weather.
    return np.repeat(np.asarray(current_disturbance)[None], len(target), axis=0)

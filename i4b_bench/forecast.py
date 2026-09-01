"""Where future disturbances come from.

Two sources behind one interface: the realized exogenous record, or the archived forecast runs a
controller would actually have had. Which channels come back follows the caller's view, so the
`realistic` view gets raw weather and `perfect` gets the building-specific gain.

This lives outside the environment because open-loop evaluation needs forecast horizons just as
much as closed-loop does. Keeping it inside the env is what pushed a second implementation of
the same publication-lag rule into a downstream repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .corpus import prepare_forecast_runs, select_forecast_disturbances
from .observation import internal_gain_profile

#: Preparing the archived runs for one building costs ~37 s -- every run is pushed through
#: `prepare_disturbances`, solar gains and all. It depends only on the location's forecasts and
#: the building's parameters, both fixed for a scenario, so every window of that scenario would
#: otherwise pay it again. Keyed by (location, building); a full benchmark touches one entry per
#: scenario.
_RUNS: dict[tuple[str, str], dict] = {}


def forecast_runs(forecasts, building_params, profile, key):
    """`prepare_forecast_runs`, memoised on (location_id, building_id)."""
    if key not in _RUNS:
        _RUNS[key] = prepare_forecast_runs(forecasts, building_params, profile)
    return _RUNS[key]


class ForecastProvider:
    """Provides exogenous forecasts (oracle or archived) at each timestep."""

    def __init__(
        self,
        exogenous: pd.DataFrame,
        disturbance_channels: tuple[str, ...],
        building_params: dict,
        use_forecast: bool,
        forecasts: pd.DataFrame | None = None,
        location_id: str | None = None,
        cache_key: tuple[str, str] | None = None,
        correction_fraction: float = 0.5,
    ):
        self._exogenous = exogenous
        self._disturbance_channels = disturbance_channels
        self._use_forecast = use_forecast
        # How far an archived run is nudged toward the current measurement. Realistic for a
        # controller correcting against its own sensor, but it makes the "forecast" partly
        # observational, so a study of forecast error proper sets this to zero.
        self._correction_fraction = correction_fraction
        self._runs: dict[pd.Timestamp, pd.DataFrame] | None = None

        if use_forecast:
            if forecasts is None or location_id is None:
                raise ValueError("forecasts and location_id are required when use_forecast=True")
            location_forecasts = forecasts[forecasts["location_id"] == location_id]
            if cache_key is None:
                # never key on id(): a fresh params dict each call would miss every time
                raise ValueError("cache_key is required when use_forecast=True")
            key = cache_key
            self._runs = forecast_runs(
                location_forecasts, building_params, internal_gain_profile(), key
            )

    def get_forecast(
        self,
        decision_time: pd.Timestamp,
        planning_steps: int,
        current_disturbance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, values) arrays for the forecast horizon.

        values has shape (planning_steps, len(disturbance_channels)).
        timestamps has shape (planning_steps,) with dtype datetime64.
        """
        delta_t = 900
        future_start = decision_time + pd.Timedelta(seconds=delta_t)
        timestamps = pd.date_range(future_start, periods=planning_steps, freq=f"{delta_t}s")

        if self._use_forecast and self._runs is not None:
            raw = select_forecast_disturbances(
                self._runs,
                decision_time,
                planning_steps,
                current_disturbance,
                channels=self._disturbance_channels,
                correction_fraction=self._correction_fraction,
            )
            # raw[0] is decision_time (current), raw[1:] is the forecast
            values = raw[1 : planning_steps + 1]
        else:
            slice_ = self._exogenous.reindex(timestamps)
            values = slice_[list(self._disturbance_channels)].to_numpy(dtype=np.float32)

        ts = timestamps.values.astype("datetime64[s]")
        return ts, values

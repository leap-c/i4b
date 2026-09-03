"""Where future disturbances come from.

Two sources behind one interface: the realized exogenous record, or the archived forecast runs a
controller would actually have had. Which channels come back follows the caller's view, so the
`realistic` view gets raw weather and `perfect` gets the building-specific gain.

This lives outside the environment because open-loop evaluation needs forecast horizons just as
much as closed-loop does. Keeping it inside the env is what pushed a second implementation of
the same publication-lag rule into a downstream repo.

Horizon alignment
-----------------
The plant integrates ``x_t + u_t + d_t -> x_(t+1)``: the disturbance driving a transition is the
one at the *start* of the interval, not the one at the state it produces. A horizon handed over
at decision time `t` therefore carries the **interval inputs** ``d_t ... d_(t+h-1)``, while its
timestamps name the **target states** ``t+1 ... t+h`` -- so `forecast[i]` is the weather that,
together with `control[i]`, produces the state stamped `timestamp[i]`. Row zero is the current
measurement rather than a prediction; at `t` it has been observed.

This used to be off by one: ``d_(t+1)`` was published against the transition driven by ``u_t``,
so every model was handed a horizon shifted one step out of the plant's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .corpus import prepare_forecast_runs, select_forecast_disturbances
from .observation import internal_gain_profile


class ForecastProvider:
    """Provides exogenous forecasts (oracle or archived) at each timestep.

    Parameters
    ----------
    exogenous : pandas.DataFrame
        The realized record, timestamp-indexed; the oracle source, and the fallback.
    disturbance_channels : tuple of str
        The view's channels, and the column order of the returned values.
    building_params : dict
        Needed to turn archived weather into this building's disturbances.
    use_forecast : bool
        Archived runs rather than the realized record.
    forecasts, location_id
        The archive, and which location to take out of it. Required when `use_forecast`.
    cache_key : tuple of str, optional
        The key into `runs_cache`. Never `id()` of a params dict: a fresh dict each call would
        miss every time.
    runs_cache : dict, optional
        Where to memoise prepared runs. Preparing one building's archive costs ~40 s, and every
        window of a scenario would otherwise pay it again. The caller owns the dict, so a memo
        cannot outlive the run that built it and then be served against a different corpus --
        which the module-level cache this replaces silently allowed.
    correction_fraction : float
        How far an archived run is nudged toward the current measurement, in [0, 1]. Realistic
        for a controller correcting against its own sensor, but it makes the "forecast" partly
        observational, so a study of forecast error proper sets this to zero.
    """

    def __init__(
        self,
        exogenous: pd.DataFrame,
        disturbance_channels: tuple[str, ...],
        building_params: dict,
        use_forecast: bool,
        forecasts: pd.DataFrame | None = None,
        location_id: str | None = None,
        cache_key: tuple[str, str] | None = None,
        runs_cache: dict | None = None,
        correction_fraction: float = 0.5,
    ):
        self._exogenous = exogenous
        self._disturbance_channels = disturbance_channels
        self._use_forecast = use_forecast
        self._correction_fraction = correction_fraction
        self._runs: dict[pd.Timestamp, pd.DataFrame] | None = None

        if use_forecast:
            if forecasts is None or location_id is None:
                raise ValueError("forecasts and location_id are required when use_forecast=True")
            if cache_key is None:
                raise ValueError("cache_key is required when use_forecast=True")
            if runs_cache is not None and cache_key in runs_cache:
                self._runs = runs_cache[cache_key]
            else:
                self._runs = prepare_forecast_runs(
                    forecasts[forecasts["location_id"] == location_id],
                    building_params,
                    internal_gain_profile(),
                )
                if runs_cache is not None:
                    runs_cache[cache_key] = self._runs

    def get_forecast(
        self,
        decision_time: pd.Timestamp,
        planning_steps: int,
        current_disturbance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The horizon as of `decision_time`, as `(timestamps, values)`.

        `values` has shape `(planning_steps, len(disturbance_channels))` and holds the interval
        inputs ``d_t ... d_(t+h-1)``; `timestamps` names the states those produce, ``t+1 ...
        t+h``. See the module docstring -- the offset between the two is the point.
        """
        delta_t = 900
        # The intervals the horizon covers start at the decision time; the states they produce
        # are stamped one step later.
        intervals = pd.date_range(decision_time, periods=planning_steps, freq=f"{delta_t}s")
        timestamps = intervals + pd.Timedelta(seconds=delta_t)

        if self._use_forecast and self._runs is not None:
            raw = select_forecast_disturbances(
                self._runs,
                decision_time,
                planning_steps,
                current_disturbance,
                channels=self._disturbance_channels,
                correction_fraction=self._correction_fraction,
            )
            # `raw` is indexed by interval start: raw[0] is the measurement at `decision_time`,
            # the input to the first transition, and raw[-1] lies one interval past the horizon.
            values = raw[:planning_steps]
        else:
            slice_ = self._exogenous.reindex(intervals)
            values = slice_[list(self._disturbance_channels)].to_numpy(dtype=np.float32)

        ts = timestamps.values.astype("datetime64[s]")
        return ts, values

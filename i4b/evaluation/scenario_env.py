"""Scenario-augmented I4B building environment for closed-loop evaluation.

Wraps ``RoomHeatEnv`` and produces structured dict observations with
state, history, and forecast. See docs/EVAL_SPEC.md.

History alignment
-----------------
A history row carries the state at its timestamp together with the action and
disturbances that **produced** that state, so a control value always lines up with
its effect. This is one step later than ``transitions.parquet``, which stores
``state_t + applied_input_t -> state_(t+1)``: corpus row ``t`` becomes history row
``t + 1`` here. Seeded and stepped rows follow the same rule, so the convention does
not change part-way through an episode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from i4b.benchmark import load_params, prepare_forecast_runs, select_forecast_disturbances

from .dataset import BenchmarkDataset, load_controller_data, load_dataset


# ---------------------------------------------------------------------------
# Observation channel definitions
# ---------------------------------------------------------------------------
ObsView = Literal["perfect", "realistic"]

STATE_CHANNELS = ("T_room", "T_wall", "T_hp_ret")
CONTROL_CHANNELS = ("T_hp_sup_applied",)

_DISTURBANCE_CHANNELS: dict[ObsView, tuple[str, ...]] = {
    "perfect": ("T_amb", "Qdot_gains"),
    "realistic": ("T_amb", "ghi", "dni", "dhi"),
}

import i4b_data

# i4b_data is a namespace package, so it has no __file__; __path__ works either way.
_INTERNAL_GAIN_PROFILE = (
    Path(next(iter(i4b_data.__path__))).resolve()
    / "profiles"
    / "InternalGains"
    / "ResidentialDetached.csv"
)


# ---------------------------------------------------------------------------
# Forecast provider
# ---------------------------------------------------------------------------

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
    ):
        self._exogenous = exogenous
        self._disturbance_channels = disturbance_channels
        self._use_forecast = use_forecast
        self._runs: dict[pd.Timestamp, pd.DataFrame] | None = None

        if use_forecast:
            if forecasts is None or location_id is None:
                raise ValueError(
                    "forecasts and location_id are required when use_forecast=True"
                )
            location_forecasts = forecasts[forecasts["location_id"] == location_id]
            self._runs = prepare_forecast_runs(
                location_forecasts,
                building_params,
                _INTERNAL_GAIN_PROFILE,
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
        timestamps = pd.date_range(
            future_start, periods=planning_steps, freq=f"{delta_t}s"
        )

        if self._use_forecast and self._runs is not None:
            raw = select_forecast_disturbances(
                self._runs,
                decision_time,
                planning_steps,
                current_disturbance,
            )
            # raw[0] is decision_time (current), raw[1:] is the forecast
            values = raw[1 : planning_steps + 1]
        else:
            slice_ = self._exogenous.reindex(timestamps)
            values = slice_[list(self._disturbance_channels)].to_numpy(
                dtype=np.float32
            )

        ts = timestamps.values.astype("datetime64[s]")
        return ts, values


# ---------------------------------------------------------------------------
# Scenario environment
# ---------------------------------------------------------------------------

class ScenarioEnv:
    """Gymnasium-style environment for closed-loop evaluation on a benchmark scenario.

    Wraps ``RoomHeatEnv`` and produces structured dict observations with
    state, history, and forecast.

    Parameters
    ----------
    view : ``"perfect"`` | ``"realistic"``
        Controls which disturbance channels are exposed to the agent.
        ``"perfect"`` provides pre-computed ``(T_amb, Qdot_gains)``; ``"realistic"``
        provides raw weather ``(T_amb, ghi, dni, dhi)`` without internal gains.
    """

    def __init__(
        self,
        scenario_id: str,
        *,
        dataset: BenchmarkDataset | None = None,
        dataset_dir: str | Path | None = None,
        initial_controller_id: str = "mpc-nominal",
        history_length: int = 96,
        planning_steps: int = 12,
        start_step: int | None = None,
        use_forecast: bool = False,
        view: ObsView = "perfect",
    ):
        if dataset is None:
            dataset = load_dataset(dataset_dir)
        self._dataset = dataset

        # Resolve view-dependent channels
        self._view = view
        self._disturbance_channels = _DISTURBANCE_CHANNELS[view]
        self._history_channels = (
            STATE_CHANNELS + CONTROL_CHANNELS + self._disturbance_channels
        )

        # Resolve scenario metadata
        scenario_row = dataset.scenarios[
            dataset.scenarios["scenario_id"] == scenario_id
        ].iloc[0]
        building_id = scenario_row["building_id"]

        # Load building params and create the inner environment
        self._building_params = load_params(dataset.buildings, building_id)
        mdot_hp = self._building_params["mdot_hp"]

        # Load exogenous for this scenario (timestamp-indexed from here on)
        exo = dataset.exogenous[
            dataset.exogenous["scenario_id"] == scenario_id
        ].copy()
        exo["timestamp_utc"] = pd.to_datetime(exo["timestamp_utc"], utc=True)
        self._exogenous = exo.set_index("timestamp_utc").sort_index()

        # Build disturbance DataFrame for RoomHeatEnv (always needs T_amb + Qdot_gains)
        from i4b.gym_interface.room_env import RoomHeatEnv

        self._env = RoomHeatEnv(
            hp_model="Heatpump_AW",
            building=None,
            method="4R3C",
            mdot_HP=mdot_hp,
            internal_gain_profile=str(_INTERNAL_GAIN_PROFILE),
            building_params=self._building_params,
            disturbances=self._exogenous[["T_amb", "Qdot_gains"]].copy(),
            integrator="linear",
            return_numpy=True,
        )

        # Load initial controller trajectory for history seeding
        # TODO: verify that trajectory is shortened for initial context
        self._initial_trajectory = load_controller_data(
            dataset, initial_controller_id, scenario_id
        )

        # Forecast provider
        self._forecast_provider = ForecastProvider(
            exogenous=self._exogenous,
            disturbance_channels=self._disturbance_channels,
            building_params=self._building_params,
            use_forecast=use_forecast,
            forecasts=dataset.forecasts,
            location_id=scenario_row.get("location_id"),
        )

        # Configuration
        # TODO: history_length -> max_context_length, initial_context_length as extra param
        self._history_length = history_length
        self._planning_steps = planning_steps

        # TODO: replace with: self._start_step = start_step if start_step is not None else initial_context_length
        self._start_step = start_step if start_step is not None else history_length
        if self._start_step < history_length:
            raise ValueError(
                f"start_step ({self._start_step}) must be >= history_length ({history_length})"
            )
        if self._start_step > len(self._initial_trajectory):
            raise ValueError(
                f"start_step ({self._start_step}) exceeds available trajectory "
                f"length ({len(self._initial_trajectory)})"
            )

        # State that gets populated on reset
        self._history_buffer: list[dict[str, float]] = []
        self._timestamps: list[np.datetime64] = []
        self._step_count = 0

    @property
    def max_steps(self) -> int:
        """Maximum number of evaluation steps from start to end of scenario."""
        return len(self._env.p) - 1 - self._start_step

    def reset(self) -> tuple[dict, dict]:
        """Reset the environment to the evaluation start point.

        Seeds the building state and history buffer from the recorded
        initial controller trajectory.
        """
        traj = self._initial_trajectory

        # Set the inner env to the correct timestep and state
        self._env.reset()
        self._env.t = self._start_step

        # Set the building state from the recorded trajectory at start_step
        state_at_start = {
            ch: float(traj.iloc[self._start_step][ch]) for ch in STATE_CHANNELS
        }
        self._env.state = self._env._build_observation(state_at_start)

        # Seed the history buffer from the recorded trajectory
        # One extra row: pairing each state with the previous row's inputs consumes one.
        history_start = max(0, self._start_step - self._history_length)
        history_slice = traj.iloc[history_start : self._start_step + 1]

        # Recorded rows are state_t + input_t; a history row is the state together with
        # the input that produced it, so the inputs move one step forward with the state.
        self._history_buffer = []
        self._timestamps = []
        input_channels = [ch for ch in self._history_channels if ch not in STATE_CHANNELS]
        rows = list(history_slice.itertuples(index=False))
        for previous, row in zip(rows, rows[1:]):
            record = {ch: float(getattr(row, ch)) for ch in STATE_CHANNELS}
            record.update({ch: float(getattr(previous, ch)) for ch in input_channels})
            self._history_buffer.append(record)
            self._timestamps.append(
                row.timestamp_utc.to_datetime64().astype("datetime64[s]")
            )

        self._step_count = 0
        return self._build_observation(), {}

    def step(self, action: float) -> tuple[dict, float, bool, bool, dict]:
        """Step the building with the given supply temperature action (Celsius).

        Returns (observation, reward, terminated, truncated, info).
        """
        normalized = self._env.normalize_action(np.array(action))
        obs, reward, terminated, truncated, info = self._env.step(normalized)

        # Record the step into the history buffer. ``obs`` is the state reached by this
        # step, so it is stamped at the end of the interval, beside the action and
        # disturbances that produced it.
        timestamp = self._env.p.index[min(self._env.t, len(self._env.p) - 1)]
        record: dict[str, float] = {
            "T_room": float(obs[0]),
            "T_wall": float(obs[1]),
            "T_hp_ret": float(obs[2]),
            "T_hp_sup_applied": float(info["u"]),
        }
        # Look up view-specific disturbance values from the exogenous data
        exo_row = self._exogenous.iloc[self._env.t - 1]
        for ch in self._disturbance_channels:
            record[ch] = float(exo_row[ch])

        self._history_buffer.append(record)
        self._timestamps.append(
            timestamp.to_datetime64().astype("datetime64[s]")
        )

        # Trim history buffer to max length
        if len(self._history_buffer) > self._history_length:
            self._history_buffer = self._history_buffer[-self._history_length :]
            self._timestamps = self._timestamps[-self._history_length :]

        self._step_count += 1
        return self._build_observation(), reward, terminated, truncated, info

    def _build_observation(self) -> dict:
        """Construct the nested observation dict."""
        # Current state
        state_vals = self._env.state[: len(STATE_CHANNELS)]
        state = {ch: float(state_vals[i]) for i, ch in enumerate(STATE_CHANNELS)}

        # History
        n = len(self._history_buffer)
        history = {"timestamp": np.array(self._timestamps, dtype="datetime64[s]")}
        for ch in self._history_channels:
            history[ch] = np.array(
                [self._history_buffer[i][ch] for i in range(n)], dtype=np.float32
            )

        # Forecast
        current_time = self._env.get_cur_time()
        exo_now = self._exogenous.loc[current_time]
        current_disturbance = np.array(
            [exo_now[ch] for ch in self._disturbance_channels], dtype=np.float32
        )
        forecast_ts, forecast_vals = self._forecast_provider.get_forecast(
            current_time, self._planning_steps, current_disturbance
        )
        forecast = {"timestamp": forecast_ts}
        for i, ch in enumerate(self._disturbance_channels):
            forecast[ch] = forecast_vals[:, i].astype(np.float32)

        return {"state": state, "history": history, "forecast": forecast}

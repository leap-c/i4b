"""Scenario-augmented I4B building environment for closed-loop evaluation.

Wraps ``RoomHeatEnv`` and produces the structured observation `i4b_bench.observation` defines.
See docs/EVAL_SPEC.md.

History alignment
-----------------
A history row carries the state at its timestamp together with the action and disturbances that
**produced** that state, so a control value always lines up with its effect. This is one step
later than ``transitions.parquet``, which stores ``state_t + applied_input_t -> state_(t+1)``:
corpus row ``t`` becomes history row ``t + 1`` here. Seeded and stepped rows follow the same
rule, so the convention does not change part-way through an episode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import load_params
from .dataset import BenchmarkDataset, load_controller_data, load_dataset
from .forecast import ForecastProvider
from .observation import (
    CONTROL_CHANNELS,
    DISTURBANCE_CHANNELS,
    STATE_CHANNELS,
    ObsView,
    build_observation,
    history_channels,
    internal_gain_profile,
)


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
    start_step : int | None
        The timestep at which to start the evaluation (time-wise steps are determined by the scenario time discretization).
    """

    def __init__(
        self,
        scenario_id: str,
        *,
        dataset: BenchmarkDataset | None = None,
        dataset_dir: str | Path | None = None,
        initial_controller_id: str = "mpc-nominal",
        max_context_length: int = 96,
        initial_context_length: int | None = None,
        planning_steps: int = 12,
        start_step: int | None = None,
        use_forecast: bool = False,
        view: ObsView = "perfect",
        build_observation: bool = True,
        forecast_correction: float = 0.5,
    ):
        if dataset is None:
            dataset = load_dataset(dataset_dir)
        self._dataset = dataset

        # Resolve view-dependent channels
        self._view = view
        self._disturbance_channels = DISTURBANCE_CHANNELS[view]
        self._history_channels = history_channels(view)

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
            internal_gain_profile=str(internal_gain_profile()),
            building_params=self._building_params,
            disturbances=self._exogenous[["T_amb", "Qdot_gains"]].copy(),
            # Pinned, not left to RoomHeatEnv's default. The corpus was integrated with the
            # legacy backend, so evaluating against it with a different integrator would score
            # controllers in a slightly different world than the recorded baselines they are
            # compared to (measured: 6.4 mK worst case). It is also ~30x faster per step, which
            # matters because the control-response metric rolls the plant once per probe.
            backend="legacy",
        )

        # Observations cost a dict of arrays per step. Generation discards them, and over
        # 35,039 steps x 1,528 trajectories that is the difference between minutes and hours.
        self._build_observations = build_observation

        # The recorded trajectory seeds the history buffer -- and it is the *only* thing here
        # that needs one, so `initial_controller_id=None` lets this environment generate the
        # trajectories a corpus is made of rather than only replay them.
        self._initial_trajectory = (
            None
            if initial_controller_id is None
            else load_controller_data(dataset, initial_controller_id, scenario_id)
        )

        # Forecast provider
        self._forecast_provider = ForecastProvider(
            exogenous=self._exogenous,
            disturbance_channels=self._disturbance_channels,
            building_params=self._building_params,
            use_forecast=use_forecast,
            forecasts=dataset.forecasts,
            location_id=scenario_row.get("location_id"),
            cache_key=(str(scenario_row.get("location_id")), str(building_id)),
            correction_fraction=forecast_correction,
        )

        # Configuration
        self._max_context_length = max_context_length
        self._initial_context_length = (
            initial_context_length if initial_context_length is not None else max_context_length
        )
        if self._initial_context_length > max_context_length:
            raise ValueError(
                f"initial_context_length ({self._initial_context_length}) must be "
                f"<= max_context_length ({max_context_length})"
            )
        self._planning_steps = planning_steps

        if self._initial_trajectory is None:
            self._initial_context_length = 0
        self._start_step = start_step if start_step is not None else self._initial_context_length
        if self._start_step < self._initial_context_length:
            raise ValueError(
                f"start_step ({self._start_step}) must be >= "
                f"initial_context_length ({self._initial_context_length})"
            )
        if self._initial_trajectory is not None and self._start_step > len(
            self._initial_trajectory
        ):
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

        if traj is not None:
            # Set the building state from the recorded trajectory at start_step
            state_at_start = {
                ch: float(traj.iloc[self._start_step][ch]) for ch in STATE_CHANNELS
            }
            self._env.state = self._env._build_observation(state_at_start)

        # Seed the history buffer from the recorded trajectory
        # One extra row: pairing each state with the previous row's inputs consumes one.
        history_start = max(0, self._start_step - self._initial_context_length)
        history_slice = (
            traj.iloc[history_start : self._start_step + 1]
            if traj is not None
            else traj  # no recorded past to seed from; the buffer fills as the episode runs
        )

        # Recorded rows are state_t + input_t; a history row is the state together with
        # the input that produced it, so the inputs move one step forward with the state.
        self._history_buffer = []
        self._timestamps = []
        input_channels = [ch for ch in self._history_channels if ch not in STATE_CHANNELS]
        rows = [] if history_slice is None else list(history_slice.itertuples(index=False))
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
        if len(self._history_buffer) > self._max_context_length:
            self._history_buffer = self._history_buffer[-self._max_context_length :]
            self._timestamps = self._timestamps[-self._max_context_length :]

        self._step_count += 1
        return self._build_observation(), reward, terminated, truncated, info

    def _build_observation(self) -> dict:
        """Construct the nested observation dict, unless the caller opted out of the cost."""
        if not self._build_observations:
            return {}
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

        return build_observation(state, history, forecast)

# Closed-Loop Controller Evaluation

Evaluate the closed-loop performance of a controller on an I4B benchmark
scenario. The dataset schema is defined in `specs/DATA_SCHEMA.md`.

## Overview

A single evaluation run:

1. Loads the scenario-augmented I4B building model for a given `scenario_id`.
2. Constructs an initial state and history from a recorded controller trajectory.
3. Steps the controller and building forward for `n_evaluation_steps` timesteps.
4. Returns primary and secondary metrics plus the full state-action trajectory.

## Observation Structure

The environment produces a nested dictionary observation at each timestep:

```python
{
    "state": {
        "T_room": float,
        "T_wall": float,
        "T_hp_ret": float,
    },
    "history": {
        "timestamp": np.ndarray,       # dtype=datetime64, shape (<= max_context_length,)
        "T_room": np.ndarray,          # shape (<= max_context_length,)
        "T_wall": np.ndarray,
        "T_hp_ret": np.ndarray,
        "T_hp_sup_applied": np.ndarray, # past actions
        "T_amb": np.ndarray,
        "Qdot_gains": np.ndarray,
    },
    "forecast": {
        "timestamp": np.ndarray,       # dtype=datetime64, shape (planning_steps,)
        "T_amb": np.ndarray,           # shape (planning_steps,)
        "Qdot_gains": np.ndarray,
    },
}
```

**Design notes:**

- `state` holds the current scalar values. These also appear as the last
  entry in `history` (intentional overlap for convenience).
- `history` includes past states, past actions, and past exogenous data.
  Its length is at most `max_context_length` rows. Before enough steps have
  elapsed, it may be shorter (padded or truncated — implementation choice).
- `forecast` contains future exogenous data starting from the next timestep.
  Its length is at most `planning_steps` rows.
- Timestamps use `numpy.datetime64` — framework-agnostic, fast, and trivially
  convertible to strings or pandas when needed.
- Forecast channels are initially limited to `T_amb` and `Qdot_gains`.
  The set of channels in all three groups should become configurable in a
  future iteration.

The observation follows `gymnasium.spaces.Dict` conventions, making it
compatible with both RL and TSFM controller wrappers.

## Scenario-Augmented Building Model

Wraps the I4B `RoomHeatEnv` for a specific `scenario_id`. Follows the
Gymnasium `step`/`reset` interface and returns the structured observation
described above.

Responsibilities:

- **Simulation**: given a control action, advance the building by one
  timestep using the I4B plant model. Returns `(observation, reward,
  terminated, truncated, info)`.
- **History assembly**: maintains a rolling buffer of past states, actions,
  and exogenous data. Composed from the recorded controller trajectory
  (for the initial context) plus new states produced during evaluation.
- **Forecast construction**: at each step, provides future exogenous data.
  A flag controls whether the forecast is constructed from
  `forecasts.parquet` (realistic archived forecasts) or from
  `exogenous.parquet` (oracle realized weather). See "Forecast
  Reconstruction" below.

## Controller Interface

A controller is a callable with the signature:

```python
def controller(
    observation: dict,
) -> Tuple[float, dict | None]:
```

**Arguments:**

- `observation`: the nested dictionary described above.

**Returns:**

- The next control action as an absolute supply temperature in Celsius.
- Optionally a dict with the controller's predicted state-action plan
  over the planning horizon. Same nested structure as `observation` but
  covering the planned future trajectory. `None` if the controller is
  planning-free (e.g. RL, heatcurve).

This signature covers both planning-based controllers (MPC, sampling-based
MPC, TSFM-based planners) and planning-free controllers (RL, rule-based).

For compatibility with `i4b.benchmark.rollout_controller`, a thin adapter
maps this signature to `(state_dict, future_disturbances) -> action_C`.

## Evaluation Parameters

| Parameter              | Type             | Default / Description                                           |
|------------------------|------------------|-----------------------------------------------------------------|
| `scenario_id`          | `str`            | Required. Identifies building + period.                         |
| `controller`           | callable         | Required. See controller interface above.                       |
| `initial_controller_id`| `str`            | `"mpc-nominal"`. Controller whose recorded trajectory provides the initial history. |
| `max_context_length`   | `int`            | Required. Maximum number of past timesteps in the history buffer. |
| `initial_context_length` | `int` or `None` | Number of past timesteps seeded from the recorded trajectory on reset. Defaults to `max_context_length`. Must be <= `max_context_length`. |
| `planning_steps`       | `int`            | Required. Number of future timesteps in the forecast.           |
| `n_evaluation_steps`   | `int`            | Number of timesteps to evaluate. Defaults to all remaining steps after the start. |
| `start_step`           | `int` or `None`  | Timestep offset at which evaluation begins. Defaults to `initial_context_length` (the earliest possible start). Must be >= `initial_context_length`. |
| `use_forecast`         | `bool`           | `True` = use archived forecasts from `forecasts.parquet`. `False` = oracle weather from `exogenous.parquet`. |

The evaluation window is determined by `start_step` and `n_evaluation_steps`.
If `n_evaluation_steps` is not specified, evaluation runs from `start_step`
to the end of the scenario. The implementation validates that enough recorded
data exists before `start_step` to fill the initial history of length
`initial_context_length`.

## Forecast Reconstruction

`forecasts.parquet` contains hourly weather forecasts indexed by
`(location_id, model, initialization_time_utc, valid_time_utc)`.

At each evaluation timestep `t`:

1. Look up the scenario's `location_id` from `scenarios.parquet`.
2. Query `forecasts.parquet` for the most recent `initialization_time_utc <= t`.
3. Select rows where `valid_time_utc` covers the next `planning_steps`
   timesteps.
4. Convert raw weather forecasts to building-specific disturbances (T_amb,
   Qdot_gains) using `i4b.benchmark.prepare_forecast_runs()` or equivalent
   utility functions.

When `use_forecast=False`, the forecast is sliced directly from
`exogenous.parquet` (oracle mode, matching the dataset generation setup).

Forecast reconstruction should be implemented as a standalone utility to keep
the evaluation loop clean.

## Metrics

### Primary

- **Energy consumption** (kWh): total heat-pump electrical energy over the
  evaluation window.
- **Comfort violation** (degree-hours): cumulative time-weighted deviation
  outside the comfort band [20, 26] C.

### Secondary (plan quality)

Only evaluated when the controller returns a non-`None` plan.

- Simulate the controller's planned action sequence on the I4B building model.
- Compare the simulated state trajectory to the controller's predicted state
  trajectory.
- Report RMSE per state channel (T_room, T_wall, T_hp_ret) per evaluated
  planning step.
- Evaluation frequency of plan quality can be reduced via a parameter to
  limit computational cost.
- Nice-to-have: if the controller provides 10th and 90th percentile
  uncertainty bounds at the planning endpoint, evaluate calibration.

### Returns

```python
{
    "energy_kwh": float,
    "comfort_violation_degree_hours": float,
    "plan_quality": pd.DataFrame | None,  # per-step per-channel RMSE
    "trajectory": pd.DataFrame,           # full state-action trajectory
}
```

## Implementation Notes

- Reuse `i4b.benchmark.rollout_controller` and `i4b.benchmark.load_params`
  where possible. If the existing rollout interface requires changes to
  support history/forecast passing, discuss before modifying.
- Use utility functions for forecast reconstruction and history assembly to
  keep the evaluation loop itself concise.
- `notebooks/MPC_example.ipynb` serves as a reference for the MPC stepping
  pattern, but note that `src.` prefixes are outdated and should be `i4b.`.

## Future Considerations

- Configurable channel selection for state, history, and forecast.
- Run an MPC baseline and a random controller in parallel to produce
  building-normalized scores that allow aggregation across scenarios.
- Electricity-price-aware objectives.

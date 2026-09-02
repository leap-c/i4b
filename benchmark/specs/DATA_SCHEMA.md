# I4B Benchmark Dataset Schema (v2)

Reference for building fine-tuning datasets and evaluation loops on the canonical
I4B benchmark corpus. What the benchmark *measures* over this data is in `EVAL_SPEC.md`.

Paths below are relative to the corpus directory, which lives at `data/corpus/` in this
repository and is what `load_dataset()` finds by default.

## Directory Layout

```
<dataset>/
  manifest.json
  buildings.parquet
  trajectories.parquet
  scenarios.parquet
  exogenous.parquet
  controllers.json
  split.parquet
  ablation-splits/
    time.parquet
  transitions/
    part-00000.parquet ... part-00037.parquet
  forecasts.parquet
  prices.parquet
```

`transitions/` is the one store of trajectory states; `trajectories.parquet` is the index into
it, and `shard_index` turns a `trajectory_id` into a single-file read. Earlier corpora also kept a
per-controller copy under `controllers/`, holding the same state columns for seven of the eleven
controllers. It saved 12% on a read, cost 2.7 GB, and meant a trajectory could exist in one store
and not the other -- which is exactly what happened, and made the excitation levels unusable as
evaluation contexts. It is gone; `load_controller_data` reads the shards for every controller.

## Key Concepts

### Scenario

A `(building_id, period_id)` pair. All controllers sharing a scenario share the
same weather, disturbances, and initial state. `scenario_id` is
`"{building_id}--{period_id}"`.

### Trajectory

A `(building_id, period_id, controller_id)` triple. Exactly one trajectory per
combination. No repeated seeds or stochastic runs. Trajectories are deterministic.

### Period

Two year-long weather periods over the same buildings:

| Period     | Range                          |
|------------|--------------------------------|
| `period_a` | 2024-04-01 through 2025-03-31  |
| `period_b` | 2025-04-01 through 2026-03-31  |

### Controllers

All MPC controllers re-solve every 15 minutes from the current state with a
12-step (3-hour) horizon. During dataset generation MPC received **oracle**
(realized) weather, not forecasts.

| `controller_id`        | Description                                      |
|------------------------|--------------------------------------------------|
| `mpc-nominal`          | No residual (baseline MPC)                       |
| `mpc-offset-plus-2K`   | Constant +2 K residual on top of MPC             |
| `mpc-offset-minus-2K`  | Constant -2 K residual on top of MPC             |
| `mpc-aprbs-low`        | Symmetric APRBS residual, amplitude 0.5 K        |
| `mpc-aprbs-medium`     | Symmetric APRBS residual, amplitude 1.5 K        |
| `mpc-aprbs-high`       | Symmetric APRBS residual, amplitude 3.0 K        |
| `open-loop-aprbs`      | Replays nominal action + 1.5 K APRBS, no re-solve|
| `open-loop-aprbs-3K`   | As `open-loop-aprbs`, amplitude 3 K              |
| `open-loop-aprbs-6K`   | As `open-loop-aprbs`, amplitude 6 K              |
| `open-loop-aprbs-12K`  | As `open-loop-aprbs`, amplitude 12 K             |
| `open-loop-aprbs-24K`  | As `open-loop-aprbs`, amplitude 24 K             |

APRBS waveforms are deterministic from building and period identity. Low, medium,
and high share the same normalized waveform (only the amplitude differs).

The four `open-loop-aprbs-<n>K` levels share the waveform of `open-loop-aprbs`, so together with
it they form an amplitude ladder over one realisation. They exist as training data for studying
how much excitation a dynamics model needs — see
`data/scripts/finalize_excitation_levels.py`. The plant clips roughly 65-70% of the commanded steps at
every level, which takes a near-constant fractional bite, so delivered excitation still scales
with the amplitude.

### Variable Categories

The dataset distinguishes three categories of variables:

| Category    | Variables                              | Description                                    |
|-------------|----------------------------------------|------------------------------------------------|
| **State**   | `T_room`, `T_wall`, `T_hp_ret`         | Building thermal states. Evolve via the plant model. |
| **Action**  | `T_hp_sup_applied`                     | Applied control input (heat-pump supply temperature). |
| **Exogenous** | `T_amb`, `Qdot_gains`               | External disturbances not influenced by the controller. |

In the v2 schema these are stored separately:

- `transitions/` contains **state + action + disturbances** per trajectory.
- `exogenous.parquet` contains **exogenous** variables per scenario (shared
  across all controllers for the same building and period).

To reconstruct a full transition row, join on `(scenario_id, timestamp_utc)`.

In the evaluation context (see `specs/EVAL_SPEC.md`):

- **`state`** in the observation holds the current scalar state values.
- **`history`** is the combination of past states, actions, and exogenous
  data over a sliding window — it requires joining controller and exogenous
  data.
- **`forecast`** contains only future exogenous variables (weather and gains),
  since future states and actions are unknown.

### Transition Semantics

Each row represents a 15-minute interval:

```
state_t + applied_input_t + exogenous_t  ->  state_(t+1)
```

The next row in the same trajectory contains the resulting state. States are not
duplicated as `*_next` columns.

Each trajectory contains **35,039 rows** (35,040 state samples with the final
state omitted because it has no following applied input).

---

## Table Schemas

### `buildings.parquet`

One row per building variant. 191 buildings across 64 families and 7 countries.

| Column                | Type   | Description                                  |
|-----------------------|--------|----------------------------------------------|
| `building_id`         | string | Unique building identifier                   |
| `building_family_id`  | string | Family grouping (used for split assignment)   |
| `country_code`        | string | Two-letter country code (BG, CY, DE, FR, IE, NO, PL) |
| `reference_area_m2`   | float  | Reference floor area                         |
| `H_tr`                | float  | Transmission heat loss coefficient [W/K]     |
| `mdot_hp`             | float  | Heat-pump mass flow [kg/s] (clamped 0.05-0.50) |
| `params_json`         | string | Full nested I4B parameter dict as JSON       |
| ...                   |        | Additional mapping provenance columns        |

**Loading a building for simulation:**

```python
from i4b.benchmark import load_params
from i4b.gym_interface.room_env import RoomHeatEnv

params = load_params(buildings, "sfh_1984_de_1")
env = RoomHeatEnv(building_params=params, mdot_HP=params["mdot_hp"])
```

### `scenarios.parquet`

One row per `(building_id, period_id)`. Controller-independent context.

| Column              | Type      | Description                              |
|---------------------|-----------|------------------------------------------|
| `scenario_id`       | string    | `"{building_id}--{period_id}"`           |
| `building_id`       | string    | References `buildings.parquet`           |
| `period_id`         | string    | `period_a` or `period_b`                |
| `country_code`      | string    | Country code                            |
| `location_id`       | string    | Weather station / location reference     |
| `timezone`          | string    | IANA timezone for local time conversion  |
| `market_id`         | string    | Electricity market (`DE-LU`)            |
| `start_time_utc`    | timestamp | Period start                            |
| `end_time_utc`      | timestamp | Period end                              |
| `timestep_seconds`  | int       | Always 900 (15 min)                     |

### `trajectories.parquet`

One row per `(building_id, period_id, controller_id)`.

| Column                              | Type      | Description                           |
|--------------------------------------|-----------|---------------------------------------|
| `trajectory_id`                      | string    | Unique trajectory identifier          |
| `scenario_id`                        | string    | References `scenarios.parquet`        |
| `building_id`                        | string    | References `buildings.parquet`        |
| `controller_id`                      | string    | One of the 7 controller IDs          |
| `period_id`                          | string    | `period_a` or `period_b`             |
| `start_time_utc`                     | timestamp | Trajectory start                     |
| `end_time_utc`                       | timestamp | Trajectory end                       |
| `timestep_seconds`                   | int       | Always 900                           |
| `row_count`                          | int       | Always 35,039                        |
| `controller_config_hash`             | string    | Deterministic config hash            |
| `horizon_steps`                      | int       | MPC horizon (12 for MPC, null for OL) |
| `forecast_source`                    | string    | Source of disturbance forecast        |
| `solver_fallback_count`              | int       | Number of solver fallback events      |
| `solver_fallback_policy`             | string    | Fallback strategy used               |
| `solver_fallback_statuses_json`      | string    | JSON dict of status code counts       |

### `exogenous.parquet`

One row per `(scenario_id, timestamp_utc)`. Shared across all controllers for the
same scenario. **Not duplicated per controller.**

| Column               | Type      | Description                                 |
|----------------------|-----------|---------------------------------------------|
| `scenario_id`        | string    | References `scenarios.parquet`              |
| `timestamp_utc`      | timestamp | Timestep                                   |
| `local_time`         | datetime  | Wall-clock local time (tz-naive)            |
| `utc_offset_minutes` | int       | UTC offset at this timestep                 |
| `is_dst`             | bool      | Whether DST is active                       |
| `temperature_2m_C`   | float     | Raw weather station temperature             |
| `ghi_W_m2`           | float     | Global horizontal irradiance                |
| `dni_W_m2`           | float     | Direct normal irradiance                    |
| `dhi_W_m2`           | float     | Diffuse horizontal irradiance               |
| `T_amb`              | float32   | Ambient temperature as used by the simulator|
| `Qdot_gains`         | float32   | Total thermal gains (internal + solar) [W]  |


### `forecasts.parquet`

Weather forecasts indexed by initialization time and valid time.

| Column                   | Type      | Description                         |
|--------------------------|-----------|-------------------------------------|
| `location_id`            | string    | Weather location                    |
| `model`                  | string    | Forecast model identifier           |
| `initialization_time_utc`| timestamp | When the forecast was issued        |
| `valid_time_utc`         | timestamp | The time being predicted            |
| `lead_hours`             | float     | `valid_time_utc - initialization_time_utc` in hours |
| `temperature_2m_C`       | float     | Predicted temperature               |
| `ghi_W_m2`, `dni_W_m2`, `dhi_W_m2` | float | Predicted irradiance     |

### `prices.parquet`

Optional DE-LU day-ahead electricity prices. Current controller trajectories
are **not** price-optimized.

| Column               | Type      | Description                          |
|----------------------|-----------|--------------------------------------|
| `price_signal_id`    | string    | Always `de-lu-day-ahead`            |
| `delivery_start_utc` | timestamp | Delivery interval start              |
| `price_eur_per_mwh`  | float     | Day-ahead price                     |
| `market_id`          | string    | `DE-LU`                             |
| `source`             | string    | Data source                         |
| `signal_type`        | string    | `common_proxy`                      |

### `split.parquet` (Primary Split)

Separates both **building families** and **time**. One row per selected trajectory.

| Column           | Type      | Description             |
|------------------|-----------|-------------------------|
| `trajectory_id`  | string    | References trajectories |
| `split`          | string    | `train`, `validation`, or `test` |
| `start_time_utc` | timestamp | Split window start      |
| `end_time_utc`   | timestamp | Split window end        |

Assignment (families stratified by country via deterministic SHA-256 ordering):

| Split        | Families          | Period / Time Window               |
|--------------|-------------------|------------------------------------|
| `train`      | 44 train families | `period_a` x [2024-04-01, 2025-01-01) |
| `validation` | 10 val families   | `period_a` x [2025-01-01, 2025-04-01) |
| `test`       | 10 test families  | `period_b` x [2025-04-01, 2026-04-01) |

No building family and no timestamp appears in more than one primary split.
Trajectories outside these windows remain in the corpus but are unassigned.

### `ablation-splits/time.parquet`

Time-only split (all families in every split):

| Split        | Time Window                  |
|--------------|------------------------------|
| `train`      | [2024-04-01, 2025-01-01)     |
| `validation` | [2025-01-01, 2025-04-01)     |
| `test`       | [2025-04-01, 2026-04-01)     |

---

## Join Keys

The v2 schema factors the data to avoid duplicating weather across all 11 controllers.
The primary join paths are:

```
    |-- trajectory_id --> trajectories.parquet --> building_id, controller_id, period_id
    |-- scenario_id   --> scenarios.parquet    --> location_id, timezone, country_code
    |-- (scenario_id, timestamp_utc) --> exogenous.parquet --> weather, gains
```

To reconstruct the full flat transition table for one controller:

```python
controller = load_controller_data(dataset, "mpc-nominal", scenario_id)
exogenous  = pd.read_parquet(f"{DATA_DIR}/exogenous.parquet")
full = controller.merge(exogenous, on=["scenario_id", "timestamp_utc"])
```

To restrict to a split:

```python
split = pd.read_parquet(f"{DATA_DIR}/split.parquet")
train_ids = split[split["split"] == "train"]["trajectory_id"]
train = full[full["trajectory_id"].isin(train_ids)]
```

## Forecast Reconstruction

The dataset stores only the **applied** action at each timestep. No MPC plans or
intermediate solver states are saved. To evaluate a TSFM or branch a new
controller from a historical state:

1. Look up the scenario's `location_id` from `scenarios.parquet`.
2. Query `forecasts.parquet` for the forecast available at the desired
   `timestamp_utc` (filtering by `initialization_time_utc <= timestamp` and
   choosing the most recent initialization).
3. Convert raw weather forecasts to building-specific disturbances using
   `i4b.benchmark.prepare_forecast_runs()`.

## Building Instantiation

Each building's full parameter dictionary is stored as JSON in
`buildings.parquet["params_json"]`. To instantiate:

```python
from i4b.benchmark import load_params
from i4b.gym_interface.room_env import RoomHeatEnv

params = load_params(buildings, building_id)
env = RoomHeatEnv(
    building_params=params,
    mdot_HP=params["mdot_hp"],
    disturbances=disturbance_df,  # optional: precomputed T_amb + Qdot_gains
)
```

## Rollout Interface

`i4b.benchmark.rollout_controller` runs a callable controller through an
environment and returns canonical transition rows:

```python
from i4b.benchmark import rollout_controller

transitions = rollout_controller(env, controller_fn, trajectory_id="my_run")
```

The controller callable receives `(state_dict, future_disturbances)` and returns
an absolute supply temperature in Celsius. The rollout helper handles action
normalization, stepping, and transition alignment.

## Implicit Defaults

These values are not stored per-row in the dataset but were fixed during
generation. They must be reproduced when instantiating buildings, running MPC,
or evaluating comfort.

### Building Model

| Parameter           | Value   | Description                                   |
|---------------------|---------|-----------------------------------------------|
| Model               | `4R3C`  | Thermal network method for all 191 buildings  |
| Initial states      | 20 °C   | All three states (T_room, T_wall, T_hp_ret)   |
| `T_offset`          | 0 K     | No static temperature offset                  |
| `T_amb_lim`         | 20 °C   | Ambient temperature limit for heating curve   |
| `c_frame`           | 0.3     | Window frame correction factor                |
| `c_shade`           | 0.6     | Window shading correction factor              |
| Timestep (`delta_t`)| 900 s   | 15-minute integration step                    |
| Internal gains      | DIN EN 16798-1 | `ResidentialDetached` profile, scaled by `area_floor` |

### Heat-Pump Sizing

| Parameter             | Value         | Description                              |
|-----------------------|---------------|------------------------------------------|
| Design room temp      | 20 °C         | For design heat loss calculation         |
| Design ambient temp   | 1st percentile| Of reference T_amb across both periods   |
| Water temperature diff| 15 K          | Supply-return design delta               |
| `mdot_hp` clamp       | [0.05, 0.50] kg/s | Hard bounds on mass flow            |

### MPC Controller

| Parameter                        | Value            | Description                         |
|----------------------------------|------------------|-------------------------------------|
| Horizon                          | 12 steps (3 h)   | Prediction/control horizon          |
| Room comfort band                | [20, 26] °C      | Soft constraint (quadratic penalty)  |
| Comfort slack weight             | 1.0              | Quadratic penalty weight on violation|
| Supply temperature bounds        | [5, 65] °C       | Hard action constraints              |
| Thermal power bounds             | [0, 26] kW       | Hard constraint on heat pump output  |
| Stage cost                       | `Qth_kW / (COP * 100)` | Minimize thermal energy weighted by COP |
| Terminal cost                    | 0.0              | No terminal penalty                 |
| Disturbance forecast             | Oracle (realized)| MPC saw perfect future weather       |
| Residual application             | Added to first MPC action before environment clipping | |
| Re-solve frequency               | Every timestep   | MPC re-solves from actual state each 15 min |

### Environment / Simulation

| Parameter                | Value   | Description                              |
|--------------------------|---------|------------------------------------------|
| `T_room_set_lower`       | 20 °C   | Default lower comfort setpoint           |
| `T_room_set_upper`       | 26 °C   | Default upper comfort setpoint           |
| `cop_scale`              | 1.0     | No COP scaling                           |
| Action space (internal)  | [0, 65] °C | Environment supply temperature bounds |

## Scale

```
191 buildings x 11 controllers x 2 periods = 4,202 trajectories
35,039 rows per trajectory
147,233,878 transitions total
~6.8 GB on disk
```

The 191 buildings span 64 families and 7 countries; a building under one weather period is a
*scenario*, so there are 382 of those. `split.parquet` assigns whole families, and covers one
period per building — 2,101 of the 4,202 trajectories — so no building is trained on in one
weather year and tested in the other.

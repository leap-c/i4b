# Build the Canonical I4B Benchmark Dataset

## Objective

Transform the acquired TABULA and reference-weather artifacts into:

- A simulation-ready catalog of 191 I4B `4R3C` buildings.
- Reproducible closed-loop trajectories from several MPC-based collection policies.
- Short, safe open-loop APRBS trajectories.
- One canonical, controller-independent transition dataset.
- A minimal environment and rollout interface reusable for later controller evaluation.
- A primary split plus alternative ablation split manifests.

Do not create Chronos-specific files in this ticket.

## Context

Source acquisition is implemented by `scripts/prepare_benchmark_data.py` and specified
in `tickets/source-data-acquisition.md`.

The pilot gate was positive. Residual excitation improved control sensitivity, so
benchmark-scale generation can proceed.

Existing relevant implementations:

- I4B plant and Gymnasium environments in `i4b/`.
- Parametric I4B MPC in `leap-c-lab`.
- Residual APRBS collector in `/home/jas/worktrees/leap-c/wp2-aprbs-collector/`.
- Earlier contextual analysis in `heat-control`.

Reuse these implementations. Do not create another simulator, MPC solver, or generic
controller framework.

## Inputs

Read generated artifacts, not the TABULA workbook directly:

```text
source-data/normalized/buildings/tabula_sfh.parquet
source-data/normalized/weather_reference/*.parquet
scripts/benchmark_source_data.json
```

Use:

```text
period_a: 2024-04-01 through 2025-03-31
period_b: 2025-04-01 through 2026-03-31
```

Exclude archived forecasts and electricity prices for now. MPC receives realized
future disturbances as oracle covariates during dataset generation.

## Building Mapping

Map each normalized TABULA row to one I4B parameter dictionary.

```text
H_ve        = ventilation_W_m2K * reference_area_m2
H_tr        = transmission_W_m2K * reference_area_m2
H_tr_light  = window_1_transmission_W_K
              + window_2_transmission_W_K
              + door_1_transmission_W_K
c_bldg      = thermal_capacity_Wh_m2K
area_floor  = reference_area_m2
height_room = room_height_m
```

Set:

```text
name      = building_id
T_offset  = 0 K
T_amb_lim = 20 degC
```

Create four vertical windows with:

```text
tilt     = 90 degrees
azimuth  = north 0, east 90, south 180, west 270
g_value  = window_g_value
c_frame  = 0.3
c_shade  = 0.6
```

Treat `c_frame` and `c_shade` as explicit benchmark defaults with provenance.

Use the configured representative location for each country. Preserve latitude,
longitude, timezone, and elevation. If elevation is not available in normalized data,
retain the Open-Meteo grid elevation in the weather manifest rather than introducing
an undocumented value.

### Norwegian Windows

For the 21 Norwegian variants with positive glazing but no cardinal orientation:

```text
cardinal area = total generic window area / 4
```

Set:

```text
window_orientation_imputed = true
window_orientation_imputation = "equal_cardinal"
```

Do not exclude these buildings or remove their solar gains.

## Heat-Pump Sizing

Provisionally derive mass flow from design heat loss:

```text
H_total_W_K = H_tr + H_ve
T_room_design_C = 20
T_ambient_design_C = first percentile of reference T_amb across both periods
Q_design_W = H_total_W_K * (T_room_design_C - T_ambient_design_C)

water_delta_T_K = 15
mdot_hp_kg_s = Q_design_W / (4181 * water_delta_T_K)
```

Clamp provisional flow to:

```text
0.05 <= mdot_hp_kg_s <= 0.50
```

Record the unclipped value and whether clipping occurred.

`TODO(Lilli): Review the design-temperature percentile, 15 K water-temperature
difference, and mass-flow bounds before dataset release.`

The same `mdot_hp` must be used by the I4B plant, heat-pump constraints, and MPC.

## Building Validation

For all 191 buildings:

- Instantiate `Building(..., method="4R3C")`.
- Require finite and positive derived capacities and conductances.
- Require `0 <= H_tr_light < H_tr`.
- Generate 15-minute discrete matrices.
- Require finite matrices with the expected `3x3`, `3x1`, and `3x2` shapes.
- Verify at least one overlapping German archetype against the existing handwritten
  I4B parameters.
- Preserve all source IDs and mapping/default provenance.

Do not add 191 Python modules or entries to `BUILDING_NAMES2CLASS`.

## Environment Loading

Make the smallest backward-compatible extension to scalar `RoomHeatEnv`:

```text
building_params: optional explicit parameter dictionary
disturbances: optional precomputed DataFrame
```

Behavior:

- Existing registry-based construction remains unchanged when these arguments are
  absent.
- `building_params` bypasses `BUILDING_NAMES2CLASS`.
- `disturbances` must have a monotonic 15-minute UTC index and exactly `T_amb` and
  `Qdot_gains`.
- Existing HP constraints, action normalization, stepping, rewards, and `info` remain
  unchanged.
- Do not modify `RoomHeatVecEnv` in this ticket.
- Do not add Parquet dependencies to I4B core; benchmark scripts load Parquet and pass
  dictionaries and DataFrames to the environment.

Add one small rollout helper accepting a callable controller. Do not add a controller
base class.

Conceptual interface:

```python
action_C = controller(state, future_disturbances)
normalized = env.normalize_action(action_C)
next_obs, reward, terminated, truncated, info = env.step(normalized)
```

The helper returns canonical transitions plus existing energy and comfort summaries.
This same path must support later closed-loop controller evaluation.

## Disturbance Preparation

For every building and period:

- Read its country's reference weather.
- Convert to a complete 15-minute UTC index.
- Forward-fill each hourly source value over its four quarter-hour intervals.
- Compute deterministic residential internal gains.
- Compute solar gains from irradiance and mapped windows.
- Set `Qdot_gains = internal + solar`.
- Produce exactly 35,040 transitions per complete 365-day period.
- Use identical disturbances and initial states for matched controller variants.

Initialize all `4R3C` states to `20 degC`. Do not add noise or random initialization.

## Collection Policies

All full-period MPC variants re-solve MPC every 15 minutes from the actual current
state. Residuals modify only the first action before standard HP projection.

```text
mpc-nominal
mpc-offset-plus-2K
mpc-offset-minus-2K
mpc-aprbs-low
mpc-aprbs-medium
mpc-aprbs-high
```

Definitions:

```text
mpc-nominal:          residual = 0 K
mpc-offset-plus-2K:   residual = +2 K continuously
mpc-offset-minus-2K:  residual = -2 K continuously
mpc-aprbs-low:        amplitude = 0.5 K
mpc-aprbs-medium:     amplitude = 1.5 K
mpc-aprbs-high:       amplitude = 3.0 K
```

Residual APRBS:

- Uses symmetric positive and negative levels.
- May include a 10% nominal zero level.
- Holds each level for 2-16 steps, equivalent to 30 minutes through 4 hours.
- Uses the same normalized waveform across low, medium, and high variants for a
  matched building and period.
- Is deterministic from building and period identity.
- Has no repeated seed dimension.

Always store the applied action after projection.

## Safe Open-Loop APRBS

Add:

```text
open-loop-aprbs-safe
```

Generate two 28-day episodes per year and location:

- One cold-season episode.
- One shoulder-season episode.
- Select non-overlapping windows deterministically from reference weather.
- Use the same windows across buildings assigned to the same location.

Derive each building's action range from its matched nominal MPC trajectory:

```text
lower = fifth percentile of applied nominal controls
upper = ninety-fifth percentile of applied nominal controls
```

Sample deterministic absolute supply-temperature levels in this range and hold them
for 1-6 hours.

Use a fixed state envelope independent of changing comfort schedules:

```text
18 degC <= T_room <= 28 degC
```

Safety intervention:

```text
T_room >= 27.5 degC -> no heating until T_room < 27 degC
T_room <= 18.5 degC -> upper APRBS level until T_room > 19 degC
```

Require safety intervention on less than 5% of transitions. If exceeded, shrink the
action range toward its midpoint and regenerate the episode.

This policy is safety-filtered and must not be described as perfectly open loop.

## Dataset Layout

```text
i4b-benchmark/
  manifest.json
  buildings.parquet
  trajectories.parquet
  controllers.json
  split.parquet

  ablation-splits/
    nominal-only.parquet
    nominal-offsets.parquet
    nominal-residual-aprbs.parquet
    nominal-open-loop-aprbs.parquet
    all-collection-policies.parquet
    temporal-a-to-b.parquet
    family-fold-0.parquet
    family-fold-1.parquet
    family-fold-2.parquet
    family-fold-3.parquet
    family-fold-4.parquet
    country-BG-holdout.parquet
    country-CY-holdout.parquet
    country-DE-holdout.parquet
    country-FR-holdout.parquet
    country-IE-holdout.parquet
    country-NO-holdout.parquet
    country-PL-holdout.parquet

  transitions/
    part-000.parquet
    part-001.parquet
    ...
```

Transitions remain physically unsplit. Split manifests select trajectories without
copying transition data.

Generated data must not be committed.

## Transition Schema

One row represents:

```text
state_t + applied input_t + disturbance_t -> state_next
```

Use exactly:

```text
trajectory_id       string
timestamp_t         timestamp[UTC]
timestamp_next      timestamp[UTC]

T_room_t            float32
T_wall_t            float32
T_hp_ret_t          float32

T_hp_sup_applied    float32
T_amb_t             float32
Qdot_gains_t        float32

T_room_next         float32
T_wall_next         float32
T_hp_ret_next       float32
```

Do not add controller diagnostics, forecasts, prices, matrices, building metadata, or
TSFM-specific columns.

## Trajectory Metadata

`trajectories.parquet` contains:

```text
trajectory_id
building_id
controller_id
period_id
start_time_utc
end_time_utc
timestep_seconds
row_count
controller_config_hash
```

No seed column is required.

`controllers.json` records the exact controller configuration, implementation
revision, horizon, objective, residual rule, action bounds, and configuration hash.

## Main Split

`split.parquet` contains:

```text
trajectory_id
split
```

Use a deterministic country-stratified building-family assignment:

| Country | Train | Validation | Test |
|---|---:|---:|---:|
| BG | 5 | 1 | 1 |
| CY | 2 | 1 | 1 |
| DE | 9 | 2 | 2 |
| FR | 6 | 2 | 2 |
| IE | 11 | 2 | 2 |
| NO | 6 | 1 | 1 |
| PL | 5 | 1 | 1 |
| Total | 44 | 10 | 10 |

Rank family IDs using a stable SHA-256 rule, not process-dependent hashing.

All variants, periods, episodes, and controllers belonging to one family inherit the
same split.

Ablation manifests may test family folds, future-time generalization, country holdout,
controller distribution, and different trajectory mixtures. They must never split
rows from one trajectory.

## Writing And Compaction

Long generation must be resumable:

1. Write each completed trajectory atomically to an ignored staging cache.
2. Validate row count, timestamp alignment, uniqueness, and finite values.
3. Compact validated trajectories into deterministic Parquet shards.
4. Target approximately 128-256 MB per final shard.
5. Verify compaction preserves every transition exactly once.
6. Write the final manifest and checksums only after successful compaction.

Expected scale:

```text
6 full-period policies: 80,311,680 transitions
safe open-loop APRBS:    2,053,632 transitions
total:                   82,365,312 transitions
estimated storage:       approximately 3-6 GB
```

## Tests

Add offline tests for:

- TABULA-to-I4B formulas.
- Norwegian equal-cardinal imputation.
- Heat-loss sizing and clipping metadata.
- All 191 buildings instantiate as `4R3C`.
- Finite discrete matrices with correct shapes.
- Dynamic `RoomHeatEnv` building and disturbance loading.
- Existing registry-based environment behavior remains unchanged.
- Environment one-step output matches direct I4B simulation.
- Rollout transition alignment.
- Continuous `+2 K` and `-2 K` residuals.
- Deterministic residual APRBS.
- Shared low/medium/high APRBS waveform.
- Safe open-loop APRBS range and intervention behavior.
- No family leakage in any split manifest.
- All controller variants for one family inherit the same main split.
- Compaction preserves keys and row counts.
- A two-building, two-day, all-controller smoke run.

Do not add live network tests.

## Repository Responsibilities

I4B owns:

- TABULA-to-`4R3C` mapping.
- Explicit building/environment loading.
- Reference disturbance preparation.
- Minimal generic rollout helper.
- Dataset schemas and validation.

The existing LEAP-C or leap-c-lab integration owns:

- Parametric MPC execution.
- MPC controller adapters.
- Benchmark generation CLI.
- Residual action policies.

`heat-control` is unchanged in this ticket. It will later consume a TSFM-specific
projection derived from this canonical dataset.

## Non-Goals

- Archived forecast integration.
- Electricity-price objectives.
- Learned TSFM controllers.
- Chronos train/validation/test materialization.
- Adding all buildings to the static Gym registry.
- Extending the vector environment before a demonstrated need.
- Generic controller classes or planner abstractions.
- Dataset publication or upload tooling.

## Acceptance Criteria

- Exactly 191 mapped buildings and 64 families are retained.
- Every mapped building produces a valid I4B `4R3C` model.
- Dynamic buildings and disturbances can be used in `RoomHeatEnv`.
- The same rollout path can evaluate any callable closed-loop controller.
- Six full-period MPC policies and safe APRBS episodes are reproducible and resumable.
- Canonical transitions use exactly the 12-column schema.
- `split.parquet` has no family leakage.
- Ablation manifests reference existing trajectories without data duplication.
- Final Parquet shards contain no duplicate transition keys.
- No forecasts, prices, learned-controller code, or TSFM-specific preprocessing are
  introduced.

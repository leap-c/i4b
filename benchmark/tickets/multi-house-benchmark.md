# Parametric MPC Multi-House Dataset and Chronos-2 Benchmark

## Status

Planned. This is the umbrella ticket across I4B, LEAP-C, and `heat-control`.

## Goal

Create a reproducible, shareable dataset of heterogeneous European houses driven by
locally perturbed optimal MPC actions, then use it to fine-tune Chronos-2 as a
controlled building-dynamics model.

The benchmark must answer two separate questions:

1. Can a model trained on other building families control a previously unseen house?
2. How much does leakage-free calibration data from the deployed house improve control?

The existing `heat-control` setup remains the canonical evaluation plant. Dataset
quality is measured by control sensitivity and closed-loop performance, not only by
forecast error.

## Confirmed Decisions

- Use 15-minute, `4R3C` trajectories initially.
- Use LEAP-C/acados MPC as the behavior policy.
- Parameterize discrete dynamics matrices so one compiled solver covers all `4R3C`
  houses.
- Re-solve MPC after every applied action during dataset generation.
- Perturb actions locally around the nominal MPC solution and retain nominal MPC
  trajectories as a control group.
- Fine-tune `amazon/chronos-2` with LoRA first.
- Use building-family splits before creating training windows.
- Exclude all `sfh_2016_now_*` variants from cross-building training and validation.
- Treat same-house fine-tuning as a separate personalization track with disjoint
  calibration and evaluation periods.
- Publish canonical trajectories as partitioned Parquet with a manifest, schema,
  provenance, and checksums.

## Pilot Scope

Before building the 191-house benchmark, run a vertical slice using only existing I4B
`4R3C` houses:

- one parametric solver checked against fixed solvers for two houses
- approximately one week of nominal and residual-APRBS data from 3-5 non-frozen
  building families
- one minimal Parquet transition table
- zero-shot, nominal-only LoRA, and residual-APRBS LoRA comparisons
- one frozen retained-gain go/no-go decision

Do not add the TABULA catalog, a new I4B dynamics abstraction, generic planner APIs,
full provenance infrastructure, same-house personalization, or publication tooling
before this gate passes.

## Non-Goals for V1

- Embedding Chronos-2 directly inside acados.
- Supporting multiple RC state dimensions in one compiled solver.
- Training a policy to imitate MPC actions.
- Full Chronos-2 fine-tuning before LoRA has been evaluated.
- Adding stochastic occupancy, window opening, shading, sensor failures, or HP
  outages.
- Treating forecast accuracy alone as evidence that a model is suitable for control.

## Post-Gate Source Building Expansion

Use national TABULA/EPISCOPE single-family-house real-example archetypes from:

```text
https://episcope.eu/fileadmin/tabula/public/calc/tabula-calculator.xlsx
```

Initial countries:

- DE: continental
- FR: oceanic
- PL: continental-cold
- NO: Nordic-cold
- CY: Mediterranean-hot
- IE: oceanic-mild
- BG: continental

Include original, standard-refurbishment, and ambitious-refurbishment variants. The
current extract contains 191 variants. Generation code must reproduce the extract
from the source workbook; the workbook itself should not be committed.

### TABULA Mapping

| I4B field | TABULA source | Derivation |
|---|---|---|
| `H_ve` | `h_Ventilation`, `A_C_Ref` | product, W/K |
| `H_tr` | transmission and thermal-bridge columns | sum, W/K |
| `H_tr_light` | window transmission | W/K |
| `area_floor` | `A_C_Ref` | square metres |
| window areas | orientation-specific window columns | square metres |
| `position` | per-country reference city | lat/lon/altitude/timezone |

Parameters absent from TABULA, including `c_bldg`, `height_room`, `g_value`,
`T_offset`, and `mdot_hp`, require documented defaults and a source/provenance field.
Do not silently copy German assumptions to other countries without recording them.

## Leakage-Free Evaluation Tracks

### Track A: Unseen Building Family

This is the primary benchmark.

- Group renovation siblings by the TABULA base archetype, excluding the final
  renovation variant from the group key.
- Assign complete groups to train, validation, or test. Never split renovation
  siblings across partitions.
- Reserve the complete `sfh_2016_now` family:
  - `sfh_2016_now_0_soc`
  - `sfh_2016_now_1_enev`
  - `sfh_2016_now_2_kfw`
- Use `sfh_2016_now_0_soc` with the existing `heat-control` configuration as the
  canonical frozen test plant.
- Do not use frozen-family states, actions, fitted scalers, target encodings, or
  validation losses during training or model selection.

Known weather for the evaluation period is a legitimate future covariate and is not
target leakage. Building identity and physical matrices are dataset metadata, not
Chronos inputs in V1.

### Track B: Same-House Personalization

This post-gate track intentionally uses calibration data from `sfh_2016_now_0_soc` but
must not reuse evaluation targets.

Preferred protocol:

- Fine-tune on a different weather year from the frozen evaluation year.
- Evaluate on the unchanged frozen year and control protocol.

Fallback if only one weather year is available:

- Define a chronological calibration interval and a later evaluation interval.
- Ensure target windows do not overlap.
- Add a gap at least as long as the maximum context plus prediction horizon.
- Report this result as personalization, never as unseen-building generalization.

The exact calibration year or fallback interval must be fixed in the release manifest
before running experiments.

## Pilot Matrix Source

I4B already computes exact zero-order-hold matrices in
`i4b/core/integrators.py::linear_discretize_batch` and uses heterogeneous matrix
batches in `i4b/core/sim.py::JaxSimulator`.

Do not make a new I4B dynamics abstraction a prerequisite for the pilot. The existing
LEAP-C I4B OCP already derives `Ad`, `Bd`, and `Ed` while constructing a fixed-building
solver. Extract that calculation into a small LEAP-C helper and use its values as the
defaults and runtime parameters of the parametric OCP.

For the current `4R3C` pilot:

```text
x[k+1] = Ad x[k] + Bd u[k] + Ed d[k]
d       = [T_amb, Qdot_gains]
u       = [T_hp_sup]
```

I4B names the disturbance matrix `Cd` and returns one-house shapes `Ad=(3, 3)`,
`Bd=(3,)`, and `Cd=(3, 2)`. The LEAP-C OCP may retain its `Ed` name, but its conversion
must explicitly set `Bd=(3, 1)` and `Ed=Cd`. `mdot_hp` is already embedded when `Bd`
is derived; the separately packed value used by HP constraints and cost must match it.

Do not inspect `JaxSimulator._linear_mats`, which is private. A stable I4B public matrix
export may be added after the pilot if it materially simplifies full benchmark
generation. If added, prefer re-exporting the existing function over introducing a
dataclass, affine term, metadata container, or future-model interface.

### Pilot Matrix Verification

- Test `4R3C` only.
- Compare a parametric solve with the existing fixed-building solve.
- Compare a two-house heterogeneous batch with two separate fixed solvers.
- Check state, control, and disturbance ordering explicitly.
- Use float32 for the pilot and document tolerances.
- Pack matrices in the column-major order expected by CasADi and test all 19 values
  (`Ad`, `Bd`, `Ed`, and `mdot_hp`) with asymmetric matrices; defer release hashes.

## Parametric LEAP-C OCP

Use one compiled OCP per state dimension/model method. For `4R3C`, every house shares
the same symbolic structure while runtime values differ.

### Constant Runtime Parameters

Pass through acados `p_global`:

- `Ad`
- `Bd`
- `Ed`
- `mdot_hp`

LEAP-C's current parameter manager calls the `p_global` interface `learnable`. These
matrix values are runtime parameters, not optimization variables for V1; callers must
pass detached tensors and must not request sensitivities with respect to them.

Matrix flattening needs a dedicated round-trip test. The current parameter manager
notes unresolved CasADi versus NumPy row/column-major ordering for matrix values.

### Stage-Wise Runtime Parameters

Pass through acados stage parameter `p`:

- `T_amb`
- `Qdot_gains`
- lower and upper comfort bounds
- grid price or objective weight

Use stage order `[T_amb, Qdot_gains, T_set_lower, T_set_upper, grid_signal]` with
batched shape `(B, N+1, 5)`.

`Qdot_gains` is a known disturbance forecast for dataset generation and should no
longer be represented as a learned policy output in the I4B planner.

### Planner Output for the Pilot

Use the existing `I4bPlanner.forward` return values and solver context directly. They
already provide the first action, state and control trajectories, objective, status,
statistics, and warm start. A generic `Planner.solve(...) -> Plan` interface is
deferred until a second planner implementation creates a concrete need.

## Action Semantics

Use physical supply temperature in degrees Celsius at planner and dataset boundaries.
Do not mix normalized Gym actions with physical actions in the collector.

Record three values:

- `T_hp_sup_mpc`: first action from the nominal optimal plan
- `T_hp_sup_requested`: nominal plus sampled perturbation
- `T_hp_sup_applied`: action after projection and heat-pump constraints

The MPC and plant must share the same definition of no-heating behavior, return-flow
constraint, supply-temperature bounds, `mdot_hp`, and thermal-power limit. Do not apply
the heatcurve-specific building `T_offset` to MPC actions.

Before collection, resolve and test the current `[5, 65]` acados bound versus I4B's
normalized `[20, 65]` action mapping. The canonical physical plant should permit
no-heating through the HP constraint without an undocumented controller offset.

## Closed-Loop Local Excitation

At every 15-minute decision step:

1. Read the actual current plant state.
2. Build the disturbance and objective forecast.
3. Solve the nominal MPC problem.
4. Read the current seeded APRBS residual.
5. Scale or project the resulting first action into action and HP constraints.
6. Apply that action to the plant.
7. Log the complete transition and collector diagnostics.
8. Warm-start and re-solve MPC from the resulting state.

Never execute a full perturbed open-loop plan. Closed-loop re-solving is what keeps the
data near the locally optimal state-action manifold.

### Perturbation Pilot

Use a constrained residual APRBS as the primary excitation strategy:

```text
T_hp_sup_requested[t] = T_hp_sup_mpc[t] + alpha[t] * aprbs[t]
```

Generate the seeded APRBS schedule per trajectory with symmetric positive and negative
amplitudes and random dwell times. Keep the residual fixed for multiple 15-minute plant
steps, but re-solve the nominal MPC at every step before adding the current residual.
Do not construct or score a perturbed horizon in the pilot: only the first projected
action is applied, and feedback remains active throughout the rollout.

Pilot small, medium, and wide amplitude bands around approximately 0.5 K, 1.5 K, and
3 K, with dwell times spanning approximately 30 minutes to 4 hours. Scale amplitudes
per house using the available HP/action margin. Report realized objective degradation
against matched nominal trajectories and retained control gain before recommending
post-gate release values; do not hard-code a relative cost threshold before observing
nominal objective scales. Use correlated Gaussian
residuals only as a later ablation, not as the primary dataset policy.

Each accepted sample must satisfy hard HP/action constraints. Log:

- requested and realized perturbation in K
- hard HP/action constraint margins
- whether the perturbation was scaled, clipped, rejected, or zero
- perturbation generator, band, hold duration, and RNG seed

Include an unperturbed nominal fraction. The release should support an ablation between
pure MPC data and locally perturbed MPC data.

## Pilot Parquet Contract

Use `pyarrow` and Parquet for the vertical slice, but keep the pilot table deliberately
small. One row represents a control decision at `t` and the state reached at
`timestamp_next`:

```text
trajectory_id
building_id
building_family_id
split
timestamp_t
timestamp_next
T_room_next
T_hp_ret_next
T_hp_sup_applied
T_amb_t
Qdot_gains_t
T_hp_sup_mpc
T_hp_sup_requested
perturbation_band
perturbation_disposition
```

This table is sufficient for alignment, leakage, fine-tuning, and excitation
ablations. A mandatory trajectory sidecar keyed by `trajectory_id` must record the
initial full state, seed, method, building/config identity, weather interval, matrices,
collector settings, and source revisions needed for deterministic replay. Detailed
planner diagnostics may be logged separately; these auxiliary records do not belong to
the Chronos input contract. The larger canonical schema below is a post-gate release
requirement.

## Canonical Dataset Schema

Store transitions without timestamp ambiguity. Each row represents a decision at `t`:

### Identity and Provenance

```text
trajectory_id
transition_index
timestamp_t
timestamp_next
building_id
building_family_id
country
split
evaluation_track
weather_source
weather_year
method
timestep_seconds
i4b_revision
leap_c_revision
matrix_hash
config_hash
seed
```

### State and Disturbance

```text
T_room_t
T_wall_t
T_hp_ret_t
T_amb_t
Qdot_gains_t
T_room_next
T_wall_next
T_hp_ret_next
```

### Control and Cost

```text
T_hp_sup_mpc
T_hp_sup_requested
T_hp_sup_applied
action_perturbation_requested_K
action_perturbation_applied_K
COP
Qdot_th_W
P_el_W
E_el_kWh
T_room_set_lower
T_room_set_upper
```

### Planner Diagnostics

```text
mpc_objective
perturbed_predicted_objective
objective_degradation_absolute
objective_degradation_normalized
solver_status
solver_iterations
solver_time_seconds
perturbation_band
perturbation_generator
perturbation_hold_steps
perturbation_disposition
comfort_margin_min
hp_action_margin_min
```

Store full nominal plans only in an optional plan-level side table keyed by
`trajectory_id`, `transition_index`, and horizon step. Avoid repeating 96-step plans in
every transition row of the primary table.

## Chronos-2 Training View

Derive a regular time-series view from pilot or canonical transitions. The conversion
must perform this exact one-step alignment independently within each trajectory:

```text
item_id     = trajectory_id
timestamp   = timestamp_next
T_room      = T_room_next
T_hp_ret    = T_hp_ret_next
T_hp_sup    = T_hp_sup_applied
T_amb       = T_amb_t
Qdot_gains  = Qdot_gains_t
```

Here `T_hp_sup_applied`, `T_amb_t`, and `Qdot_gains_t` are the inputs applied over
`[timestamp_t, timestamp_next)` and are stamped at `timestamp_next` beside the state
they produced. Never shift across `trajectory_id` boundaries. Use the same convention
when constructing known-future inference covariates.

V1 model inputs must match `heat-control` inference:

- Targets: `T_room`, `T_hp_ret`
- Past covariates: `T_hp_sup`, `T_amb`, `Qdot_gains`
- Known-future covariates: `T_hp_sup`, `T_amb`, `Qdot_gains`

Do not use `T_hp_sup_mpc`, perturbation labels, setpoints, grid price, building ID, RC
parameters, or discrete matrices as Chronos inputs in V1. They remain available for
analysis and later ablations.

Before Chronos preprocessing, project strictly to these columns so metadata cannot be
treated as model covariates:

```text
item_id
timestamp
T_room
T_hp_ret
T_hp_sup
T_amb
Qdot_gains
```

Use this Chronos 2.3.1 preprocessing contract:

```python
from_data_frame(
    df,
    target_columns=["T_room", "T_hp_ret"],
    prediction_length=96,
    known_covariates_names=["T_hp_sup", "T_amb", "Qdot_gains"],
)
```

Pass trajectories only after the building split; `Chronos2Dataset` performs random
training-window sampling internally. Validation uses the terminal prediction window
of each item, so create several non-overlapping validation `item_id` segments per
family if broader coverage is needed. Do not add a custom dataset, data loader,
trainer, scaler, or window sampler.

Initial fine-tuning settings:

- model: `amazon/chronos-2`
- API: `Chronos2Pipeline.fit`
- argument: `finetune_mode="lora"`
- dependencies: `peft` and `pyarrow`; absence of `peft` is an error because Chronos
  otherwise falls back to full fine-tuning
- prediction length: 96 steps / 24 hours
- context length: at most 2,048 steps; 21 days is 2,016 steps
- minimum past: at least 96 steps
- validation inputs: complete held-out validation building families
- checkpoint selection: validation loss and control diagnostics from validation
  families only, never the frozen test plant

Chronos batch size counts target and covariate variates, not only houses. Account for
five variates per trajectory when sizing batches.

## Evaluation

### Post-Gate Forecasting

- MAE for `T_room` and `T_hp_ret`
- quantile loss and empirical interval coverage
- horizons of 1, 6, 12, and 24 hours
- metrics stratified by building family, country, heating activity, and perturbation
  band

### Control Sensitivity

Reuse the `heat-control` sensitivity and APRBS diagnostics:

- simulated versus predicted response to supply-temperature offsets
- RMS retained control gain from `gain_summary.retained()` by forecast horizon
- cosine/alignment metrics

Expose checkpoint `model_id` and an experiment tag in evaluation commands and cache
filenames so zero-shot and fine-tuned runs cannot reuse or overwrite each other. Use
the same accepted, unsaturated plant sweep for every checkpoint comparison.

This is the pilot gate. Existing experiments show that low forecast error can coexist
with control blindness under pure closed-loop MPC context.

### Post-Gate Closed-Loop Control

Compare:

- I4B/acados MPC oracle
- zero-shot Chronos-2 plus CEM
- Chronos-2 LoRA trained on pure MPC trajectories
- Chronos-2 LoRA trained on locally perturbed MPC trajectories
- same-house LoRA personalization, reported separately

Report energy, discomfort, maximum comfort violation, objective value, planning time,
solver/model failures, and control-gain retention.

Freeze the `heat-control` replanning interval in the benchmark configuration. Its
README describes six-hour replanning while `LoopConfig` currently defaults to 15
minutes; benchmark results must not depend on this unresolved discrepancy.

## Post-Gate Release Layout

```text
dataset/
  manifest.json
  schema.json
  checksums.txt
  transitions/
    split=train/country=.../*.parquet
    split=validation/country=.../*.parquet
    split=test/country=.../*.parquet
    split=personalization/country=DE/*.parquet
  plans/                         # optional
  configs/
  provenance/
```

Publish the immutable release through Hugging Face Hub and archive a versioned copy
with a DOI through Zenodo.

## Repository Responsibilities

### Working Repositories and Fork Flow

| Workstream | Local checkout | Development repository | Upstream target |
|---|---|---|---|
| I4B core, buildings, and matrix export | `/home/jas/projects/i4b` | `leap-c/i4b` | `lfrison/i4b` after validation |
| Parametric MPC and trajectory collector | separate worktree from `/home/jas/projects/leap-c` | `dirkpr/leap-c`, based on its `i4b` branch | `leap-c/leap-c` |
| Chronos fine-tuning and evaluation | `/home/jas/projects/heat-control` | `JasperHoffmann/heat-control` (private) | `MazenAmria/heat-control` |

Repository-specific branch flow:

- Create the I4B benchmark branch from the current `gpu-leap` line and push it to the
  `leap` remote (`leap-c/i4b`), not the read-only `origin` remote
  (`lfrison/i4b`).
- Base LEAP-C work on `dirkpr/leap-c:i4b` commit
  `04478bb1f66882d326cbdc7c4ed4db5327ac1638`, where the existing I4B OCP and planner
  live. Fetch that remote and use a new branch rather than committing to `i4b`
  directly. Develop in a separate worktree because the existing
  `/home/jas/projects/leap-c` checkout is used for unrelated work.
- Base `heat-control` changes on `feat/dynamics-models`, which contains the existing
  Chronos wrapper, sensitivity diagnostics, APRBS experiment, and CEM loop. Push
  experiment and review branches to the private `JasperHoffmann/heat-control`
  `origin`; keep `MazenAmria/heat-control` as `upstream` for later cleaned PRs.
- Update `heat-control/external/i4b` only after the required I4B changes are committed;
  the submodule URL already points to `leap-c/i4b`.
- Do not put benchmark work in `leap-c/leap-c-developer`; it is an unrelated HVAC
  developer/demo repository.
- Do not fork or modify Amazon's Chronos repository for V1. Consume its public
  `Chronos2Pipeline.fit` API from `heat-control`.

### I4B

- Pilot: provide the existing building models and physical simulator without a required
  production change.
- Post-gate: add reproducible TABULA extraction and the expanded catalog.
- Post-gate: expose existing matrix generation publicly only if the full generator
  needs it.

### LEAP-C `i4b` Branch

- Update stale I4B imports and APIs.
- Parameterize `Ad`, `Bd`, `Ed`, and `mdot_hp`.
- Move `Qdot_gains` from learned `p_global` to stage-wise runtime parameters.
- Add matrix-layout and heterogeneous-batch tests.
- Implement closed-loop residual APRBS collection and pilot Parquet generation using
  existing planner outputs.

### `heat-control`

- Consume pilot and released Parquet without importing LEAP-C.
- Build leakage-safe Chronos inputs.
- Add LoRA fine-tuning through `Chronos2Pipeline.fit`; reuse existing checkpoint
  loading.
- Make sensitivity evaluation checkpoint- and tag-aware.
- Freeze unseen-building and same-house evaluation protocols.
- Run sensitivity evaluation in the pilot; forecasting and closed-loop CEM are
  post-gate benchmark work.

## Pilot Work Packages

### WP1: Parametric `4R3C` Solver

- [ ] Fetch and branch from the pinned `dirkpr/leap-c:i4b` commit.
- [ ] Update LEAP-C to current I4B package paths and environment APIs.
- [ ] Extract the existing fixed-building `Ad`, `Bd`, and `Ed` calculation into a
  small LEAP-C helper.
- [ ] Move `Qdot_gains` to stage-wise non-learnable parameters.
- [ ] Parameterize one `4R3C` OCP with `Ad`, `Bd`, `Ed`, and `mdot_hp`.
- [ ] Define one explicit flat packing order and round-trip asymmetric matrices in a
  test.
- [ ] Freeze state, control, objective, and replay tolerances before comparison runs.
- [ ] Verify one parametric solve against the existing fixed-building OCP.
- [ ] Verify heterogeneous batched solves against separate solvers.

### WP2: Residual APRBS Parquet Pilot

- [ ] Freeze one planner/collector/evaluation contract for physical action bounds,
  no-heating, HP projection, and `mdot_hp`, without adding heatcurve `T_offset`.
- [ ] Implement nominal closed-loop MPC collection using existing planner outputs.
- [ ] Implement seeded constrained residual APRBS and projection.
- [ ] Confirm the LEAP-C collector dependency explicitly includes `pyarrow` rather
  than relying on a transitive install.
- [ ] Write the minimal pilot contract with `pyarrow`.
- [ ] Write the mandatory deterministic-replay trajectory sidecar.
- [ ] Generate one-week nominal and excited trajectories for 3-5 existing non-frozen
  building families.
- [ ] Match nominal and excited cohorts by building, weather interval, sample count,
  fine-tuning steps, and seeds.
- [ ] Pre-register the minimum applied-offset variation required for an excitation
  cohort to be accepted.
- [ ] Validate transition alignment and deterministic replay.
- [ ] Characterize candidate perturbation bands and recommend post-gate values from
  observed objective and constraint statistics.

### WP3: Chronos-2 LoRA Pilot

- [ ] Add explicit `peft` and `pyarrow` Chronos optional dependencies.
- [ ] Add one strict pilot-transition to Chronos conversion command.
- [ ] Test exact next-state alignment and no shifting across trajectories.
- [ ] Test strict seven-column projection and metadata exclusion.
- [ ] Test family-level train/validation/frozen-test disjointness.
- [ ] Test that all three controls/disturbances are marked known-future covariates.
- [ ] Check that `peft` imports before `fit` and fail instead of silently running a
  full fine-tune.
- [ ] Record a zero-shot baseline.
- [ ] Fine-tune LoRA on nominal-only pilot data.
- [ ] Fine-tune LoRA on nominal plus residual-APRBS pilot data.

### WP4: Checkpoint-Aware Gain Evaluation

- [ ] Add `--model-id` and `--tag` to sensitivity evaluation.
- [ ] Include the tag in caches and result filenames.
- [ ] Reuse one accepted, unsaturated plant sweep for all checkpoints.
- [ ] Report RMS retained control gain, cosine, and alignment by horizon.
- [ ] Compare zero-shot, nominal-only LoRA, and residual-APRBS LoRA.
- [ ] Before running checkpoint comparisons, pre-register the target/horizon
  aggregation, number of sweeps or seeds, minimum improvement, and allowed alignment
  regression used by the gate.
- [ ] Make and record the pilot go/no-go decision.

## Go/No-Go Gate

Proceed to benchmark-scale work only if locally excited fine-tuning meets the
pre-registered WP4 threshold over both zero-shot and nominal-only fine-tuning for:

- RMS retained control gain
- control-response sign/cosine alignment

Record a negative result and stop if the pilot does not meet the gate.

## Deferred Benchmark-Scale Todos

- [ ] Add and validate the 191-building TABULA catalog.
- [ ] Freeze deterministic family-level train/validation/test assignments.
- [ ] Add a public I4B matrix export only if full generation requires it.
- [ ] Generate full trajectories across buildings, seasons, and excitation bands.
- [ ] Finalize canonical Parquet partitions, manifests, schema, hashes, and checksums.
- [ ] Run leakage, duplicate, replay, and provenance checks.
- [ ] Freeze and run closed-loop CEM evaluation.
- [ ] Define and evaluate same-house personalization as a separate track.
- [ ] Reproduce metrics from immutable configurations.
- [ ] Publish dataset, model adapters, revisions, and model cards.
- [ ] Archive the release and document known limitations.

## Pilot Acceptance Criteria

- A single compiled `4R3C` acados solver produces matching solutions for at least two
  heterogeneous houses when supplied different runtime matrices.
- Matrix packing is covered by an asymmetric round-trip test.
- Pilot Parquet rows replay deterministically from recorded configurations and seeds.
- Train, validation, and frozen-test `building_family_id` values are pairwise disjoint
  before Chronos windows are created.
- No `sfh_2016_now` family trajectory contributes to Track A training, validation,
  preprocessing statistics, or model selection.
- The pilot contains meaningful applied-action variation without hard
  constraint violations.
- Chronos training uses applied controls, not nominal controls, as dynamics covariates.
- Residual-APRBS fine-tuning is compared with both zero-shot and nominal-only
  fine-tuning.
- Zero-shot and fine-tuned evaluations use distinct tags and the same plant sweep.
- The go/no-go result is based on the pre-registered RMS retained-gain and alignment
  thresholds.

## Remaining Decisions Before Full Generation

- Select the same-house calibration year, or define the chronological fallback split.
- Freeze the CEM replanning interval for the canonical evaluation.
- Freeze release perturbation bands from the pilot recommendation.
- Confirm whether weather locations vary with country or are controlled as a separate
  benchmark factor.

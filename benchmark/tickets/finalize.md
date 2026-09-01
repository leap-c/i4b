# Finalize The I4B TSFM Benchmark

## Current Decision

Keep the current benchmark collection. Its six MPC policies use a 12-step, three-hour
planning horizon, while the intended TSFM prediction and control horizon is 96 steps,
or 24 hours. Full-year trajectories remain suitable for learning 24-hour dynamics, but
the collection action distribution may underrepresent day-ahead preheating.

Do not rerun the complete corpus solely because of this horizon mismatch. First compare
three-hour and 24-hour MPC on 10-20 representative buildings spanning countries,
families, thermal mass, and renovation levels. Rerun or supplement collection only if
the 24-hour controller materially changes action distributions, comfort, energy use, or
preheating. A price-aware objective is a separate strong reason to collect 24-hour MPC
trajectories because the current objective has no day-ahead price signal.

## Dataset Completion

- Resume the 18 incomplete building-period MPC jobs. The first production pass stopped
  at 364/382 jobs after acados returned `ACADOS_MINSTEP` for
  `mpc-offset-minus-2K` at step 28,092 in one job.
- Diagnose or add a deterministic retry policy for the failing solve; do not silently
  substitute an action without recording the behavior.
- Generate one full-period `open-loop-aprbs` trajectory per building-period using the
  fixed 1.5 K excitation policy.
- Require exactly six MPC and one open-loop APRBS trajectory, each with 35,039 rows, for
  all 382 building-period pairs before compaction.
- Validate final shards, split manifests, controller metadata, and transition counts.

## Forecast Archive

Acquire archived ECMWF IFS forecasts separately from the canonical transition corpus:

- Seven representative country locations.
- Period A and period B.
- 00Z and 12Z initialization runs; 06Z and 18Z are unavailable for the complete range.
- A 48-hour horizon, retaining initialization time, valid time, and lead time.
- Temperature plus GHI, DNI, and DHI with raw responses and normalized Parquet files.

The configured archive contains 1,460 runs and is estimated by the downloader at about
20,000 weighted Open-Meteo calls, potentially requiring multiple days on the free tier.
Run `--max-forecast-runs 1` first to validate current availability, schema, size, and
credentials. The full acquisition is resumable and must run as a supervised background
job without `--force`.

Do not flatten forecast leads into channels. Represent forecasts as future-time tokens
with shared weather features, issue time, valid time, and lead time. Use only runs issued
at or before each decision time.

## Forecast Integration

Keep measured and forecast ambient temperature as separate channels:

- `T_amb_measured` is the authoritative historical and current outdoor-temperature
  measurement.
- `T_amb_forecast` is an ECMWF future covariate and must not overwrite the measurement.
- Include forecast issue time, valid time, lead time, forecast age, and missing-value or
  fallback masks.
- Select the newest forecast whose issue time is at or before the decision time. Begin
  future forecast tokens strictly after the decision timestamp.
- Map each building to its configured country location. Convert forecast GHI, DNI, and
  DHI into building-specific solar gains with the same physical transformation used by
  `prepare_disturbances`, then add only internal gains that are known at decision time.
- Keep forecasts in a normalized location/run table and join them to trajectories by
  location and time. Do not duplicate all forecast rows into every trajectory.

The lead-zero ECMWF temperature was compared with the normalized historical weather at
the same UTC timestamp for all 1,455 available runs and seven locations. Every forecast
file aligned to the correct historical timestamp. Four lead-zero temperature values per
location were null. The remaining values were close but not identical, confirming that
the historical series and individual forecast runs are distinct products and should not
share a channel.

| Location | Lead-zero MAE (C) | RMSE (C) | Correlation |
| --- | ---: | ---: | ---: |
| Sofia | 0.63 | 0.99 | 0.995 |
| Nicosia | 0.49 | 0.87 | 0.996 |
| Freiburg | 0.56 | 0.88 | 0.994 |
| Paris | 0.62 | 1.00 | 0.990 |
| Dublin | 0.33 | 0.51 | 0.994 |
| Oslo | 0.53 | 0.80 | 0.996 |
| Warsaw | 0.56 | 0.93 | 0.995 |

The aggregate lead-zero MAE is approximately 0.53 C. At the history/forecast boundary,
retain the latest measured `T_amb` and expose later ECMWF values as future tokens rather
than replacing the measurement with forecast lead zero.

## Evaluation Views

Keep the canonical full-state dataset and derive views without regeneration:

- Oracle: all states and realized disturbances.
- Sensor-realistic: `T_room`, `T_hp_ret`, applied action, ambient temperature, and
  calendar features; hide `T_wall` and exact gains.
- Minimal: `T_room`, applied action, ambient temperature, and calendar features.
- Forecast-conditioned: historical observations or estimates plus archived future
  forecast tokens and source/lead indicators.

Add sensor noise, bias, missingness, actuator delay, and forecast-error experiments as
derived views. Do not overwrite canonical transitions.

## Evaluation Plan

- Compare frozen TSFMs against persistence, local ARX/state-space models, small neural
  baselines, nominal MPC, and known `4R3C` dynamics where appropriate.
- Evaluate one-step and 24-hour rollout RMSE by intervention and state variable.
- Measure zero-shot and context-only performance at 1, 3, 7, 28, and 90 days.
- Run closed-loop evaluation on held-out families and period B, reporting comfort,
  energy, stability, action clipping, and inference latency.
- Inspect learned thermal lag, decay rates, intervention response, residual
  autocorrelation, and qualitative measured-versus-predicted trajectories.

## Release Decision

Treat the synthetic benchmark as mechanistic evidence for transfer and data efficiency,
not as proof of safe real-building control. Before a real-world claim, repeat evaluation
with realistic observability and archived forecasts, then use independent real data and
a supervised shadow-mode pilot.

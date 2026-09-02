# What the benchmark measures

Two evaluations over one corpus. **Open loop** asks a model to predict and scores accuracy and
control response; **closed loop** asks a controller to act and scores what acting cost. Both hand
the method the same observation, so a model written for one runs in the other unchanged.

The corpus itself is specified in `DATA_SCHEMA.md`; which problems a run covers is in
`../src/i4b_bench/config/README.md`. This file is the part that must not drift: what is handed
over, what is computed from it, and what the numbers mean.

## The observation

Built by one function, `i4b_bench.observation.build_observation`, in both loops.

```python
{
    "state":    {channel: float},              # STATE_CHANNELS[view]
    "history":  {"timestamp": datetime64[n], channel: float32[n]},   # history_channels(view)
    "forecast": {"timestamp": datetime64[h], channel: float32[h]},   # DISTURBANCE_CHANNELS[view]
}
```

`n` is at most the context length and `h` is the planning horizon. A history row pairs a state
with **the input that produced it** — the recorded corpus stores `state_t + input_t -> state_t+1`,
and the shift is applied on the way in, so `history["T_hp_sup_applied"][-1]` is the action that
led to `history["T_room"][-1]`.

Candidate future controls are deliberately *not* in the observation. Open loop passes them
alongside, which keeps the observation identical between the loops and makes it a description of
what happened rather than of what to try.

### Views

| | `perfect` | `realistic` |
| --- | --- | --- |
| state | `T_room`, `T_wall`, `T_hp_ret` | `T_room`, `T_hp_ret` |
| disturbances | `T_amb`, `Qdot_gains` | `T_amb`, `ghi`, `dni`, `dhi` |

`perfect` is the oracle: the wall node and the building-specific heat gain the simulator used,
neither measurable on a real site, together making the 4R3C dynamics fully observable.
`realistic` is what an installation could be instrumented for. The gap between a method's two
scores is the price of the missing information.

`use_forecast` is orthogonal to the view. `False` slices the realised weather from
`exogenous.parquet`; `True` assembles the horizon from the archived forecast runs in
`forecasts.parquet` — the most recent initialisation at or before the current step, extended by
the next when it runs out. `forecast_correction` in `[0, 1]` pulls the archived horizon toward
the current sensor reading; the closed-loop setting uses 0.5, and the open-loop setting uses 0,
because forecast error is part of what it measures.

## Open loop

```python
Predictor = Callable[[list[dict], list[np.ndarray]], list[dict[str, np.ndarray]]]
```

`controls[i]` is `(probes, horizon)` and `returns[i]` maps a channel name to `(probes, horizon)`.
Predictors are batched because prediction is pure.

For each window the harness drives the plant from the window's anchor under `probes` control
trajectories, evenly spaced over `±probe_amplitude` about the plan the corpus applied and clipped
to the actuator range `[5, 65] C`. Everything else — weather, initial state, building — is held
fixed, so the probes differ **only** in the control.

**The predictor is given the control the plant applied, not the control that was requested.**
`check_hp` collapses the supply temperature whenever the pump idles, so the requested residual is
frequently not the intervention: across a year-spread set only ~40% of the requested spread was
realised. Scoring a model's answer about a requested plan against the plant's response to a
clipped one inflates gain by the reciprocal of that fraction. `realized_share` reports it per
window.

### Metrics

All are computed on `T_room` (`SCORED_CHANNEL`); a predictor may return more channels and they
are ignored, but omitting this one is an error.

Let `A` be the plant's response and `P` the model's, both `(probes, horizon)`, and let `i` be the
nominal probe (the middle one, whose offset is zero).

| column | definition |
| --- | --- |
| `mae_K` | `mean(abs(P[i] - A[i]))` — point accuracy on the nominal plan |
| `bias_K` | `mean(P[i] - A[i])` — signed, so a systematic offset is visible |
| `response_K` | `sqrt(mean((A - mean_probes(A))^2))` — how far the probes moved the *plant* |
| `gain` | slope of the model's deviation on the plant's, below |

```
plant = A - A.mean(axis=0)          # deviations about the mean across probes
model = P - P.mean(axis=0)
gain  = sum(model * plant) / sum(plant ** 2)
```

Taking deviations about the probe mean cancels everything the probes shared — weather, the
starting state, any constant bias — so only the response to *differences* in control survives.
`gain = 1.0` moves exactly as the plant does; `0.0` ignores the control entirely.

Numerator and denominator are exposed separately (`gain_terms`) so windows can be **pooled** by
summing each before dividing. Averaging per-window gains instead weights a window that barely
moved as heavily as one that moved a lot.

`gain` is `NaN` when `response_K < MIN_RESPONSE_K` (1e-3 K): the pump was clipped or idle
throughout, every probe produced the same trajectory, and the ratio would be arbitrarily large.

## Closed loop

```python
Controller = Callable[[dict], tuple[float, dict | None]]
```

Returns a supply-temperature setpoint in Celsius, and optionally the plan behind it. **The plan is
recorded but not scored.** A plan's per-channel error is dominated by weather — the control
accounts for a few percent of a room's daily movement — so it does not discriminate control
quality. Measuring that needs a counterfactual, which is what the open loop does.

| column | definition |
| --- | --- |
| `energy_kwh` | heat-pump electrical energy over the episode, summed from the plant's `E_el` |
| `comfort_violation_degree_hours` | `sum(max(20 - T_room, 0) * dt/3600)` over the episode |
| `planning_seconds_mean` | wall-clock per `controller()` call |

The comfort band is `[20, 26] C`, and the metric is **one-sided**: it counts undercooling only.
Overheating is computed by the plant (`dev_pos_sum`) but not reported, because the heat pump can
only add heat — penalising it for a warm room would score the weather.

Planning time is reported because a controller that wins on comfort by thinking for a minute a
step has not solved the problem.

## What must be held fixed for two results to be comparable

The settings dataclasses, one per loop, and every field of them is written out in the YAML rather
than defaulted: the split, the view, `use_forecast`, the horizon, the context lengths, the probe
count and amplitude, `forecast_correction`, and the named problem instances themselves.

The plant is pinned to the `legacy` integrator. The corpus was generated with it, and evaluating
against the corpus under a different integrator would score a controller in a slightly different
world than the recorded baselines it is compared to.

## Known limitations

`IMPLICIT_ASSUMPTIONS.md` tracks what is true but unenforced — the 15-minute step assumed in
several places independently, and the two superseded disturbance conventions corpus v2 was built
under.

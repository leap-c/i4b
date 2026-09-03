# What the benchmark measures

Two evaluations over one corpus. **Open loop** asks a model to predict and scores accuracy and
control response; **closed loop** asks a controller to act and scores what acting cost. Both hand
the method the same observation, so a model written for one runs in the other unchanged.

The corpus itself is specified in `DATA_SCHEMA.md`; which problems a run covers is in each
evaluation set's `definition.yaml` (open loop) and in `../src/i4b_bench/config/README.md`
(closed loop). This file is the part that must not drift: what is handed over, what is computed
from it, and what the numbers mean.

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

The forecast follows the same rule, one step the other way. The plant integrates
`x_t + u_t + d_t -> x_(t+1)`: the disturbance driving a transition is the one at the **start** of
the interval. So a horizon published at decision time `t` carries the interval inputs
`d_t ... d_(t+h-1)`, while its timestamps name the **target states** `t+1 ... t+h` — `forecast[i]`
is the weather that, together with `control[i]`, produces the state stamped `timestamp[i]`. Row
zero is the current measurement rather than a prediction; at `t` it has been observed. This was
off by one until the case artifact was introduced: `d_(t+1)` was published against the transition
driven by `u_t`, handing every model a horizon shifted out of the world the plant ran in.

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
from i4b_bench import eval_benchmark_open_loop

results = eval_benchmark_open_loop(predictor, evaluation_set="benchmark-v1")
```

```python
Predictor = Callable[[list[dict], list[np.ndarray]], list[dict[str, np.ndarray]]]
```

`controls[i]` is `(plans, horizon)` and `returns[i]` maps a channel name to `(plans, horizon)`.
Predictors are batched because prediction is pure; batch size is a memory knob and does not
change a number.

### The evaluation set

An open-loop run scores a **compiled evaluation set**, not the corpus. The set is a definition,
the cases compiled from it, and a manifest fingerprinting both — see `DATA_SCHEMA.md` for the
directory and the case schema. Evaluation loads the cases, slices the history to each context
length, calls the predictor, and computes the metrics. It runs no simulation, transforms no
forecast, and joins nothing.

That separation is what makes a number reproducible. The plant is driven exactly once, when the
set is compiled; every later evaluation is arithmetic on the same stored trajectories. A new
ablation is a new artifact, not a flag.

### The probes

For each case the compiler drove the plant from the anchor under **five** control trajectories.
Everything else — weather, initial state, building — is held fixed, so the plans differ **only**
in the control.

| `plan_role` | trajectory |
| --- | --- |
| `nominal` | the plan the corpus applied |
| `offset_minus` / `offset_plus` | nominal ∓ `probe_amplitude`, held over the whole horizon |
| `aprbs_minus` / `aprbs_plus` | nominal ∓ one pseudo-random waveform, held `probe_hold_steps` at a time |

All five are clipped to the actuator range `[5, 65] C`.

The set is a proxy for one question: is this model worth putting inside an MPC? That takes two
different perturbations. The constant offsets measure the steady-state response — shift the
supply temperature for a day and the room has to follow. The APRBS pair measures timing, which a
model can get wrong while having exactly the right daily gain, and timing is what an MPC exploits
when it moves heat around a price or comfort window. Pure offsets miss a model whose dynamics are
wrong but whose integral is right; pure APRBS is more aggressive than anything an MPC would
choose, and mixes the two questions together.

Each perturbation appears with **both signs**, so the deviations `gain` regresses are balanced
about the nominal and a model's asymmetry does not read as a slope. The waveform is derived
deterministically from the case's identity, so rebuilding the artifact reproduces it.

**The predictor is given the control that was requested.** The actuator does not always deliver
it — `check_hp` collapses the supply temperature whenever the pump idles — and the case artifact
keeps what it did deliver as `applied_control`, for provenance and diagnostics. Handing that over
would tell a model how the plant reacted to the intervention as part of the question about how
the plant reacts to the intervention: the applied supply temperature depends on the true future
return temperature, so it leaks the future. `realized_share` reports how much of the requested
spread the actuator let through, per case.

### Metrics

All are computed on `T_room` (`SCORED_CHANNEL`); a predictor may return more channels and they
are ignored, but omitting this one is an error. A prediction of the wrong shape, or holding a
non-finite value, is an error rather than a `NaN` in the results.

Let `A` be the plant's response and `P` the model's, both `(plans, horizon)`, and let `i` be the
plan tagged `nominal` — looked up by role, never by position.

| column | definition |
| --- | --- |
| `mae_K` | `mean(abs(P[i] - A[i]))` — point accuracy on the nominal plan |
| `bias_K` | `mean(P[i] - A[i])` — signed, so a systematic offset is visible |
| `response_K` | `sqrt(mean((A - mean_plans(A))^2))` — how far the probes moved the *plant* |
| `gain` | slope of the model's deviation on the plant's, below |

```
plant = A - A.mean(axis=0)          # deviations about the mean across plans
model = P - P.mean(axis=0)
gain  = sum(model * plant) / sum(plant ** 2)
```

Taking deviations about the mean over plans cancels everything they shared — weather, the
starting state, any constant bias — so only the response to *differences* in control survives.
`gain = 1.0` moves exactly as the plant does; `0.0` ignores the control entirely.

Numerator and denominator are returned separately, as `gain_cross` and `gain_square`, so a set of
rows is **pooled** by summing each before dividing:

```python
pooled = results["gain_cross"].sum() / results["gain_square"].sum()
```

Averaging per-case gains instead weights a case that barely moved as heavily as one that moved a
lot.

`gain` is `NaN` when `response_K < MIN_RESPONSE_K` (1e-3 K): the pump was clipped or idle
throughout, every plan produced the same trajectory, and the ratio would be arbitrarily large.

### Results

One row per case and context length — 240 x 4 = 960 for `benchmark-v1` — each carrying enough
provenance to be read on its own: the evaluation set, the case and its scenario, the context
length, the split, the view, the horizon and probe settings, and the two fingerprints
(`definition_sha256`, `corpus_manifest_sha256`) that say which artifact and which corpus produced
it.

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

Every field is written out in the YAML rather than defaulted: the split, the view,
`use_forecast`, the horizon, the context lengths, the probe count, amplitude and hold length,
`forecast_correction`, and the named problem instances themselves. Open loop goes one step
further and freezes the *result* of applying them — `cases.parquet` — so two people scoring the
same predictor are answering the same questions about the same trajectories, and the manifest
says so.

The plant is pinned to the `legacy` integrator. The corpus was generated with it, and evaluating
against the corpus under a different integrator would score a controller in a slightly different
world than the recorded baselines it is compared to.

## Known limitations

`IMPLICIT_ASSUMPTIONS.md` tracks what is true but unenforced — the 15-minute step assumed in
several places independently, and the two superseded disturbance conventions corpus v2 was built
under.

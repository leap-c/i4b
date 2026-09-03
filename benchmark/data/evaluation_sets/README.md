# Evaluation sets

What an open-loop evaluation covers, and the artifact it reads.

```
open_loop/
  benchmark-v1/           the official set: 240 cases
    definition.yaml         the windows and the settings -- tracked
    cases.parquet           compiled from the corpus and the plant -- generated, gitignored
    manifest.json           fingerprints of the definition, the corpus, and cases.parquet
  excitation-ladder-v1/   the excitation ablation: 270 cases, 30 buildings x 9 rungs
  fast-eval/              3 cases, for CI and for checking a method runs at all
```

```python
from i4b_bench import eval_benchmark_open_loop

results = eval_benchmark_open_loop(predictor, evaluation_set="benchmark-v1")
```

## The three files

**`definition.yaml` is the decision.** It names the problem instances and states every setting
that must not vary between two runs — the split, the view, `use_forecast`, the horizon, the
context ladder, the probe count, amplitude and hold length, `forecast_correction` — with nothing
inherited from a default you would have to go and look up. It is version-controlled, because
which problems a benchmark covers is a decision someone makes and records.

**`cases.parquet` is generated, and immutable for a release.** Compiling it resolves each window
against the corpus, seeds its context, assembles its forecast, and drives the plant under the
five probe plans, storing everything the plant answered. See `specs/DATA_SCHEMA.md` for the
schema.

**`manifest.json` is written last**, once the case file has been read back and validated. It
carries the case count and the semantics, and three fingerprints: the definition, the corpus
manifest, and `cases.parquet` itself. Evaluation checks the last one before it scores anything,
so an artifact edited underneath its manifest raises rather than producing numbers.

Rebuild a set with:

```bash
uv run python data/scripts/build_open_loop_cases.py \
    data/evaluation_sets/open_loop/benchmark-v1/definition.yaml \
    data/evaluation_sets/open_loop/benchmark-v1
```

## Why compile at all

Evaluation used to resolve windows against the corpus, prepare forecasts, and roll the plant on
every run, across a process pool, behind caches. That made a benchmark run cost half an hour of
simulation, made it depend on a 7 GB corpus nobody scoring a model otherwise needed, and left
every one of those steps free to drift between two people's numbers.

Compiling moves all of it to one place that runs once. Downstream evaluation loads the cases,
slices the history, calls the predictor, and computes metrics — no simulator, no corpus, no
forecast transformation, no join. **A new ablation is therefore a new artifact**, not a flag:
change the probes, the horizon, the view or the windows, and you compile a new set under a new
name rather than quietly changing what an old number meant.

## What is in the official set

30 buildings x 4 windows x 2 controllers = 240 cases, each scored at four context lengths, so
960 result rows.

`controller` names the recorded run whose history seeds the context — not the method under test,
which is the argument you pass in. The set uses **two** of them, `mpc-nominal` and
`open-loop-aprbs`, because measured zero-shot they span almost the whole range of control
response on their own: gain 0.15 → 0.44 across the context ladder under the first against 0.89 →
0.99 under the second. An MPC's action is a function of the state it is reacting to, so its
effect is confounded with the state; an open-loop APRBS is not. The five intermediate MPC
excitations landed between those two and cost five sevenths of the runtime to say so. The corpus
also holds four `open-loop-aprbs-<n>K` levels; those are training data, not problem instances,
and no definition names them.

Every window sits in **December to February**. A probe cannot move a room whose pump is off, and
pump activity across the test split runs 66 / 75 / 62 % of steps in those months against 2 % in
June–August. Spreading windows through the year left a fifth of them unable to measure control
response at all; in the heating season that falls to 0.4 %, and the plant's response to a probe
roughly triples, so the same number of windows buys a far quieter estimate.

Each window also names a time of day, dealt round-robin within each controller over eight slots
three hours apart, so excitation is not confounded with the thermal regime a window starts in.
120 windows per controller give a standard error near 0.016 on mean gain.

All four context lengths are kept because each is informative on at least one metric: 1 d and 2 d
are redundant for gain (paired t = 0.6) but well separated on accuracy (t = −5.9), while 5 d and
21 d are the reverse (gain t = 4.1, accuracy t = −2.8). They are also nearly free — only the
longest context is stored, and a shorter one is exactly its tail.

## The excitation ladder

`benchmark-v1` scores a method. `excitation-ladder-v1` asks a different question: **how much
control response is identifiable from a context at all, as a function of how hard that context
was excited?**

It holds the window fixed and sweeps only the recorded controller whose history seeds it, across
nine rungs from an MPC holding its comfort band to a 24 K open-loop APRBS:

| rung | `excitation` | what the history looks like |
| --- | --- | --- |
| `mpc-nominal` | none | an MPC reacting to the state; its action is a function of what it is correcting |
| `mpc-aprbs-low` / `-medium` / `-high` | low / medium / high | the same MPC with an APRBS residual added to its action |
| `open-loop-aprbs` | open loop 1.5K | no feedback at all: APRBS about the heating curve |
| `open-loop-aprbs-3K` / `-6K` / `-12K` / `-24K` | 3K … 24K | the amplitude ladder |

30 buildings x 1 anchor x 9 rungs = 270 cases, scored at four contexts, so 1,080 rows. The nine
rungs are exactly the nine categories of the ordered `excitation` dtype the results carry, in
that order, so grouping on it comes out as a curve rather than as an alphabetised list.

The design is **paired**: the nine cases of a building share its timestamp, its weather and its
forecast, so the comparison between rungs is within-window rather than across the corpus. An
unpaired sweep would confound excitation with which building and which week it landed on, and
the between-building spread in gain is larger than the effect being measured. Group on
`(scenario_id, start)` to recover a rung's siblings:

```python
results = eval_benchmark_open_loop(predictor, evaluation_set="excitation-ladder-v1")
ladder = results.groupby(["excitation", "context_days"], observed=True).apply(
    lambda g: g["gain_cross"].sum() / g["gain_square"].sum(), include_groups=False
)
```

What a rung **cannot** share is its own past. A different recorded controller ran a different
year, so it arrives at the anchor in a different state and with a different plan to probe
around: the probes and the plant's response to them differ between rungs, and a rung's number is
not purely a function of the excitation. That is inherent — you cannot vary the history a model
is given without varying where that history ended up — and it is the same residual confound
`benchmark-v1` carries between its two controllers. It is bounded by the pairing: the weather,
the building, the season and the probe design are held fixed, so what moves is the trajectory
and its excitation, not the problem.

One further difference between rungs is the probe waveform itself: the APRBS is seeded from the
case's identity, and a case's identity includes its controller, so the nine rungs of a window are
probed with nine different waveforms about nine different baselines. `gain` is a slope, and the
quantity it estimates does not depend on which particular waveform was drawn, so this adds
sampling noise rather than bias, and 30 windows per rung average it out. Seeding on the window
instead of the case would make the pairing tighter still; that is a change to the compiler, and
so a change to every artifact, and it is left for a `-v2` if the noise turns out to matter.

One thing the ladder is emphatically **not** ordered by is how much the supply temperature
*moved*. On the first window of the corpus, `mpc-nominal` has a slightly larger context-control
standard deviation than `open-loop-aprbs` (8.70 K against 8.63 K), and roughly a sixth of its
measured gain. An MPC cycles hard, but it cycles *because of* the state it is correcting, so its
variation is confounded with the response it produced and identifies almost nothing. That is the
result this set exists to quantify, and it is why "excitation" here means uncorrelated
excitation, not amplitude.

### What it measures, on Chronos-2 zero-shot

Pooled gain, 270 cases x 4 contexts, 52 s on one GPU:

| rung | 1 d | 2 d | 5 d | 21 d |
| --- | --- | --- | --- | --- |
| none (`mpc-nominal`) | 0.002 | 0.003 | 0.006 | 0.009 |
| low | 0.001 | 0.002 | 0.003 | 0.007 |
| medium | 0.003 | 0.005 | 0.006 | 0.011 |
| high | 0.006 | 0.013 | 0.016 | 0.026 |
| open loop 1.5K | 0.038 | 0.101 | 0.168 | 0.246 |
| open loop 3K | 0.075 | 0.206 | 0.361 | 0.521 |
| open loop 6K | 0.109 | 0.329 | 0.583 | 0.809 |
| open loop 12K | 0.183 | 0.500 | 0.812 | **1.046** |
| open loop 24K | 0.294 | 0.667 | **1.008** | **1.222** |

Three things this says that the two-controller comparison could not:

**The four MPC rungs are flat.** Adding an APRBS residual to an MPC's action moves gain from
0.009 to 0.026 at a 21 d context — still two orders of magnitude below what the model needs. The
step change is at the transition to *open loop*, not at any amplitude: `open-loop-aprbs` at
1.5 K scores ten times `mpc-aprbs-high`. Excitation that a feedback law chose is nearly worthless,
however much of it there is, because it is a function of the state whose response it is meant to
reveal.

**Gain overshoots one.** At 12 K and above the model *over*-responds — 1.05 and 1.22 — so more
excitation is not monotonically better. A context full of large control swings biases the model
toward attributing room movement to the control, past the point where the plant actually does.
There is an optimum somewhere around 6 K, and nothing below 3 K gets close to it.

**Accuracy moves the other way.** `mae_K` at 21 d runs 0.134 (none) → 0.257 (1.5 K) → 0.658
(24 K): the more excited the context, the worse the point prediction. Part of that is a harder
trajectory rather than a worse model — a building being driven with 24 K of APRBS is genuinely
less predictable — but it means the ladder trades accuracy against response, and reading either
metric alone would pick a different rung.

Two rungs are deliberately outside what a real installation would do. `open-loop-aprbs-24K` is a
system-identification experiment rather than a heating strategy; it is there to bound what *any*
amount of excitation could buy, so that a flat curve below it reads as "the model cannot use
excitation" rather than "there was not enough of it". The two `mpc-offset-*` controllers are
omitted: they shift the comfort bound rather than the excitation, so they would be a third
"none" rung.

The corpus calls the four amplitude levels training data, and `benchmark-v1` names none of them.
That is not a contradiction — a rung is a *context* here, not a problem instance to be scored on
and reported as a headline. Keeping the ablation in its own artifact is exactly the point of
compiling sets: it cannot quietly change what a `benchmark-v1` number means.

## Writing a window

```yaml
scenarios:
  window001:
    building: BG.N.SFH.02.Gen.ReEx.001.001--period_b
    start: 2025-12-01 00:00:00
    controller: mpc-nominal
```

`start` is either a plain date, meaning midnight UTC, or a datetime naming any point on the
corpus' 15-minute grid. Time of day matters: the same building on the same date behaves very
differently anchored at an early-morning recovery than at an afternoon coast.

The compiler refuses a window it cannot resolve exactly, rather than compiling something
approximate: the building must exist once, the named controller must have been recorded for it,
and the **whole** interval — from the start of the longest context to the end of the horizon —
must lie inside the declared split, on that trajectory. Another controller of the same building
being in the split says nothing about this one.

A building id is one house, in one thermal state, under one year of weather; the segments are
decoded in `src/i4b_bench/config/README.md`.

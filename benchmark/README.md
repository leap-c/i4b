# The I4B benchmark

A benchmark for **dynamics models and controllers of heat-pump-heated buildings**, over a corpus
of 191 single-family houses drawn from the TABULA typology, each simulated for two years of real
weather under eleven recorded controllers — 382 building-years, 4,202 trajectories, 147 million
transitions.

Everything the benchmark needs lives in this directory: the library, the corpus construction, the
settings, the scripts, the specs, the notebooks and the data. It depends on `i4b` — the simulator
next door — and nothing in `i4b` depends on it, which a test enforces. Extracting this into its
own repository is meant to be a directory move plus dropping three lines from `pyproject.toml`.

## Two evaluations

```python
from i4b_bench import eval_benchmark_open_loop, eval_benchmark_closed_loop

results = eval_benchmark_open_loop(predictor, evaluation_set="benchmark-v1")
closed_loop = eval_benchmark_closed_loop(controller)  # what did acting on it cost?
```

Both hand over the same observation, so a model written for one runs in the other unchanged. Both
return a DataFrame and write nothing.

Open loop scores a **compiled evaluation set**: windows whose contexts, forecasts, probe plans
and plant responses were computed once and stored in `cases.parquet`. Scoring a predictor reads
that file and nothing else — no corpus, no simulator, no rollout — so a run takes seconds and two
people scoring the same model are answering the same questions. Three sets ship:
`benchmark-v1` (240 cases, the official one), `excitation-ladder-v1` (270 cases sweeping how
excited the context was, paired within window) and `fast-eval` (3 cases, for CI). See
`data/evaluation_sets/README.md`.

The `view` decides how much of the plant that observation reveals: `perfect` shows the wall node
and the true heat gain — neither measurable on a real site, but together they make the dynamics
fully observable — while `realistic` shows only raw weather and the two temperatures a heat pump
and a room sensor report. See `src/i4b_bench/config/README.md`.

**Open loop** gives a predictor a context and *candidate control trajectories*, and scores two
things. `mae_K` asks whether the prediction tracks the building. `gain` asks whether it moves when
the control moves: the plant was rolled under five perturbed control plans — the recorded one, a
signed pair of constant offsets, and a signed pair of APRBS waveforms — and `gain` regresses the
model's predicted deviation on the plant's. 1.0 responds exactly as the plant does, 0.0 ignores
the control entirely. The offsets ask about the steady-state response and the APRBS about timing,
because an MPC needs both. Both numbers are needed, because a model can be accurate and still
useless inside a controller. On this corpus the two come apart sharply, and in opposite
directions:

| `benchmark-v1`, 21 d context | `mae_K` | `gain` |
| --- | --- | --- |
| Chronos-2, zero-shot | **0.213** | 0.132 |
| multi-step linear | 0.304 | **0.608** |

The foundation model predicts the room a third more accurately and has roughly a fifth of the
control response. A benchmark reporting only accuracy would rank them the wrong way round for
anything that has to plan — and in an earlier evaluation the linear model duly beat Chronos-2 in
the closed loop, 1095 against 1596.

Context and excitation both matter, and for different metrics. Zero-shot Chronos-2:

| context | 1 d | 2 d | 5 d | 21 d |
| --- | --- | --- | --- | --- |
| `mae_K` | 0.294 | 0.272 | 0.231 | 0.213 |
| `gain`, nominal-MPC context | 0.005 | 0.003 | 0.007 | 0.013 |
| `gain`, open-loop-APRBS context | 0.035 | 0.092 | 0.185 | 0.260 |

Accuracy improves steadily with more history; control response stays near zero unless that
history was *excited*. An MPC's action is a function of the state it reacts to, so its effect is
confounded with the state; an open-loop APRBS is not. `excitation-ladder-v1` measures that
across nine excitation levels rather than these two.

**Closed loop** runs a controller against the plant for an episode and reports energy, comfort
violation, and planning time — the last because a controller that wins on comfort by thinking for
a minute a step has not solved the problem.

Predictors are batched (prediction is pure); controllers are not (each action depends on the state
the last one produced).

```python
Predictor:  (observations: list[dict], controls: list[np.ndarray]) -> list[dict[str, np.ndarray]]
Controller: (observation: dict) -> tuple[float, dict | None]
```

Both are plain callables — there is no base class to inherit. A controller reads the observation
and returns a supply temperature, optionally with the plan it had in mind:

```python
def thermostat(observation):
    too_cold = observation["state"]["T_room"] < 21.0
    return (45.0 if too_cold else 25.0), None

eval_benchmark_closed_loop(thermostat)
```

`notebooks/mpc_eval.py` does the same with i4b's CasADi MPC, which is the interesting case: it
shows how a planner that re-solves each step is wrapped into this signature.

## Layout

```
src/i4b_bench/        the library
  open_loop_eval.py     eval_benchmark_open_loop, inspect_case
  closed_loop_eval.py   eval_scenario_closed_loop, eval_benchmark_closed_loop, ClosedLoopBenchmark
  cases.py              the compiled case artifact: its schema, and what makes one valid
  evaluation_set.py     an open-loop set: the definition, and where its artifact lives
  control_gain.py       the five probe plans and the gain
  observation.py        the views: which channels a model may see
  forecast.py           archived forecast runs, and how a horizon is assembled from them
  scenario_env.py       the plant, wrapped so generation, closed loop and case compilation
                        drive the same one
  dataset.py            reading the finished corpus
  corpus.py             building parameters, disturbances, reference weather
  generation.py         how the corpus is *made*; the rest of the package never imports it
  config/closed_loop/   the closed-loop settings -- see config/README.md
data/                 the dataset and everything that builds it -- see data/README.md
  scripts/              the build pipeline, and build_open_loop_cases.py
  source/               TABULA workbooks and downloaded weather -- fetched, gitignored
  corpus/               the built dataset -- generated, gitignored
  evaluation_sets/      the open-loop sets: definitions tracked, cases generated
specs/                DATA_SCHEMA.md (what the corpus and the case artifact contain),
                      EVAL_SPEC.md (what is measured), IMPLICIT_ASSUMPTIONS.md (what is true
                      but unenforced)
notebooks/            open_loop_eval.py, closed_loop_eval.py -- each loop run end to end
  data/                 reviews of the corpus, the source data and the forecasts
tests/                the contract: the import boundary, the views, gain calibration, the
                      one-step alignment, the case artifact, leakage
```

## Running it

The repository is a uv workspace, so one sync installs both packages:

```bash
uv sync --all-packages --extra mpc --extra cpu
uv run pytest                     # from the repository root, or from here
```

Tests needing the corpus skip themselves when `data/corpus/` is absent; point `I4B_BENCHMARK` at
another copy to use one.

## Building the corpus and the evaluation sets

Only needed to regenerate. The corpus is the physical-data authority; the open-loop sets are
compiled from it, and are what evaluation reads.

```bash
cd data
uv run python scripts/prepare_benchmark_data.py --output-dir source   # TABULA + weather
uv run python scripts/prepare_i4b_catalog.py                          # the building catalog
uv run python scripts/generate_benchmark_dataset.py --collector ...   # the trajectories
uv run python scripts/finalize_benchmark_dataset.py corpus            # shards, splits, manifest

uv run python scripts/build_open_loop_cases.py \
    evaluation_sets/open_loop/benchmark-v1/definition.yaml \
    evaluation_sets/open_loop/benchmark-v1                            # the 240 cases
```

The last step is the only one that has to be repeated when an open-loop definition changes, and
it takes about half an hour.

`data/README.md` walks through each step; `specs/DATA_SCHEMA.md` describes what comes out. The split assigns whole building *families* —
all three refurbishment levels of a house land on the same side of the train/test boundary, so a
model cannot have seen the insulated version of a building it is tested on.

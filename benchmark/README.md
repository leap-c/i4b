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

open_loop = eval_benchmark_open_loop(predictor)     # does the model predict, and does it respond?
closed_loop = eval_benchmark_closed_loop(controller)  # what did acting on it cost?
```

Both hand over the same observation, so a model written for one runs in the other unchanged. Both
return a DataFrame and write nothing.

The `view` decides how much of the plant that observation reveals: `perfect` shows the wall node
and the true heat gain — neither measurable on a real site, but together they make the dynamics
fully observable — while `realistic` shows only raw weather and the two temperatures a heat pump
and a room sensor report. See `src/i4b_bench/config/README.md`.

**Open loop** gives a predictor a context and *candidate control trajectories*, and scores two
things. `mae_K` asks whether the prediction tracks the building. `gain` asks whether it moves when
the control moves: roll the plant under perturbed controls, and regress the model's predicted
deviation on the plant's — 1.0 responds exactly as the plant does, 0.0 ignores the control
entirely. Both numbers are needed, because a model can be accurate and still useless inside a
controller. Chronos-2 zero-shot scores MAE 0.265 / gain 0.129 against a multi-step linear model's
0.402 / 0.492, and the linear model wins the closed loop 1095 to 1596. Ranking on accuracy alone
gets that backwards.

**Closed loop** runs a controller against the plant for an episode and reports energy, comfort
violation, and planning time — the last because a controller that wins on comfort by thinking for
a minute a step has not solved the problem.

Predictors are batched (prediction is pure); controllers are not (each action depends on the state
the last one produced).

```python
Predictor:  (observations: list[dict], controls: list[np.ndarray]) -> list[dict[str, np.ndarray]]
Controller: (observation: dict) -> tuple[float, dict | None]
```

## Layout

```
src/i4b_bench/        the library
  open_loop_eval.py     eval_scenario_open_loop, eval_benchmark_open_loop, OpenLoopBenchmark
  closed_loop_eval.py   eval_scenario_closed_loop, eval_benchmark_closed_loop, ClosedLoopBenchmark
  control_gain.py       the probe plans and the gain
  observation.py        the views: which channels a model may see
  forecast.py           archived forecast runs, and how a horizon is assembled from them
  scenario_env.py       the plant, wrapped so both loops and generation drive the same one
  dataset.py            reading the finished corpus
  corpus.py             building parameters, disturbances, reference weather
  generation.py         how the corpus is *made*; the rest of the package never imports it
  config/               the settings, one YAML per named set per loop -- see config/README.md
scripts/              building and finalizing the corpus
examples/             a worked controller, run end to end
specs/                DATA_SCHEMA.md (what the corpus contains), EVAL_SPEC.md (what is
                      measured), IMPLICIT_ASSUMPTIONS.md (what is true but unenforced)
notebooks/            marimo notebooks over the corpus and its source data
tests/                the contract: the import boundary, the views, gain calibration, leakage
production/           the corpus itself -- generated, gitignored
source-data/          TABULA workbooks and downloaded weather -- fetched, gitignored
```

## Running it

The repository is a uv workspace, so one sync installs both packages:

```bash
uv sync --all-packages --extra mpc --extra cpu
uv run pytest                     # from the repository root, or from here
```

Tests needing the corpus skip themselves when `production/` is absent; point `I4B_BENCHMARK` at
another copy to use one.

## Building the corpus

Only needed to regenerate, not to evaluate.

```bash
uv run python scripts/prepare_benchmark_data.py --output-dir source-data   # TABULA + weather
uv run python scripts/prepare_i4b_catalog.py                              # the building catalog
uv run python scripts/generate_benchmark_dataset.py --collector ... --dataset ... --output ...
uv run python scripts/finalize_benchmark_dataset.py production            # shards, splits, manifest
```

`specs/DATA_SCHEMA.md` describes what comes out. The split assigns whole building *families* —
all three refurbishment levels of a house land on the same side of the train/test boundary, so a
model cannot have seen the insulated version of a building it is tested on.

# `data/` — the dataset, and everything that makes it

Nothing here is needed to *run* the benchmark. Evaluating a method needs only a built corpus in
`corpus/` and the library in `../src/`. This directory exists so that the corpus is reproducible:
`scripts/` builds `corpus/` from `source/`, and both data directories are gitignored because they
are large and regenerable.

```
data/
  scripts/    the build pipeline; run from this directory
  source/     what was downloaded — raw/, normalized/, manifests/   (~200 MB, gitignored)
  corpus/     what the pipeline produced                            (~7 GB, gitignored)
```

If you were given a corpus rather than building one, extract it into `corpus/` and stop reading:
`load_dataset()` finds it there by default.

## Building a corpus

Run everything from this directory — the scripts default their paths to `source/` and `corpus/`
relative to the working directory.

```bash
cd benchmark/data
```

**1. Acquire the source data** (network; a few hours, resumable).

```bash
uv run python scripts/prepare_benchmark_data.py \
    --config scripts/benchmark_source_data.json --output-dir source
```

Fetches TABULA building records, Open-Meteo reference weather and archived forecast runs, and
Energy-Charts day-ahead prices for the seven locations named in the config. Writes `source/raw/`
exactly as returned, `source/normalized/` as typed Parquet, and a checksummed manifest per file.

**2. Build the building catalog.**

```bash
uv run python scripts/prepare_i4b_catalog.py
```

Maps the TABULA rows onto I4B `4R3C` parameters and writes `corpus/buildings.parquet` — 191
buildings across 64 families. Validates that every one instantiates.

**3. Collect the trajectories.** This is the expensive step: 2,674 MPC runs of a full year each,
which need an external solver, plus the open-loop APRBS runs.

```bash
uv run python scripts/generate_benchmark_dataset.py \
    --collector /path/to/mpc_collector.py --dataset corpus --output corpus --workers 30
```

Orchestrates the MPC collector, then `generate_safe_aprbs.py`, then finalization. Resumable:
finished trajectories are cached in `corpus/.staging/` and skipped on a re-run.

**4. Finalize** (folded into step 3, but runnable alone after a partial collection).

```bash
uv run python scripts/finalize_benchmark_dataset.py corpus   # validate, shard, split, manifest
uv run python scripts/finalize_benchmark_tables.py           # exogenous, forecasts, prices
```

## Adding excitation levels to an existing corpus

The four `open-loop-aprbs-<n>K` levels are training data, appended after the fact rather than
collected with the rest. They replay each building's nominal actions with an APRBS residual, so
they need no solver and take minutes.

```bash
uv run python scripts/generate_excitation_levels.py --amplitude 3 6 12 24 --workers 30
uv run python scripts/finalize_excitation_levels.py --dataset corpus
```

Finalization is additive: existing shards keep their bytes, and the new controllers inherit the
split assignment of `open-loop-aprbs`, so a held-out building stays held out at every amplitude.

**Anything appended to a corpus must read that corpus' own `exogenous.parquet` rather than
re-deriving disturbances from raw weather** — `_published_disturbances` in
`generate_excitation_levels.py` is the pattern. `../specs/IMPLICIT_ASSUMPTIONS.md` explains what
goes wrong otherwise.

## Checking a corpus

`corpus/manifest.json` records `trajectory_count`, `transition_count` and the shard list; they
should agree with the tables. `../notebooks/i4b_benchmark_dataset.py` plots the corpus, and
`../notebooks/benchmark_source_data.py` plots what it was built from.

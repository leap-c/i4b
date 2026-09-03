# `corpus/` — the built dataset

This directory is **not tracked**: it is ~7 GB, and it is reproducible from `../source/` by the
pipeline in `../scripts/`. Extract a released corpus here, or build one (see `../README.md`), and
the benchmark finds it without configuration — it is the default `dataset_dir`.

Point somewhere else with `load_dataset("/path/to/corpus")`, or set `I4B_BENCHMARK` for the tests.

```
corpus/
  buildings.parquet        one row per building variant (191)
  scenarios.parquet        one row per building x weather period (382)
  trajectories.parquet     one row per recorded run (4,202); the index into transitions/
  transitions/             part-*.parquet, the states themselves (147,233,878 rows)
  controllers.json         how each controller was configured, with provenance hashes
  exogenous.parquet        the disturbances every trajectory was rolled under
  forecasts.parquet        archived forecast runs, for use_forecast=True
  prices.parquet           day-ahead electricity prices
  split.parquet            train / validation / test, assigned by building family
  ablation-splits/         alternative splits, e.g. time.parquet
  manifest.json            counts, shard list, and the schema version
```

`../../specs/DATA_SCHEMA.md` documents every column. `manifest.json` is the thing to check first if
a corpus looks wrong: its `trajectory_count` and `transition_count` should match the tables.

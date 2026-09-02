# `source/` — what the corpus is built from

This directory is **not tracked**: it is ~200 MB and it is re-downloadable. Populate it with

```bash
cd benchmark/data
uv run python scripts/prepare_benchmark_data.py \
    --config scripts/benchmark_source_data.json --output-dir source
```

which acquires TABULA building records, Open-Meteo reference weather and archived forecast runs,
and Energy-Charts day-ahead prices — then normalizes each into typed Parquet with a checksummed
manifest beside it.

```
source/
  raw/          exactly what the provider returned: tabula/, open_meteo/, energy_charts/
  normalized/   typed Parquet, in this benchmark's column names:
                  buildings/  weather_reference/  weather_forecasts/  electricity_prices/
  manifests/    one JSON per normalized file: source URL, checksum, when it was fetched
```

Keeping `raw/` means a normalization bug is fixable without re-downloading, and the manifests are
what make a rebuild checkable rather than merely repeatable.

## Locations

Weather and price files are named `<location_id>_<period_id>.parquet`, e.g.
`de_freiburg_period_a.parquet`. The seven locations — id, country, latitude, longitude, altitude,
timezone and price market — are defined in `../scripts/benchmark_source_data.json`, which is the
single registry the acquisition reads; the periods are there too (`period_a` 2024-04 → 2025-03,
`period_b` 2025-04 → 2026-03).

These are the *benchmark's* locations, and are unrelated to `i4b_data/cities.yaml` at the
repository root — that is the simulator's own registry, used by `i4b.disturbances` to fetch PVGIS
and DWD weather for interactive use. Nothing in the benchmark reads it.

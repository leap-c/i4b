# Multi-House Time-Series Benchmark for I4B

## Goal

Generate a multi-house synthetic dataset for benchmarking time-series foundation
models. The dataset is produced by the existing I4B simulation pipeline (full-year
2015, 15-min timestep, 4R3C RC model, heatcurve controller, fixed weather location)
over a diverse set of Tabula single-family-house (SFH) archetypes drawn from 7
climate-diverse European countries, with a deterministic train/test split recorded
in a manifest.

## Motivation

The repo currently bundles 31 buildings (30 German Tabula SFH from periods 03–12 ×
3 renovation levels, plus the bespoke `i4c` two-family house). This is a reasonable
evaluation set but is single-country (one climate, Potsdam) and single building
category. Adding Tabula SFH archetypes from 6 more countries gives cross-climate and
cross-construction-tradition diversity, which is what a time-series foundation-model
benchmark needs in order to test generalization. A dedicated CLI makes the benchmark
reproducible and provides a manifest with a train/test split so downstream eval can
filter cleanly.

## Scope (confirmed)

- **Countries (7, climate-diverse)**: DE (continental, Potsdam), FR (oceanic, Paris),
  PL (continental-cold, Warsaw), NO (Nordic-cold, Oslo), CY (Mediterranean-hot,
  Nicosia), IE (oceanic-mild, Dublin), BG (continental, Sofia). These span
  Nordic → Mediterranean and oceanic → continental.
- **Building category**: SFH only, national `.N.SFH.` archetypes (real-example
  "ReEx" rows). Terraced (TH), Multi-Family (MFH), and Apartment Block (AB) are
  deferred (MFH/AB raise single-zone modelling concerns; the repo's RC model is
  single-zone).
- **Renovation variants**: all 3 per building — `0_soc` (original), `1_enev`
  (standard refurbishment), `2_kfw` (ambitious refurbishment) — matching the
  existing repo convention and Tabula variant numbers 1/2/3.
- **Benchmark knobs**: only `building_name` varies. Everything else is fixed by
  default: location = freiburg, year = 2015, profile = ResidentialDetached.csv,
  method = 4R3C, hp_model = Heatpump_AW, controller = heatcurve,
  night_setback = off, full year 2015-01-01 to 2016-01-01, timestep = 900 s.
- **Missing parameters** (not in the Tabula xlsx): `c_bldg`, `height_room`,
  `g_value`, `T_offset`, `mdot_hp` — filled with sensible defaults per renovation
  level, mirroring the values already used in the repo's existing DE buildings.

## Data Source

`tabula-calculator.xlsx` (33 MB) published by the TABULA/EPISCOPE project:

```
https://episcope.eu/fileadmin/tabula/public/calc/tabula-calculator.xlsx
```

The relevant sheet is `Calc.Set.Building` (3292 rows × 333 cols). Row 1 holds the
column headers; data starts at row 11. The `Code_BuildingVariant` column uses the
scheme `CC.N.SFH.NN.Gen.ReEx.001.NNN` where `CC` is the 2-letter country code, `NN`
is the construction-period class, and the final `.NNN` is the renovation variant
(001 = original, 002 = standard refurbishment, 003 = ambitious refurbishment).

### Verified field mapping (xlsx → i4b dict)

The xlsx data has been cross-checked against the repo's existing
`sfh_1984_1994_0_soc` and matches within rounding:

| i4b dict field         | xlsx column(s)                                          | Derivation                         |
|------------------------|---------------------------------------------------------|------------------------------------|
| `H_ve`                 | `h_Ventilation` × `A_C_Ref`                             | H_ve = h_ve × A_C_Ref  [W/K]       |
| `H_tr`                | `H_Transmission_Wall_1..3`, `Roof_1..2`, `Floor_1..2`,  | sum of all H_Transmission_* +      |
|                        | `Window_1`, `ThermalBridging`                            | ThermalBridging  [W/K]             |
| `H_tr_light`           | `H_Transmission_Window_1`                               | window transmission term  [W/K]    |
| `area_floor`           | `A_C_Ref`                                                | TABULA reference floor area [m²]   |
| `windows[E/S/W/N].area`| `A_Window_East/South/West/North`                        | per-orientation window area [m²]   |
| `position`             | (per-country reference city)                             | lat/long/altitude/timezone         |
| `c_bldg`               | (default = 45)                                           | thermal capacity [Wh/(m²K)]        |
| `height_room`          | (default = 2.5)                                           | room height [m]                    |
| `windows[].g_value`    | (default 0.75 / 0.6 / 0.5 for 0_soc/1_enev/2_kfw)         | solar heat gain coefficient        |
| `T_offset`, `mdot_hp`  | (defaults per reno level, from repo conventions)         | heating-curve offset, mass flow   |

### Available SFH buildings (7 chosen countries, ReEx, variants 1/2/3)

Extracted from the xlsx (full extract at `/tmp/opencode/tabula_sfh_i4b_ready.csv`
during planning; the generator script reproduces this from the vendored xlsx):

```
Country  Periods  Buildings (0_soc / 1_enev / 2_kfw)
DE        13       13 / 13 / 13   (39)   <- includes 2 new pre-1919 periods (01, 02)
FR        10       10 / 10 / 10   (30)
PL         7        7 /  7 /  7   (21)
NO         7        8 /  8 /  8   (24)
CY         4        4 /  4 /  4   (12)
IE        10       15 / 14 / 15   (44)
BG         6        7 /  7 /  7   (21)
-------------------------------------------------
TOTAL    57       191 buildings
```

The repo's existing 30 DE buildings are a subset of DE's 39 (periods 03–12). The
generator adds the 9 missing DE buildings (periods 01, 02) and all 152 buildings
from the other 6 countries. Net new buildings committed to the repo: 161.

## Train / Test / OOD Split

Splitting is by **building** (house-level), not by time, because a house-level split
tests cross-building generalization (the property foundation models claim), and a
temporal split on the same houses would allow per-house memorization.

- **Test (~20%, ~40 buildings)**: 1 building per (country × ~2 periods), cycling
  renovation levels, stratified so the test set spans old → new across all 7
  countries. The model never sees these buildings during training.
- **Train (~80%, ~160 buildings)**: all remaining buildings.
- **OOD (1 building)**: the existing `i4c` two-family house (non-Tabula source, from
  an EnEV certificate) — flagged `split=ood` as an extra out-of-distribution
  transfer test.

The split is a **manifest annotation only**; the simulation/generation is identical
for every building. Downstream eval filters with, e.g.,
`manifest[manifest.split == "test"]`.

## Files to Add / Change

### New

1. `data/buildings/sfh_<CC>_<y1>_<y2>.py` — one file per (country, period), each
   defining 3 dicts (`_0_soc`, `_1_enev`, `_2_kfw`) in the same structure as the
   existing `data/buildings/sfh_1984_1994.py`. Generated from the xlsx.
2. `scripts/fetch_tabula_xlsx.py` — downloads `tabula-calculator.xlsx` into
   `data/tabula/` if missing (download-on-demand). Used by the generator.
3. `scripts/build_tabula_buildings.py` — reads the vendored xlsx, applies the field
   mapping and per-reno defaults above, and emits the `data/buildings/sfh_<CC>_*.py`
   files. Run once; its output is committed. Kept in the repo for reproducibility.
4. `examples/generate_benchmark.py` — the benchmark CLI (argparse, runnable via
   `python -m examples.generate_benchmark`). No changes to `src/data_generation.py`.

### Changed

5. `data/buildings/__init__.py` — import and export all new `sfh_<CC>_*` buildings in
   `__all__`.
6. `src/gym_interface/__init__.py` — register the new buildings in
   `BUILDING_NAMES2CLASS`.
7. `src/data_generation.py` — add the 7 country reference locations to
   `DEFAULT_LOCATIONS` so the CLI can take `--location de`, `--location fr`, etc.
8. `DATA_GENERATION.md` — add a "## Benchmark CLI" section (usage, defaults, the
   7-country set, split logic, manifest schema, xlsx note) and update the
   bundled-buildings list.
9. `.gitignore` — add `data/tabula/` (the 33 MB xlsx is never committed; fetched on
   demand).
10. `README.md` (stretch) — update the building-count note to reflect the expanded
    ~200-building 7-country set.

### Unchanged

- `src/data_generation.py` core logic — the CLI calls
  `generate_building_data_file(...)` directly.
- No packaging (`pyproject.toml` / `setup.py`); runs via
  `python -m examples.generate_benchmark`, matching `examples/run_mpc.py`.
- The notebook `notebooks/DataGeneration.ipynb` stays the interactive tool.
- `src/utils.py:131` (`startswith('sfh')`) — already matches all new names; no
  change needed (verify at implementation).
- The existing 30 DE buildings + `i4c` are untouched.

## CLI Design (`examples/generate_benchmark.py`)

Flags:

```
--buildings {all,train,test,ood,<comma-list>}   default: all
--location <name>                               default: freiburg
--year <int>                                    default: 2015
--profile <filename>                            default: ResidentialDetached.csv
--start-date <ISO>                              default: 2015-01-01
--end-date <ISO>                                default: 2016-01-01
--timestep-seconds <int>                        default: 900
--method <RC model>                             default: 4R3C
--hp-model <name>                              default: Heatpump_AW
--controller <name>                            default: heatcurve
--output-dir <path>                            default: data/generated/benchmark
--night-setback                                flag, default off
--repo-root <path>                             default: parent of script dir
```

Behavior: resolve the building list from the chosen `--buildings` filter, loop and
call `dg.generate_building_data_file(...)` per building, collect per-scenario
metadata, then write:

- `manifest.csv` — one row per building, columns:
  `building, country, split, location, weather_year, internal_gain_profile,
  method, hp_model, controller, timestep_seconds, start_date, end_date,
  night_setback, rows, first_row, last_row, csv_path, metadata_path`
- `manifest.json` — the full run config for reproducibility.

Prints `[i/N] building -> csv_path` progress and a final summary (total rows,
total bytes, per-split counts). Idempotent: re-runs overwrite cleanly because
filenames are deterministic (existing slug scheme).

The `SPLIT_ASSIGNMENTS` dict (building_name → `train`/`test`/`ood`) is hardcoded
in the CLI.

## Benchmark Size (defaults)

```
Buildings:          ~200
Rows per building:   35,040  (full year 2015, 15-min)
Total rows:          ~7.0 M
CSV size:            ~2 GB
Generation time:     ~3-4 h  (sequential; ~60 s/building/year)
```

## Verification

1. **Smoke test**: `python -m examples.generate_benchmark --buildings test
   --start-date 2015-01-01 --end-date 2015-01-08` (1 week, ~40 test buildings,
   seconds). Confirm `manifest.csv` has ~40 rows with `split=test`, correct
   csv/metadata paths, and each CSV loads with a `datetime` index and the expected
   columns (`T_room`, `T_amb`, `Qdot_gains`, `P_el`, `COP`, etc.).
2. **Full run**: `python -m examples.generate_benchmark` (all ~200 buildings, full
   year). Confirm `manifest.csv` splits sum to the building count and the per-split
   row totals are as expected.

## Out of Scope (deferred)

- Terraced (TH), Multi-Family (MFH), Apartment Block (AB) archetypes — present in
  the xlsx but raise single-zone modelling concerns; deferred.
- Non-residential buildings (Office, School) — TABULA has a separate
  non-residential subsection; not extracted.
- Regional SFH from ES (`.ME.`), GB (`.ENG.`), IT (`.MidClim.`) — excluded by the
  national `.N.SFH.` filter; could be added later by relaxing the filter.
- Stochastic disturbances (randomized occupancy, window opening, shading, sensor
  noise, heat-pump outages) — listed in `DATA_GENERATION.md` "Things To Extend";
  not part of this benchmark.
- Parallel generation (`--workers`) — left for a future trivial add; sequential by
  default to avoid PVGIS weather-cache races.

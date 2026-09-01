# Benchmark Source Data Acquisition and Normalization

## Status

Ready for implementation. This ticket covers source-data preparation only. Integration
with I4B models, LEAP-C planners, foundation-model training views, dataset splits,
metrics, and benchmark trajectories will be designed separately after the prepared data
has been reviewed.

## Project Context

This work spans three repositories, but this ticket changes only I4B.

| Repository | Local checkout | Role |
|---|---|---|
| I4B | `/home/jas/projects/i4b` | Own this acquisition script, its configuration, tests, and normalized building/weather/price contracts. |
| leap-c-lab | `/home/jas/projects/leap-c-lab` | Later consume I4B building parameters and weather/price artifacts in the parametric MPC and collector. Do not change it in this ticket. |
| heat-control | `/home/jas/projects/heat-control` | Later consume generated trajectory Parquet and create foundation-model-specific training views. Do not change it in this ticket. |

Relevant I4B code and data:

- `i4b/models/model_buildings.py`: legacy building model and current parameter-dict
  contract
- `i4b/core/dynamics.py`, `i4b/core/params.py`, and `i4b/core/integrators.py`: current
  RC dynamics, derived parameters, and discrete-matrix generation
- `i4b/disturbances.py`: current weather loading and solar/internal-gain calculations
- `i4b_data/buildings/`: existing hand-written German TABULA building dictionaries
- `i4b_data/cities.yaml`: current small city registry; this ticket must not expand or
  depend on it
- `tickets/multi-house-benchmark.md`: umbrella benchmark design and repository
  responsibilities

Relevant downstream code for later integration:

- `/home/jas/projects/leap-c-lab/leapc_lab/i4b/planner.py`: parametric I4B planner
- `/home/jas/projects/leap-c-lab/leapc_lab/i4b/acados_ocp.py`: runtime `Ad`, `Bd`, `Ed`,
  and `mdot_hp` MPC parameters
- `/home/jas/worktrees/leap-c/wp2-aprbs-collector/`: earlier residual-APRBS collector
  prototype to port later, not an implementation target for this ticket
- `/home/jas/projects/heat-control/src/heat_control/plant.py`: current I4B plant wrapper
- `/home/jas/worktrees/heat-control/wp4-gain-evaluation/src/heat_control/chronos_finetune.py`:
  existing trajectory-to-Chronos projection and fine-tuning code

An older LEAP-C HVAC example already contains useful source-specific download code at
`/home/jas/worktrees/leap-c/wp1-parametric-solver/leap_c/examples/hvac/dataset.py`:

- Energy-Charts price endpoint and UTC conversion
- Open-Meteo historical-forecast endpoint
- hourly-price to 15-minute forward filling

Use it only as a reference. Do not copy its environment/dataset abstraction. In
particular, do not copy its negative-price clipping, implicit live download on cache
miss, unvalidated units, CSV cache, UTC-based occupancy features, or benchmark-time
resampling.

## Current Decisions

- Normalize 191 TABULA variants belonging to 64 base building families.
- Store source data by modality in Parquet; do not repeat building records per house
  trajectory.
- Use JSON only for configuration and small provenance manifests.
- Keep raw downloads locally during development, then decide their release retention
  separately.
- Acquire two complete 12-month periods so country-specific heating seasons can be
  selected later.
- Use one representative synthetic location per country initially.
- Download reference weather and archived ECMWF IFS forecasts from Open-Meteo.
- Download six supported day-ahead price zones from Energy-Charts; Cyprus remains
  explicitly unsupported for prices.
- Do not impute Norwegian window orientations during source normalization.
- Do not map normalized records into I4B buildings or create trajectories in this
  ticket.

## Handoff Boundary

A coding agent starting this ticket should:

1. Work only in `/home/jas/projects/i4b`.
2. Add `scripts/prepare_benchmark_data.py` and
   `scripts/benchmark_source_data.json`.
3. Add focused offline tests and only the dependency/ignore changes required by those
   files.
4. Produce and inspect small local Parquet artifacts before attempting the configured
   two-year acquisition.
5. Stop after acquisition and normalization work is complete.

Do not modify building registries, RC models, disturbance APIs, planners, collectors,
Chronos code, split manifests, or trajectory schemas under this ticket.

## Goal

Provide small, reproducible scripts that download and normalize the source data needed
by the future multi-building benchmark:

- TABULA building records
- reference weather and archived weather forecasts
- day-ahead electricity prices

The scripts should produce typed, checksummed artifacts that can be inspected without
changing existing I4B behavior.

## Repository-Friendly Approach

- Keep acquisition scripts outside the `i4b` runtime package.
- Do not change existing building definitions or registries in this work.
- Do not add a provider framework, plugin system, catalog class, or generic data API.
- Use plain functions and `argparse` scripts.
- Add shared helpers only after real duplication appears.
- Keep acquisition dependencies optional and unavailable to normal I4B imports.
- Never download data implicitly during simulation or benchmark execution.

Use one user-facing script and one declarative configuration initially:

```text
scripts/prepare_benchmark_data.py
scripts/benchmark_source_data.json
```

The JSON file is the reproducible acquisition specification. It contains periods,
representative locations, weather model and run cadence, and electricity markets. The
script keeps download, normalization, and validation as plain source-specific
functions. Do not create a shared base class or provider interface. Split the file only
if its size or independent reuse becomes a concrete maintenance problem.

Suggested acquisition-only dependencies are `openpyxl`, `pyarrow`, and `requests`.
Put them in a dedicated dependency group rather than normal project dependencies.

## Command-Line Interface

The normal user interface is one automated command:

```bash
python scripts/prepare_benchmark_data.py \
  --config scripts/benchmark_source_data.json \
  --output-dir /path/to/source-data
```

The command processes TABULA, all configured locations, forecast runs, and supported
markets. For development, `--only` may restrict work to one of `tabula`,
`weather-reference`, `weather-forecast`, or `prices`. `--force` permits replacing an
existing artifact. Without `--force`, valid existing raw and normalized artifacts are
reused so a long acquisition can be resumed safely.

Do not require users to enter city coordinates or enumerate forecast runs manually.
Optional one-location or one-run overrides may be added only if they materially help
debugging.

The initial configuration should have this direct shape, without a generic provider
schema:

```json
{
  "schema_version": 1,
  "periods": [
    {"id": "period_a", "start": "2024-04-01", "end": "2025-03-31"},
    {"id": "period_b", "start": "2025-04-01", "end": "2026-03-31"}
  ],
  "weather_model": "ecmwf_ifs",
  "forecast_run_hours_utc": [0, 6, 12, 18],
  "forecast_horizon_hours": 48,
  "locations": [
    {
      "id": "de_freiburg", "country": "DE", "latitude": 48.0252,
      "longitude": 7.7184, "timezone": "Europe/Berlin", "market": "DE-LU"
    },
    {
      "id": "fr_paris", "country": "FR", "latitude": 48.8566,
      "longitude": 2.3522, "timezone": "Europe/Paris", "market": "FR"
    },
    {
      "id": "pl_warsaw", "country": "PL", "latitude": 52.2297,
      "longitude": 21.0122, "timezone": "Europe/Warsaw", "market": "PL"
    },
    {
      "id": "no_oslo", "country": "NO", "latitude": 59.9139,
      "longitude": 10.7522, "timezone": "Europe/Oslo", "market": "NO1"
    },
    {
      "id": "cy_nicosia", "country": "CY", "latitude": 35.1856,
      "longitude": 33.3823, "timezone": "Asia/Nicosia", "market": null
    },
    {
      "id": "ie_dublin", "country": "IE", "latitude": 53.3498,
      "longitude": -6.2603, "timezone": "Europe/Dublin", "market": "IE(SEM)"
    },
    {
      "id": "bg_sofia", "country": "BG", "latitude": 42.6977,
      "longitude": 23.3219, "timezone": "Europe/Sofia", "market": "BG"
    }
  ]
}
```

These are representative synthetic locations, not known coordinates of TABULA houses.
Using one location per country is the initial YAGNI choice. A future location pool can
extend the configuration without changing downloader logic.

## Storage

Store normalized source data by modality, not per house:

```text
<output>/
  raw/
    tabula/
    open_meteo/
    energy_charts/
  normalized/
    buildings/tabula_sfh.parquet
    weather_reference/<location>_<period>.parquet
    weather_forecasts/<location>/<model>_<run>.parquet
    electricity_prices/<market>_<period>.parquet
  manifests/
    ... matching JSON sidecars ...
```

Building records occur once and are keyed by `building_id`. Weather is keyed by
`location_id`, and prices are keyed by `market_id`. Future trajectories can reference
these identifiers instead of duplicating source records for every house.

Use Parquet for normalized tables because it preserves numeric and UTC timestamp types
and allows later code to read only required columns. Use JSON only for small manifests.
Keep original XLSX and API responses locally while developing so normalization can be
repeated and debugged without new network requests. Treat these raw files as a temporary
development cache, not as part of the normalized dataset contract.

Every manifest must contain:

```text
schema_version
artifact_type
source_url
request_parameters
retrieved_at_utc
raw_sha256
normalized_sha256
row_count
time_range, when applicable
column_units
license_note
```

Calculate `normalized_sha256` after writing the Parquet file. Do not include a checksum
of the manifest itself.

Raw and normalized artifacts are generated outputs and should not be committed to the
repository.

TODO before publishing a dataset release: define the raw-data retention policy. The
default should be to remove development caches after validating the normalized files.
Retain or publish a raw source file only when it is needed for reproducibility and its
license permits redistribution; otherwise keep its source request and checksum in the
manifest.

## TABULA Buildings

Source:

```text
https://episcope.eu/fileadmin/tabula/public/calc/tabula-calculator.xlsx
```

Use `openpyxl.load_workbook(..., read_only=True, data_only=True)`. TABULA contains
formula cells, and the normalizer needs their cached values rather than formula text.
Read headers from row 1 and data from row 7 onward. Select columns by header name, not
Excel column letter.

Apply exactly this filter:

```text
Code_StatusDataset == "Typology"
Code_Country in {"BG", "CY", "DE", "FR", "IE", "NO", "PL"}
Code_BuildingSizeClass == "SFH"
Code_DataType_Building == "ReEx"
Number_BuildingVariant in {1, 2, 3}
```

The script should then:

1. Download the workbook and record its checksum.
2. Read the `Calc.Building` sheet.
3. Set `building_id` from `Code_BuildingVariant`.
4. Set `building_family_id` from `Code_Building`.
5. Normalize selected numeric columns to numeric Parquet types.
6. Sort by `country_code`, `building_family_id`, and `variant_number`.
7. Write one flat building table.

Write these normalized columns initially:

| Normalized column | TABULA column | Unit/type |
|---|---|---|
| `building_id` | `Code_BuildingVariant` | string |
| `building_family_id` | `Code_Building` | string |
| `country_code` | `Code_Country` | string |
| `variant_number` | `Number_BuildingVariant` | integer |
| `dataset_status` | `Code_StatusDataset` | string |
| `data_type` | `Code_DataType_Building` | string |
| `building_size_class` | `Code_BuildingSizeClass` | string |
| `year_start` | `Year1_Building` | nullable integer |
| `year_end` | `Year2_Building` | nullable integer |
| `roof_type` | `Code_RoofType` | string |
| `attached_neighbours` | `Code_AttachedNeighbours` | string |
| `reference_area_m2` | `A_C_Ref` | float |
| `room_height_m` | `h_room` | float |
| `thermal_capacity_Wh_m2K` | `c_m` | float |
| `transmission_W_m2K` | `h_Transmission` | float |
| `ventilation_W_m2K` | `h_Ventilation` | float |
| `window_1_area_m2` | `A_Window_1` | float |
| `window_2_area_m2` | `A_Window_2` | float |
| `window_east_area_m2` | `A_Window_East` | float |
| `window_south_area_m2` | `A_Window_South` | float |
| `window_west_area_m2` | `A_Window_West` | float |
| `window_north_area_m2` | `A_Window_North` | float |
| `window_g_value` | `g_gl_n` | float |
| `window_1_transmission_W_K` | `H_Transmission_Window_1` | float |
| `window_2_transmission_W_K` | `H_Transmission_Window_2` | float |
| `door_1_transmission_W_K` | `H_Transmission_Door_1` | float |
| `window_orientation_missing` | derived validation flag | boolean |

Set `window_orientation_missing` when total window area is positive but all four
cardinal areas are zero. This identifies 21 Norwegian variants. Do not distribute or
otherwise impute those areas during normalization.

Expected counts provide immediate diagnostics:

| Country | Variants | Families |
|---|---:|---:|
| BG | 21 | 7 |
| CY | 12 | 4 |
| DE | 39 | 13 |
| FR | 30 | 10 |
| IE | 44 | 15 |
| NO | 24 | 8 |
| PL | 21 | 7 |
| Total | 191 | 64 |

The Irish family `IE.N.SFH.10.Gen.ReEx.001` has only variants 1 and 3. Do not enforce
exactly three variants per family. The workbook currently reports `c_m = 45` and
`h_room = 2.5` for all selected rows; this lack of variation is valid and should not be
treated as an extraction failure.

Do not derive `H_tr`, `H_ve`, `H_tr_light`, window imputations, I4B parameter
dictionaries, or simulation defaults in this phase.

Keep the implementation direct:

```text
download_workbook(url, destination)
read_building_rows(workbook_path)
select_sfh_rows(rows)
normalize_building_rows(rows)
main()
```

## Weather

Use Open-Meteo for both reference weather and archived forecast runs:

```text
https://archive-api.open-meteo.com/v1/archive
https://single-runs-api.open-meteo.com/v1/forecast
```

Request these exact Open-Meteo hourly variables:

- `temperature_2m`
- `shortwave_radiation`
- `direct_normal_irradiance`
- `diffuse_radiation`

Wind speed is not needed by the current model and is excluded for now.

Always request `timezone=GMT`. Read location identifiers and coordinates from
`benchmark_source_data.json`; do not add locations to the I4B city registry in this
work.

Reference-weather output columns:

```text
location_id                    string
valid_time_utc                 timestamp[UTC]
model                          string
temperature_2m_C               float
ghi_W_m2                       float
dni_W_m2                       float
dhi_W_m2                       float
```

Forecast-run output columns:

```text
location_id                    string
model                          string
initialization_time_utc        timestamp[UTC]
valid_time_utc                 timestamp[UTC]
lead_hours                     float
temperature_2m_C               float
ghi_W_m2                       float
dni_W_m2                       float
dhi_W_m2                       float
```

For forecasts, read the model, periods, run hours, and horizon from the configuration.
Enumerate UTC initialization times deterministically. The Open-Meteo example model
identifier for ECMWF IFS is `ecmwf_ifs`. Individual ECMWF IFS HRES runs are documented
from 2024-03-14; do not silently substitute another model when a requested run is
unavailable.

Implement two straightforward request functions and one normalizer; do not introduce a
provider class:

```text
download_reference_weather(...)
download_forecast_run(...)
normalize_weather_response(..., initialization_time=None)
iter_configured_forecast_runs(config)
```

Download only the periods and runs specified by the checked-in configuration. Process
runs sequentially at first, cache each raw response independently, print progress, and
make reruns resumable. Do not add concurrency before sequential acquisition has been
validated. Do not infer model publication time yet; retain initialization time and
defer forecast-availability policy to benchmark integration.

Open-Meteo accepts multiple comma-separated coordinates. Request all seven configured
locations together for each reference period or forecast run, then split the returned
location responses into the per-location normalized files. This avoids making seven
otherwise identical API calls for every run.

The provisional source-data coverage is two consecutive forecast-backed 12-month
periods:

```text
period A: 2024-04-01 through 2025-03-31
period B: 2025-04-01 through 2026-03-31
```

Full-year acquisition avoids imposing one calendar-based heating season on countries
with different climates, particularly Norway and Cyprus. It also preserves passive and
summer dynamics for later analysis. This does not require generating closed-loop
trajectories for every downloaded timestamp; trajectory-period selection remains a
separate benchmark decision.

The two periods support a later chronological generalization test and stay within the
documented ECMWF IFS run archive. Before bulk download, run a small completeness check
for the selected model, variables, and locations. Keep the CLI period-agnostic so these
dates can be adjusted without changing code.

## Electricity Prices

Use the existing public Energy-Charts endpoint for the initial downloader:

```text
https://api.energy-charts.info/price
```

Call the endpoint with:

```text
bzn=<zone>
start=<YYYY-MM-DD>T00:00
end=<YYYY-MM-DD>T00:00
```

The response contains `unix_seconds`, `price`, and `unit`. Initially support the
confirmed zones `DE-LU`, `FR`, `PL`, `BG`, `IE(SEM)`, and `NO1`. Energy-Charts rejects
`CY`; reject that zone locally with a clear message rather than adding a second provider
now.

Output columns:

```text
market_id                      string
delivery_start_utc             timestamp[UTC]
price_eur_per_mwh              float
source                         string, always "energy_charts"
```

Price normalization should:

- retain market and delivery timestamps in UTC
- preserve the provider's native temporal resolution
- normalize to an explicit `price_eur_per_mwh` column
- preserve negative prices
- validate the returned unit
- reject duplicate or non-increasing delivery timestamps
- record zone-specific licensing restrictions

Do not resample or forward-fill prices during acquisition. Those are benchmark policy
decisions and belong in the future loader.

Energy-Charts marks DE-LU, FR, and PL data as CC BY 4.0. Its API documentation limits
BG, IE(SEM), and NO1 to private/internal use. Record this in each manifest and do not
commit or publish downloaded price artifacts.

Keep the implementation direct:

```text
download_prices(zone, start, end)
normalize_price_response(payload, zone)
main()
```

## Normalization Rules

- Store authoritative timestamps in UTC.
- Use explicit column names containing units where practical.
- Preserve native source resolution.
- Do not interpolate, forward-fill, or backward-fill missing data.
- Sort records deterministically and reject duplicate keys.
- Fail clearly on malformed responses or unsupported locations.
- Record source URLs, request parameters, retrieval times, and checksums.
- Do not silently replace existing artifacts with newly downloaded data.
- Use an explicit `--force` flag if a command needs to replace an existing artifact.
- Write to a temporary path and rename only after validation succeeds.

## Tests

Tests must use small local fixtures and must not access the network. Keep downloading
and normalization as separate functions so normalization tests can pass dictionaries or
small temporary workbooks directly. Cover only the important source boundaries:

- TABULA selection count and family grouping
- unit and UTC timestamp normalization
- forecast initialization and lead-time preservation
- negative-price preservation
- missing and duplicate interval rejection
- unsupported bidding zones
- configuration validation, including unique country/location IDs and market mapping
- provenance and checksum generation

Do not add live API tests. The script's `--help` output and the command examples in this
ticket are sufficient initial usage documentation.

## Implementation Order

Implement and review one source at a time:

1. Add the acquisition-only dependency group and ignore rules for generated output.
2. Implement TABULA download, normalization, and fixture-based tests.
3. Add and validate the checked-in acquisition configuration.
4. Implement Open-Meteo reference weather and one forecast-run download.
5. Add deterministic, resumable enumeration of configured forecast runs.
6. Implement Energy-Charts prices for the six configured markets.
7. Check that `--help` describes the configuration, output, `--only`, and `--force`.

Do not start the next source if the current source cannot produce a validated Parquet
file and manifest.

## Acceptance Criteria

- The scripts run independently of I4B simulation code.
- Existing I4B imports and behavior remain unchanged.
- TABULA normalization produces exactly 191 variants and 64 families.
- Weather reference data and forecast runs remain distinguishable.
- Price data preserves native resolution and negative values.
- Each normalized artifact has a provenance sidecar and checksum.
- No test or benchmark execution requires network access.
- Only the ticket, acquisition scripts, their focused tests, dependency declarations,
  and necessary ignore/documentation changes are included in the implementation PR.

## Deferred Decisions

- I4B building catalog representation and loading
- TABULA-to-4R3C parameter mapping and defaults
- country reference locations and final trajectory-period selection
- forecast availability rules at MPC decision time
- benchmark resampling and alignment
- Cyprus and alternative electricity-price providers
- HVAC sizing and planner parameters
- family splits and trajectory generation
- trajectory identifiers and transition-table layout
- foundation-model-specific training projections
- benchmark metrics and generalization reporting
- publication and redistribution of source artifacts

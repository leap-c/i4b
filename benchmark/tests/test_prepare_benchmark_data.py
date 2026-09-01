"""Offline tests for benchmark/scripts/prepare_benchmark_data.py.

Deliberately small. The normalized schema, the country set, the periods, and
the cohort counts are all still moving, so tests that pin those would just be
work to redo on every change - the acquisition run validates them against the
real sources anyway, and the notebook plots them.

What is tested here is the part that is expensive to get wrong and cheap to
break silently: caching and resumability semantics, plus two pieces of fiddly
arithmetic that already caused real bugs. No test touches the network.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from scripts import prepare_benchmark_data as pbd


def _location(loc_id: str = "de_freiburg", latitude: float = 48.0252) -> pbd.Location:
    return pbd.Location(
        id=loc_id,
        country="DE",
        latitude=latitude,
        longitude=7.7184,
        timezone="Europe/Berlin",
        market="DE-LU",
    )


def _weather_payload(times: list[str]) -> list[dict]:
    n = len(times)
    return [
        {
            "hourly": {
                "time": times,
                "temperature_2m": [10.0 + i for i in range(n)],
                "shortwave_radiation": [100.0 + i for i in range(n)],
                "direct_normal_irradiance": [200.0 + i for i in range(n)],
                "diffuse_radiation": [50.0 + i for i in range(n)],
            }
        }
    ]


# ---------------------------------------------------------------------------
# Fiddly arithmetic that already bit us
# ---------------------------------------------------------------------------


def test_window_orientation_missing_flag():
    """Window area present but no cardinal split - flags the Norwegian variants,
    which must be marked rather than imputed."""
    with_split = dict.fromkeys(["window_1_area_m2", "window_2_area_m2"], 20.0) | dict.fromkeys(
        [
            "window_east_area_m2",
            "window_south_area_m2",
            "window_west_area_m2",
            "window_north_area_m2",
        ],
        5.0,
    )
    without_split = with_split | dict.fromkeys(
        [
            "window_east_area_m2",
            "window_south_area_m2",
            "window_west_area_m2",
            "window_north_area_m2",
        ],
        0.0,
    )
    no_windows = dict.fromkeys(without_split, 0.0)

    assert pbd._orientation_missing(without_split) is True
    assert pbd._orientation_missing(with_split) is False
    assert pbd._orientation_missing(no_windows) is False


def test_price_period_covers_the_whole_final_day():
    """Energy-Charts' `end` is an inclusive instant, so it must be the following
    midnight - otherwise prices stop 23h short of the weather for the same
    period, and the trailing row leaks into the next period."""
    period = pbd.Period(id="p", start=date(2024, 4, 1), end=date(2025, 3, 31))

    assert pbd.price_request("DE-LU", period)["end"] == "2025-04-01T00:00"

    last, boundary = (
        pd.Timestamp("2025-03-31T23:00Z"),
        pd.Timestamp("2025-04-01T00:00Z"),
    )
    payload = {
        "unix_seconds": [int(last.timestamp()), int(boundary.timestamp())],
        "price": [10.0, -11.0],
        "unit": "EUR / MWh",
    }

    df = pbd.normalize_price_response(payload, "DE-LU", end_exclusive=date(2025, 4, 1))

    assert len(df) == 1
    assert df["delivery_start_utc"].max() == last


# ---------------------------------------------------------------------------
# Caching and resumability
# ---------------------------------------------------------------------------


def _artifact(tmp_path: Path, request: dict | None) -> tuple[Path, Path]:
    parquet_path, manifest_path = tmp_path / "out.parquet", tmp_path / "out.json"
    checksum = pbd.write_dataframe_atomic(pd.DataFrame({"a": [1, 2, 3]}), parquet_path)
    pbd.write_json(manifest_path, {"normalized_sha256": checksum, "request": request})
    return parquet_path, manifest_path


def test_artifact_up_to_date(tmp_path: Path):
    request = {"lat": 48.0}
    parquet_path, manifest_path = _artifact(tmp_path, request)

    assert pbd.artifact_up_to_date(parquet_path, manifest_path, request)
    # An edited config must invalidate, not silently relabel someone else's data.
    assert not pbd.artifact_up_to_date(parquet_path, manifest_path, {"lat": 49.0})
    assert not pbd.artifact_up_to_date(tmp_path / "nope.parquet", manifest_path, request)

    pbd.write_json(manifest_path, {"normalized_sha256": "stale", "request": request})
    assert not pbd.artifact_up_to_date(parquet_path, manifest_path, request)


def test_raw_cache_roundtrip_and_invalidation(tmp_path: Path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text("{}")
    request = {"latitude": 48.0}

    assert pbd.read_raw_cache(raw_path, request, force=False) is None  # no sidecar yet

    pbd.write_raw_cache(
        raw_path,
        source_url="https://x.test",
        request=request,
        retrieved_at_utc="2024-01-01T00:00:00Z",
    )

    # A hit returns the ORIGINAL timestamp, so rebuilding from cache does not
    # claim the data was fetched just now.
    assert pbd.read_raw_cache(raw_path, request, force=False) == "2024-01-01T00:00:00Z"
    assert pbd.read_raw_cache(raw_path, {"latitude": 49.0}, force=False) is None
    assert pbd.read_raw_cache(raw_path, request, force=True) is None


def test_weather_reference_refetches_when_coordinates_change(tmp_path, monkeypatch):
    """A location kept under the same id but moved must not reuse old data."""
    calls: list[float] = []

    def fake_download(locations, start, end):
        calls.append(locations[0].latitude)
        return _weather_payload(["2024-04-01T00:00"])

    monkeypatch.setattr(pbd, "download_reference_weather", fake_download)

    def config_at(latitude: float) -> pbd.BenchmarkConfig:
        return pbd.BenchmarkConfig(
            periods=[pbd.Period(id="p", start=date(2024, 4, 1), end=date(2024, 4, 1))],
            weather_model="ecmwf_ifs",
            forecast_run_hours_utc=[0],
            forecast_horizon_hours=2,
            locations=[_location(latitude=latitude)],
        )

    pbd.run_weather_reference(config_at(48.0), tmp_path, force=False)
    pbd.run_weather_reference(config_at(48.0), tmp_path, force=False)  # cached
    pbd.run_weather_reference(config_at(49.0), tmp_path, force=False)  # moved

    assert calls == [48.0, 49.0]


def test_weather_forecasts_skip_preflight_when_fully_cached(tmp_path, monkeypatch):
    """A completed acquisition must stay reviewable without network access."""
    location = _location()
    run = datetime(2024, 4, 1, tzinfo=timezone.utc)
    config = pbd.BenchmarkConfig(
        periods=[pbd.Period(id="p", start=date(2024, 4, 1), end=date(2024, 4, 1))],
        weather_model="ecmwf_ifs",
        forecast_run_hours_utc=[0],
        forecast_horizon_hours=2,
        locations=[location],
    )
    name = f"ecmwf_ifs_{run.strftime('%Y%m%dT%H')}"

    df = pbd.normalize_weather_response(
        _weather_payload(["2024-04-01T00:00", "2024-04-01T01:00"]),
        [location],
        model="ecmwf_ifs",
        initialization_time=run,
    )[location.id]
    checksum = pbd.write_dataframe_atomic(
        df,
        tmp_path / "normalized" / "weather_forecasts" / location.id / f"{name}.parquet",
    )
    pbd.write_json(
        tmp_path / "manifests" / "weather_forecasts" / location.id / f"{name}.json",
        {
            "normalized_sha256": checksum,
            "request": pbd.narrow_request(
                pbd.forecast_request([location], "ecmwf_ifs", run, 2), location
            ),
        },
    )

    def fail(*args, **kwargs):
        raise AssertionError("no network access expected when artifacts are cached")

    monkeypatch.setattr(pbd, "download_forecast_run", fail)

    pbd.run_weather_forecasts(config, tmp_path, force=False)  # must not raise


def test_weather_forecast_shards_are_disjoint_and_complete():
    config = pbd.BenchmarkConfig(
        periods=[
            pbd.Period(id="p1", start=date(2024, 4, 1), end=date(2024, 4, 3)),
            pbd.Period(id="p2", start=date(2025, 4, 1), end=date(2025, 4, 3)),
        ],
        weather_model="ecmwf_ifs",
        forecast_run_hours_utc=[0, 12],
        forecast_horizon_hours=2,
        locations=[_location()],
    )
    runs = list(pbd.iter_configured_forecast_runs(config))
    shards = [runs[index::3] for index in range(3)]

    shard_runs = [run for shard in shards for run in shard]

    assert len(runs) == 12
    assert len(shard_runs) == len(runs)
    assert {run for _, run in shard_runs} == {run for _, run in runs}
    assert all(len(shard) == 4 for shard in shards)

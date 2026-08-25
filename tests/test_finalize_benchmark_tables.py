from pathlib import Path

import pandas as pd

from scripts.finalize_benchmark_tables import _aligned_source, _write_prices


def test_aligned_source_interpolates_temperature_to_benchmark_timestep():
    source = pd.DataFrame(
        {
            "valid_time_utc": pd.date_range(
                "2025-01-01", periods=2, freq="1h", tz="UTC"
            ),
            "temperature_2m_C": [1.0, 2.0],
            "ghi_W_m2": [100.0, 200.0],
        }
    )
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")

    aligned = _aligned_source(source, timestamps, "valid_time_utc")

    assert aligned["temperature_2m_C"].tolist() == [
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.0,
        2.0,
        2.0,
    ]
    assert aligned["ghi_W_m2"].tolist() == [100.0] * 4 + [200.0] * 4


def test_write_prices_keeps_the_proxy_signal_separate(tmp_path: Path):
    source = tmp_path / "source-data"
    price_dir = source / "normalized" / "electricity_prices"
    price_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "market_id": ["DE-LU"],
            "delivery_start_utc": [pd.Timestamp("2025-01-01", tz="UTC")],
            "price_eur_per_mwh": [50.0],
            "source": ["energy_charts"],
        }
    )
    frame.to_parquet(price_dir / "DE-LU_period_a.parquet", index=False)

    output = tmp_path / "prices.parquet"
    _write_prices(source, output)
    prices = pd.read_parquet(output)

    assert prices.loc[0, "price_signal_id"] == "de-lu-day-ahead"
    assert prices.loc[0, "signal_type"] == "common_proxy"

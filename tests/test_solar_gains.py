"""Solar gains scale with the glazed share (1 - c_frame) and the shading factor c_shade."""

import numpy as np
import pandas as pd

from i4b.disturbances import get_solar_gains

POSITION = {"lat": 48.0, "long": 7.85, "altitude": 280, "timezone": "Europe/Berlin"}


def _gains(c_frame, c_shade):
    index = pd.date_range("2025-03-21 08:00", periods=9, freq="h", tz="Europe/Berlin")
    weather = pd.DataFrame({"ghi": 600.0, "dni": 500.0, "dhi": 150.0}, index=index)
    window = {"area": 10.0, "tilt": 90, "azimuth": 180, "g_value": 0.5,
              "c_frame": c_frame, "c_shade": c_shade}
    return np.asarray(get_solar_gains(weather, {"windows": [window], "position": POSITION}), float)


def test_gains_are_glazed_share_times_shading_factor():
    reference = _gains(0.0, 1.0)  # no frame, no shading: area * g * irradiance
    assert reference.max() > 0
    np.testing.assert_allclose(_gains(0.3, 0.6), 0.7 * 0.6 * reference, rtol=1e-12)


def test_less_shading_means_more_gain():
    # c_shade is the transmitted fraction: an open site (0.9) gets more sun than the default 0.6.
    assert _gains(0.3, 0.9).sum() > _gains(0.3, 0.6).sum()
    assert _gains(0.4, 0.6).sum() < _gains(0.3, 0.6).sum()

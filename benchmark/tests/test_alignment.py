"""The one-step conventions everything else is built on.

Three separate pieces of code have to agree about which disturbance drives which transition: the
plant, the horizon a model is handed, and the history a context is seeded from. They agreed
about the first and the third and not the second, so a model was scored against a forecast
shifted one step out of the world the plant integrated.

These tests use synthetic disturbances with a distinct value at every timestep, so an off-by-one
cannot hide behind weather that barely moves between quarter hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from i4b_bench.forecast import ForecastProvider
from i4b_bench.observation import internal_gain_profile

STEP = pd.Timedelta(minutes=15)


@pytest.fixture
def disturbances() -> pd.DataFrame:
    """A frame whose every value identifies its own timestep."""
    index = pd.date_range("2025-01-01", periods=64, freq="900s", tz="UTC", name="timestamp_utc")
    return pd.DataFrame(
        {
            "T_amb": -np.arange(64, dtype=float),
            "Qdot_gains": 10.0 * np.arange(64, dtype=float),
        },
        index=index,
    )


def test_the_plant_integrates_the_disturbance_at_the_start_of_the_interval(disturbances):
    """``x_t + u_t + d_t -> x_(t+1)``, asserted against the simulator rather than assumed.

    Everything downstream -- the forecast horizon, the history shift, the case artifact -- is
    aligned to this. If the plant ever changed to consume `d_(t+1)`, this is the test that says
    so, and all three would have to move together.
    """
    from i4b.gym_interface.room_env import RoomHeatEnv

    env = RoomHeatEnv(
        hp_model="Heatpump_AW",
        building="sfh_1919_1948_0_soc",
        method="4R3C",
        mdot_HP=0.2,
        internal_gain_profile=str(internal_gain_profile()),
        disturbances=disturbances,
        backend="legacy",
    )
    env.reset()
    env.t = 3
    state = dict(zip(env.obs_keys, [float(v) for v in env.state]))

    _, _, _, _, info = env.step(env.normalize_action(np.array(40.0)))

    at_t = env.simulator.get_next_state(state, info["u"], disturbances.iloc[3].to_dict())
    at_next = env.simulator.get_next_state(state, info["u"], disturbances.iloc[4].to_dict())
    assert info["T_room"] == pytest.approx(at_t["state"]["T_room"])
    assert info["T_room"] != pytest.approx(at_next["state"]["T_room"])


def test_the_horizon_publishes_interval_inputs_against_target_timestamps(disturbances):
    """`forecast[i]` is `d_(t+i)`; `forecast.timestamp[i]` is the state it produces, `t+i+1`.

    The timestamps name target states because that is what a model predicts. The values are the
    inputs to those transitions, which is one step earlier -- and publishing `d_(t+1)` against
    `u_t`, as this used to, hands the model a horizon the plant never ran.
    """
    provider = ForecastProvider(
        exogenous=disturbances,
        disturbance_channels=("T_amb", "Qdot_gains"),
        building_params={},
        use_forecast=False,
    )
    anchor = 10
    decision_time = disturbances.index[anchor]
    horizon = 8
    timestamps, values = provider.get_forecast(
        decision_time, horizon, disturbances.iloc[anchor].to_numpy()
    )

    assert len(timestamps) == horizon
    assert values.shape == (horizon, 2)
    for lead in range(horizon):
        # the value drives the transition into the state its timestamp names
        assert pd.Timestamp(timestamps[lead], tz="UTC") == decision_time + (lead + 1) * STEP
        assert values[lead, 0] == pytest.approx(disturbances["T_amb"].iloc[anchor + lead])
        assert values[lead, 1] == pytest.approx(disturbances["Qdot_gains"].iloc[anchor + lead])
    # the first row is what has been measured at the decision time, not a prediction
    assert values[0, 0] == pytest.approx(disturbances["T_amb"].iloc[anchor])

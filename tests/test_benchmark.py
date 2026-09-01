from __future__ import annotations

import numpy as np
import pandas as pd

from i4b_bench.corpus import (
    TRANSITION_COLUMNS,
    aprbs,
    design_mdot_hp,
    make_split_manifests,
    map_tabula_row,
    prepare_disturbances,
    residual_controller,
    rollout_controller,
    safe_aprbs_controller,
    select_forecast_disturbances,
    validate_buildings,
)


def _row(building_id: str = "NO.N.SFH.01.Gen.ReEx.001") -> dict:
    return {
        "building_id": building_id,
        "reference_area_m2": 100.0,
        "transmission_W_m2K": 0.5,
        "ventilation_W_m2K": 0.2,
        "thermal_capacity_Wh_m2K": 45.0,
        "room_height_m": 2.5,
        "window_1_area_m2": 20.0,
        "window_2_area_m2": 0.0,
        "window_north_area_m2": 0.0,
        "window_east_area_m2": 0.0,
        "window_south_area_m2": 0.0,
        "window_west_area_m2": 0.0,
        "window_g_value": 0.6,
        "window_1_transmission_W_K": 10.0,
        "window_2_transmission_W_K": 0.0,
        "door_1_transmission_W_K": 1.0,
        "window_orientation_missing": True,
    }


def _location() -> dict:
    return {
        "latitude": 59.9,
        "longitude": 10.7,
        "timezone": "Europe/Oslo",
    }


def test_tabula_mapping_imputes_missing_cardinal_orientation():
    params = map_tabula_row(_row(), _location(), 0.25)

    assert params["H_tr"] == 50.0
    assert params["H_ve"] == 20.0
    assert params["H_tr_light"] == 11.0
    assert [window["area"] for window in params["windows"]] == [5.0] * 4
    assert params["position"]["long"] == 10.7


def test_heat_pump_sizing_records_clipping():
    sizing = design_mdot_hp(_row(), pd.Series([-20.0, -10.0, 0.0]), delta_t_K=15.0)

    assert sizing["mdot_hp"] == 0.05
    assert sizing["mdot_hp_clipped"] is True
    assert sizing["mdot_hp_raw"] < 0.05


def test_mapped_building_has_valid_discrete_matrices():
    params = map_tabula_row(_row(), _location(), 0.25)

    ad, bd, cd = validate_buildings({params["name"]: params})

    assert ad.shape == (1, 3, 3)
    assert bd.shape == (1, 3, 1)
    assert cd.shape == (1, 3, 2)


def test_disturbance_internal_gains_follow_local_time_across_dst(
    tmp_path, monkeypatch
):
    profile = pd.DataFrame(
        {
            "hour": range(24),
            "user [W/m^2]": range(24),
            "appliances [W/m^2]": 0.0,
            "workday_user": 1.0,
            "workday_appliances": 0.0,
            "weekend_user": 1.0,
            "weekend_appliances": 0.0,
        }
    )
    profile_path = tmp_path / "internal-gains.csv"
    profile.to_csv(profile_path, sep=";", index=False)
    weather = pd.DataFrame(
        {
            "valid_time_utc": pd.date_range(
                "2025-03-30T00:00Z", periods=2, freq="1h"
            ),
            "T_amb": [5.0, 6.0],
            "ghi": 0.0,
            "dni": 0.0,
            "dhi": 0.0,
        }
    )
    params = {"area_floor": 10.0, "position": {"timezone": "Europe/Berlin"}}
    monkeypatch.setattr(
        "i4b.disturbances.get_solar_gains",
        lambda frame, building: pd.Series(0.0, index=frame.index),
    )

    disturbances = prepare_disturbances(weather, params, profile_path)

    assert disturbances.index.tz is not None
    assert disturbances["Qdot_gains"].tolist() == [10.0] * 4 + [30.0] * 4


def test_forecast_selection_uses_latest_complete_as_of_run():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    index = pd.date_range(start, periods=9, freq="15min")
    old = pd.DataFrame({"T_amb": 1.0, "Qdot_gains": 10.0}, index=index)
    recent = pd.DataFrame({"T_amb": 2.0, "Qdot_gains": 20.0}, index=index[:5])

    selected = select_forecast_disturbances(
        {start: old, start + pd.Timedelta(minutes=30): recent},
        start + pd.Timedelta(minutes=30),
        horizon_steps=4,
        current_disturbance=np.array([3.0, 30.0]),
        availability_delay_hours=0,
        correction_fraction=0,
    )

    assert selected.shape == (5, 2)
    assert selected[0].tolist() == [3.0, 30.0]
    assert selected[1:].tolist() == [[2.0, 20.0]] * 4


def test_forecast_selection_holds_last_value_when_run_ends_early():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    run = pd.DataFrame(
        {"T_amb": [1.0, 2.0], "Qdot_gains": [10.0, 20.0]},
        index=pd.date_range(start, periods=2, freq="15min"),
    )

    selected = select_forecast_disturbances(
        {start: run},
        start,
        horizon_steps=3,
        current_disturbance=np.array([1.0, 10.0]),
        availability_delay_hours=0,
        correction_fraction=0,
    )

    assert selected.tolist() == [[1.0, 10.0], [2.0, 20.0], [2.0, 20.0], [2.0, 20.0]]


def test_split_manifests_separate_family_and_time():
    rows = []
    country_counts = {"BG": 7, "CY": 4, "DE": 13, "FR": 10, "IE": 15, "NO": 8, "PL": 7}
    for country, count in country_counts.items():
        for family_index in range(count):
            family = f"{country}.family.{family_index}"
            for period in ("period_a", "period_b"):
                start = pd.Timestamp(
                    "2024-04-01" if period == "period_a" else "2025-04-01",
                    tz="UTC",
                )
                rows.append(
                    {
                        "trajectory_id": f"{family}-{period}",
                        "building_family_id": family,
                        "country_code": country,
                        "period_id": period,
                        "start_time_utc": start,
                        "end_time_utc": start + pd.DateOffset(years=1),
                    }
                )
    trajectories = pd.DataFrame(rows)

    primary, time_only = make_split_manifests(trajectories)

    assert set(primary["split"]) == {"train", "validation", "test"}
    assert len(primary) < len(trajectories)
    assert set(time_only["split"]) == {"train", "validation", "test"}
    assert len(time_only) == 3 * sum(country_counts.values())
    assert (
        primary.loc[primary["split"] == "test", "trajectory_id"]
        .str.endswith("period_b")
        .all()
    )
    intervals = primary.groupby("split")[["start_time_utc", "end_time_utc"]].first()
    assert (
        intervals.loc["train", "end_time_utc"]
        <= intervals.loc["validation", "start_time_utc"]
    )
    assert (
        intervals.loc["validation", "end_time_utc"]
        <= intervals.loc["test", "start_time_utc"]
    )

    family_by_trajectory = trajectories.set_index("trajectory_id")["building_family_id"]
    family_sets = {
        split: set(
            primary.loc[primary["split"] == split, "trajectory_id"].map(
                family_by_trajectory
            )
        )
        for split in ("train", "validation", "test")
    }
    assert family_sets["train"].isdisjoint(family_sets["validation"])
    assert family_sets["train"].isdisjoint(family_sets["test"])
    assert family_sets["validation"].isdisjoint(family_sets["test"])


class _FakeEnv:
    obs_keys = ("T_room", "T_wall", "T_hp_ret")
    max_t = 2

    def __init__(self):
        index = pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC")
        self.p = pd.DataFrame(
            {"T_amb": [-5.0, -4.0, -3.0], "Qdot_gains": [100.0, 110.0, 120.0]},
            index=index,
        )
        self.t = 0
        self.state = np.array([20.0, 20.0, 25.0])

    def reset(self):
        return np.r_[self.state, self.p.iloc[0].to_numpy()], {}

    def get_cur_time(self):
        return self.p.index[self.t]

    def get_cur_p(self):
        return self.p.iloc[self.t].to_dict()

    def normalize_action(self, action):
        return action

    def step(self, action):
        self.state = self.state + np.array([0.1, 0.05, 0.2])
        self.t += 1
        observation = np.r_[self.state, self.p.iloc[self.t].to_numpy()]
        return observation, 0.0, False, self.t >= self.max_t, {"u": action}


def test_rollout_stores_state_and_applied_input_without_duplicate_next_state():
    env = _FakeEnv()
    result = rollout_controller(
        env,
        lambda state, future: 30.0,
        trajectory_id="demo",
    )

    assert list(result.columns) == list(TRANSITION_COLUMNS)
    assert len(result) == 2
    np.testing.assert_allclose(result["T_room"], [20.0, 20.1])
    assert "T_room_next" not in result
    assert (result[list(TRANSITION_COLUMNS[2:])].dtypes == "float32").all()


def test_split_manifests_intersect_short_trajectory_bounds():
    rows = []
    country_counts = {"BG": 7, "CY": 4, "DE": 13, "FR": 10, "IE": 15, "NO": 8, "PL": 7}
    for country, count in country_counts.items():
        for family_index in range(count):
            family = f"{country}.family.{family_index}"
            for period, start in (
                ("period_a", pd.Timestamp("2024-04-01", tz="UTC")),
                ("period_b", pd.Timestamp("2025-04-01", tz="UTC")),
            ):
                rows.append(
                    {
                        "trajectory_id": f"{family}-{period}",
                        "building_family_id": family,
                        "country_code": country,
                        "period_id": period,
                        "start_time_utc": start,
                        "end_time_utc": start + pd.DateOffset(years=1),
                    }
                )
    trajectories = pd.DataFrame(rows)
    primary, _ = make_split_manifests(trajectories)
    train_trajectory = primary.loc[primary["split"] == "train", "trajectory_id"].iloc[0]
    train_family = trajectories.set_index("trajectory_id").loc[
        train_trajectory, "building_family_id"
    ]
    country = train_family.split(".")[0]
    episodes = pd.DataFrame(
        [
            {
                "trajectory_id": "safe-shoulder",
                "building_family_id": train_family,
                "country_code": country,
                "period_id": "period_a",
                "start_time_utc": pd.Timestamp("2024-11-01", tz="UTC"),
                "end_time_utc": pd.Timestamp("2024-11-29", tz="UTC"),
            },
            {
                "trajectory_id": "safe-cold",
                "building_family_id": train_family,
                "country_code": country,
                "period_id": "period_a",
                "start_time_utc": pd.Timestamp("2025-02-01", tz="UTC"),
                "end_time_utc": pd.Timestamp("2025-03-01", tz="UTC"),
            },
        ]
    )

    primary, time_only = make_split_manifests(pd.concat([trajectories, episodes]))

    shoulder = primary.loc[primary["trajectory_id"] == "safe-shoulder"].iloc[0]
    assert shoulder["start_time_utc"] == pd.Timestamp("2024-11-01", tz="UTC")
    assert shoulder["end_time_utc"] == pd.Timestamp("2024-11-29", tz="UTC")
    assert "safe-cold" not in set(primary["trajectory_id"])
    cold = time_only.loc[time_only["trajectory_id"] == "safe-cold"].iloc[0]
    assert cold["split"] == "validation"
    assert cold["start_time_utc"] == pd.Timestamp("2025-02-01", tz="UTC")


def test_aprbs_is_deterministic_and_scales_through_controller():
    index = pd.date_range("2025-01-01", periods=20, freq="15min", tz="UTC")
    first = aprbs(
        index,
        "house-period-residual",
        low=-1.0,
        high=1.0,
        min_hold_steps=2,
        max_hold_steps=4,
    )
    second = aprbs(
        index,
        "house-period-residual",
        low=-1.0,
        high=1.0,
        min_hold_steps=2,
        max_hold_steps=4,
    )
    controller = residual_controller(lambda state, future: 30.0, 2.0 * first)

    pd.testing.assert_series_equal(first, second)
    assert controller({}, pd.DataFrame(index=index[:1])) == 30.0 + 2.0 * first.iloc[0]


def test_safe_aprbs_controller_uses_fixed_temperature_guards():
    index = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
    levels = pd.Series([30.0, 35.0], index=index)
    controller = safe_aprbs_controller(levels)
    future = pd.DataFrame(index=index[:1])

    assert controller({"T_room": 28.0, "T_hp_ret": 24.0}, future) == 24.0
    assert controller({"T_room": 26.0, "T_hp_ret": 24.0}, future) == 30.0
    assert controller({"T_room": 18.0, "T_hp_ret": 24.0}, future) == 35.0


def test_room_env_accepts_explicit_building_and_disturbances():
    from i4b.gym_interface.room_env import RoomHeatEnv

    params = map_tabula_row(_row("demo"), _location(), 0.25)
    index = pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC")
    disturbances = pd.DataFrame(
        {"T_amb": [-5.0, -4.0, -3.0], "Qdot_gains": [100.0, 110.0, 120.0]},
        index=index,
    )
    env = RoomHeatEnv(
        hp_model="Heatpump_AW",
        building=None,
        building_params=params,
        disturbances=disturbances,
        method="4R3C",
        mdot_HP=0.25,
        internal_gain_profile="unused",
        backend="legacy",
    )

    observation, _ = env.reset()
    next_observation, _, _, _, info = env.step(env.normalize_action(30.0))

    assert observation.shape == (5,)
    assert next_observation.shape == (5,)
    assert env.building["name"] == "demo"
    assert info["u"] == 30.0

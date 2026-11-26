"""RoomHeatEnv builds its own disturbances from the packaged data."""

import os

import pytest

from i4b.gym_interface.room_env import RoomHeatEnv, resolve_profile_path

PROFILE = "i4b/data/profiles/InternalGains/ResidentialDetached.csv"


def test_packaged_profile_resolves_from_any_working_directory(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert resolve_profile_path(PROFILE).is_file()
    finally:
        os.chdir(cwd)


def test_absolute_profile_path_is_used_as_given(tmp_path):
    custom = tmp_path / "gains.csv"
    assert resolve_profile_path(str(custom)) == custom


@pytest.mark.parametrize("method", ["4R3C"])
def test_env_loads_weather_and_gains_without_injected_disturbances(tmp_path, method):
    # Regression: _load_disturbances referred to an undefined `repo_root`
    # whenever the env built the disturbances itself.
    cwd = os.getcwd()
    os.chdir(tmp_path)  # not the repository root
    try:
        env = RoomHeatEnv(
            hp_model="Heatpump_AW",
            building="sfh_1919_1948_0_soc",
            method=method,
            mdot_HP=0.25,
            internal_gain_profile=PROFILE,
            days=1,
        )
    finally:
        os.chdir(cwd)

    # The env keeps the whole weather year; `days` only sets the episode length.
    assert list(env.p.columns) == ["T_amb", "Qdot_gains"]
    assert len(env.p) >= 96
    assert not env.p.isna().any().any()

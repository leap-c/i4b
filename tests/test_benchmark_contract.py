"""The contracts the benchmark makes, checked without pinning any number.

Golden metric values would break on any deliberate corpus or view change and become noise. These
assert properties instead: the package boundary, that the gain metric is calibrated, that the
realistic view really withholds what it claims to, and that every view times forecast-source
combination returns the channels its view declares -- the last of which is the check that would
have caught the crash fixed in 861cf96.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from i4b_bench import DISTURBANCE_CHANNELS, ScenarioEnv, control_gain, load_dataset
from i4b_bench.open_loop_eval import eval_scenario_open_loop

DATASET = Path(
    os.environ.get("I4B_BENCHMARK", Path(__file__).resolve().parents[1] / "production")
)
needs_dataset = pytest.mark.skipif(
    not (DATASET / "trajectories.parquet").exists(), reason="benchmark dataset not present"
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def scenario(dataset):
    return dataset.scenarios["scenario_id"].iloc[0]


def test_i4b_does_not_import_the_benchmark():
    """The dependency runs one way, so the benchmark stays extractable and `i4b` stays upstream's."""
    root = Path(__file__).resolve().parents[1] / "i4b"
    offenders = [
        path.relative_to(root.parent)
        for path in root.rglob("*.py")
        if "i4b_bench" in path.read_text()
    ]
    assert not offenders, f"i4b must not import i4b_bench: {offenders}"


def test_gain_is_calibrated():
    """A model that copies the plant scores 1, one that ignores the control scores 0."""
    rng = np.random.default_rng(0)
    actual = rng.normal(size=(5, 96))
    assert control_gain(actual, actual) == pytest.approx(1.0)
    assert control_gain(actual, np.tile(actual.mean(axis=0), (5, 1))) == pytest.approx(0.0)
    assert control_gain(actual, 0.5 * actual) == pytest.approx(0.5)


@needs_dataset
@pytest.mark.parametrize("view", sorted(DISTURBANCE_CHANNELS))
@pytest.mark.parametrize("use_forecast", [False, True])
def test_forecast_matches_the_view_channels(dataset, scenario, view, use_forecast):
    """Every view x source combination returns the channels its view declares.

    The fallback path preserves whatever width it is handed, so this has to run far enough for a
    published run to become eligible -- at reset the bug this guards was invisible.
    """
    env = ScenarioEnv(
        scenario, dataset=dataset, view=view, use_forecast=use_forecast,
        max_context_length=96, planning_steps=12,
    )
    observation, _ = env.reset()
    for _ in range(400):
        observation, *_ = env.step(35.0)
    channels = {k for k in observation["forecast"] if k != "timestamp"}
    assert channels == set(DISTURBANCE_CHANNELS[view])


@needs_dataset
def test_realistic_forecast_is_not_the_realised_weather(dataset, scenario):
    """A forecast that matched the realised record would be leaking the future."""
    kwargs = dict(dataset=dataset, view="realistic", max_context_length=96, planning_steps=96)
    truth = ScenarioEnv(scenario, use_forecast=False, start_step=500, **kwargs)
    guess = ScenarioEnv(scenario, use_forecast=True, start_step=500, **kwargs)
    a, _ = truth.reset()
    b, _ = guess.reset()
    assert not np.allclose(a["forecast"]["T_amb"], b["forecast"]["T_amb"])


@needs_dataset
def test_a_model_never_sees_the_wall_temperature(dataset, scenario):
    """Nothing measures a wall, so it never reaches the covariates a predictor is handed."""
    seen = {}

    def predictor(observations, controls):
        seen["forecast"] = set(observations[0]["forecast"])
        return [{"T_room": np.zeros(u.shape)} for u in controls]

    eval_scenario_open_loop(scenario, predictor, dataset=dataset, seeds=1, use_forecast=False)
    assert "T_wall" not in seen["forecast"]


@needs_dataset
def test_prepare_disturbances_matches_the_plant_contract(dataset):
    """`RoomHeatEnv` rejects any frame that is not exactly its two channels.

    Adding the irradiance the `realistic` forecast needs to this frame silently broke every
    generation script, because the plant validates the column set rather than selecting from it.
    """
    from i4b.gym_interface.room_env import RoomHeatEnv
    from i4b_bench.corpus import load_params, prepare_disturbances, read_reference_weather
    from i4b_bench.observation import internal_gain_profile

    row = dataset.scenarios.iloc[0]
    params = load_params(dataset.buildings, row["building_id"])
    weather = read_reference_weather(
        Path(__file__).resolve().parents[1]
        / "source-data/normalized/weather_reference"
        / f"{row['location_id']}_{row['period_id']}.parquet"
    )
    plain = prepare_disturbances(weather, params, internal_gain_profile())
    assert list(plain.columns) == ["T_amb", "Qdot_gains"]
    RoomHeatEnv(  # would raise if the contract drifted
        hp_model="Heatpump_AW", building=None, method="4R3C",
        mdot_HP=float(params["mdot_hp"]), internal_gain_profile="unused",
        building_params=params, disturbances=plain, backend="legacy",
    )
    wide = prepare_disturbances(weather, params, internal_gain_profile(), keep_irradiance=True)
    assert set(wide.columns) == {"T_amb", "Qdot_gains", "ghi", "dni", "dhi"}

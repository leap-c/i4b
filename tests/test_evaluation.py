"""The observation contract, checked against the dataset."""

import os
from pathlib import Path

import numpy as np
import pytest

from i4b.evaluation import ScenarioEnv, evaluation_scenarios, load_dataset, run_evaluation

DATASET = Path(
    os.environ.get("I4B_BENCHMARK", Path(__file__).resolve().parents[1] / "production")
)
pytestmark = pytest.mark.skipif(
    not (DATASET / "trajectories.parquet").exists(), reason="benchmark dataset not present"
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def scenario(dataset):
    return evaluation_scenarios(dataset, "test")[0]


@pytest.mark.parametrize("history_length", [96, 672])
def test_current_state_is_the_last_history_row(dataset, scenario, history_length):
    """EVAL_SPEC: `state` values "also appear as the last entry in `history`".

    The seeded and the generated halves of the buffer have to agree about this, or a
    controller's context changes convention part-way through an episode.
    """
    env = ScenarioEnv(
        scenario, dataset=dataset, history_length=history_length, planning_steps=96,
        start_step=1000,
    )
    obs, _ = env.reset()
    assert len(obs["history"]["T_room"]) == history_length

    for channel in ("T_room", "T_wall", "T_hp_ret"):
        assert np.isclose(obs["history"][channel][-1], obs["state"][channel], atol=1e-4)

    for _ in range(3):  # across the seam between seeded and generated rows
        obs, *_ = env.step(30.0)
        for channel in ("T_room", "T_wall", "T_hp_ret"):
            assert np.isclose(obs["history"][channel][-1], obs["state"][channel], atol=1e-4)


def test_history_is_contiguous_and_forecast_follows_it(dataset, scenario):
    env = ScenarioEnv(
        scenario, dataset=dataset, history_length=96, planning_steps=96, start_step=1000
    )
    obs, _ = env.reset()
    history, forecast = obs["history"]["timestamp"], obs["forecast"]["timestamp"]
    step = np.timedelta64(15, "m")

    assert (np.diff(history) == step).all()
    assert len(forecast) == 96
    assert forecast[0] == history[-1] + step  # the forecast starts at the next timestep


def test_run_evaluation_reports_the_agreed_quantities(dataset, scenario):
    result = run_evaluation(
        scenario, lambda obs: (30.0, None), dataset=dataset, history_length=96,
        planning_steps=96, n_evaluation_steps=8,
    )
    assert set(result) == {
        "energy_kwh", "comfort_violation_degree_hours", "planning_seconds_mean", "trajectory"
    }
    assert len(result["trajectory"]) == 8
    assert {"T_room", "T_hp_ret", "planning_seconds"} <= set(result["trajectory"].columns)

"""The contracts the benchmark makes, checked without pinning any number.

Golden metric values would break on any deliberate corpus or view change and become noise. These
assert properties instead: the package boundary, that the gain metric is calibrated, that the
realistic view really withholds what it claims to, and that every view times forecast-source
combination returns the channels its view declares -- the last of which is the check that would
have caught the crash fixed in 861cf96.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from i4b_bench import (
    DISTURBANCE_CHANNELS,
    TARGET_CHANNELS,
    ScenarioEnv,
    control_gain,
    load_dataset,
)
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
    """Nothing measures a wall, so `T_wall` is not among the channels a predictor is scored on."""
    assert "T_wall" not in TARGET_CHANNELS
    seen = {}

    def predictor(observations, controls):
        seen["forecast"] = set(observations[0]["forecast"])
        return [np.zeros((u.shape[0], u.shape[1], len(TARGET_CHANNELS))) for u in controls]

    eval_scenario_open_loop(scenario, predictor, dataset=dataset, seeds=1, use_forecast=False)
    assert "T_wall" not in seen["forecast"]

"""The contracts the benchmark makes, checked without pinning any number.

Golden metric values would break on any deliberate corpus or view change and become noise. These
assert properties instead: the package boundary, that the gain metric is calibrated, that the
realistic view really withholds what it claims to, and that every view times forecast-source
combination returns the channels its view declares -- the last of which is the check that would
have caught the crash fixed in 861cf96.
"""

import importlib.util
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from i4b_bench import (
    DISTURBANCE_CHANNELS,
    STATE_CHANNELS,
    ScenarioEnv,
    control_gain,
    load_controller_data,
    load_dataset,
)
from i4b_bench.cases import load_cases
from i4b_bench.evaluation_set import CASES_FILE, load_definition
from i4b_bench.observation import history_channels
from i4b_bench.open_loop_eval import eval_benchmark_open_loop

DATASET = Path(os.environ.get("I4B_BENCHMARK", Path(__file__).resolve().parents[1] / "data" / "corpus"))
needs_dataset = pytest.mark.skipif(
    not (DATASET / "trajectories.parquet").exists(), reason="benchmark dataset not present"
)
FAST = Path(__file__).resolve().parents[1] / "data/evaluation_sets/open_loop/fast-eval"
needs_fast = pytest.mark.skipif(
    not (FAST / CASES_FILE).exists(), reason="fast-eval has not been compiled"
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def scenario(dataset):
    return dataset.scenarios["scenario_id"].iloc[0]


def test_i4b_does_not_import_the_benchmark():
    """The dependency runs one way, so the benchmark stays extractable and `i4b` stays upstream's."""
    root = Path(importlib.util.find_spec("i4b").origin).parent
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
        scenario,
        dataset=dataset,
        view=view,
        use_forecast=use_forecast,
        max_context_length=96,
        planning_steps=12,
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
@pytest.mark.parametrize("view", sorted(STATE_CHANNELS))
def test_the_wall_temperature_is_exposed_only_by_the_perfect_view(dataset, scenario, view):
    """`perfect` is the oracle view and shows the wall node; `realistic` cannot -- nothing
    measures a wall, so a method scored there must infer the thermal mass from its own history."""
    env = ScenarioEnv(
        scenario, dataset=dataset, view=view, max_context_length=96, planning_steps=12
    )
    observation, _ = env.reset()
    for _ in range(20):
        observation, *_ = env.step(35.0)
    exposed = (
        set(observation["state"]) | set(observation["history"]) | set(observation["forecast"])
    )
    assert ("T_wall" in exposed) == (view == "perfect")
    assert set(observation["state"]) == set(STATE_CHANNELS[view])


@needs_fast
def test_the_predictor_is_handed_only_the_declared_channels():
    """Whatever a view declares is exactly what arrives -- no extra channel leaks through."""
    seen = {}

    def predictor(observations, controls):
        seen["history"] = {k for k in observations[0]["history"] if k != "timestamp"}
        seen["forecast"] = {k for k in observations[0]["forecast"] if k != "timestamp"}
        return [{"T_room": np.zeros(u.shape)} for u in controls]

    definition = load_definition(FAST)
    eval_benchmark_open_loop(predictor, evaluation_set=FAST)
    assert seen["history"] == set(history_channels(definition.view))
    assert seen["forecast"] == set(DISTURBANCE_CHANNELS[definition.view])


@needs_dataset
def test_prepare_disturbances_matches_the_plant_contract(dataset):
    """`RoomHeatEnv` rejects any frame that is not exactly its two channels.

    Adding the irradiance the `realistic` forecast needs to this frame silently broke every
    generation script, because the plant validates the column set rather than selecting from it.
    """
    from i4b_bench.corpus import load_params, prepare_disturbances, read_reference_weather
    from i4b_bench.observation import internal_gain_profile

    from i4b.gym_interface.room_env import RoomHeatEnv

    row = dataset.scenarios.iloc[0]
    params = load_params(dataset.buildings, row["building_id"])
    weather = read_reference_weather(
        Path(__file__).resolve().parents[1]
        / "data/source/normalized/weather_reference"
        / f"{row['location_id']}_{row['period_id']}.parquet"
    )
    plain = prepare_disturbances(weather, params, internal_gain_profile())
    assert list(plain.columns) == ["T_amb", "Qdot_gains"]
    RoomHeatEnv(  # would raise if the contract drifted
        hp_model="Heatpump_AW",
        building=None,
        method="4R3C",
        mdot_HP=float(params["mdot_hp"]),
        internal_gain_profile="unused",
        building_params=params,
        disturbances=plain,
        backend="legacy",
    )
    wide = prepare_disturbances(weather, params, internal_gain_profile(), keep_irradiance=True)
    assert set(wide.columns) == {"T_amb", "Qdot_gains", "ghi", "dni", "dhi"}


@needs_fast
def test_a_predictor_that_replays_the_plant_scores_a_perfect_gain():
    """The end-to-end calibration: replay the plant and the harness must say so.

    `test_gain_is_calibrated` checks the formula on synthetic arrays. This checks the whole path
    -- which control the predictor is handed, which plan is nominal, how history is sliced for
    each context -- against the plant's own answer, and demands it come back exact.

    That the answer replayed here is the *stored* one is the point: an artifact whose
    trajectories drifted from the simulator would still score 1.0 here, which is why
    `test_case_generation.py` rolls the plant again and compares.
    """
    definition = load_definition(FAST)
    table = load_cases(FAST / CASES_FILE, definition.view)
    actual = {
        row["case_id"]: np.array([p["actual_T_room"] for p in row["plans"]], dtype=float)
        for row in table.to_pylist()
    }
    order = table.column("case_id").to_pylist()
    seen = itertools.cycle(order)

    def oracle(observations, controls):
        return [{"T_room": actual[next(seen)]} for _ in controls]

    frame = eval_benchmark_open_loop(oracle, evaluation_set=FAST)
    assert frame["mae_K"].max() == pytest.approx(0.0, abs=1e-6)
    assert frame["gain"].min() == pytest.approx(1.0, abs=1e-9)


@needs_dataset
def test_every_recorded_controller_can_seed_a_context(dataset):
    """Every trajectory the corpus records must be loadable.

    The corpus used to keep a second, partial copy of these columns under `controllers/`, and
    `load_controller_data` read only that -- so naming an excitation level in a setting raised
    FileNotFoundError. `transitions/` is now the one store, and this asserts it covers everything.
    """
    scenario = dataset.trajectories["scenario_id"].iloc[0]
    controllers = sorted(dataset.trajectories["controller_id"].unique())
    for controller in controllers:
        frame = load_controller_data(dataset, controller, scenario)
        assert not frame.empty, controller


@needs_dataset
def test_controllers_are_interchangeable_as_a_context(dataset):
    """Every controller yields the same schema, so a setting may name any of them.

    The excitation levels used to differ from the rest by which file they lived in; nothing
    about them should differ now.
    """
    scenario = dataset.trajectories["scenario_id"].iloc[0]
    frames = {
        controller: load_controller_data(dataset, controller, scenario)
        for controller in ("mpc-nominal", "open-loop-aprbs", "open-loop-aprbs-6K")
    }
    shapes = {name: (list(f.columns), len(f)) for name, f in frames.items()}
    assert len(set(map(str, shapes.values()))) == 1, shapes

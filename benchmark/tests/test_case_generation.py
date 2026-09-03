"""That a compiled case is what driving the plant directly would have produced.

The case artifact is an optimisation: it lets evaluation skip the simulator. That is only sound
while the stored trajectories are exactly what the simulator would answer, so these tests drive
the plant themselves and compare, rather than trusting the compiler that wrote the file.

They need the corpus, and they are the slow tests in this suite -- one window's five probes plus
its building's archived forecast runs.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from i4b_bench import ScenarioEnv, load_controller_data, load_dataset
from i4b_bench.cases import load_cases
from i4b_bench.control_gain import NOMINAL_ROLE, PLAN_ROLES
from i4b_bench.dataset import step_of
from i4b_bench.evaluation_set import CASES_FILE, load_definition
from scripts import build_open_loop_cases as build

DATASET = Path(
    os.environ.get("I4B_BENCHMARK", Path(__file__).resolve().parents[1] / "data" / "corpus")
)
SETS = Path(__file__).resolve().parents[1] / "data" / "evaluation_sets" / "open_loop"
FAST = SETS / "fast-eval"

needs_dataset = pytest.mark.skipif(
    not (DATASET / "trajectories.parquet").exists(), reason="benchmark dataset not present"
)
needs_fast = pytest.mark.skipif(
    not (FAST / CASES_FILE).exists(), reason="fast-eval has not been compiled"
)



@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def compiled():
    """The first `fast-eval` case, with the definition and window that produced it."""
    definition = load_definition(FAST)
    table = load_cases(FAST / CASES_FILE, definition.view)
    case = table.to_pylist()[0]
    return case, definition, definition.scenarios[case["case_id"]]


@needs_dataset
@needs_fast
def test_the_stored_trajectories_are_what_the_plant_answers(dataset, compiled):
    """Roll the plant again, from the same anchor under the same requested controls.

    Room temperature and applied control must come back identical -- this is what makes
    evaluating off the artifact the same experiment as evaluating against the simulator.
    """
    case, definition, window = compiled
    anchor = step_of(dataset, window.building, window.start)
    env = ScenarioEnv(
        window.building,
        dataset=dataset,
        initial_controller_id=window.controller,
        max_context_length=definition.max_context_steps,
        planning_steps=definition.horizon_steps,
        start_step=anchor,
        use_forecast=definition.use_forecast,
        view=definition.view,
        forecast_correction=definition.forecast_correction,
        build_observation=False,
    )
    for plan in case["plans"]:
        env.reset()
        rolled, applied = [], []
        for action in plan["requested_control"]:
            _, _, _, _, info = env.step(float(action))
            rolled.append(info["T_room"])
            applied.append(info["u"])
        assert plan["actual_T_room"] == pytest.approx(rolled, abs=1e-5)
        # the applied control is `info["u"]` -- kept for diagnostics, and only that
        assert plan["applied_control"] == pytest.approx(applied, abs=1e-5)


@needs_dataset
@needs_fast
def test_the_stored_forecast_is_the_weather_the_plant_used(dataset, compiled):
    """`forecast[i]` is the disturbance driving the transition into `forecast.timestamp[i]`.

    Checked against the realised record rather than against the provider that produced it, so
    the two cannot agree on a shared mistake. `use_forecast` is on for these sets, so the stored
    ambient temperature is an archived prediction and only the timestamps are compared there;
    the irradiance channels come from the same archive and are compared the same way.
    """
    case, definition, window = compiled
    step = pd.Timedelta(minutes=15)
    anchor = pd.Timestamp(case["start_timestamp"])
    exogenous = dataset.exogenous[dataset.exogenous["scenario_id"] == window.building].copy()
    exogenous["timestamp_utc"] = pd.to_datetime(exogenous["timestamp_utc"], utc=True)
    exogenous = exogenous.set_index("timestamp_utc").sort_index()

    for lead, row in enumerate(case["forecast"]):
        assert pd.Timestamp(row["timestamp"]) == anchor + (lead + 1) * step
    # row zero is the measurement at the decision time, not a prediction: it is the input to the
    # first transition, and the plant integrated exactly this value
    assert case["forecast"][0]["T_amb"] == pytest.approx(
        float(exogenous.loc[anchor, "T_amb"]), abs=1e-4
    )


@needs_dataset
@needs_fast
def test_the_history_ends_at_the_anchor_and_pairs_a_state_with_its_input(dataset, compiled):
    """A history row carries the state, and the input that produced it -- one step earlier.

    The same convention as `transitions.parquet` read through `aligned`, so a context built from
    the corpus and one built by the environment agree.
    """
    case, definition, window = compiled
    trajectory = load_controller_data(dataset, window.controller, window.building)
    anchor = step_of(dataset, window.building, window.start)
    step = pd.Timedelta(minutes=15)

    last = case["history"][-1]
    assert pd.Timestamp(last["timestamp"]) == pd.Timestamp(case["start_timestamp"])
    assert pd.Timestamp(last["timestamp"]) == trajectory["timestamp_utc"].iloc[anchor]
    assert last["T_room"] == pytest.approx(float(trajectory["T_room"].iloc[anchor]), abs=1e-3)
    assert last["T_hp_sup_applied"] == pytest.approx(
        float(trajectory["T_hp_sup_applied"].iloc[anchor - 1]), abs=1e-3
    )
    assert last["T_amb"] == pytest.approx(float(trajectory["T_amb"].iloc[anchor - 1]), abs=1e-3)
    assert case["state"]["T_room"] == pytest.approx(last["T_room"], abs=1e-4)
    first = case["history"][0]
    assert pd.Timestamp(first["timestamp"]) == pd.Timestamp(
        case["start_timestamp"]
    ) - (definition.max_context_steps - 1) * step


@needs_fast
def test_the_probes_are_the_five_named_plans(compiled):
    """Nominal, a signed pair of constant offsets, and a signed pair of APRBS waveforms.

    The offsets probe the steady-state response and the APRBS the timing; each appears with both
    signs so the deviations the gain regresses are balanced about the nominal.
    """
    case, definition, _ = compiled
    plans = {plan["plan_role"]: np.array(plan["requested_control"]) for plan in case["plans"]}
    assert list(plans) == list(PLAN_ROLES)
    nominal = plans[NOMINAL_ROLE]
    amplitude = definition.probe_amplitude

    # constant offsets, up to the actuator's clipping
    for role, sign in (("offset_minus", -1), ("offset_plus", 1)):
        offset = plans[role] - nominal
        unclipped = np.abs(offset) > 1e-9
        assert np.allclose(offset[unclipped], sign * amplitude)

    # the two APRBS plans are the same waveform mirrored, held for `probe_hold_steps`
    wave = plans["aprbs_plus"] - nominal
    assert np.allclose(plans["aprbs_minus"] - nominal, -wave, atol=1e-4)
    assert np.abs(wave).max() <= amplitude + 1e-6
    hold = definition.probe_hold_steps
    for start in range(0, len(wave) - hold, hold):
        block = wave[start : start + hold]
        assert np.allclose(block, block[0], atol=1e-4)


@needs_dataset
def test_a_window_outside_its_split_is_refused(dataset):
    """The interval is checked on the exact trajectory, and end to end.

    A sibling controller of the same building being in the split says nothing about this one,
    and a window that resolves is no use if its context reaches back before the split opens or
    its horizon runs past the end of it.
    """
    definition = load_definition(FAST)
    good = next(iter(definition.scenarios.values()))
    scenario = dataset.scenarios[dataset.scenarios["scenario_id"] == good.building].iloc[0]

    def resolve(window, **overrides):
        return build.resolve_windows(
            dataset, replace(definition, scenarios={"window001": window}, **overrides)
        )

    assert len(resolve(good)) == 1

    with pytest.raises(ValueError, match="matches 0 scenarios"):
        resolve(replace(good, building="NO.SUCH.BUILDING--period_b"))
    with pytest.raises(ValueError, match="recorded trajectories"):
        resolve(replace(good, controller="mpc-does-not-exist"))
    with pytest.raises(ValueError, match="rows in the 'train' split"):
        resolve(good, split="train")
    # a context that reaches back before the scenario opens
    with pytest.raises(ValueError, match="before the test split opens"):
        resolve(replace(good, start=scenario["start_time_utc"] + pd.Timedelta(days=1)))
    # a horizon that runs past the end of it
    with pytest.raises(ValueError, match="past the end of"):
        resolve(replace(good, start=scenario["end_time_utc"] - pd.Timedelta(hours=6)))


@needs_dataset
def test_the_compiled_case_matches_a_hand_rolled_one(dataset):
    """The compiler end to end, on one window, against the plant driven by hand.

    Belt and braces over the artifact tests above: those check the shipped file, this checks the
    code that would write the next one.
    """
    definition = load_definition(FAST)
    definition = replace(
        definition, scenarios={"window001": definition.scenarios["window001"]}
    )
    window = definition.scenarios["window001"]
    windows = build.resolve_windows(dataset, definition)
    rows = build.compile_cases(dataset, definition, windows)
    assert len(rows) == 1
    case = rows[0]
    assert case["case_id"] == "window001"
    assert len(case["history"]) == definition.max_context_steps
    assert len(case["forecast"]) == definition.horizon_steps
    assert [p["plan_role"] for p in case["plans"]] == list(PLAN_ROLES)

    anchor = step_of(dataset, window.building, window.start)
    env = ScenarioEnv(
        window.building,
        dataset=dataset,
        initial_controller_id=window.controller,
        max_context_length=definition.max_context_steps,
        planning_steps=definition.horizon_steps,
        start_step=anchor,
        use_forecast=definition.use_forecast,
        view=definition.view,
        forecast_correction=definition.forecast_correction,
        build_observation=False,
    )
    for plan in case["plans"]:
        env.reset()
        rolled = [
            env.step(float(action))[4]["T_room"] for action in plan["requested_control"]
        ]
        assert plan["actual_T_room"] == pytest.approx(rolled, abs=1e-5)

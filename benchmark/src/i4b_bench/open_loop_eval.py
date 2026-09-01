"""Open-loop evaluation: how well does a model predict, and does it respond to the control?

The closed-loop side asks a controller to act; this side asks a model to predict. Both are handed
the same observation, so a model written for one works in the other unchanged. The difference is
that a predictor is also handed *candidate control trajectories* -- prediction here is a question
about an intervention, and the control is what is intervened on.

The unit is a window: one building, one date, and the recorded run whose history fills the
context. Each is named in the setting, so the harness executes rather than chooses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import yaml

from .control_gain import gain_terms, probe_plans
from .dataset import (
    EXCITATION,
    EXCITATION_DTYPE,
    PER_DAY,
    BenchmarkDataset,
    evaluation_scenarios,
    load_controller_data,
    load_dataset,
    scenario_metadata,
    step_of,
)
from .scenario_env import ScenarioEnv

CONFIG = Path(__file__).parent / "config" / "open_loop"

#: The channel the metrics are computed on. A predictor may return more; anything else is ignored,
#: and one that omits this is an error rather than a silent zero.
SCORED_CHANNEL = "T_room"

#: A window whose probes moved the room less than this in RMS carries no information about control
#: response -- the pump was clipped or idle throughout, so every probe produced the same
#: trajectory. Reporting a ratio there yields arbitrarily large nonsense.
MIN_RESPONSE_K = 1e-3


class Predictor(Protocol):
    """A batch of observations with their candidate controls, in; predictions out.

    `controls[i]` is `(k, horizon)`, and `returns[i]` maps a channel name to a `(k, horizon)`
    array. Naming the channels rather than returning a positional block means a model predicting
    one channel and one predicting five need no different handling, and no caller can get the
    ordering backwards.
    """

    def __call__(
        self, observations: list[dict], controls: list[np.ndarray]
    ) -> list[dict[str, np.ndarray]]: ...


@dataclass(frozen=True)
class Window:
    """One problem instance: a building, a date, and whose recorded run fills the context."""

    building: str
    date: date
    controller: str


@dataclass(frozen=True)
class OpenLoopBenchmark:
    """What must not vary between open-loop runs for two results to be comparable."""

    split: str
    view: str
    use_forecast: bool
    horizon_hours: float
    #: Context lengths to sweep. The same window at one day and at three weeks is one window
    #: under two conditions, so this is a column in the results rather than part of a window.
    context_days: tuple[float, ...]
    probes: int
    probe_amplitude: float
    #: Zero, unlike the closed loop: forecast error is what this measures, so pulling the
    #: forecast toward the current sensor reading would correct away the thing under test.
    forecast_correction: float
    scenarios: dict[str, Window]


def open_loop_setting(name: str = "benchmark") -> OpenLoopBenchmark:
    """Load a named setting from `config/open_loop/<name>.yaml`."""
    path = CONFIG / f"{name}.yaml"
    if not path.exists():
        have = sorted(p.stem for p in CONFIG.glob("*.yaml"))
        raise KeyError(f"unknown setting {name!r}, have {have}")
    body = yaml.safe_load(path.read_text())
    common = dict(body["common"])
    common["context_days"] = tuple(common["context_days"])
    windows = {name: Window(**entry) for name, entry in body["scenarios"].items()}
    return OpenLoopBenchmark(**common, scenarios=windows)


def eval_benchmark_open_loop(
    predictor: Predictor,
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir=None,
    setting: OpenLoopBenchmark | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Score a predictor on the open-loop benchmark.

    Runs every window in the setting at every context length, asking two things of each: does the
    prediction track the building, and does it move when the control moves. A model can do the
    first while failing the second, which makes it useless inside a controller.

    Returns one row per window and context length. `gain` is the slope of the model's predicted
    deviation on the plant's, over counterfactual probes: 1.0 moves exactly as the plant does,
    0.0 ignores the control. Windows whose probes barely moved the room report `NaN`.
    """
    setting = setting or open_loop_setting()
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    _check_split(dataset, setting)

    horizon = round(setting.horizon_hours * 4)
    rows = []
    for days in setting.context_days:
        context = round(days * PER_DAY)
        for start in range(0, len(setting.scenarios), batch_size):
            batch = list(setting.scenarios.items())[start : start + batch_size]
            rows += _score(dataset, setting, batch, context, horizon, predictor, days)
    frame = pd.DataFrame(rows)
    frame["excitation"] = frame["excitation"].astype(EXCITATION_DTYPE)
    return frame


def eval_scenario_open_loop(
    window: Window,
    predictor: Predictor,
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir=None,
    setting: OpenLoopBenchmark | None = None,
    context_days: float | None = None,
) -> dict:
    """Score a predictor on one window, at one context length.

    Returns the row `eval_benchmark_open_loop` would have produced for it. Useful for debugging a
    predictor against a single building; the benchmark number is the benchmark-level function.
    """
    setting = setting or open_loop_setting()
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    days = context_days if context_days is not None else setting.context_days[0]
    horizon = round(setting.horizon_hours * 4)
    batch = [(f"{window.building}@{window.date}", window)]
    return _score(dataset, setting, batch, round(days * PER_DAY), horizon, predictor, days)[0]


def _score(dataset, setting, batch, context, horizon, predictor, days) -> list[dict]:
    """Roll the plant for one batch of windows, ask the predictor once, and score."""
    observations, controls, realised, meta = [], [], [], []
    for name, window in batch:
        anchor = step_of(dataset, window.building, window.date)
        trajectory = load_controller_data(dataset, window.controller, window.building)
        if anchor < context or anchor + horizon >= len(trajectory):
            raise ValueError(f"{name}: {window.date} leaves no room for {days} d of context")

        env = ScenarioEnv(
            window.building,
            dataset=dataset,
            initial_controller_id=window.controller,
            max_context_length=context,
            planning_steps=horizon,
            start_step=anchor,
            use_forecast=setting.use_forecast,
            view=setting.view,
            forecast_correction=setting.forecast_correction,
        )
        observation, _ = env.reset()
        nominal = trajectory["T_hp_sup_applied"].to_numpy(float)[anchor : anchor + horizon]
        plans = probe_plans(
            np.random.default_rng(anchor), nominal, setting.probe_amplitude, setting.probes
        )
        rolled = np.empty((len(plans), horizon), dtype=float)
        for k, plan in enumerate(plans):
            env.reset()
            for t, action in enumerate(plan):
                _, _, _, _, info = env.step(float(action))
                rolled[k, t] = info["T_room"]

        observations.append(observation)
        controls.append(plans)
        realised.append(rolled)
        meta.append(
            {
                "window": name,
                "controller": window.controller,
                "date": str(window.date),
                "context_days": days,
                **scenario_metadata(dataset, window.building),
            }
        )

    predictions = predictor(observations, controls)
    if len(predictions) != len(observations):
        raise ValueError(f"predictor returned {len(predictions)} blocks for {len(observations)}")

    nominal_index = setting.probes // 2
    rows = []
    for info, actual, prediction in zip(meta, realised, predictions):
        if SCORED_CHANNEL not in prediction:
            raise ValueError(
                f"predictor returned {sorted(prediction)}, without {SCORED_CHANNEL!r}"
            )
        predicted = np.asarray(prediction[SCORED_CHANNEL], dtype=float)
        if predicted.shape != actual.shape:
            raise ValueError(f"expected {actual.shape} predictions, got {predicted.shape}")
        error = predicted[nominal_index] - actual[nominal_index]
        cross, square = gain_terms(actual, predicted)
        response = float(np.sqrt(square / actual.size))
        rows.append(
            {
                **info,
                "excitation": EXCITATION.get(info["controller"]),
                "mae_K": float(np.abs(error).mean()),
                "bias_K": float(error.mean()),
                "response_K": response,
                "gain": cross / square if response >= MIN_RESPONSE_K else float("nan"),
            }
        )
    return rows


def _check_split(dataset, setting) -> None:
    """Every named building must belong to the declared split.

    Otherwise `split` would be inert, and a training building pasted into a held-out set would
    quietly produce a number nobody can trust.
    """
    available = set(evaluation_scenarios(dataset, setting.split))
    stray = sorted({w.building for w in setting.scenarios.values()} - available)
    if stray:
        raise ValueError(
            f"{len(stray)} building(s) are not in the {setting.split!r} split, e.g. {stray[0]}"
        )

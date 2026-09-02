"""Open-loop evaluation: how well does a model predict, and does it respond to the control?

A predictor gets the same observation a controller gets in the closed loop, plus candidate
control trajectories to predict under. The unit is a window: one building, one date, and the
recorded run whose history fills the context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
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

#: The channel the metrics are computed on. A predictor may return more; omitting it is an error.
SCORED_CHANNEL = "T_room"

#: Minimum RMS room response, in K, for a window's gain to be reported. Below it the probes moved
#: the plant too little to divide by -- the pump was clipped or idle throughout.
MIN_RESPONSE_K = 1e-3


class Predictor(Protocol):
    """A batch of observations with their candidate controls, in; predictions out.

    `controls[i]` is `(k, horizon)`; `returns[i]` maps a channel name to a `(k, horizon)` array.
    """

    def __call__(
        self, observations: list[dict], controls: list[np.ndarray]
    ) -> list[dict[str, np.ndarray]]: ...


@dataclass(frozen=True)
class Window:
    """One problem instance: a building, when it starts, and whose run fills the context."""

    building: str
    #: Where the horizon begins. A date means midnight UTC; a datetime names any time on the
    #: corpus' 15-minute grid.
    start: date | datetime
    #: The recorded run whose history seeds the context.
    controller: str


@dataclass(frozen=True)
class OpenLoopBenchmark:
    """What must not vary between open-loop runs for two results to be comparable."""

    split: str
    view: str
    use_forecast: bool
    horizon_hours: float
    #: Context lengths to sweep. Every window is run at each, and it becomes a results column.
    context_days: tuple[float, ...]
    probes: int
    probe_amplitude: float
    #: How far an archived forecast is pulled toward the current sensor reading, in [0, 1]. Zero
    #: here: forecast error is part of what this measures.
    forecast_correction: float
    scenarios: dict[str, Window]


def open_loop_setting(name: str = "benchmark") -> OpenLoopBenchmark:
    """Load a named setting from `config/open_loop/<name>.yaml`.

    Parameters
    ----------
    name : str
        Setting stem, e.g. ``"benchmark"`` or ``"fast_eval"``.

    Returns
    -------
    OpenLoopBenchmark
        The `common` block plus one `Window` per named scenario.
    """
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

    Runs every window at every context length in the setting.

    Parameters
    ----------
    predictor : Predictor
        Called once per batch as ``predictor(observations, controls)``; see `Predictor`.
    dataset : BenchmarkDataset, optional
        An already-loaded corpus. Loaded from `dataset_dir` when omitted.
    dataset_dir : str or Path, optional
        Corpus directory. Defaults to the bundled `production/`.
    setting : OpenLoopBenchmark, optional
        What to run. Defaults to `open_loop_setting("benchmark")`.
    batch_size : int
        Windows per predictor call. Trades peak memory against call overhead.

    Returns
    -------
    pandas.DataFrame
        One row per window and context length, with `mae_K` and `bias_K` (point accuracy on the
        nominal plan), `response_K` (RMS movement the probes produced in the plant), `gain`, and
        the window's provenance. `gain` is the slope of the model's predicted deviation on the
        plant's over the probes: 1.0 moves as the plant does, 0.0 ignores the control, `NaN`
        when `response_K` fell below `MIN_RESPONSE_K`.
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

    Parameters
    ----------
    window : Window
        The building, date and history-seeding controller to score on.
    predictor : Predictor
        As for `eval_benchmark_open_loop`.
    dataset, dataset_dir, setting
        As for `eval_benchmark_open_loop`.
    context_days : float, optional
        Context length. Defaults to the setting's first.

    Returns
    -------
    dict
        The row `eval_benchmark_open_loop` would have produced for this window.
    """
    setting = setting or open_loop_setting()
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    days = context_days if context_days is not None else setting.context_days[0]
    horizon = round(setting.horizon_hours * 4)
    batch = [(f"{window.building}@{window.start}", window)]
    return _score(dataset, setting, batch, round(days * PER_DAY), horizon, predictor, days)[0]


def _score(dataset, setting, batch, context, horizon, predictor, days) -> list[dict]:
    """Roll the plant for one batch of windows, ask the predictor once, and score."""
    observations, controls, realised, meta = [], [], [], []
    for name, window in batch:
        anchor = step_of(dataset, window.building, window.start)
        trajectory = load_controller_data(dataset, window.controller, window.building)
        if anchor < context or anchor + horizon >= len(trajectory):
            raise ValueError(f"{name}: {window.start} leaves no room for {days} d of context")

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
                "start": str(window.start),
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
    """Every named building must belong to the declared split."""
    available = set(evaluation_scenarios(dataset, setting.split))
    stray = sorted({w.building for w in setting.scenarios.values()} - available)
    if stray:
        raise ValueError(
            f"{len(stray)} building(s) are not in the {setting.split!r} split, e.g. {stray[0]}"
        )

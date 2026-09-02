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
    workers: int = 1,
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
    workers : int
        Processes driving the plant. The probes are the expensive part and windows are
        independent, so this scales close to linearly. The predictor stays in the parent and is
        still called once per batch, so a GPU model is neither copied nor contended for. Batches
        are grouped by building, because preparing a building's archived forecast runs costs
        ~40 s and is cached per process.

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
    longest = round(max(setting.context_days) * PER_DAY)
    # Group by building so a worker prepares each building's forecast runs once, not per window.
    ordered = sorted(setting.scenarios.items(), key=lambda kv: (kv[1].building, kv[0]))
    batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]

    # The plant is driven once per window, at the longest context. Nothing about the probes
    # depends on the context length -- same anchor, same plans, same weather -- and a shorter
    # history is exactly the tail of a longer one, so the other rungs are a slice rather than
    # another set of rollouts.
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(dataset,)
        ) as pool:
            futures = [
                pool.submit(_roll_in_worker, setting, batch, longest, horizon) for batch in batches
            ]
            rolled = [future.result() for future in futures]
    else:
        rolled = [_roll(dataset, setting, batch, longest, horizon) for batch in batches]

    rows = []
    for days in setting.context_days:
        context = round(days * PER_DAY)
        for prepared in rolled:
            rows += _score([_at_context(item, context, days) for item in prepared], predictor)
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
    context = round(days * PER_DAY)
    prepared = _roll(dataset, setting, batch, context, horizon)
    return _score([_at_context(item, context, days) for item in prepared], predictor)[0]


_WORKER_DATASET: BenchmarkDataset | None = None


def _init_worker(dataset: BenchmarkDataset) -> None:
    """Hand each worker the corpus once, rather than with every batch."""
    global _WORKER_DATASET
    _WORKER_DATASET = dataset


def _roll_in_worker(setting, batch, context, horizon) -> list[dict]:
    return _roll(_WORKER_DATASET, setting, batch, context, horizon)


def _at_context(item: dict, context: int, days: float) -> dict:
    """The same rollout, seen through a shorter history. Verified equal to rolling at `context`."""
    history = {name: values[-context:] for name, values in item["observation"]["history"].items()}
    return {
        **item,
        "observation": {**item["observation"], "history": history},
        "meta": {**item["meta"], "context_days": days},
    }


def _roll(dataset, setting, batch, context, horizon) -> list[dict]:
    """Drive the plant for one batch of windows: the observation, the probes, the responses.

    The plant half of scoring, and the expensive one. It needs no predictor, so it is what a
    worker process runs.
    """
    prepared = []
    for name, window in batch:
        anchor = step_of(dataset, window.building, window.start)
        trajectory = load_controller_data(dataset, window.controller, window.building)
        if anchor < context or anchor + horizon >= len(trajectory):
            raise ValueError(f"{name}: {window.start} leaves no room for {context} steps")

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
        # The probes below only read `T_room` out of `info`; assembling the full observation on
        # each of their `probes * horizon` steps copies the whole context every time, and costs
        # about ten times the rollout itself.
        env.build_observations = False
        nominal = trajectory["T_hp_sup_applied"].to_numpy(float)[anchor : anchor + horizon]
        plans = probe_plans(
            np.random.default_rng(anchor), nominal, setting.probe_amplitude, setting.probes
        )
        rolled = np.empty((len(plans), horizon), dtype=float)
        # What the plant applied, which is not what was requested: `check_hp` collapses the
        # supply temperature whenever the pump idles. The predictor is asked about the applied
        # sequence, because that is the intervention its response is compared against. Handing
        # it the requested one instead inflates gain by the reciprocal of the realized
        # fraction -- about 2.5x on this corpus.
        applied = np.empty_like(rolled)
        for k, plan in enumerate(plans):
            env.reset()
            for t, action in enumerate(plan):
                _, _, _, _, info = env.step(float(action))
                rolled[k, t] = info["T_room"]
                applied[k, t] = info["u"]

        requested_spread = float(plans.std(axis=0).mean())
        prepared.append(
            {
                "observation": observation,
                "plans": applied,
                "requested": plans,
                "actual": rolled,
                "meta": {
                    "window": name,
                    "controller": window.controller,
                    "start": str(window.start),
                    # How much of the probe the actuator let through; 0 means the pump never
                    # moved, so the window carries no control-response information.
                    "realized_share": (
                        float(applied.std(axis=0).mean()) / requested_spread
                        if requested_spread > 0
                        else 0.0
                    ),
                    **scenario_metadata(dataset, window.building),
                },
            }
        )
    return prepared


def _score(prepared: list[dict], predictor: Predictor) -> list[dict]:
    """Ask the predictor about a prepared batch, and turn its answer into result rows."""
    observations = [item["observation"] for item in prepared]
    controls = [item["plans"] for item in prepared]
    predictions = predictor(observations, controls)
    if len(predictions) != len(observations):
        raise ValueError(f"predictor returned {len(predictions)} blocks for {len(observations)}")

    nominal_index = len(controls[0]) // 2
    rows = []
    for item, prediction in zip(prepared, predictions):
        if SCORED_CHANNEL not in prediction:
            raise ValueError(
                f"predictor returned {sorted(prediction)}, without {SCORED_CHANNEL!r}"
            )
        actual = item["actual"]
        predicted = np.asarray(prediction[SCORED_CHANNEL], dtype=float)
        if predicted.shape != actual.shape:
            raise ValueError(f"expected {actual.shape} predictions, got {predicted.shape}")
        error = predicted[nominal_index] - actual[nominal_index]
        cross, square = gain_terms(actual, predicted)
        response = float(np.sqrt(square / actual.size))
        rows.append(
            {
                **item["meta"],
                "excitation": EXCITATION.get(item["meta"]["controller"]),
                "mae_K": float(np.abs(error).mean()),
                "bias_K": float(error.mean()),
                "response_K": response,
                "gain": cross / square if response >= MIN_RESPONSE_K else float("nan"),
            }
        )
    return rows


def inspect_window(
    window: Window,
    predictor: Predictor,
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir=None,
    setting: OpenLoopBenchmark | None = None,
    context_days: float | None = None,
) -> dict:
    """Everything one window's scoring saw, for looking at rather than aggregating.

    Returns the context handed over, the probes as requested and as the actuator applied them,
    the plant's response, the model's prediction, and the row those produce. `eval_*` reduce all
    of this to four numbers; this is what to plot when one of them looks wrong.

    Parameters
    ----------
    window : Window
        The window to inspect.
    predictor : Predictor
        Called once, on this window alone.
    dataset, dataset_dir, setting
        As for `eval_benchmark_open_loop`.
    context_days : float, optional
        Context length. Defaults to the setting's longest.

    Returns
    -------
    dict
        ``history`` and ``forecast`` frames, ``requested`` and ``applied`` probe arrays
        `(probes, horizon)`, ``actual`` and ``predicted`` room temperatures of the same shape,
        the horizon ``timestamps``, and the scored ``row``.
    """
    setting = setting or open_loop_setting()
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    days = context_days if context_days is not None else max(setting.context_days)
    context, horizon = round(days * PER_DAY), round(setting.horizon_hours * 4)
    name = f"{window.building}@{window.start}"
    item = _at_context(
        _roll(dataset, setting, [(name, window)], context, horizon)[0], context, days
    )

    prediction = predictor([item["observation"]], [item["plans"]])[0]
    row = _score([item], lambda _o, _c: [prediction])[0]
    observation = item["observation"]
    return {
        "history": pd.DataFrame(observation["history"]).set_index("timestamp"),
        "forecast": pd.DataFrame(observation["forecast"]).set_index("timestamp"),
        "requested": item["requested"],
        "applied": item["plans"],
        "actual": item["actual"],
        "predicted": np.asarray(prediction[SCORED_CHANNEL], dtype=float),
        "timestamps": pd.DatetimeIndex(observation["forecast"]["timestamp"]),
        "row": row,
    }


def _check_split(dataset, setting) -> None:
    """Every named building must belong to the declared split."""
    available = set(evaluation_scenarios(dataset, setting.split))
    stray = sorted({w.building for w in setting.scenarios.values()} - available)
    if stray:
        raise ValueError(
            f"{len(stray)} building(s) are not in the {setting.split!r} split, e.g. {stray[0]}"
        )

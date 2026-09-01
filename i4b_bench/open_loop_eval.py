"""Open-loop evaluation: how well does a model predict, and does it respond to the control?

The closed-loop side asks a controller to act; this side asks a model to predict. Both are handed
the same observation, so a model written for one works in the other unchanged. The difference is
that a predictor is also handed *candidate control trajectories* -- prediction is a question about
an intervention, and the control is what is being intervened on.

Batching is the top-level unit here, which it cannot be for control: prediction is a pure function
of context and control, so a whole benchmark reaches the GPU in a handful of calls, while every
closed-loop action depends on the state the last one produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import tomllib

from . import scenario
from .control_gain import gain_terms, probe_plans
from .dataset import (
    EXCITATION,
    EXCITATION_DTYPE,
    PER_DAY,
    BenchmarkDataset,
    load_controller_data,
    load_dataset,
    timestamp_of,
)
from .scenario import Scenario
from .scenario_env import ScenarioEnv

#: The channel the metrics are computed on. A predictor may return more; anything else is
#: ignored, and a predictor that omits this one is an error rather than a silent zero.
SCORED_CHANNEL = "T_room"

#: A window whose probes moved the room less than this in RMS carries no information about
#: control response -- the pump was clipped or idle for the whole horizon, so every probe
#: produced the same trajectory. Reporting a ratio there yields arbitrarily large nonsense.
MIN_RESPONSE_K = 1e-3


class Predictor(Protocol):
    """A batch of observations with their candidate controls, in; predictions out.

    `controls[i]` is `(k_i, horizon)`, and `returns[i]` maps a channel name to a
    `(k_i, horizon)` array. Naming the channels rather than returning a positional
    `(k, horizon, n_targets)` block means a model that predicts one channel and one that
    predicts five need no different handling, and no caller can get the ordering backwards.

    A caller with one window and one control passes lists of length one.
    """

    def __call__(
        self, observations: list[dict], controls: list[np.ndarray]
    ) -> list[dict[str, np.ndarray]]: ...


@dataclass(frozen=True)
class OpenLoopBenchmark:
    """What must not vary between open-loop runs for two results to be comparable.

    `split`, `view` and `use_forecast` are deliberately repeated in `ClosedLoopBenchmark`
    rather than shared through module constants: the two are independent descriptions of a
    problem, and results are only comparable across them if they happen to agree, which is a
    thing to check rather than to enforce by construction.
    """

    split: str
    #: The scenarios to run, named outright; None runs the whole split. Named rather than
    #: counted so that adding buildings to the corpus cannot silently change which problems a
    #: benchmark covers. Generate a set with `python -m i4b_bench.select_scenarios`.
    scenarios: tuple[str, ...] | None
    view: str
    use_forecast: bool
    #: How far ahead a prediction runs, in hours.
    horizon_hours: float
    #: The context lengths to sweep, in days.
    history_days: tuple[float, ...]
    seeds: int
    probes: int
    probe_amplitude: float
    #: Zero, unlike the closed loop: forecast error is what this measures, so pulling the
    #: forecast toward the current sensor reading would correct away the thing under test.
    forecast_correction: float


#: Named settings live as readable TOML beside the code, one file per set, so a configuration is
#: a thing to send someone, diff against theirs, and cite. They are in the package rather than in
#: the dataset directory because they define what the benchmark *measures*, which is a decision
#: that belongs under version control -- the dataset directory is not.
CONFIG = Path(__file__).parent / "config" / "open_loop"


def open_loop_setting(name: str = "benchmark") -> OpenLoopBenchmark:
    """Load a named setting, e.g. `fast`, `benchmark`, `full`."""
    path = CONFIG / f"{name}.toml"
    if not path.exists():
        have = sorted(p.stem for p in CONFIG.glob("*.toml"))
        raise KeyError(f"unknown setting {name!r}, have {have}")
    body = tomllib.loads(path.read_text())
    body = {k: tuple(v) if isinstance(v, list) else v for k, v in body.items()}
    return OpenLoopBenchmark(**body)


def eval_benchmark_open_loop(
    predictor: Predictor,
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir=None,
    setting: OpenLoopBenchmark | None = None,
) -> pd.DataFrame:
    """Score a predictor on the open-loop benchmark.

    Runs every scenario in the setting at every context length, asking two things of each
    window: does the prediction track the building, and does it move when the control moves. A
    model can do the first while failing the second, which makes it useless inside a controller.

    Returns one row per scenario and context length. Gain is pooled within a scenario rather
    than averaged over its windows, so a window where the control barely moved contributes
    little instead of a loud ratio; those windows report NaN and are counted separately.
    """
    setting = setting or open_loop_setting()
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    scenarios = _resolve_scenarios(setting)

    rows = []
    for days in setting.history_days:
        history_length = round(days * PER_DAY)
        for chosen in scenarios:
            result = eval_scenario_open_loop(
                chosen.corpus_id,
                predictor,
                dataset=dataset,
                history_length=history_length,
                planning_steps=round(setting.horizon_hours * 4),
                view=setting.view,
                use_forecast=setting.use_forecast,
                seeds=setting.seeds,
                probes=setting.probes,
                probe_amplitude=setting.probe_amplitude,
                forecast_correction=setting.forecast_correction,
            )
            rows.append(
                {
                    **chosen.metadata(),
                    "history_days": days,
                    "view": setting.view,
                    "n_scenarios": len(scenarios),
                    "use_forecast": setting.use_forecast,
                    "windows": len(result["windows"]),
                    "mae_K": result["mae_K"],
                    "bias_K": result["bias_K"],
                    "gain": result["gain"],
                }
            )
    return pd.DataFrame(rows)


def eval_scenario_open_loop(
    scenario_id: str,
    predictor: Predictor,
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir=None,
    history_length: int = 96,
    planning_steps: int = 96,
    view: str = "realistic",
    use_forecast: bool = True,
    seeds: int = 10,
    probe_amplitude: float = 6.0,
    probes: int = 5,
    probe_kind: str = "offset",
    forecast_correction: float = 0.0,
) -> dict[str, Any]:
    """Score one scenario: point accuracy on the recorded control, and control response.

    Returns `mae_K`, `bias_K`, `gain` and a per-window frame. Accuracy is measured on the control
    the corpus actually applied; control response on counterfactual probes around it, each rolled
    through the plant so the comparison is against what the building would really have done.
    """
    if dataset is None:
        dataset = load_dataset(dataset_dir)

    observations: list[dict] = []
    controls: list[np.ndarray] = []
    realised: list[np.ndarray] = []
    meta: list[dict] = []

    for index, controller_id in _windows(dataset, scenario_id, seeds):
        trajectory = load_controller_data(dataset, controller_id, scenario_id)
        first = history_length
        last = len(trajectory) - planning_steps - 1
        if last <= first:
            continue
        # Evenly spaced through the year rather than sampled: weather dominates these metrics,
        # so coverage of the seasons matters more than randomness, and it needs no seed to
        # reproduce. `index` is the window's position in the requested set.
        start = first + round((last - first) * (index + 0.5) / seeds)

        env = ScenarioEnv(
            scenario_id,
            dataset=dataset,
            initial_controller_id=controller_id,
            max_context_length=history_length,
            planning_steps=planning_steps,
            start_step=start,
            use_forecast=use_forecast,
            view=view,
            forecast_correction=forecast_correction,
        )
        observation, _ = env.reset()
        nominal = trajectory["T_hp_sup_applied"].to_numpy(float)[start : start + planning_steps]
        plans = probe_plans(
            np.random.default_rng(index), nominal, probe_amplitude, probes, probe_kind
        )

        rolled = np.empty((len(plans), planning_steps), dtype=float)
        for k, plan in enumerate(plans):
            env.reset()
            for t, action in enumerate(plan):
                _, _, _, _, info = env.step(float(action))
                rolled[k, t] = info["T_room"]

        observations.append(observation)
        controls.append(plans)
        realised.append(rolled)
        # A window names itself: trajectory (building, weather year, and the controller whose
        # history is handed over) plus where in that trajectory it starts. The context *length*
        # stays a separate column -- the same window scored at 96 and at 2016 steps is one
        # window under two conditions, not two windows.
        anchor = timestamp_of(dataset, scenario_id, start)
        meta.append(
            {
                "window_id": f"{scenario_id}--{controller_id}@{anchor:%Y-%m-%d}",
                "controller_id": controller_id,
                "start_step": start,
            }
        )

    if not observations:
        raise ValueError(f"no usable windows for {scenario_id} at history_length={history_length}")

    predictions = predictor(observations, controls)
    if len(predictions) != len(observations):
        raise ValueError(f"predictor returned {len(predictions)} blocks for {len(observations)}")

    nominal_index = probes // 2
    rows, cross, square = [], 0.0, 0.0
    for info, plans, actual, prediction in zip(meta, controls, realised, predictions):
        if SCORED_CHANNEL not in prediction:
            raise ValueError(
                f"predictor returned {sorted(prediction)}, without {SCORED_CHANNEL!r}"
            )
        predicted = np.asarray(prediction[SCORED_CHANNEL], dtype=float)
        if predicted.shape != actual.shape:
            raise ValueError(f"expected {actual.shape} predictions, got {predicted.shape}")
        error = predicted[nominal_index] - actual[nominal_index]
        window_cross, window_square = gain_terms(actual, predicted)
        cross += window_cross
        square += window_square
        response_rms = float(np.sqrt(window_square / actual.size))
        informative = response_rms >= MIN_RESPONSE_K
        rows.append(
            {
                **info,
                "excitation": EXCITATION.get(info["controller_id"]),
                "mae_K": float(np.abs(error).mean()),
                "bias_K": float(error.mean()),
                "response_K": response_rms,
                "gain": window_cross / window_square if informative else float("nan"),
            }
        )

    frame = pd.DataFrame(rows)
    frame["excitation"] = frame["excitation"].astype(EXCITATION_DTYPE)
    return {
        "scenario_id": scenario_id,
        "informative_windows": int(frame["gain"].notna().sum()),
        "mae_K": float(frame["mae_K"].mean()),
        "bias_K": float(frame["bias_K"].mean()),
        # one pooled slope, not a mean of per-window slopes: a window the control barely moved
        # contributes little to both sums instead of a noisy ratio
        "gain": cross / square if square > 0 else float("nan"),
        "windows": frame,
    }


def _windows(dataset: BenchmarkDataset, scenario_id: str, seeds: int):
    """One window per seed, cycling the controllers so every excitation regime is represented.

    The corpus runs the same building under the same weather with different controllers, and it
    is that matched variation -- identical conditions, different control -- that carries what a
    control signal does.
    """
    rows = dataset.trajectories[dataset.trajectories["scenario_id"] == scenario_id]
    controllers = sorted(rows["controller_id"].unique())
    if not controllers:
        raise ValueError(f"no trajectories for {scenario_id}")
    return [(i, controllers[i % len(controllers)]) for i in range(seeds)]


def _resolve_scenarios(setting) -> list[Scenario]:
    """The scenarios to run, loaded from their files and checked against the declared split.

    Named scenarios make `split` a guard rather than a selector: pasting a training scenario
    into a held-out set fails here instead of quietly producing a number nobody can trust.
    """
    names = setting.scenarios if setting.scenarios is not None else scenario.available()
    chosen = [scenario.load(name) for name in names]
    stray = sorted(s.name for s in chosen if s.split != setting.split)
    if stray:
        raise ValueError(
            f"{len(stray)} scenario(s) are not in the {setting.split!r} split, e.g. {stray[0]}"
        )
    return chosen

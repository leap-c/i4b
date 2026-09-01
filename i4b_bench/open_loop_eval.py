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

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .control_gain import gain_terms, probe_plans
from .dataset import (
    EXCITATION,
    EXCITATION_DTYPE,
    BenchmarkDataset,
    evaluation_scenarios,
    load_controller_data,
    load_dataset,
)
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

    split: str = "test"
    #: How many scenarios of the split to run, evenly spaced; None runs all of them. Part of
    #: the setting because a ten-scenario run and a thirty-scenario run are not comparable, so
    #: the thing that records what was run has to know.
    n_scenarios: int | None = None
    view: str = "realistic"
    use_forecast: bool = True
    horizon: int = 96
    history_lengths: tuple[int, ...] = (96, 2 * 96, 5 * 96, 21 * 96)
    seeds: int = 10
    probes: int = 5
    probe_amplitude: float = 6.0
    #: Zero, unlike the closed loop: forecast error is what this measures, so pulling the
    #: forecast toward the current sensor reading would correct away the thing under test.
    forecast_correction: float = 0.0


OPEN_LOOP = OpenLoopBenchmark()


#: Named settings live as readable JSON beside the code, one file per set, so a configuration is
#: a thing to send someone, diff against theirs, and cite. They are in the package rather than in
#: the dataset directory because they define what the benchmark *measures*, which is a decision
#: that belongs under version control -- the dataset directory is not.
CONFIG = Path(__file__).parent / "config" / "open_loop"


def open_loop_setting(name: str = "benchmark") -> OpenLoopBenchmark:
    """Load a named setting, e.g. `fast`, `benchmark`, `full`."""
    path = CONFIG / f"{name}.json"
    if not path.exists():
        have = sorted(p.stem for p in CONFIG.glob("*.json"))
        raise KeyError(f"unknown setting {name!r}, have {have}")
    body = json.loads(path.read_text())
    body.pop("description", None)
    body = {k: tuple(v) if isinstance(v, list) else v for k, v in body.items()}
    return replace(OPEN_LOOP, **body)


def eval_benchmark_open_loop(
    predictor: Predictor,
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir=None,
    setting: OpenLoopBenchmark = OPEN_LOOP,
) -> pd.DataFrame:
    """Score a predictor on the open-loop benchmark.

    Runs every scenario in the setting at every context length, asking two things of each
    window: does the prediction track the building, and does it move when the control moves. A
    model can do the first while failing the second, which makes it useless inside a controller.

    Returns one row per scenario and context length. Gain is pooled within a scenario rather
    than averaged over its windows, so a window where the control barely moved contributes
    little instead of a loud ratio; those windows report NaN and are counted separately.
    """
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    scenarios = evaluation_scenarios(dataset, setting.split, limit=setting.n_scenarios)

    rows = []
    for history_length in setting.history_lengths:
        for scenario_id in scenarios:
            result = eval_scenario_open_loop(
                scenario_id,
                predictor,
                dataset=dataset,
                history_length=history_length,
                planning_steps=setting.horizon,
                view=setting.view,
                use_forecast=setting.use_forecast,
                seeds=setting.seeds,
                probes=setting.probes,
                probe_amplitude=setting.probe_amplitude,
                forecast_correction=setting.forecast_correction,
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "history_length": history_length,
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
    seed: int = 0,
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

    for controller_id, rng in _windows(dataset, scenario_id, seeds, seed):
        trajectory = load_controller_data(dataset, controller_id, scenario_id)
        first = history_length
        last = len(trajectory) - planning_steps - 1
        if last <= first:
            continue
        start = int(rng.integers(first, last))

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
        plans = probe_plans(rng, nominal, probe_amplitude, probes, probe_kind)

        rolled = np.empty((len(plans), planning_steps), dtype=float)
        for k, plan in enumerate(plans):
            env.reset()
            for t, action in enumerate(plan):
                _, _, _, _, info = env.step(float(action))
                rolled[k, t] = info["T_room"]

        observations.append(observation)
        controls.append(plans)
        realised.append(rolled)
        meta.append({"controller_id": controller_id, "start_step": start})

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


def _windows(dataset: BenchmarkDataset, scenario_id: str, seeds: int, seed: int):
    """One window per seed, cycling the controllers so every excitation regime is represented.

    The corpus runs the same building under the same weather with different controllers, and it
    is that matched variation -- identical conditions, different control -- that carries what a
    control signal does.
    """
    rows = dataset.trajectories[dataset.trajectories["scenario_id"] == scenario_id]
    controllers = sorted(rows["controller_id"].unique())
    if not controllers:
        raise ValueError(f"no trajectories for {scenario_id}")
    rng = np.random.default_rng(seed)
    return [(controllers[i % len(controllers)], rng) for i in range(seeds)]

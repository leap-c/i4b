"""Closed-loop controller evaluation on I4B benchmark scenarios.

See docs/EVAL_SPEC.md for the full specification.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import BenchmarkDataset, evaluation_scenarios, load_dataset
from .scenario_env import ScenarioEnv


@dataclass(frozen=True)
class ClosedLoopBenchmark:
    """What must not vary between closed-loop runs.

    These were defaults on `eval_scenario_closed_loop` rather than configuration, so a
    benchmark run could not pin them and two runs could differ silently on the context length
    or the controller their history was seeded from.
    """

    split: str = "test"
    #: How many scenarios of the split to run, evenly spaced; None runs all of them. Part of
    #: the setting because a ten-scenario run and a thirty-scenario run are not comparable, so
    #: the thing that records what was run has to know. Ten is a night's run for a controller
    #: that plans in ~2 s a step, or under an hour if the scenarios are distributed.
    n_scenarios: int | None = 10
    view: str = "realistic"
    use_forecast: bool = True
    planning_steps: int = 96
    #: Generous on purpose. The buffer costs only memory, and a context shorter than a method
    #: wants silently penalises it -- a foundation model asking for three weeks got one day.
    max_context_length: int = 21 * 96
    initial_context_length: int | None = None
    #: Excited rather than nominal: the handed-over history then contains control variation the
    #: room actually responded to, which is what makes the dynamics identifiable from it at all.
    #: It is also less of a head start for controllers that happen to resemble the MPC that
    #: produced the state. Deliberately one of the original controllers -- the open-loop
    #: excitation levels carry a weather inconsistency that would propagate into every run.
    initial_controller_id: str = "mpc-aprbs-high"
    #: Two weeks. Weather dominates the level of these metrics -- a colder week costs 5x the
    #: comfort violation of a mild one -- so a short window measures the week as much as the
    #: controller. Comparisons stay paired (same scenario, same window) so weather cancels in
    #: differences, but absolute numbers are only meaningful alongside the window they came from.
    evaluation_steps: int | None = 14 * 96
    #: A controller would correct an archived forecast against its own sensor, so unlike the
    #: open loop this is not zero.
    forecast_correction: float = 0.5


CLOSED_LOOP = ClosedLoopBenchmark()


#: Named settings live as readable JSON beside the code, one file per set, so a configuration is
#: a thing to send someone, diff against theirs, and cite. They are in the package rather than in
#: the dataset directory because they define what the benchmark *measures*, which is a decision
#: that belongs under version control -- the dataset directory is not.
CONFIG = Path(__file__).parent / "config" / "closed_loop"


def closed_loop_setting(name: str = "benchmark") -> ClosedLoopBenchmark:
    """Load a named setting, e.g. `fast`, `benchmark`, `full`."""
    path = CONFIG / f"{name}.json"
    if not path.exists():
        have = sorted(p.stem for p in CONFIG.glob("*.json"))
        raise KeyError(f"unknown setting {name!r}, have {have}")
    body = json.loads(path.read_text())
    body.pop("description", None)
    body = {k: tuple(v) if isinstance(v, list) else v for k, v in body.items()}
    return replace(CLOSED_LOOP, **body)


def eval_benchmark_closed_loop(
    controller: Callable[[dict], tuple[float, dict | None]],
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir: str | Path | None = None,
    setting: ClosedLoopBenchmark = CLOSED_LOOP,
) -> pd.DataFrame:
    """Score a controller on the closed-loop benchmark.

    Runs the controller on every scenario in the setting and reports what it cost: energy and
    comfort. Unlike the open-loop side this cannot batch, since each action depends on the state
    the last one produced -- so planning time is reported too, because a controller that wins on
    comfort by thinking for a minute a step has not solved the problem.
    """
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    scenarios = evaluation_scenarios(dataset, setting.split, limit=setting.n_scenarios)

    rows = []
    for scenario_id in scenarios:
        result = eval_scenario_closed_loop(
            scenario_id,
            controller,
            dataset=dataset,
            initial_controller_id=setting.initial_controller_id,
            max_context_length=setting.max_context_length,
            initial_context_length=setting.initial_context_length,
            planning_steps=setting.planning_steps,
            n_evaluation_steps=setting.evaluation_steps,
            use_forecast=setting.use_forecast,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "view": setting.view,
                "n_scenarios": len(scenarios),
                "use_forecast": setting.use_forecast,
                "steps": len(result["trajectory"]),
                "energy_kwh": result["energy_kwh"],
                "comfort_violation_degree_hours": result["comfort_violation_degree_hours"],
                "planning_seconds_mean": result["planning_seconds_mean"],
            }
        )
    return pd.DataFrame(rows)


def eval_scenario_closed_loop(
    scenario_id: str,
    controller: Callable[[dict], tuple[float, dict | None]],
    *,
    dataset_dir: str | Path | None = None,
    dataset: BenchmarkDataset | None = None,
    initial_controller_id: str = "mpc-nominal",
    max_context_length: int = 96,
    initial_context_length: int | None = None,
    planning_steps: int = 96,
    n_evaluation_steps: int | None = None,
    start_step: int | None = None,
    use_forecast: bool = False,
) -> dict[str, Any]:
    """Run a controller on one benchmark scenario, closed loop.

    `controller` takes an observation and returns `(action_celsius, plan_or_none)`.
    `initial_controller_id` names the recorded trajectory the history is seeded from, and
    `use_forecast` chooses archived forecasts over the realised weather.

    Returns energy, comfort violation, mean planning time, and the trajectory.

    The controller may return a plan; nothing scores it. A plan's per-channel error is dominated
    by weather -- the control accounts for a few percent of a room's daily movement -- so it does
    not discriminate control quality. Scoring one wants a counterfactual: replay a *perturbed*
    plan and compare the response, which is what `eval_scenario_open_loop` does.
    """
    if dataset is None:
        dataset = load_dataset(dataset_dir)

    env = ScenarioEnv(
        scenario_id,
        dataset=dataset,
        initial_controller_id=initial_controller_id,
        max_context_length=max_context_length,
        initial_context_length=initial_context_length,
        planning_steps=planning_steps,
        start_step=start_step,
        use_forecast=use_forecast,
    )

    n_steps = n_evaluation_steps if n_evaluation_steps is not None else env.max_steps
    n_steps = min(n_steps, env.max_steps)

    obs, _ = env.reset()

    trajectory_rows = []

    for step_i in range(n_steps):
        started = time.perf_counter()
        action, plan = controller(obs)
        planning_seconds = time.perf_counter() - started
        obs, _reward, terminated, truncated, info = env.step(action)

        # Record trajectory
        trajectory_rows.append(
            {
                "step": step_i,
                "timestamp_utc": obs["history"]["timestamp"][-1],
                "T_room": info["T_room"],
                "T_hp_ret": obs["state"]["T_hp_ret"],
                "T_hp_sup_applied": info["u"],
                "Q_el_kWh": info.get("Q_el_kWh", 0.0),
                "comfort_violation_dh": info.get("dev_sum", 0.0),
                "dev_neg_max": info.get("dev_max", 0.0),
                "planning_seconds": planning_seconds,
            }
        )

        if terminated or truncated:
            break

    trajectory = pd.DataFrame(trajectory_rows)

    return {
        "energy_kwh": trajectory["Q_el_kWh"].sum(),
        "planning_seconds_mean": trajectory["planning_seconds"].mean(),
        "comfort_violation_degree_hours": trajectory["comfort_violation_dh"].sum(),
        "trajectory": trajectory,
    }

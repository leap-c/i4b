"""Closed-loop controller evaluation on I4B benchmark scenarios.

See specs/EVAL_SPEC.md for the full specification.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .dataset import (
    PER_DAY,
    BenchmarkDataset,
    evaluation_scenarios,
    load_dataset,
    scenario_metadata,
    step_of,
)
from .scenario_env import ScenarioEnv

CONFIG = Path(__file__).parent / "config" / "closed_loop"


@dataclass(frozen=True)
class Episode:
    """One problem instance: a building, the dates it runs between, and what seeded its history."""

    building: str
    start: date
    end: date
    #: Whose recorded run fills the context handed over at the start. Excited rather than
    #: nominal: the history then contains control variation the room actually responded to,
    #: which is what makes the dynamics identifiable from it, and it is less of a head start for
    #: controllers that happen to resemble the MPC that produced the state.
    controller: str


@dataclass(frozen=True)
class ClosedLoopBenchmark:
    """What must not vary between closed-loop runs for two results to be comparable."""

    split: str
    view: str
    use_forecast: bool
    planning_horizon: int
    #: Generous on purpose. The buffer costs only memory, and a context shorter than a method
    #: wants silently penalises it -- a foundation model asking for three weeks got one day.
    context_days: float
    #: A controller would correct an archived forecast against its own sensor, so unlike the
    #: open loop this is not zero.
    forecast_correction: float
    scenarios: dict[str, Episode]


def closed_loop_setting(name: str = "benchmark") -> ClosedLoopBenchmark:
    """Load a named setting from `config/closed_loop/<name>.yaml`."""
    path = CONFIG / f"{name}.yaml"
    if not path.exists():
        have = sorted(p.stem for p in CONFIG.glob("*.yaml"))
        raise KeyError(f"unknown setting {name!r}, have {have}")
    body = yaml.safe_load(path.read_text())
    episodes = {name: Episode(**entry) for name, entry in body["scenarios"].items()}
    return ClosedLoopBenchmark(**body["common"], scenarios=episodes)


def eval_benchmark_closed_loop(
    controller: Callable[[dict], tuple[float, dict | None]],
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir: str | Path | None = None,
    setting: ClosedLoopBenchmark | None = None,
) -> pd.DataFrame:
    """Score a controller on the closed-loop benchmark.

    Runs the controller on every scenario in the setting and reports what it cost: energy and
    comfort. Unlike the open-loop side this cannot batch, since each action depends on the state
    the last one produced -- so planning time is reported too, because a controller that wins on
    comfort by thinking for a minute a step has not solved the problem.
    """
    setting = setting or closed_loop_setting()
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    _check_split(dataset, setting)

    context = round(setting.context_days * PER_DAY)
    rows = []
    for name, episode in setting.scenarios.items():
        start = step_of(dataset, episode.building, episode.start)
        end = step_of(dataset, episode.building, episode.end)
        result = eval_scenario_closed_loop(
            episode.building,
            controller,
            dataset=dataset,
            initial_controller_id=episode.controller,
            max_context_length=context,
            initial_context_length=context,
            planning_steps=setting.planning_horizon,
            n_evaluation_steps=end - start,
            start_step=start,
            use_forecast=setting.use_forecast,
        )
        rows.append(
            {
                "scenario": name,
                **scenario_metadata(dataset, episode.building),
                "start": str(episode.start),
                "end": str(episode.end),
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


def _check_split(dataset, setting) -> None:
    """Every named building must belong to the declared split."""
    available = set(evaluation_scenarios(dataset, setting.split))
    stray = sorted({e.building for e in setting.scenarios.values()} - available)
    if stray:
        raise ValueError(
            f"{len(stray)} building(s) are not in the {setting.split!r} split, e.g. {stray[0]}"
        )

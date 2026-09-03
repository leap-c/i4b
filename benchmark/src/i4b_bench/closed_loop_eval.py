"""Closed-loop controller evaluation on I4B benchmark scenarios.

See specs/EVAL_SPEC.md for the full specification.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
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
    #: Bounds of the episode. A date means midnight UTC; a datetime names any time on the
    #: corpus' 15-minute grid.
    start: date | datetime
    end: date | datetime
    #: The recorded run whose history seeds the context at `start`.
    controller: str


@dataclass(frozen=True)
class ClosedLoopBenchmark:
    """What must not vary between closed-loop runs for two results to be comparable."""

    split: str
    view: str
    use_forecast: bool
    planning_horizon: int
    #: How much history the context holds. Sized for the most demanding method, since a shorter
    #: context silently penalises anything that wanted more.
    context_days: float
    #: How far an archived forecast is pulled toward the current sensor reading, in [0, 1].
    forecast_correction: float
    scenarios: dict[str, Episode]


def closed_loop_setting(name: str = "benchmark") -> ClosedLoopBenchmark:
    """Load a named setting from `config/closed_loop/<name>.yaml`.

    Parameters
    ----------
    name : str
        Setting stem, e.g. ``"benchmark"`` or ``"fast_eval"``.

    Returns
    -------
    ClosedLoopBenchmark
        The `common` block plus one `Episode` per named scenario.
    """
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

    Episodes run sequentially -- each action depends on the state the last one produced -- but
    they are independent of one another, so a caller may distribute them.

    Parameters
    ----------
    controller : callable
        ``controller(observation) -> (action_celsius, plan_or_none)``.
    dataset : BenchmarkDataset, optional
        An already-loaded corpus. Loaded from `dataset_dir` when omitted.
    dataset_dir : str or Path, optional
        Corpus directory. Defaults to the bundled `production/`.
    setting : ClosedLoopBenchmark, optional
        What to run. Defaults to `closed_loop_setting("benchmark")`.

    Returns
    -------
    pandas.DataFrame
        One row per episode: `energy_kwh`, `comfort_violation_degree_hours`,
        `planning_seconds_mean`, the dates run, and the building's provenance.
    """
    setting = setting or closed_loop_setting()
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    _check_split(dataset, setting)

    context = round(setting.context_days * PER_DAY)
    # One cache for this call, shared by every episode: two episodes on the same building would
    # otherwise each pay ~40 s to prepare the same archived forecast runs. It dies with the
    # call, so a memo can never be served against a corpus it was not built from.
    forecast_cache: dict = {}
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
            forecast_cache=forecast_cache,
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
    forecast_cache: dict | None = None,
) -> dict[str, Any]:
    """Run a controller on one benchmark scenario, closed loop.

    Parameters
    ----------
    scenario_id : str
        The building to run, e.g. ``"BG.N.SFH.02.Gen.ReEx.001.001--period_b"``.
    controller : callable
        ``controller(observation) -> (action_celsius, plan_or_none)``. The plan is recorded but
        not scored.
    dataset_dir : str or Path, optional
        Corpus directory. Defaults to the bundled `production/`.
    dataset : BenchmarkDataset, optional
        An already-loaded corpus. Loaded from `dataset_dir` when omitted.
    initial_controller_id : str
        Recorded trajectory whose history seeds the context.
    max_context_length : int
        Steps of history kept in the rolling buffer.
    initial_context_length : int, optional
        Steps of history seeded at reset. Defaults to `max_context_length`.
    planning_steps : int
        Forecast horizon handed over each step, in steps.
    start_step : int, optional
        Step index to start at. Defaults to the scenario's own start.
    use_forecast : bool
        Use archived forecast runs rather than the realised weather.
    forecast_cache : dict, optional
        A caller-owned dict memoising prepared forecast runs across episodes; see `ScenarioEnv`.

    Returns
    -------
    dict
        `energy_kwh`, `comfort_violation_degree_hours`, `planning_seconds_mean`, and the
        per-step `trajectory` frame.
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
        forecast_cache=forecast_cache,
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

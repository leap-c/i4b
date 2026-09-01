"""Closed-loop controller evaluation on I4B benchmark scenarios.

See docs/EVAL_SPEC.md for the full specification.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from .dataset import (
    BENCHMARK,
    BenchmarkDataset,
    BenchmarkSetting,
    evaluation_scenarios,
    load_dataset,
)
from .scenario_env import ScenarioEnv


def eval_scenario_closed_loop(
    scenario_id: str,
    controller: Callable[[dict], Tuple[float, dict | None]],
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

    Parameters
    ----------
    scenario_id : str
        Identifies the building + period combination.
    controller : callable
        Takes an observation dict, returns ``(action_celsius, plan_or_none)``.
    dataset_dir : str, Path, or None
        Path to the benchmark dataset directory. Defaults to
        ``<repo>/production``. Ignored if ``dataset`` is provided.
    dataset : BenchmarkDataset or None
        Pre-loaded dataset. If provided, ``dataset_dir`` is ignored.
        Useful to avoid reloading when running multiple evaluations.
    initial_controller_id : str
        Controller whose recorded trajectory provides the initial history.
    max_context_length : int
        Maximum number of past timesteps in the history buffer.
    initial_context_length : int or None
        Number of past timesteps seeded from the recorded trajectory on reset.
        Defaults to ``max_context_length``.
    planning_steps : int
        Number of future timesteps in the forecast.
    n_evaluation_steps : int or None
        Number of steps to evaluate. None = run to end of scenario.
    start_step : int or None
        Timestep offset to begin evaluation. None = ``initial_context_length``.
    use_forecast : bool
        True = archived forecasts, False = oracle weather.

    Returns
    -------
    dict with keys: energy_kwh, comfort_violation_degree_hours,
    planning_seconds_mean, trajectory (DataFrame).

    The controller still returns a plan; nothing scores it here. A plan's per-channel RMSE is
    dominated by weather -- over this corpus the control accounts for a few percent of the
    room's daily movement -- so it does not discriminate control quality. Scoring a plan
    wants a counterfactual: replay a *perturbed* plan and compare the response, which the
    snapshot/restore in the removed helper already had the hard part of.
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


def eval_benchmark_closed_loop(
    controller: Callable[[dict], Tuple[float, dict | None]],
    *,
    dataset: BenchmarkDataset | None = None,
    dataset_dir: str | Path | None = None,
    setting: BenchmarkSetting = BENCHMARK,
    scenarios: Sequence[str] | None = None,
    limit: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run a controller over the agreed benchmark, one row per scenario.

    The open-loop counterpart batches its model calls; this one cannot. Every action depends on
    the state the previous action produced, so the scenarios are simply run in turn.
    """
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    if scenarios is None:
        scenarios = evaluation_scenarios(dataset, setting.split, limit=limit)

    rows = []
    for scenario_id in scenarios:
        result = eval_scenario_closed_loop(
            scenario_id,
            controller,
            dataset=dataset,
            planning_steps=setting.planning_steps,
            use_forecast=setting.use_forecast,
            **kwargs,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "view": setting.view,
                "use_forecast": setting.use_forecast,
                "steps": len(result["trajectory"]),
                "energy_kwh": result["energy_kwh"],
                "comfort_violation_degree_hours": result["comfort_violation_degree_hours"],
                "planning_seconds_mean": result["planning_seconds_mean"],
            }
        )
    return pd.DataFrame(rows)

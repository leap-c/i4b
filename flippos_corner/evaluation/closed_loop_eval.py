"""Closed-loop controller evaluation on I4B benchmark scenarios.

See specs/EVAL_SPEC.md for the full specification.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from dataset import BenchmarkDataset, load_dataset
from scenario_env import STATE_CHANNELS, ScenarioEnv


def run_evaluation(
    scenario_id: str,
    controller: Callable[[dict], Tuple[float, dict | None]],
    *,
    dataset_dir: str | Path | None = None,
    dataset: BenchmarkDataset | None = None,
    initial_controller_id: str = "mpc-nominal",
    history_length: int = 96,
    planning_steps: int = 12,
    n_evaluation_steps: int | None = None,
    start_step: int | None = None,
    use_forecast: bool = False,
    plan_eval_frequency: int | None = None,
) -> dict[str, Any]:
    """Run a closed-loop evaluation of a controller on a benchmark scenario.

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
    history_length : int
        Number of past timesteps provided to the controller.
    planning_steps : int
        Number of future timesteps in the forecast.
    n_evaluation_steps : int or None
        Number of steps to evaluate. None = run to end of scenario.
    start_step : int or None
        Timestep offset to begin evaluation. None = ``history_length``.
    use_forecast : bool
        True = archived forecasts, False = oracle weather.
    plan_eval_frequency : int or None
        Evaluate plan quality every N steps. None = skip plan evaluation.

    Returns
    -------
    dict with keys: energy_kwh, comfort_violation_degree_hours,
    plan_quality (DataFrame or None), trajectory (DataFrame).
    """
    if dataset is None:
        dataset = load_dataset(dataset_dir)

    env = ScenarioEnv(
        scenario_id,
        dataset=dataset,
        initial_controller_id=initial_controller_id,
        history_length=history_length,
        planning_steps=planning_steps,
        start_step=start_step,
        use_forecast=use_forecast,
    )

    n_steps = n_evaluation_steps if n_evaluation_steps is not None else env.max_steps
    n_steps = min(n_steps, env.max_steps)

    obs, _ = env.reset()

    trajectory_rows = []
    plan_quality_rows = []

    for step_i in range(n_steps):
        action, plan = controller(obs)
        obs, _reward, terminated, truncated, info = env.step(action)

        # Record trajectory
        trajectory_rows.append(
            {
                "step": step_i,
                "timestamp_utc": obs["history"]["timestamp"][-1],
                "T_room": info["T_room"],
                "T_hp_sup_applied": info["u"],
                "Q_el_kWh": info.get("Q_el_kWh", 0.0),
                "comfort_violation_dh": info.get("dev_sum", 0.0),
                "dev_neg_max": info.get("dev_max", 0.0),
            }
        )

        # Plan quality evaluation
        if (
            plan is not None
            and plan_eval_frequency is not None
            and step_i % plan_eval_frequency == 0
        ):
            plan_rmse = _evaluate_plan(env, plan, planning_steps)
            if plan_rmse is not None:
                plan_rmse["eval_step"] = step_i
                plan_quality_rows.append(plan_rmse)

        if terminated or truncated:
            break

    trajectory = pd.DataFrame(trajectory_rows)
    plan_quality = pd.DataFrame(plan_quality_rows) if plan_quality_rows else None

    return {
        "energy_kwh": trajectory["Q_el_kWh"].sum(),
        "comfort_violation_degree_hours": trajectory["comfort_violation_dh"].sum(),
        "plan_quality": plan_quality,
        "trajectory": trajectory,
    }


def _evaluate_plan(
    env: ScenarioEnv, plan: dict, planning_steps: int
) -> dict[str, float] | None:
    """Simulate the planned actions on the building and compare to the plan's
    predicted states. Returns per-channel RMSE or None if plan is incomplete."""
    if "T_hp_sup_applied" not in plan or "T_room" not in plan:
        return None

    planned_actions = plan["T_hp_sup_applied"]
    n = min(len(planned_actions), planning_steps)
    if n == 0:
        return None

    # Snapshot current env state to restore after simulation
    saved_t = env._env.t
    saved_state = env._env.state.copy()  # type: ignore[union-attr]

    simulated: dict[str, list[float]] = {ch: [] for ch in STATE_CHANNELS}
    for i in range(n):
        action = float(planned_actions[i])
        normalized = env._env.normalize_action(np.array(action))
        obs, _, terminated, _, _ = env._env.step(normalized)
        for j, ch in enumerate(STATE_CHANNELS):
            simulated[ch].append(float(obs[j]))
        if terminated:
            break

    # Restore env state
    env._env.t = saved_t
    env._env.state = saved_state

    actual_n = len(simulated["T_room"])
    result = {}
    for ch in STATE_CHANNELS:
        sim_arr = np.array(simulated[ch][:actual_n])
        if ch in plan and len(plan[ch]) >= actual_n:
            plan_arr = np.array(plan[ch][:actual_n], dtype=np.float32)
            result[f"{ch}_rmse"] = float(np.sqrt(np.mean((sim_arr - plan_arr) ** 2)))

    return result

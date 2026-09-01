#!/usr/bin/env python3
"""Example: evaluate the i4b CasADi MPC controller on a benchmark scenario.

Usage:
    python scripts/evaluation/example_mpc_eval.py \
        --dataset production \
        --scenario "sfh_1984_de_1--period_a" \
        --steps 96

The MPC controller uses only the forecast (oracle weather) and ignores
history — it re-solves from the current state at every timestep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import i4b.models.model_hvac as model_hvac
from i4b_bench.corpus import load_params
from i4b.models.model_buildings import Building
from i4b.controller.mpc.casadi_framework import MPC_solver

from i4b_bench import load_dataset
from i4b_bench import eval_scenario_closed_loop


def make_mpc_controller(building_params: dict, horizon: int = 12):
    """Create an MPC controller callable compatible with ScenarioEnv.

    The returned callable takes an observation dict and returns
    (action_celsius, None).
    """
    mdot_hp = building_params["mdot_hp"]

    building_model = Building(
        params=building_params,
        mdot_hp=mdot_hp,
        method="4R3C",
        T_room_set_lower=20,
        T_room_set_upper=26,
    )
    hp_model = model_hvac.Heatpump_AW(mdot_HP=mdot_hp)

    # MPC problem dimensions for 4R3C
    nx = 3   # T_room, T_wall, T_hp_ret
    npar = 4 # T_amb, Qdot_gains, T_room_set_lower, grid_signal
    nc = 4   # constraints
    ns = 2   # slack variables
    h = 900  # timestep in seconds

    mpc = MPC_solver(
        resultdir="/tmp",
        resultfile="eval_mpc",
        hp_model=hp_model,
        building_model=building_model,
        nx=nx,
        npar=npar,
        nc=nc,
        ns=ns,
        h=h,
        nk=horizon,
        ws=1,
    )

    def controller(obs: dict) -> tuple[float, dict | None]:
        # Extract current state
        state = obs["state"]
        xk = np.array([state["T_room"], state["T_wall"], state["T_hp_ret"]])

        # Build the parameter matrix P for the MPC horizon
        # P has shape (nk+1, 4): [T_amb, Qdot_gains_kW, T_room_set_lower, grid]
        forecast = obs["forecast"]
        n_forecast = len(forecast["T_amb"])
        n_needed = horizon + 1

        # Pad forecast if shorter than horizon
        T_amb_forecast = np.zeros(n_needed, dtype=np.float32)
        Qdot_forecast = np.zeros(n_needed, dtype=np.float32)

        n_copy = min(n_forecast, n_needed)
        T_amb_forecast[:n_copy] = forecast["T_amb"][:n_copy]
        Qdot_forecast[:n_copy] = forecast["Qdot_gains"][:n_copy]

        # Hold last value if forecast is shorter
        if n_copy < n_needed:
            T_amb_forecast[n_copy:] = T_amb_forecast[n_copy - 1]
            Qdot_forecast[n_copy:] = Qdot_forecast[n_copy - 1]

        P = np.column_stack([
            T_amb_forecast,
            Qdot_forecast / 1000.0,      # MPC expects kW
            np.full(n_needed, 20.0),      # T_room_set_lower
            np.ones(n_needed),            # grid signal (constant = energy-efficient mode)
        ])

        # Solve MPC
        mpc.update_NLP(xk)
        result = mpc.solve_NLP(P)
        uk = result[0]
        action = float(np.array(uk).flatten()[0])

        return action, None

    return controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("production"))
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--steps", type=int, default=96,
                        help="Number of evaluation steps (default: 96 = 1 day)")
    parser.add_argument("--horizon", type=int, default=12,
                        help="MPC horizon in steps (default: 12 = 3 hours)")
    parser.add_argument("--max-context-length", type=int, default=96)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    dataset = load_dataset(args.dataset)

    # Extract building params for this scenario
    scenario_row = dataset.scenarios[
        dataset.scenarios["scenario_id"] == args.scenario
    ].iloc[0]
    building_id = scenario_row["building_id"]
    building_params = load_params(dataset.buildings, building_id)

    print(f"Building: {building_id}")
    print(f"Scenario: {args.scenario}")
    print(f"MPC horizon: {args.horizon} steps ({args.horizon * 15 / 60:.1f} h)")
    print(f"Evaluation steps: {args.steps} ({args.steps * 15 / 60:.1f} h)")

    controller = make_mpc_controller(building_params, horizon=args.horizon)

    results = eval_scenario_closed_loop(
        dataset=dataset,
        scenario_id=args.scenario,
        controller=controller,
        max_context_length=args.max_context_length,
        planning_steps=args.horizon + 1,  # forecast needs horizon+1 for MPC
        n_evaluation_steps=args.steps,
        use_forecast=False,  # oracle weather, same as dataset generation
    )

    print(f"\n--- Results ({args.steps} steps) ---")
    print(f"Energy consumption: {results['energy_kwh']:.3f} kWh")
    print(f"Comfort violation:  {results['comfort_violation_degree_hours']:.4f} degree-hours")

    traj = results["trajectory"]
    print(f"\nTrajectory: {len(traj)} rows")
    print(f"  T_room range: [{traj['T_room'].min():.2f}, {traj['T_room'].max():.2f}] C")
    print(f"  Supply range:  [{traj['T_hp_sup_applied'].min():.2f}, {traj['T_hp_sup_applied'].max():.2f}] C")


if __name__ == "__main__":
    main()

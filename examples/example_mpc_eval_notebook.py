import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, mo, np


@app.cell
def _(mo):
    mo.md("""
    # MPC Controller Evaluation

    Evaluate the i4b CasADi MPC controller on a benchmark scenario using the
    closed-loop evaluation framework. The MPC re-solves from the current state
    at every timestep using oracle (realized) weather as forecast.
    """)
    return


@app.cell
def _(Path, mo):
    try:
        default = Path(__file__).resolve().parents[2] / "production"
    except NameError:
        default = Path("production").resolve()
    dataset_input = mo.ui.text(
        value=str(default), label="Dataset directory", full_width=True
    )
    dataset_input
    return (dataset_input,)


@app.cell
def _(Path, dataset_input):
    import sys

    eval_dir = Path(__file__).resolve().parent if "__file__" in dir() else Path(".").resolve()
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    from i4b.evaluation import load_dataset

    dataset = load_dataset(Path(dataset_input.value).expanduser())
    scenarios = dataset.scenarios
    print(f"Loaded {len(scenarios)} scenarios")
    return dataset, scenarios


@app.cell
def _(mo, scenarios):
    scenario_options = sorted(scenarios["scenario_id"].tolist())
    scenario_select = mo.ui.dropdown(
        options=scenario_options,
        value=scenario_options[0] if scenario_options else None,
        label="Scenario",
        searchable=True,
    )
    horizon_select = mo.ui.slider(
        start=4, stop=48, step=4, value=12, label="MPC horizon (steps)"
    )
    steps_select = mo.ui.slider(
        start=12, stop=672, step=12, value=96, label="Evaluation steps"
    )
    history_select = mo.ui.slider(
        start=12, stop=672, step=12, value=96, label="Max context length"
    )
    mo.vstack([
        mo.md("## Configuration"),
        scenario_select,
        mo.hstack([horizon_select, steps_select, history_select]),
    ])
    return history_select, horizon_select, scenario_select, steps_select


@app.cell
def _(horizon_select, mo, steps_select):
    horizon = horizon_select.value
    n_steps = steps_select.value
    mo.md(
        f"**MPC horizon:** {horizon} steps ({horizon * 15 / 60:.1f} h) · "
        f"**Evaluation:** {n_steps} steps ({n_steps * 15 / 60:.1f} h)"
    )
    return horizon, n_steps


@app.cell
def _(dataset, mo, scenario_select):
    from i4b.benchmark import load_params
    from i4b.models.model_buildings import Building
    import i4b.models.model_hvac as model_hvac
    from i4b.controller.mpc.casadi_framework import MPC_solver

    scenario_id = scenario_select.value
    scenario_row = dataset.scenarios[
        dataset.scenarios["scenario_id"] == scenario_id
    ].iloc[0]
    building_id = scenario_row["building_id"]
    building_params = load_params(dataset.buildings, building_id)

    mo.md(f"**Building:** `{building_id}` · **Scenario:** `{scenario_id}`")
    return Building, MPC_solver, building_params, model_hvac, scenario_id


@app.cell
def _(Building, MPC_solver, building_params, horizon, model_hvac, np):
    def make_mpc_controller(params: dict, nk: int):
        mdot_hp = params["mdot_hp"]
        bldg = Building(
            params=params, mdot_hp=mdot_hp, method="4R3C",
            T_room_set_lower=20, T_room_set_upper=26,
        )
        hp = model_hvac.Heatpump_AW(mdot_HP=mdot_hp)
        mpc = MPC_solver(
            resultdir="/tmp", resultfile="eval_mpc",
            hp_model=hp, building_model=bldg,
            nx=3, npar=4, nc=4, ns=2, h=900, nk=nk, ws=1,
        )

        def controller(obs: dict) -> tuple[float, dict | None]:
            state = obs["state"]
            xk = np.array([state["T_room"], state["T_wall"], state["T_hp_ret"]])

            forecast = obs["forecast"]
            n_forecast = len(forecast["T_amb"])
            n_needed = nk + 1

            T_amb_f = np.zeros(n_needed, dtype=np.float32)
            Qdot_f = np.zeros(n_needed, dtype=np.float32)
            n_copy = min(n_forecast, n_needed)
            T_amb_f[:n_copy] = forecast["T_amb"][:n_copy]
            Qdot_f[:n_copy] = forecast["Qdot_gains"][:n_copy]
            if n_copy < n_needed:
                T_amb_f[n_copy:] = T_amb_f[n_copy - 1]
                Qdot_f[n_copy:] = Qdot_f[n_copy - 1]

            P = np.column_stack([
                T_amb_f,
                Qdot_f / 1000.0,
                np.full(n_needed, 20.0),
                np.ones(n_needed),
            ])

            mpc.update_NLP(xk)
            result = mpc.solve_NLP(P)
            action = float(np.array(result[0]).flatten()[0])
            return action, None

        return controller

    mpc_controller = make_mpc_controller(building_params, horizon)
    return (mpc_controller,)


@app.cell
def _(
    dataset,
    history_select,
    horizon,
    mo,
    mpc_controller,
    n_steps,
    scenario_id,
):
    from i4b.evaluation import run_evaluation

    mo.status.spinner(title="Running MPC evaluation...")

    results = run_evaluation(
        dataset=dataset,
        scenario_id=scenario_id,
        controller=mpc_controller,
        max_context_length=history_select.value,
        planning_steps=horizon + 1,
        n_evaluation_steps=n_steps,
        use_forecast=False,
    )
    return (results,)


@app.cell
def _(mo, results):
    energy = results["energy_kwh"]
    comfort = results["comfort_violation_degree_hours"]
    n_rows = len(results["trajectory"])

    mo.vstack([
        mo.md("## Results"),
        mo.hstack([
            mo.stat(label="Energy", value=f"{energy:.2f} kWh"),
            mo.stat(label="Comfort violation", value=f"{comfort:.4f} dh"),
            mo.stat(label="Trajectory rows", value=str(n_rows)),
        ]),
    ])
    return


@app.cell
def _(go, make_subplots, mo, results):
    traj = results["trajectory"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Room temperature", "Supply temperature", "Energy per step"),
    )
    fig.add_trace(
        go.Scatter(x=traj["step"], y=traj["T_room"], name="T_room", line={"color": "#4477aa"}),
        row=1, col=1,
    )
    fig.add_hline(y=20, line_dash="dash", line_color="gray", row=1, col=1, annotation_text="Comfort lower")
    fig.add_hline(y=26, line_dash="dash", line_color="gray", row=1, col=1, annotation_text="Comfort upper")

    fig.add_trace(
        go.Scatter(x=traj["step"], y=traj["T_hp_sup_applied"], name="T_hp_sup", line={"color": "#cc6677"}),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=traj["step"], y=traj["Q_el_kWh"], name="Q_el", line={"color": "#228833"}),
        row=3, col=1,
    )
    fig.update_yaxes(title_text="Temperature [C]", row=1, col=1)
    fig.update_yaxes(title_text="Temperature [C]", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_layout(
        height=750, hovermode="x unified",
        margin={"l": 55, "r": 20, "t": 40, "b": 40},
    )
    mo.vstack([mo.md("## Trajectory"), fig])
    return


if __name__ == "__main__":
    app.run()

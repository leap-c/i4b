import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from i4b_bench import eval_benchmark_open_loop, load_dataset, open_loop_setting

    return (
        eval_benchmark_open_loop,
        go,
        load_dataset,
        mo,
        np,
        open_loop_setting,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Open-loop evaluation

    Scores a *predictor* rather than a controller: given a context and candidate control
    trajectories, how well does it predict, and does it move when the control moves.

    The point of this notebook is that those are two different questions. A model can track the
    room accurately while ignoring the heat pump entirely, which makes it useless inside a
    controller — so the benchmark reports `mae_K` and `gain` side by side.
    """)
    return


@app.cell
def _(load_dataset, mo, open_loop_setting):
    dataset = load_dataset()
    setting = open_loop_setting("fast_eval")
    mo.md(f"**{len(setting.scenarios)} windows**, view `{setting.view}`, "
          f"horizon {setting.horizon_hours:g} h, {setting.probes} probes at "
          f"±{setting.probe_amplitude:g} K")
    return dataset, setting


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Two predictors

    A predictor takes a batch of observations and their candidate controls, and returns a
    `{channel: (n_plans, horizon)}` dict per window. Both of these are deliberately poor
    forecasters; they differ in whether they react to the control at all.
    """)
    return


@app.cell
def _(np):
    def persistence(observations, controls):
        """Hold the last observed room temperature. Ignores the control completely."""
        return [
            {"T_room": np.full(u.shape, o["history"]["T_room"][-1])}
            for o, u in zip(observations, controls)
        ]

    def integrating(observations, controls, gain_per_step=0.0037):
        """Persistence, plus the accumulated departure of the plan from the last action.

        Crude, but it responds to the control in roughly the way the building does: a room
        integrates its heat input rather than following it instantaneously.
        """
        out = []
        for observation, plans in zip(observations, controls):
            last_room = observation["history"]["T_room"][-1]
            last_action = observation["history"]["T_hp_sup_applied"][-1]
            drift = np.cumsum(plans - last_action, axis=1)
            out.append({"T_room": last_room + gain_per_step * drift})
        return out

    return integrating, persistence


@app.cell
def _(dataset, eval_benchmark_open_loop, integrating, pd, persistence, setting):
    _rows = []
    for _name, _predictor in (("persistence", persistence), ("integrating", integrating)):
        _frame = eval_benchmark_open_loop(_predictor, dataset=dataset, setting=setting)
        _rows.append(
            {"predictor": _name, "mae_K": _frame.mae_K.mean(), "gain": _frame.gain.mean()}
        )
    scores = pd.DataFrame(_rows)
    scores.round(3)
    return (scores,)


@app.cell(hide_code=True)
def _(mo, scores):
    _p = scores.set_index("predictor")
    mo.md(f"""
    `persistence` is the **more accurate** of the two — MAE {_p.loc["persistence", "mae_K"]:.2f} K
    against {_p.loc["integrating", "mae_K"]:.2f} K — and it is useless for control: its gain is
    {_p.loc["persistence", "gain"]:.2f}, meaning its prediction does not move at all when the plan
    does. `integrating` is worse on accuracy and scores a gain of
    {_p.loc["integrating", "gain"]:.2f}, close to the 1.0 of a model that moves exactly as the
    plant does.

    Ranking these two on accuracy alone puts them in the wrong order for anything that has to
    plan.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How gain is measured

    From one anchor the plant is rolled under several perturbed control trajectories — same
    weather, same starting state, only the control differs. Deviations are taken about the mean
    across probes, so everything the probes shared cancels, and `gain` is the slope of the
    model's predicted deviation on the plant's actual one.
    """)
    return


@app.cell
def _(go, np, setting):
    _probes = np.linspace(-setting.probe_amplitude, setting.probe_amplitude, setting.probes)
    _steps = np.arange(round(setting.horizon_hours * 4))
    _fig = go.Figure()
    for _offset in _probes:
        _fig.add_scatter(
            x=_steps / 4,
            y=0.0037 * np.cumsum(np.full_like(_steps, _offset, dtype=float)),
            mode="lines",
            name=f"{_offset:+.0f} K",
        )
    _fig.update_layout(
        title="Predicted room deviation under each probe (integrating predictor)",
        xaxis_title="hours into the horizon",
        yaxis_title="deviation from the nominal plan [K]",
        height=380,
    )
    _fig
    return


if __name__ == "__main__":
    app.run()

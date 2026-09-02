import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from i4b_bench import inspect_window, load_dataset, open_loop_setting

    # Categorical slots 1 and 2, in fixed order: the plant is always blue, the model always
    # orange. Signed probe offsets get the diverging blue<->red pair, gray at the nominal.
    PLANT, MODEL = "#2a78d6", "#eb6834"
    PROBES = ["#184f95", "#6da7ec", "#7a7a76", "#ef8b8a", "#e34948"]
    GRID, INK = "#e4e3df", "#52514e"

    def style(fig, title, xaxis, yaxis, height=380):
        fig.update_layout(
            title=title, height=height, template="simple_white",
            xaxis_title=xaxis, yaxis_title=yaxis,
            font=dict(color=INK, size=12), margin=dict(l=60, r=20, t=50, b=45),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
        )
        fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
        return fig

    return (
        GRID, INK, MODEL, PLANT, PROBES, go, inspect_window, load_dataset,
        mo, np, open_loop_setting, pd, style,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Open-loop evaluation, window by window

    `eval_benchmark_open_loop` reduces each window to four numbers. This notebook shows what
    produced them: the history a model was given, the probes it was asked about, and how its
    answer compares to what the plant did.

    Use it to check that a number is measuring what it claims to.
    """)
    return


@app.cell
def _(load_dataset, mo, open_loop_setting):
    dataset = load_dataset()
    setting = open_loop_setting("benchmark")
    windows = dict(sorted(setting.scenarios.items()))
    picker = mo.ui.dropdown(
        options={f"{n} - {w.controller} - {w.start}": n for n, w in windows.items()},
        value=next(f"{n} - {w.controller} - {w.start}" for n, w in windows.items()),
        label="window",
    )
    context_pick = mo.ui.dropdown(
        options={f"{d:g} d": d for d in setting.context_days},
        value=f"{max(setting.context_days):g} d",
        label="context",
    )
    mo.hstack([picker, context_pick], justify="start", gap=2)
    return context_pick, dataset, picker, setting, windows


@app.cell
def _(context_pick, dataset, inspect_window, np, picker, setting, windows):
    window = windows[picker.value]

    def persistence(observations, controls):
        """Stand-in so the notebook runs anywhere. Swap in a real model below."""
        return [
            {"T_room": np.full(u.shape, o["history"]["T_room"][-1])}
            for o, u in zip(observations, controls)
        ]

    try:  # a real model if this environment has one
        from heat_control.adapter import as_predictor
        from heat_control.models import load as load_model

        predictor = as_predictor(load_model("chronos_2", ["T_room", "T_hp_ret"]), view=setting.view)
        model_name = "chronos_2 (zero-shot)"
    except Exception:
        predictor, model_name = persistence, "persistence (no model available)"

    detail = inspect_window(
        window, predictor, dataset=dataset, setting=setting, context_days=context_pick.value
    )
    return detail, model_name, window


@app.cell(hide_code=True)
def _(detail, mo, model_name, window):
    _r = detail["row"]
    mo.md(f"""
    **{window.building}** &nbsp;·&nbsp; seeded from `{window.controller}` &nbsp;·&nbsp;
    model: `{model_name}`

    | mae_K | bias_K | gain | response_K | realized_share |
    |---|---|---|---|---|
    | {_r["mae_K"]:.3f} | {_r["bias_K"]:+.3f} | {_r["gain"]:.3f} | {_r["response_K"]:.3f} |
    {_r["realized_share"]:.2f} |

    `realized_share` is how much of the requested probe spread the actuator let through. When it
    is near zero the pump never moved and `gain` means nothing -- which is why the set sits in
    the heating season.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## What the model was given: the control in its context")
    return


@app.cell
def _(GRID, INK, detail, go, style):
    _h = detail["history"]
    _fig = go.Figure()
    _fig.add_scatter(x=_h.index, y=_h["T_hp_sup_applied"], mode="lines",
                     line=dict(width=1.6, color="#2a78d6"), name="supply temperature")
    style(_fig, "Applied control over the context", "", "T_hp_sup_applied [C]", height=300)
    _fig.update_layout(showlegend=False)
    _fig
    return


@app.cell(hide_code=True)
def _(detail, mo, np):
    _u = detail["history"]["T_hp_sup_applied"].to_numpy()
    mo.md(f"""
    Standard deviation of the control across the context: **{_u.std():.2f} K**. This is the
    single strongest predictor of whether the dynamics are identifiable from the history -- a
    controller that holds the supply temperature steady leaves nothing to learn the response
    from. Zero-shot gain runs about 0.15 under `mpc-nominal` and 0.89 under `open-loop-aprbs`
    at a one-day context for exactly this reason.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Which controls were tested

    Five counterfactual plans around the one the corpus applied, evenly spaced over
    ±`probe_amplitude`. The plant clips them: `check_hp` collapses the supply temperature
    whenever the pump idles, so the dashed lines are what was asked for and the solid lines are
    what actually happened. **The model is asked about the solid ones** -- comparing its answer
    to a response the plant never produced would inflate the gain.
    """)
    return


@app.cell
def _(PROBES, detail, go, style):
    _t = detail["timestamps"]
    _fig = go.Figure()
    for _i in range(detail["applied"].shape[0]):
        _c = PROBES[_i * (len(PROBES) - 1) // max(detail["applied"].shape[0] - 1, 1)]
        _fig.add_scatter(x=_t, y=detail["requested"][_i], mode="lines", showlegend=False,
                         line=dict(width=1.2, color=_c, dash="dot"), opacity=0.55)
        _fig.add_scatter(x=_t, y=detail["applied"][_i], mode="lines",
                         line=dict(width=2, color=_c), name=f"probe {_i + 1}")
    style(_fig, "Probe plans: requested (dotted) against applied (solid)", "", "T_hp_sup [C]")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How the predictions look

    The plant's room temperature under each probe, against the model's prediction for the same
    probe. Blue is the plant, orange the model; one pair of lines per probe.
    """)
    return


@app.cell
def _(MODEL, PLANT, detail, go, style):
    _t = detail["timestamps"]
    _fig = go.Figure()
    for _i in range(detail["actual"].shape[0]):
        _fig.add_scatter(x=_t, y=detail["actual"][_i], mode="lines",
                         line=dict(width=2, color=PLANT), opacity=0.85,
                         name="plant" if _i == 0 else None, showlegend=_i == 0)
        _fig.add_scatter(x=_t, y=detail["predicted"][_i], mode="lines",
                         line=dict(width=2, color=MODEL, dash="dash"), opacity=0.85,
                         name="model" if _i == 0 else None, showlegend=_i == 0)
    style(_fig, "Room temperature under each probe", "", "T_room [C]", height=420)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What `gain` actually regresses

    Deviations are taken about the mean across probes, so everything the probes shared --
    weather, the starting state, any constant bias -- cancels, and only the response to
    *differences* in control survives. `gain` is the slope of the model's deviation on the
    plant's: 1.0 moves exactly as the plant does, 0.0 ignores the control.
    """)
    return


@app.cell
def _(INK, MODEL, PLANT, detail, go, np, style):
    _plant = detail["actual"] - detail["actual"].mean(axis=0)
    _model = detail["predicted"] - detail["predicted"].mean(axis=0)
    _fig = go.Figure()
    _fig.add_scatter(x=_plant.ravel(), y=_model.ravel(), mode="markers",
                     marker=dict(size=5, color=MODEL, opacity=0.5), name="probe x lead")
    _lim = float(np.abs(_plant).max()) * 1.05 or 1.0
    _fig.add_scatter(x=[-_lim, _lim], y=[-_lim, _lim], mode="lines", name="gain = 1",
                     line=dict(width=1.5, color=PLANT, dash="dot"))
    _g = detail["row"]["gain"]
    _fig.add_scatter(x=[-_lim, _lim], y=[-_lim * _g, _lim * _g], mode="lines",
                     name=f"fitted, gain = {_g:.2f}", line=dict(width=2, color=INK))
    style(_fig, "Model deviation against plant deviation, over probes and leads",
          "plant deviation [K]", "model deviation [K]", height=420)
    _fig
    return


if __name__ == "__main__":
    app.run()

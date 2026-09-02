import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, mo, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # I4B benchmark dataset

    Minimal review of the mapped building catalog, trajectory metadata, split
    manifests, and canonical transitions. This notebook only reads generated
    Parquet files.
    """)


@app.cell(hide_code=True)
def _(Path, mo):
    try:
        default = Path(__file__).resolve().parents[1] / "data" / "corpus"
    except NameError:
        default = Path("data/corpus").resolve()
    directory_input = mo.ui.text(value=str(default), label="Dataset directory", full_width=True)
    directory_input
    return (directory_input,)


@app.cell(hide_code=True)
def _(Path, directory_input, pd):
    directory = Path(directory_input.value).expanduser()

    def read(name):
        path = directory / name
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    buildings = read("buildings.parquet")
    trajectories = read("trajectories.parquet")
    split = read("split.parquet")
    time_split = read("ablation-splits/time.parquet")
    transition_parts = sorted((directory / "transitions").glob("*.parquet"))
    return buildings, directory, split, time_split, trajectories, transition_parts


@app.cell(hide_code=True)
def _(buildings, mo, split, time_split, trajectories, transition_parts):
    inventory = {
        "buildings": len(buildings),
        "trajectories": len(trajectories),
        "primary split rows": len(split),
        "time split rows": len(time_split),
        "transition parts": len(transition_parts),
    }
    mo.ui.table(
        [{"artifact": name, "rows/files": value} for name, value in inventory.items()],
        pagination=False,
        selection=None,
    )


@app.cell(hide_code=True)
def _(buildings, mo):
    if len(buildings):
        _building_summary = (
            buildings.groupby("country_code")
            .agg(
                variants=("building_id", "size"),
                families=("building_family_id", "nunique"),
                median_area_m2=("reference_area_m2", "median"),
                median_mdot_hp=("mdot_hp", "median"),
            )
            .reset_index()
        )
        _building_view = mo.ui.table(_building_summary, pagination=False, selection=None)
    else:
        _building_view = mo.callout("No mapped building catalog found.", kind="warn")
    mo.vstack([mo.md("## Buildings"), _building_view])


@app.cell(hide_code=True)
def _(buildings, px):
    (
        px.scatter(
            buildings,
            x="H_tr" if "H_tr" in buildings else "reference_area_m2",
            y="mdot_hp",
            color="country_code",
            hover_name="building_id",
            render_mode="svg",
            title="Mapped buildings and heat-pump sizing",
        )
        if len(buildings)
        else None
    )


@app.cell(hide_code=True)
def _(buildings, mo, split, trajectories):
    if len(buildings) and len(split) and len(trajectories):
        coverage = split.merge(trajectories, on="trajectory_id", how="left").merge(
            buildings[["building_id", "building_family_id", "country_code"]],
            on="building_id",
            how="left",
        )
        _split_summary = (
            coverage.groupby(["split", "country_code"], dropna=False)
            .agg(
                trajectories=("trajectory_id", "size"),
                families=("building_family_id", "nunique"),
            )
            .reset_index()
        )
        _split_view = mo.ui.table(_split_summary, pagination=False, selection=None)
    else:
        _split_view = mo.callout("No split and trajectory metadata found.", kind="warn")
    mo.vstack([mo.md("## Primary split"), _split_view])


@app.cell(hide_code=True)
def _(mo, trajectories):
    if len(trajectories):
        _controllers = (
            trajectories.groupby("controller_id")
            .size()
            .rename("trajectories")
            .reset_index()
        )
        _controller_view = mo.vstack(
            [
                mo.md("## Controllers"),
                mo.ui.table(_controllers, pagination=False, selection=None),
            ]
        )
    else:
        _controller_view = None
    _controller_view


@app.cell(hide_code=True)
def _(mo, trajectories):
    building_options = sorted(trajectories["building_id"].unique()) if len(trajectories) else []
    period_options = sorted(trajectories["period_id"].unique()) if len(trajectories) else []
    controller_options = (
        sorted(trajectories["controller_id"].unique()) if len(trajectories) else []
    )
    building_select = mo.ui.dropdown(
        options=building_options,
        value=building_options[0] if building_options else None,
        label="Building",
        searchable=True,
    )
    period_select = mo.ui.dropdown(
        options=period_options,
        value=period_options[0] if period_options else None,
        label="Period",
    )
    controller_select = mo.ui.multiselect(
        options=controller_options,
        value=controller_options,
        label="Controllers",
    )
    resolution_select = mo.ui.dropdown(
        options={"15 minutes": 1, "Hourly": 4, "6-hourly": 24, "Daily": 96},
        value="15 minutes",
        label="Detail resolution",
    )
    comparison_controls = mo.vstack(
        [
            mo.md("## Compare controllers by building"),
            mo.hstack(
                [
                    building_select,
                    period_select,
                    resolution_select,
                ],
                widths=[3, 1, 1],
            ),
            controller_select,
        ]
    )
    comparison_controls if building_options else None
    return building_select, controller_select, period_select, resolution_select


@app.cell(hide_code=True)
def _(mo, pd, period_select):
    period_bounds = {
        "period_a": ("2024-04-01", "2025-03-31"),
        "period_b": ("2025-04-01", "2026-03-31"),
    }
    period_start, period_end = period_bounds[period_select.value]
    detail_start = pd.Timestamp(period_start).date()
    detail_end = (pd.Timestamp(period_start) + pd.Timedelta(days=13)).date()
    detail_window = mo.ui.date_range(
        start=period_start,
        stop=period_end,
        value=(detail_start, detail_end),
        label="Detailed comparison window",
        full_width=True,
    )
    smoothing_select = mo.ui.dropdown(
        options={"Daily": "1D", "Weekly": "7D", "30-day": "30D"},
        value="Weekly",
        label="Annual control smoothing",
    )
    mo.hstack([detail_window, smoothing_select], widths=[3, 1])
    return detail_window, smoothing_select


@app.cell(hide_code=True)
def _(
    building_select,
    controller_select,
    detail_window,
    directory,
    pd,
    period_select,
    resolution_select,
    smoothing_select,
    trajectories,
):
    selected_metadata = trajectories[
        (trajectories["building_id"] == building_select.value)
        & (trajectories["period_id"] == period_select.value)
        & (trajectories["controller_id"].isin(controller_select.value))
    ][["trajectory_id", "controller_id"]]
    selected_ids = selected_metadata["trajectory_id"].tolist()
    if selected_ids and (directory / "transitions").exists():
        window_start = pd.Timestamp(detail_window.value[0], tz="UTC")
        window_end = pd.Timestamp(detail_window.value[1], tz="UTC") + pd.Timedelta(days=1)
        controller_comparison = pd.read_parquet(
            directory / "transitions",
            filters=[
                ("trajectory_id", "in", selected_ids),
                ("timestamp_utc", ">=", window_start),
                ("timestamp_utc", "<", window_end),
            ],
        )
        controller_comparison = controller_comparison.merge(
            selected_metadata,
            on="trajectory_id",
            how="left",
            validate="many_to_one",
        )
        controller_comparison = controller_comparison[
            controller_comparison.groupby("trajectory_id").cumcount()
            % resolution_select.value
            == 0
        ].sort_values(["controller_id", "timestamp_utc"])

        annual_controls = pd.read_parquet(
            directory / "transitions",
            columns=["trajectory_id", "timestamp_utc", "T_hp_sup_applied"],
            filters=[("trajectory_id", "in", selected_ids)],
        ).merge(
            selected_metadata,
            on="trajectory_id",
            how="left",
            validate="many_to_one",
        )
        annual_controls["time_bin"] = annual_controls["timestamp_utc"].dt.floor(
            smoothing_select.value
        )
        annual_control_summary = (
            annual_controls.groupby(["controller_id", "time_bin"], as_index=False)
            .agg(
                control_mean_C=("T_hp_sup_applied", "mean"),
                control_min_C=("T_hp_sup_applied", "min"),
                control_max_C=("T_hp_sup_applied", "max"),
            )
            .sort_values(["controller_id", "time_bin"])
        )
    else:
        controller_comparison = pd.DataFrame()
        annual_control_summary = pd.DataFrame()
    return annual_control_summary, controller_comparison


@app.cell(hide_code=True)
def _(annual_control_summary, go, make_subplots, mo, px):
    if len(annual_control_summary):
        _controllers = sorted(annual_control_summary["controller_id"].unique())
        _figure = make_subplots(
            rows=len(_controllers),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            subplot_titles=_controllers,
        )
        _colors = px.colors.qualitative.Plotly
        for _row, _controller in enumerate(_controllers, start=1):
            _source = annual_control_summary[
                annual_control_summary["controller_id"] == _controller
            ]
            _color = _colors[(_row - 1) % len(_colors)]
            _red, _green, _blue = (
                int(_color[_index : _index + 2], 16) for _index in (1, 3, 5)
            )
            _figure.add_trace(
                go.Scatter(
                    x=_source["time_bin"],
                    y=_source["control_min_C"],
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=_row,
                col=1,
            )
            _figure.add_trace(
                go.Scatter(
                    x=_source["time_bin"],
                    y=_source["control_max_C"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=f"rgba({_red}, {_green}, {_blue}, 0.2)",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=_row,
                col=1,
            )
            _figure.add_trace(
                go.Scatter(
                    x=_source["time_bin"],
                    y=_source["control_mean_C"],
                    mode="lines",
                    line={"color": _color, "width": 1.5},
                    name=_controller,
                    showlegend=False,
                ),
                row=_row,
                col=1,
            )
            _figure.update_yaxes(title_text="Supply (C)", row=_row, col=1)
        _figure.update_layout(
            title="Annual applied control: smoothed mean and min/max envelope",
            height=max(500, 190 * len(_controllers)),
            hovermode="x unified",
            autosize=True,
            margin={"l": 55, "r": 20, "t": 80, "b": 55},
        )
        annual_control_plot = mo.vstack([mo.md("## Annual control overview"), _figure])
    else:
        annual_control_plot = None
    return (annual_control_plot,)


@app.cell(hide_code=True)
def _(controller_comparison, mo, px):
    measurements = {
        "T_room": ("Room temperature", "Temperature (C)"),
        "T_wall": ("Wall temperature", "Temperature (C)"),
        "T_hp_ret": ("Heat-pump return temperature", "Temperature (C)"),
        "T_hp_sup_applied": ("Applied supply temperature", "Temperature (C)"),
        "T_amb": ("Ambient temperature", "Temperature (C)"),
        "Qdot_gains": ("Thermal gains", "Thermal gains (W)"),
    }
    if len(controller_comparison):
        _plots = []
        for _column, (_title, _unit) in measurements.items():
            _figure = px.line(
                controller_comparison,
                x="timestamp_utc",
                y=_column,
                color="controller_id",
                render_mode="webgl",
                title=_title,
                labels={
                    "timestamp_utc": "Time (UTC)",
                    _column: _unit,
                    "controller_id": "Controller",
                },
            )
            _figure.update_layout(
                hovermode="x unified",
                autosize=True,
                height=620,
                margin={"l": 55, "r": 20, "t": 65, "b": 175},
                legend={
                    "title_text": "",
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.22,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )
            _figure.update_xaxes(automargin=True, nticks=6)
            _figure.update_yaxes(automargin=True)
            _plots.append(_figure)
        measurement_plots = mo.vstack(_plots)
    else:
        measurement_plots = mo.callout("Select at least one controller.", kind="warn")
    return (measurement_plots,)


@app.cell(hide_code=True)
def _(annual_control_plot, measurement_plots, mo):
    mo.ui.tabs(
        {
            "Detailed window": measurement_plots,
            "Annual controls": annual_control_plot,
        }
    )


if __name__ == "__main__":
    app.run()

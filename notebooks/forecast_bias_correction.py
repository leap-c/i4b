import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import pyarrow.dataset as ds
    from plotly.subplots import make_subplots

    return Path, ds, go, json, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <style>
      :root { --paper: #f7f6f2; --ink: #242424; --rule: #d8d8d4; }
      body { background: var(--paper); color: var(--ink); }
      .forecast-note { border-left: 3px solid #4477aa; padding: 0.7rem 1rem;
        background: #ffffff; margin: 0.5rem 0 1rem 0; }
    </style>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Weather forecasts as seen by the MPC

    This notebook compares the **measured ambient temperature**, the archived
    forecast available to the controller, and a leakage-free online bias
    correction. Future observations are displayed only for retrospective
    evaluation; they are never supplied to the MPC.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correction model

    At decision time $t$, the newest eligible forecast run is

    $$r^* = \max\{r : r + d \leq t\},$$

    where $r$ is initialization time and $d$ is the assumed publication delay.
    The observed forecast offset is

    $$e_t = T_{obs}(t) - \hat{T}_{r^*}(t),$$

    $$w(h) = \max\left(1 - \frac{h}{\tau}, 0\right),$$

    $$\hat{T}_{corr}(t+h) = \hat{T}_{r^*}(t+h) + \alpha\,w(h)\,e_t.$$

    Here, $\alpha$ controls how much is applied and $\tau$ controls how quickly
    it disappears over forecast lead. Hourly temperature forecasts are linearly
    interpolated to 15 minutes. Hourly irradiance values are held over each set
    of four 15-minute stages. No future observation appears in the correction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    parameters = [
        {
            "parameter": "Location",
            "meaning": "Representative weather location and local timezone",
            "control": "Selector",
        },
        {
            "parameter": "Forecast initializations r",
            "meaning": "Newest eligible run at each three-hour decision",
            "control": "Selected automatically over a one-day window",
        },
        {
            "parameter": "Publication delay d",
            "meaning": "Hours after initialization before a run may be used",
            "control": "0–8 h slider; 6 h proposed",
        },
        {
            "parameter": "Current mismatch applied α",
            "meaning": "Fraction of the current observed forecast offset applied",
            "control": "0–100% slider",
        },
        {
            "parameter": "Correction duration τ",
            "meaning": "Hours until the current offset correction reaches zero",
            "control": "1–12 h slider; 6 h proposed",
        },
        {
            "parameter": "MPC horizon H",
            "meaning": "Prediction horizon",
            "control": "Fixed at 24 h / 96 stages",
        },
        {
            "parameter": "MPC interval Δ",
            "meaning": "Control and simulation timestep",
            "control": "Fixed at 15 min",
        },
        {
            "parameter": "Forecast resolution",
            "meaning": "Native spacing of archived forecast values",
            "control": "1 h; temperature linear, irradiance held",
        },
    ]
    mo.ui.table(parameters, pagination=False, selection=None)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    try:
        default_source = Path(__file__).resolve().parents[1] / "source-data"
    except NameError:
        default_source = Path("source-data").resolve()
    source_input = mo.ui.text(
        value=str(default_source),
        label="Normalized source-data directory",
        full_width=True,
    )
    source_input
    return (source_input,)


@app.cell(hide_code=True)
def _(Path, ds, json, pd, source_input):
    source = Path(source_input.value).expanduser()
    normalized = source / "normalized"
    forecast_columns = [
        "location_id",
        "model",
        "initialization_time_utc",
        "valid_time_utc",
        "lead_hours",
        "temperature_2m_C",
        "ghi_W_m2",
        "dni_W_m2",
        "dhi_W_m2",
    ]
    forecasts = ds.dataset(
        normalized / "weather_forecasts", format="parquet"
    ).to_table(columns=forecast_columns).to_pandas()
    forecasts["initialization_time_utc"] = pd.to_datetime(
        forecasts["initialization_time_utc"], utc=True
    )
    forecasts["valid_time_utc"] = pd.to_datetime(
        forecasts["valid_time_utc"], utc=True
    )
    references = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["location_id", "valid_time_utc", "temperature_2m_C"],
            )
            for path in sorted((normalized / "weather_reference").glob("*.parquet"))
        ],
        ignore_index=True,
    ).drop_duplicates(["location_id", "valid_time_utc"])
    references["valid_time_utc"] = pd.to_datetime(
        references["valid_time_utc"], utc=True
    )
    config = json.loads(
        (source.parent / "scripts" / "benchmark_source_data.json").read_text()
    )
    location_names = {
        item["id"]: f"{item['country']} · {item['id'].split('_', 1)[1].title()}"
        for item in config["locations"]
    }
    location_timezones = {
        item["id"]: item["timezone"] for item in config["locations"]
    }
    return forecasts, location_names, location_timezones, references


@app.cell(hide_code=True)
def _(location_names, mo):
    location_options = {
        location_names[location]: location for location in sorted(location_names)
    }
    location_select = mo.ui.dropdown(
        options=location_options,
        value=next(iter(location_options)),
        label="Location",
        searchable=True,
        full_width=True,
    )
    delay_slider = mo.ui.slider(
        start=0,
        stop=8,
        step=1,
        value=6,
        label="",
        show_value=True,
        full_width=True,
    )
    delay_control = mo.vstack(
        [mo.md(r"Publication delay $d$ [h]"), delay_slider], gap=0
    )
    decay_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=6,
        label="Correction duration τ [h]",
        show_value=True,
        full_width=True,
    )
    correction_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=50,
        label="Current mismatch applied α [%]",
        show_value=True,
        full_width=True,
    )
    return (
        correction_slider,
        decay_slider,
        delay_control,
        delay_slider,
        location_select,
    )


@app.cell(hide_code=True)
def _(
    correction_slider,
    decay_slider,
    delay_control,
    forecasts,
    location_select,
    mo,
    pd,
):
    available_runs = (
        forecasts.loc[
            forecasts["location_id"] == location_select.value,
            "initialization_time_utc",
        ]
        .drop_duplicates()
        .sort_values()
    )
    _first_day = (available_runs.iloc[0] + pd.Timedelta(days=1)).date()
    _last_start_day = (available_runs.iloc[-1] - pd.Timedelta(days=1)).date()
    _default_day = available_runs.iloc[len(available_runs) // 2].date()
    start_day_select = mo.ui.date(
        start=_first_day,
        stop=_last_start_day,
        value=min(_default_day, _last_start_day),
        label="One-day window starts",
        full_width=True,
    )
    mo.vstack(
        [
            mo.hstack(
                [location_select, start_day_select],
                widths="equal",
                wrap=True,
            ),
            delay_control,
            decay_slider,
            correction_slider,
        ],
        gap=1,
    )
    return (start_day_select,)


@app.cell(hide_code=True)
def _(pd, start_day_select):
    window_start = pd.Timestamp(start_day_select.value, tz="UTC")
    scenario_decisions = [
        timestamp.isoformat()
        for timestamp in pd.date_range(window_start, periods=8, freq="3h")
    ]
    return (scenario_decisions,)


@app.cell(hide_code=True)
def _(
    correction_slider,
    decay_slider,
    delay_slider,
    forecasts,
    location_select,
    location_timezones,
    np,
    pd,
    references,
    scenario_decisions,
):
    location_id = location_select.value
    _location_forecasts = forecasts[
        forecasts["location_id"] == location_id
    ].copy()
    initialization_times = sorted(
        _location_forecasts["initialization_time_utc"].unique()
    )
    forecast_runs = {
        initialization: run.set_index("valid_time_utc").sort_index()
        for initialization, run in _location_forecasts.groupby(
            "initialization_time_utc"
        )
    }
    observed_source = (
        references[references["location_id"] == location_id]
        .set_index("valid_time_utc")
        .sort_index()["temperature_2m_C"]
    )

    def interpolate_temperature(series, target_index):
        _expanded = series.index.union(target_index)
        return (
            series.reindex(_expanded)
            .sort_index()
            .interpolate(method="time", limit_area="inside")
            .reindex(target_index)
        )

    def select_initialization(decision_time):
        _eligible = [
            initialization
            for initialization in initialization_times
            if initialization + pd.Timedelta(hours=delay_slider.value)
            <= decision_time
        ]
        if not _eligible:
            raise ValueError(f"no forecast is available at {decision_time}")
        return _eligible[-1]

    def build_scenario(decision_value):
        _decision_time = pd.Timestamp(decision_value)
        _initialization = select_initialization(_decision_time)
        _horizon_index = pd.date_range(_decision_time, periods=97, freq="15min")
        _raw = interpolate_temperature(
            forecast_runs[_initialization]["temperature_2m_C"], _horizon_index
        )
        _observed = interpolate_temperature(
            observed_source, _horizon_index
        )
        _error = float(_observed.iloc[0] - _raw.iloc[0])
        _lead_hours = np.arange(len(_horizon_index)) / 4.0
        _weights = np.maximum(1.0 - _lead_hours / decay_slider.value, 0.0)
        _strength = correction_slider.value / 100.0
        _scenario = pd.DataFrame(
            {
                "timestamp_utc": _horizon_index,
                "observed_C": _observed.to_numpy(),
                "raw_forecast_C": _raw.to_numpy(),
                "corrected_forecast_C": (
                    _raw.to_numpy() + _error * _weights * _strength
                ),
                "lead_hours": _lead_hours,
            }
        )
        _scenario["local_time"] = (
            _scenario["timestamp_utc"]
            .dt.tz_convert(location_timezones[location_id])
            .dt.tz_localize(None)
        )
        return _scenario, _error, _decision_time, _initialization

    def build_irradiance(decision_value):
        _decision_time = pd.Timestamp(decision_value)
        _initialization = select_initialization(_decision_time)
        _horizon_index = pd.date_range(_decision_time, periods=97, freq="15min")
        _run = forecast_runs[_initialization]
        _irradiance = _run[["ghi_W_m2", "dni_W_m2", "dhi_W_m2"]].reindex(
            _horizon_index, method="ffill"
        )
        _irradiance = _irradiance.clip(lower=0).reset_index(
            names="timestamp_utc"
        )
        _irradiance["lead_hours"] = np.arange(len(_irradiance)) / 4.0
        return _irradiance, _run, _decision_time, _initialization

    return build_irradiance, build_scenario


@app.cell(hide_code=True)
def _(build_scenario, go, make_subplots, scenario_decisions):
    comparison_scenarios = [
        build_scenario(decision) for decision in scenario_decisions
    ]
    comparison_titles = [
        (
            f"{_decision:%d %b · %H:%M}"
            f"<br><span style='font-size:0.8em'>"
            f"forecast age {(_decision - _initialization).total_seconds() / 3600:.0f} h"
            f"</span>"
        )
        for _, _, _decision, _initialization in comparison_scenarios
    ]
    comparison = make_subplots(
        rows=2,
        cols=4,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=comparison_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.04,
    )
    for _index, (_scenario, _, _, _) in enumerate(comparison_scenarios):
        _row, _column = divmod(_index, 4)
        _row += 1
        _column += 1
        for _name, _field, _color, _dash in (
            ("Ground truth", "observed_C", "#7a7a76", "dot"),
            ("Prediction", "raw_forecast_C", "#cc6677", "solid"),
            ("Correction", "corrected_forecast_C", "#4477aa", "solid"),
        ):
            comparison.add_trace(
                go.Scatter(
                    x=_scenario["lead_hours"],
                    y=_scenario[_field],
                    name=_name,
                    legendgroup=_name,
                    showlegend=_index == 0,
                    fill="tonexty" if _name == "Correction" else None,
                    fillcolor=(
                        "rgba(68, 119, 170, 0.14)"
                        if _name == "Correction"
                        else None
                    ),
                    line={
                        "color": _color,
                        "width": 3 if _name == "Correction" else 2,
                        "dash": _dash,
                    },
                ),
                row=_row,
                col=_column,
            )
    for _column_index in range(1, 5):
        comparison.update_xaxes(title_text="Lead [h]", row=2, col=_column_index)
    for _row_index in range(1, 3):
        comparison.update_yaxes(
            title_text="Temperature [°C]", row=_row_index, col=1
        )
    comparison.update_layout(
        title="One day of as-of 24-hour forecasts · UTC",
        height=800,
        margin={"l": 60, "r": 25, "t": 85, "b": 85},
        paper_bgcolor="#f7f6f2",
        plot_bgcolor="#ffffff",
        font={"color": "#242424"},
        legend={"orientation": "h", "y": -0.10, "x": 0.5, "xanchor": "center"},
        hovermode="x unified",
    )
    comparison
    return


@app.cell(hide_code=True)
def _(build_irradiance, go, pd, scenario_decisions):
    irradiance, source_run, decision_time, _ = build_irradiance(
        scenario_decisions[0]
    )
    horizon_end = decision_time + irradiance["lead_hours"].max() * pd.Timedelta(
        hours=1
    )
    hourly = source_run.loc[decision_time:horizon_end]
    irradiance_figure = go.Figure()
    colors = {
        "ghi_W_m2": "#4477aa",
        "dni_W_m2": "#cc6677",
        "dhi_W_m2": "#ddaa33",
    }
    labels = {
        "ghi_W_m2": "GHI",
        "dni_W_m2": "DNI",
        "dhi_W_m2": "DHI",
    }
    for _field in colors:
        irradiance_figure.add_trace(
            go.Scatter(
                x=irradiance["lead_hours"],
                y=irradiance[_field],
                name=f"{labels[_field]} · 15-minute hold",
                line={"color": colors[_field], "width": 2, "shape": "hv"},
            )
        )
        irradiance_figure.add_trace(
            go.Scatter(
                x=(hourly.index - decision_time).total_seconds() / 3600,
                y=hourly[_field],
                name=f"{labels[_field]} · hourly source",
                mode="markers",
                marker={"color": colors[_field], "size": 6},
                showlegend=False,
            )
        )
    irradiance_figure.update_layout(
        title="Hourly irradiance held over 15-minute MPC stages",
        height=440,
        margin={"l": 60, "r": 25, "t": 65, "b": 85},
        paper_bgcolor="#f7f6f2",
        plot_bgcolor="#ffffff",
        font={"color": "#242424"},
        legend={"orientation": "h", "y": -0.20, "x": 0.5, "xanchor": "center"},
        hovermode="x unified",
        xaxis_title="Lead [h]",
        yaxis_title="Irradiance [W/m²]",
    )
    irradiance_figure
    return


@app.cell(hide_code=True)
def _(delay_slider, forecasts, location_select, pd, references):
    location_forecasts = forecasts[
        forecasts["location_id"] == location_select.value
    ].copy()
    location_forecasts = location_forecasts[
        location_forecasts["initialization_time_utc"]
        + pd.Timedelta(hours=delay_slider.value)
        <= location_forecasts["valid_time_utc"]
    ]
    latest_as_of = (
        location_forecasts.sort_values(
            ["valid_time_utc", "initialization_time_utc"]
        )
        .groupby("valid_time_utc", as_index=False)
        .tail(1)
    )
    location_reference = references[
        references["location_id"] == location_select.value
    ][["valid_time_utc", "temperature_2m_C"]].rename(
        columns={"temperature_2m_C": "observed_C"}
    )
    errors = latest_as_of.merge(
        location_reference, on="valid_time_utc", validate="one_to_one"
    )
    errors["error_C"] = errors["temperature_2m_C"] - errors["observed_C"]
    errors["absolute_error_C"] = errors["error_C"].abs()
    return (errors,)


@app.cell(hide_code=True)
def _(errors, go, make_subplots):
    diagnostics = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("As-of forecast error", "Absolute error by forecast lead"),
    )
    diagnostics.add_trace(
        go.Histogram(
            x=errors["error_C"],
            nbinsx=60,
            marker_color="#4477aa",
            name="Current-time error",
        ),
        row=1,
        col=1,
    )
    by_lead = errors.groupby("lead_hours", as_index=False)["absolute_error_C"].mean()
    diagnostics.add_trace(
        go.Scatter(
            x=by_lead["lead_hours"],
            y=by_lead["absolute_error_C"],
            mode="lines+markers",
            marker={"size": 5},
            line={"color": "#cc6677", "width": 2},
            name="Mean absolute error",
        ),
        row=1,
        col=2,
    )
    diagnostics.update_xaxes(title_text="Forecast − observed [K]", row=1, col=1)
    diagnostics.update_xaxes(title_text="Lead time [h]", row=1, col=2)
    diagnostics.update_yaxes(title_text="Hourly observations", row=1, col=1)
    diagnostics.update_yaxes(title_text="MAE [K]", row=1, col=2)
    diagnostics.update_layout(
        height=430,
        margin={"l": 60, "r": 25, "t": 65, "b": 65},
        paper_bgcolor="#f7f6f2",
        plot_bgcolor="#ffffff",
        font={"color": "#242424"},
        showlegend=False,
    )
    diagnostics
    return


@app.cell(hide_code=True)
def _(errors, mo):
    absolute = errors["absolute_error_C"]
    summary = [
        {"metric": "Mean absolute error", "temperature error [K]": absolute.mean()},
        {"metric": "Median", "temperature error [K]": absolute.median()},
        {"metric": "95th percentile", "temperature error [K]": absolute.quantile(0.95)},
        {"metric": "99th percentile", "temperature error [K]": absolute.quantile(0.99)},
        {"metric": "Maximum", "temperature error [K]": absolute.max()},
    ]
    for row in summary:
        row["temperature error [K]"] = round(row["temperature error [K]"], 2)
    mo.vstack(
        [
            mo.md("## Error summary for the selected location"),
            mo.ui.table(summary, pagination=False, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Leakage boundary

    **Available to the MPC:** current measured thermal state, current measured
    disturbances, a forecast run whose assumed publication time precedes the
    decision, deterministic local-time internal-gain schedules, and the online
    correction calculated from the current measurement.

    **Evaluation only:** reference weather after the decision time. It appears in
    gray in the notebook so forecast errors can be inspected, but it is never used
    to create controller actions.
    """)
    return


if __name__ == "__main__":
    app.run()

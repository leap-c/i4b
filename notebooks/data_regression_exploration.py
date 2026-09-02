import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import pickle
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, mo, np, pd, pickle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `DataRegression.pkl` — Sébastien's heat-pump measurements

    Interactive exploration of the regression data consumed by
    `TrainMLModel_Desired_to_Indoor_Homotopy.py`. The file contains measurements
    from a ground-source heat-pump installation (Oslo timezone), sampled at
    20 minutes over roughly 131 days (**mid April – late August 2026**).
    """)
    return


@app.cell
def _(Path, mo):
    repo = Path(__file__).resolve().parents[1]
    path_ui = mo.ui.text(
        value=str(repo / "DataRegression.pkl"),
        label="Regression data file",
    )
    mo.vstack([mo.md("## Load"), path_ui])
    return path_ui, repo


@app.cell
def _(mo, path_ui, pickle):
    with open(path_ui.value, "rb") as _f:
        raw = pickle.load(_f)
    mo.md(
        f"Keys: `{', '.join(raw.keys())}` — sampling period "
        f"`{raw['Sampling']}` min."
    )
    return (raw,)


@app.cell
def _(mo, np, pd, raw):
    _channels = [
        "IndoorPump", "Indoor", "Supply", "Return", "Outdoor", "Water",
        "Desired", "WellIn", "WellOut", "Solar", "SetTemp", "WaterHeatOn",
    ]
    frame = pd.DataFrame({"Time": raw["Time"]})
    for _c in _channels:
        frame[_c] = pd.Series(raw[_c], dtype="float64")
    frame["Power"] = np.nan
    frame.loc[: len(raw["Power"]) - 1, "Power"] = np.asarray(
        raw["Power"], dtype="float64"
    )
    mo.md(
        f"`Power` has `{len(raw['Power'])}` samples vs `{len(frame)}` time "
        "stamps; it is aligned positionally to the leading samples "
        "(see the data-quality cell for gap analysis)."
    )
    return (frame,)


@app.cell
def _(frame, mo, pd):
    _units = {
        "IndoorPump": "°C",
        "Indoor": "°C",
        "Supply": "°C",
        "Return": "°C",
        "Outdoor": "°C",
        "Water": "°C (domestic hot water)",
        "Desired": "°C",
        "WellIn": "°C (ground loop in)",
        "WellOut": "°C (ground loop out)",
        "Solar": "W/m² (assumed)",
        "SetTemp": "°C",
        "Power": "kW (assumed, electrical)",
        "WaterHeatOn": "0/1",
    }
    _rows = []
    for _col, _unit in _units.items():
        _s = frame[_col].astype("float64")
        _rows.append(
            {
                "channel": _col,
                "unit": _unit,
                "count": int(_s.count()),
                "missing": int(_s.isna().sum()),
                "min": round(float(_s.min()), 3),
                "max": round(float(_s.max()), 3),
                "mean": round(float(_s.mean()), 3),
                "std": round(float(_s.std()), 3),
            }
        )
    mo.vstack(
        [
            mo.md("## Channel summary"),
            mo.ui.table(pd.DataFrame(_rows), pagination=False, selection=None),
        ]
    )
    return


@app.cell
def _(frame, mo, pd):
    _t = frame["Time"]
    _diffs = _t.diff().dropna()
    _gaps = _diffs[_diffs != pd.Timedelta(minutes=20)]
    _gap_table = (
        pd.DataFrame(
            {
                "gap_after": _t.iloc[_gaps.index - 1].tolist(),
                "duration": _gaps.to_numpy(),
            }
        )
        if len(_gaps)
        else None
    )
    _gap_display = (
        mo.ui.table(_gap_table, selection=None)
        if _gap_table is not None
        else mo.md("*No gaps: all consecutive samples are exactly 20 min apart.*")
    )
    mo.vstack(
        [
            mo.md(f"""## Data quality

- Timezone: `{_t.iloc[0].tzinfo}`; window `{_t.iloc[0]}` → `{_t.iloc[-1]}`;
  **{(_t.iloc[-1] - _t.iloc[0]).days} days**.
- Sampling step: 20 min (asserted by the training script).
- Non-uniform gaps: **{len(_gaps)}** events listed below.
"""),
            _gap_display,
        ]
    )
    return


@app.cell
def _(frame, go, make_subplots, mo):
    _rows = [
        (["Indoor", "SetTemp", "Desired"], "Room & setpoints (°C)"),
        (["Supply", "Return"], "Heat-pump circuit (°C)"),
        (["Outdoor"], "Ambient (°C)"),
        (["Power"], "Electrical power (kW)"),
        (["Solar"], "Solar irradiance (W/m²)"),
        (["Water", "WaterHeatOn"], "Domestic hot water (°C / on)"),
        (["WellIn", "WellOut"], "Ground loop (°C)"),
    ]
    _colors = {
        "Indoor": "green",
        "SetTemp": "black",
        "Desired": "magenta",
        "Supply": "red",
        "Return": "darkred",
        "Outdoor": "royalblue",
        "Power": "darkorange",
        "Solar": "goldenrod",
        "Water": "brown",
        "WaterHeatOn": "silver",
        "WellIn": "teal",
        "WellOut": "navy",
    }
    _fig = make_subplots(
        rows=len(_rows),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[_t for _, _t in _rows],
        vertical_spacing=0.03,
    )
    for _r, (_cols, _title) in enumerate(_rows, start=1):
        for _c in _cols:
            _fig.add_trace(
                go.Scatter(
                    x=frame["Time"],
                    y=frame[_c],
                    name=_c,
                    line=dict(color=_colors[_c], width=1),
                ),
                row=_r,
                col=1,
            )
    _fig.update_layout(
        height=1100,
        margin=dict(t=60, b=30, l=10, r=10),
        title="Full timeline overview",
    )
    mo.vstack([mo.md("## Full timeline"), mo.as_html(_fig)])
    return


@app.cell
def _(frame, mo, pd):
    _t0 = frame["Time"].iloc[0]
    _t1 = frame["Time"].iloc[-1]
    range_ui = mo.ui.date_range(
        start=_t0.date(),
        stop=_t1.date(),
        value=(_t0.date(), (_t0 + pd.Timedelta(days=7)).date()),
        label="Zoom window",
    )
    mo.vstack([mo.md("## Zoomed view"), range_ui])
    return (range_ui,)


@app.cell
def _(frame, go, make_subplots, mo, pd, range_ui):
    _t0 = frame["Time"].iloc[0]
    _start, _stop = range_ui.value
    _tz = _t0.tzinfo
    _mask = (frame["Time"] >= pd.Timestamp(_start, tz=_tz)) & (
        frame["Time"] < pd.Timestamp(_stop, tz=_tz) + pd.Timedelta(days=1)
    )
    _zoom = frame[_mask]

    _rows = [
        (["Indoor", "SetTemp", "Desired"], "Room & setpoints (°C)"),
        (["Supply", "Return"], "Heat-pump circuit (°C)"),
        (["Outdoor"], "Ambient (°C)"),
        (["Power"], "Electrical power (kW)"),
    ]
    _colors = {
        "Indoor": "green",
        "SetTemp": "black",
        "Desired": "magenta",
        "Supply": "red",
        "Return": "darkred",
        "Outdoor": "royalblue",
        "Power": "darkorange",
    }
    _fig = make_subplots(
        rows=len(_rows),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[_t for _, _t in _rows],
        vertical_spacing=0.06,
    )
    for _r, (_cols, _title) in enumerate(_rows, start=1):
        for _c in _cols:
            _fig.add_trace(
                go.Scatter(
                    x=_zoom["Time"],
                    y=_zoom[_c],
                    name=_c,
                    line=dict(color=_colors[_c], width=1.5),
                ),
                row=_r,
                col=1,
            )
    _fig.update_layout(
        height=720,
        margin=dict(t=60, b=30, l=10, r=10),
        title=f"Zoom: {_start} → {_stop} ({len(_zoom)} samples)",
    )
    mo.as_html(_fig)


@app.cell
def _(frame, go, mo, pd):
    daily = frame.set_index("Time").resample("D").agg(
        energy_kwh=("Power", lambda s: float(s.sum() * 20 / 60)),
        outdoor_mean=("Outdoor", "mean"),
    )
    _fig = go.Figure()
    _fig.add_trace(go.Bar(x=daily.index, y=daily["energy_kwh"], name="daily kWh"))
    _fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["outdoor_mean"],
            name="mean Outdoor (°C)",
            yaxis="y2",
            line=dict(color="royalblue"),
        )
    )
    _fig.update_layout(
        height=340,
        margin=dict(t=60, b=30, l=10, r=10),
        title="Daily electrical consumption and outdoor mean",
        yaxis=dict(title="kWh"),
        yaxis2=dict(title="°C", overlaying="y", side="right"),
    )
    mo.vstack([mo.md("## Daily energy"), mo.as_html(_fig)])
    return (daily,)


@app.cell
def _(frame, go, mo):
    _fig1 = go.Figure()
    _fig1.add_trace(
        go.Scatter(
            x=frame["Outdoor"],
            y=frame["Desired"],
            mode="markers",
            marker=dict(size=3, opacity=0.3, color="magenta"),
            name="Desired vs Outdoor",
        )
    )
    _fig1.update_layout(
        height=330,
        margin=dict(t=40, b=30, l=10, r=10),
        xaxis_title="Outdoor (°C)",
        yaxis_title="Desired (°C)",
        title="Desired supply temperature vs ambient",
    )
    _fig2 = go.Figure()
    _fig2.add_trace(
        go.Scatter(
            x=frame["Outdoor"],
            y=frame["Supply"],
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.3,
                color=frame["Indoor"],
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title="Indoor (°C)"),
            ),
            name="Supply vs Outdoor",
        )
    )
    _fig2.update_layout(
        height=330,
        margin=dict(t=40, b=30, l=10, r=10),
        xaxis_title="Outdoor (°C)",
        yaxis_title="Supply (°C)",
        title="Measured supply vs ambient (colored by indoor)",
    )
    mo.vstack([mo.md("## Heat-curve views"), mo.as_html(_fig1), mo.as_html(_fig2)])
    return


@app.cell
def _(frame, go, mo, pd):
    _df = frame.copy()
    _df["hour"] = _df["Time"].apply(lambda t: t.hour + t.minute / 60)
    _df["month"] = _df["Time"].apply(lambda t: t.month)
    _figs = []
    for _c, _title in (
        ("Indoor", "Indoor (°C)"),
        ("Outdoor", "Outdoor (°C)"),
        ("Power", "Power (kW)"),
    ):
        _piv = _df.pivot_table(
            index="hour", columns="month", values=_c, aggfunc="mean"
        )
        _figs.append(
            go.Figure(
                data=go.Heatmap(
                    z=_piv.T.to_numpy(),
                    x=_piv.index.to_numpy(),
                    y=[str(_m) for _m in _piv.columns],
                    colorscale="Viridis",
                    colorbar=dict(title=_title),
                ),
                layout=dict(
                    title=f"Hour-of-day × month: {_c}",
                    height=280,
                    margin=dict(t=40, b=30, l=10, r=10),
                    xaxis_title="hour",
                    yaxis_title="month",
                ),
            )
        )
    mo.vstack([mo.md("## Diurnal patterns"), *[mo.as_html(_f) for _f in _figs]])
    return


@app.cell
def _(frame, go, mo):
    _cols = [
        "Indoor", "IndoorPump", "Supply", "Return", "Outdoor", "Desired",
        "SetTemp", "Solar", "Power", "Water", "WellIn", "WellOut",
    ]
    _df = frame[_cols].copy()
    _df["dT_circuit"] = _df["Supply"] - _df["Return"]
    _df["dT_ground"] = _df["WellIn"] - _df["WellOut"]
    _corr = _df.corr(method="spearman")
    _fig = go.Figure(
        data=go.Heatmap(
            z=_corr.to_numpy(),
            x=_corr.columns.tolist(),
            y=_corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="ρ"),
        )
    )
    _fig.update_layout(
        height=700,
        margin=dict(t=60, b=30, l=10, r=10),
        title="Spearman correlation (20-min samples)",
    )
    mo.vstack([mo.md("## Correlations"), mo.as_html(_fig)])
    return


@app.cell
def _(go, mo, np, pd, daily):
    _xy = daily.dropna(subset=["energy_kwh", "outdoor_mean"])
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_xy["outdoor_mean"],
            y=_xy["energy_kwh"],
            mode="markers",
            marker=dict(size=6, opacity=0.6, color="darkred"),
            name="daily",
        )
    )
    _w = np.polyfit(_xy["outdoor_mean"], _xy["energy_kwh"], 1)
    _xs = np.linspace(
        _xy["outdoor_mean"].min(), _xy["outdoor_mean"].max(), 50
    )
    _fig.add_trace(
        go.Scatter(
            x=_xs,
            y=np.polyval(_w, _xs),
            mode="lines",
            name=f"linear fit ({_w[0]:.2f} kWh per °C·day)",
            line=dict(color="black", dash="dash"),
        )
    )
    _fig.update_layout(
        height=340,
        margin=dict(t=60, b=30, l=10, r=10),
        title="Daily electrical energy vs mean outdoor temperature",
        xaxis_title="mean Outdoor (°C)",
        yaxis_title="daily kWh",
    )
    mo.vstack([mo.md("## Heating demand"), mo.as_html(_fig)])
    return


@app.cell
def _(frame, go, mo, pd):
    _duty = float(frame["WaterHeatOn"].mean())
    _t = frame.set_index("Time")["WaterHeatOn"].resample("D").mean()
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_t.index,
            y=_t.to_numpy() * 100,
            name="DHW duty (%)",
            line=dict(color="brown"),
        )
    )
    _fig.update_layout(
        height=280,
        margin=dict(t=60, b=30, l=10, r=10),
        title=f"Domestic hot-water heating duty (overall {_duty * 100:.1f}%)",
        yaxis_title="% of day",
    )
    mo.vstack([mo.md("## Domestic hot water"), mo.as_html(_fig)])
    return


@app.cell
def _(go, mo, np, pd, raw):
    _rt = np.array([_t.timestamp() for _t in raw["RawPower"]["Time"]])
    _rp = np.asarray(raw["RawPower"]["Power"], dtype="float64")
    _step_s = float(np.median(np.diff(_rt))) if len(_rt) > 1 else float("nan")
    _fig = go.Figure()
    _fig.add_trace(
        go.Scattergl(
            x=pd.to_datetime(_rt, unit="s", utc=True).tz_convert("Europe/Oslo"),
            y=_rp,
            name="RawPower",
            line=dict(color="darkorange", width=1),
        )
    )
    _fig.update_layout(
        height=300,
        margin=dict(t=60, b=30, l=10, r=10),
        title=f"RawPower: {len(_rp)} samples, median step {_step_s:.0f} s",
    )
    mo.vstack([mo.md("## Raw power channel"), mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the training script uses this data

    From `TrainMLModel_Desired_to_Indoor_Homotopy.py` with the 20 min sampling:

    - **Past inputs** $\mathcal{P}$: `Indoor`, `Supply`, `Return`
    - **Future inputs** $\mathcal{F}$: `Outdoor`, `Feature = Desired`
    - **Target**: `Indoor`, deviations `Indoor[k+1:k+H] − Indoor[k]`
    - **Depth** `6·60/20 = 18` steps (6 h memory)
    - **Horizon** `36·60/20 = 108` steps ⇒ **36 h**, despite the source comment
      "Prediction 24h"
    - **Discount** `γ = 0.98/day` relative to the script's run date
    - **Models**: `L2` (pure quadratic), `L1Above1` $(w_+,w_-)=(0.99,0.01)$
      → 1%-quantile, `L1Below1` $(w_+,w_-)=(0.01,0.99)$ → 99%-quantile,
      with weight homotopy from $(0.8,0.2)$ / $(0.2,0.8)$ when not warm-started

    The companion notebook `desired_to_indoor_homotopy.py` re-implements this
    pipeline on the I4B standard environment with identical equations.
    """)
    return


if __name__ == "__main__":
    app.run()

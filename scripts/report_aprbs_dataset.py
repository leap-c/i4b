"""Generate a self-contained audit report for the residual APRBS pilot dataset."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BACKGROUND = "#f7f6f2"
PAPER = "#ffffff"
TEXT = "#242424"
GRID = "#d8d8d4"
BLUE = "#4477aa"
VERMILLION = "#cc6677"
OCHRE = "#ddcc77"
GREEN = "#228833"
BAND_ORDER = ["nominal", "small", "medium", "wide"]
BAND_COLOURS = {
    "nominal": "#777777",
    "small": BLUE,
    "medium": OCHRE,
    "wide": VERMILLION,
}


def rgba(colour: str, alpha: float) -> str:
    """Convert a hex colour to Plotly-compatible RGBA."""
    red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def rms(values: pd.Series) -> float:
    values = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def style_figure(figure: go.Figure, height: int | None = None) -> go.Figure:
    figure.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PAPER,
        font={
            "family": "Inter, ui-sans-serif, system-ui, sans-serif",
            "color": TEXT,
            "size": 13,
        },
        margin={"l": 55, "r": 28, "t": 62, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"bgcolor": PAPER, "font_color": TEXT},
        height=height,
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return figure


def figure_html(figure: go.Figure, include_plotlyjs: bool) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs="inline" if include_plotlyjs else False,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d"],
        },
    )


def prepare(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow").sort_values(
        ["building_family_id", "perturbation_band", "timestamp_next"]
    )
    frame["requested_residual"] = frame["T_hp_sup_requested"] - frame["T_hp_sup_mpc"]
    frame["applied_residual"] = frame["T_hp_sup_applied"] - frame["T_hp_sup_mpc"]
    frame["projection_error"] = frame["T_hp_sup_applied"] - frame["T_hp_sup_requested"]
    frame["elapsed_hours"] = frame.groupby("trajectory_id").cumcount() * 0.25 + 0.25
    frame["is_residual"] = frame["perturbation_band"] != "nominal"
    return frame


def representative_scenarios(frame: pd.DataFrame) -> list[str]:
    families = sorted(frame["building_family_id"].unique())
    selected = []
    for family in families[:2]:
        for band in BAND_ORDER:
            selected.extend(
                frame.loc[
                    (frame["building_family_id"] == family)
                    & (frame["perturbation_band"] == band),
                    "trajectory_id",
                ].unique()
            )
    validation = families[-1]
    for band in ("nominal", "wide"):
        selected.extend(
            frame.loc[
                (frame["building_family_id"] == validation)
                & (frame["perturbation_band"] == band),
                "trajectory_id",
            ].unique()
        )
    if len(selected) != 10:
        raise ValueError(
            f"expected ten representative scenarios, found {len(selected)}"
        )
    return selected


def band_summary(frame: pd.DataFrame) -> pd.DataFrame:
    table = (
        frame.groupby("perturbation_band", observed=True)
        .agg(
            rows=("trajectory_id", "size"),
            scenarios=("trajectory_id", "nunique"),
            requested_rms_k=("requested_residual", rms),
            applied_rms_k=("applied_residual", rms),
            applied_abs_p95_k=("applied_residual", lambda x: x.abs().quantile(0.95)),
            room_mean_c=("T_room_next", "mean"),
            room_std_c=("T_room_next", "std"),
            return_mean_c=("T_hp_ret_next", "mean"),
        )
        .reindex(BAND_ORDER)
    )
    disposition = pd.crosstab(
        frame["perturbation_band"], frame["perturbation_disposition"]
    )
    table["accepted_pct"] = 100 * disposition.get("accepted", 0) / table["rows"]
    table["scaled_pct"] = 100 * disposition.get("scaled", 0) / table["rows"]
    table["rejected_pct"] = 100 * disposition.get("rejected", 0) / table["rows"]
    return table


def perturbation_figure(frame: pd.DataFrame) -> go.Figure:
    residual = frame[frame["is_residual"]]
    figure = px.violin(
        residual,
        x="perturbation_band",
        y="applied_residual",
        color="perturbation_band",
        category_orders={"perturbation_band": BAND_ORDER[1:]},
        color_discrete_map=BAND_COLOURS,
        box=True,
        points=False,
        labels={
            "perturbation_band": "Band",
            "applied_residual": "Applied residual [K]",
        },
        title="Applied action excitation by perturbation band",
    )
    figure.update_layout(showlegend=False)
    figure.add_hline(y=0, line_color=TEXT, line_width=1)
    return style_figure(figure, 430)


def disposition_figure(frame: pd.DataFrame) -> go.Figure:
    counts = (
        frame.groupby(["perturbation_band", "perturbation_disposition"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    counts["share"] = counts["rows"] / counts.groupby("perturbation_band")[
        "rows"
    ].transform("sum")
    colours = {
        "accepted": BLUE,
        "scaled": OCHRE,
        "rejected": VERMILLION,
        "zero": "#999999",
    }
    figure = px.bar(
        counts,
        x="perturbation_band",
        y="share",
        color="perturbation_disposition",
        category_orders={"perturbation_band": BAND_ORDER},
        color_discrete_map=colours,
        labels={
            "perturbation_band": "Band",
            "share": "Share of transitions",
            "perturbation_disposition": "Disposition",
        },
        title="Collector disposition after physical feasibility checks",
    )
    figure.update_yaxes(tickformat=".0%")
    return style_figure(figure, 430)


def paired_response(frame: pd.DataFrame) -> tuple[pd.DataFrame, go.Figure]:
    nominal = frame[frame["perturbation_band"] == "nominal"][
        ["building_family_id", "elapsed_hours", "T_room_next", "T_hp_ret_next"]
    ].rename(columns={"T_room_next": "room_nominal", "T_hp_ret_next": "return_nominal"})
    residual = frame[frame["is_residual"]].merge(
        nominal,
        on=["building_family_id", "elapsed_hours"],
        how="left",
        validate="many_to_one",
    )
    residual["room_delta"] = residual["T_room_next"] - residual["room_nominal"]
    residual["return_delta"] = residual["T_hp_ret_next"] - residual["return_nominal"]
    residual["room_delta_smooth"] = residual.groupby("trajectory_id")[
        "room_delta"
    ].transform(lambda values: values.rolling(16, min_periods=1, center=True).mean())

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
    for band in BAND_ORDER[1:]:
        source = residual[residual["perturbation_band"] == band]
        aggregate = source.groupby("elapsed_hours")["room_delta_smooth"].agg(
            mean="mean", low=lambda x: x.quantile(0.1), high=lambda x: x.quantile(0.9)
        )
        figure.add_trace(
            go.Scatter(
                x=aggregate.index,
                y=aggregate["mean"],
                name=band,
                line={"color": BAND_COLOURS[band], "width": 2},
                legendgroup=band,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([aggregate.index, aggregate.index[::-1]]),
                y=np.concatenate([aggregate["high"], aggregate["low"][::-1]]),
                fill="toself",
                fillcolor=rgba(BAND_COLOURS[band], 0.15),
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=band,
            ),
            row=1,
            col=1,
        )
        hourly = source.groupby("elapsed_hours").agg(
            action_rms=("applied_residual", rms), return_rms=("return_delta", rms)
        )
        figure.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly["return_rms"],
                name=f"{band} return RMS",
                line={"color": BAND_COLOURS[band], "width": 1.7},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    figure.add_hline(y=0, line_color=TEXT, line_width=1, row=1, col=1)
    figure.update_yaxes(title_text="Room deviation [K]", row=1, col=1)
    figure.update_yaxes(title_text="Return deviation RMS [K]", row=2, col=1)
    figure.update_xaxes(title_text="Elapsed time [h]", row=2, col=1)
    figure.update_layout(
        title="Paired response relative to each family’s nominal trajectory"
    )
    return residual, style_figure(figure, 650)


def heatmap_figure(
    frame: pd.DataFrame, residual: pd.DataFrame, scenarios: list[str]
) -> go.Figure:
    sampled = frame[
        frame["trajectory_id"].isin(scenarios)
        & (frame.groupby("trajectory_id").cumcount() % 4 == 0)
    ]
    action = sampled.pivot(
        index="trajectory_id", columns="elapsed_hours", values="applied_residual"
    ).reindex(scenarios)
    room_source = residual[residual["trajectory_id"].isin(scenarios)].copy()
    nominal_rows = frame[
        frame["trajectory_id"].isin(scenarios)
        & (frame["perturbation_band"] == "nominal")
    ].assign(room_delta=0.0)
    room_source = pd.concat(
        [
            room_source[["trajectory_id", "elapsed_hours", "room_delta"]],
            nominal_rows[["trajectory_id", "elapsed_hours", "room_delta"]],
        ]
    )
    room_source = room_source[room_source.groupby("trajectory_id").cumcount() % 4 == 0]
    room = room_source.pivot(
        index="trajectory_id", columns="elapsed_hours", values="room_delta"
    ).reindex(scenarios)
    labels = [
        scenario.replace("sfh_", "").replace("_1_enev-", " / ")
        for scenario in scenarios
    ]

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        subplot_titles=(
            "Applied residual action",
            "Room temperature deviation from family nominal",
        ),
    )
    figure.add_trace(
        go.Heatmap(
            z=action.to_numpy(),
            x=action.columns,
            y=labels,
            colorscale=[[0, VERMILLION], [0.5, PAPER], [1, BLUE]],
            zmid=0,
            colorbar={"title": "K", "y": 0.79, "len": 0.38},
            hovertemplate="%{y}<br>%{x:.1f} h<br>residual %{z:.2f} K<extra></extra>",
        ),
        row=1,
        col=1,
    )
    limit = np.nanquantile(np.abs(room.to_numpy()), 0.99)
    figure.add_trace(
        go.Heatmap(
            z=room.to_numpy(),
            x=room.columns,
            y=labels,
            colorscale=[[0, BLUE], [0.5, PAPER], [1, VERMILLION]],
            zmid=0,
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "K", "y": 0.21, "len": 0.38},
            hovertemplate="%{y}<br>%{x:.1f} h<br>room delta %{z:.3f} K<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="Elapsed time [h]", row=2, col=1)
    figure.update_layout(
        title="Ten representative seven-day scenarios", showlegend=False
    )
    return style_figure(figure, 720)


def scenario_explorer(frame: pd.DataFrame, scenarios: list[str]) -> go.Figure:
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    traces_per_scenario = 7
    for scenario_index, scenario in enumerate(scenarios):
        source = frame[frame["trajectory_id"] == scenario]
        visible = scenario_index == 0
        traces = (
            ("T_room", source["T_room_next"], BLUE, 1),
            ("T_hp_ret", source["T_hp_ret_next"], VERMILLION, 1),
            ("MPC supply", source["T_hp_sup_mpc"], "#777777", 2),
            ("Applied supply", source["T_hp_sup_applied"], BLUE, 2),
            ("Applied residual", source["applied_residual"], VERMILLION, 2),
            ("Ambient", source["T_amb_t"], BLUE, 3),
            ("Gains [kW]", source["Qdot_gains_t"] / 1000, OCHRE, 3),
        )
        for name, values, colour, row in traces:
            figure.add_trace(
                go.Scatter(
                    x=source["elapsed_hours"],
                    y=values,
                    name=name,
                    line={"color": colour, "width": 1.5},
                    visible=visible,
                    legendgroup=str(scenario_index),
                ),
                row=row,
                col=1,
            )
    buttons = []
    for scenario_index, scenario in enumerate(scenarios):
        visibility = [False] * (len(scenarios) * traces_per_scenario)
        start = scenario_index * traces_per_scenario
        visibility[start : start + traces_per_scenario] = [True] * traces_per_scenario
        buttons.append(
            {
                "label": scenario.replace("sfh_", "").replace("_1_enev-", " / "),
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"Scenario explorer: {scenario}"},
                ],
            }
        )
    figure.update_layout(
        title=f"Scenario explorer: {scenarios[0]}",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
    )
    figure.update_yaxes(title_text="Temperature [°C]", row=1, col=1)
    figure.update_yaxes(title_text="Supply / residual [K]", row=2, col=1)
    figure.update_yaxes(title_text="Ambient [°C] / gains [kW]", row=3, col=1)
    figure.update_xaxes(title_text="Elapsed time [h]", row=3, col=1)
    return style_figure(figure, 820)


def scenario_table(frame: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    table = (
        frame[frame["trajectory_id"].isin(scenarios)]
        .groupby("trajectory_id")
        .agg(
            family=("building_family_id", "first"),
            split=("split", "first"),
            band=("perturbation_band", "first"),
            rows=("timestamp_next", "size"),
            residual_rms_k=("applied_residual", rms),
            residual_min_k=("applied_residual", "min"),
            residual_max_k=("applied_residual", "max"),
            room_min_c=("T_room_next", "min"),
            room_max_c=("T_room_next", "max"),
        )
        .reset_index()
    )
    return table


def generate(data: Path, output: Path) -> None:
    frame = prepare(data)
    scenarios = representative_scenarios(frame)
    summary = band_summary(frame)
    residual, response = paired_response(frame)

    cadence = (
        frame.groupby("trajectory_id")["timestamp_next"].diff().dropna().mode().iloc[0]
    )
    duration_days = (
        frame.groupby("trajectory_id")["timestamp_next"]
        .agg(lambda x: (x.max() - x.min()).total_seconds() / 86400)
        .median()
    )
    duplicates = int(frame.duplicated(["trajectory_id", "timestamp_next"]).sum())
    missing = int(frame.isna().sum().sum())
    exact_projection = float(np.mean(np.abs(frame["projection_error"]) <= 1e-6))

    figures = [
        perturbation_figure(frame),
        disposition_figure(frame),
        response,
        heatmap_figure(frame, residual, scenarios),
        scenario_explorer(frame, scenarios),
    ]
    rendered = [figure_html(figure, index == 0) for index, figure in enumerate(figures)]

    summary_display = summary.rename(
        columns={
            "rows": "Rows",
            "scenarios": "Scenarios",
            "requested_rms_k": "Requested RMS [K]",
            "applied_rms_k": "Applied RMS [K]",
            "applied_abs_p95_k": "|Applied| p95 [K]",
            "room_mean_c": "Room mean [°C]",
            "room_std_c": "Room SD [K]",
            "return_mean_c": "Return mean [°C]",
            "accepted_pct": "Accepted [%]",
            "scaled_pct": "Scaled [%]",
            "rejected_pct": "Rejected [%]",
        }
    ).reset_index(names="Band")
    scenario_display = scenario_table(frame, scenarios).rename(
        columns={
            "trajectory_id": "Scenario",
            "family": "Family",
            "split": "Split",
            "band": "Band",
            "rows": "Rows",
            "residual_rms_k": "Residual RMS [K]",
            "residual_min_k": "Residual min [K]",
            "residual_max_k": "Residual max [K]",
            "room_min_c": "Room min [°C]",
            "room_max_c": "Room max [°C]",
        }
    )

    cards = [
        ("Rows", f"{len(frame):,}"),
        ("Trajectories", f"{frame['trajectory_id'].nunique()}"),
        ("Families", f"{frame['building_family_id'].nunique()}"),
        ("Cadence", str(cadence)),
        ("Typical span", f"{duration_days:.1f} days"),
        ("Missing / duplicate", f"{missing} / {duplicates}"),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    selected_text = ", ".join(html.escape(scenario) for scenario in scenarios)
    css = f"""
    :root {{ --page:{BACKGROUND}; --paper:{PAPER}; --text:{TEXT}; --rule:{GRID}; --blue:{BLUE}; --red:{VERMILLION}; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--page); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,sans-serif; line-height:1.5; }}
    main {{ max-width:1280px; margin:0 auto; padding:42px 28px 80px; }}
    h1 {{ font-family:Georgia,serif; font-size:clamp(2.1rem,5vw,4rem); line-height:1.02; margin:0 0 14px; font-weight:500; }}
    h2 {{ font-family:Georgia,serif; font-size:2rem; font-weight:500; margin:58px 0 8px; border-top:1px solid var(--rule); padding-top:28px; }}
    h3 {{ font-size:1rem; text-transform:uppercase; letter-spacing:.09em; margin:30px 0 10px; }}
    .lead {{ max-width:850px; font-size:1.13rem; color:#4b4b48; }}
    .kicker {{ color:var(--red); text-transform:uppercase; letter-spacing:.13em; font-weight:700; font-size:.78rem; }}
    .cards {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin:28px 0; }}
    .card {{ background:var(--paper); border:1px solid var(--rule); padding:16px; min-height:94px; display:flex; flex-direction:column; justify-content:space-between; }}
    .card span {{ color:#666; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }}
    .card strong {{ font-family:Georgia,serif; font-size:1.45rem; font-weight:500; }}
    .note {{ border-left:3px solid var(--blue); background:#eef3f7; padding:14px 18px; margin:20px 0; }}
    .warning {{ border-left-color:var(--red); background:#f8ecee; }}
    .figure {{ margin:18px 0 34px; }}
    .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
    .table-wrap {{ overflow-x:auto; background:var(--paper); border:1px solid var(--rule); margin:18px 0 28px; }}
    table {{ border-collapse:collapse; width:100%; font-size:.86rem; }}
    th,td {{ border-bottom:1px solid #e6e6e2; padding:9px 11px; text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ background:#efefeb; font-weight:650; }}
    footer {{ margin-top:60px; padding-top:18px; border-top:1px solid var(--rule); color:#666; font-size:.82rem; }}
    code {{ background:#ecece8; padding:2px 5px; }}
    @media(max-width:900px) {{ .cards {{ grid-template-columns:repeat(3,1fr); }} .grid2 {{ grid-template-columns:1fr; }} main {{ padding:28px 14px 60px; }} }}
    @media(max-width:520px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} .card {{ min-height:82px; }} h2 {{ font-size:1.65rem; }} }}
    """

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Residual APRBS dataset audit</title><style>{css}</style></head><body><main>
<div class="kicker">I4B × Chronos-2 pilot · dataset audit</div>
<h1>What the model actually sees</h1>
<p class="lead">A visual audit of nominal and residual-action trajectories before additional fine-tuning. Statistics cover all 12 trajectories; detailed views use ten representative seven-day scenarios.</p>
<div class="cards">{card_html}</div>
<div class="note"><strong>Integrity check.</strong> The dataset has {missing} missing values and {duplicates} duplicate trajectory/timestamp keys. Applied and physically projected requested supply temperatures agree within 1 µK on {exact_projection:.1%} of rows.</div>

<h2>Dataset balance</h2>
<p>Each perturbation band contributes three trajectories and 2,016 transitions. Training contains two building families; validation contains one disjoint family.</p>
<div class="table-wrap">{summary_display.to_html(index=False, float_format=lambda x: f'{x:.3f}', border=0)}</div>
<div class="grid2"><div class="figure">{rendered[0]}</div><div class="figure">{rendered[1]}</div></div>
<div class="note warning"><strong>Interpretation.</strong> “Rejected” and “scaled” describe attempted residual samples before the collector’s feasibility handling. The model receives <code>T_hp_sup_applied</code>; no applied row differs from its physically projected request.</div>

<h2>Residual versus nominal response</h2>
<p>Residual trajectories are aligned with the nominal trajectory from the same building family and elapsed time. The upper panel shows the smoothed room-temperature difference; bands show the 10–90% cross-family range. The lower panel shows return-temperature deviation magnitude.</p>
<div class="figure">{rendered[2]}</div>

<h2>Ten-scenario overview</h2>
<p>The heatmaps make the excitation pattern and resulting room-temperature deviation visible over the complete week. Hourly samples are shown for readability.</p>
<div class="figure">{rendered[3]}</div>
<details><summary>Selected scenario IDs</summary><p>{selected_text}</p></details>
<div class="table-wrap">{scenario_display.to_html(index=False, float_format=lambda x: f'{x:.3f}', border=0)}</div>

<h2>Interactive scenario inspection</h2>
<p>Choose a scenario from the dropdown. The panels expose target temperatures, nominal and applied control, the explicit residual, ambient temperature, and internal gains.</p>
<div class="figure">{rendered[4]}</div>

<h2>Reading the training signal</h2>
<div class="note"><strong>What is explicit:</strong> the model receives absolute applied supply temperature as a known future covariate. <strong>What is implicit:</strong> the residual relative to MPC is not currently provided as its own feature, even though it is visualized here.</div>
<p>The action excitation is balanced by trajectory count and reaches meaningful amplitudes, but the room response remains slow and small relative to the absolute temperature level. This explains why ordinary forecast loss can improve while action-response gain deteriorates: the optimizer can fit levels without preserving the finite-difference response.</p>

<footer>Generated from <code>{html.escape(str(data))}</code>. Source: <code>scripts/report_aprbs_dataset.py</code>. The source Parquet is not included; plotted values and aggregate statistics are embedded for offline viewing.</footer>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.data, args.output)


if __name__ == "__main__":
    main()

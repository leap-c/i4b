import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, json, make_subplots, mo, np, pd, px


@app.cell
def _(mo):
    mo.md(
        """
        # Residual-APRBS building dataset review

        Interactive inspection of cohort balance, action semantics, excitation, response,
        operating coverage, and replay metadata. The notebook reads persisted Parquet/JSON
        artifacts and never recollects data when opened.
        """
    )


@app.cell
def _(Path, mo):
    default = Path("/tmp/opencode/i4b-aprbs-pilot.parquet")
    data_path = mo.ui.text(value=str(default), label="Pilot Parquet")
    mo.output.replace(data_path)
    return (data_path,)


@app.cell
def _(Path, data_path, json, pd):
    primary_path = Path(data_path.value).expanduser()
    diagnostics_path = primary_path.with_name(
        f"{primary_path.stem}.diagnostics.parquet"
    )
    sidecar_path = primary_path.with_name(f"{primary_path.stem}.trajectories.json")
    available = all(
        path.exists() for path in (primary_path, diagnostics_path, sidecar_path)
    )
    if available:
        primary = pd.read_parquet(primary_path).assign(
            transition_index=lambda frame: frame.groupby(
                "trajectory_id", sort=False
            ).cumcount()
        )
        diagnostics = pd.read_parquet(diagnostics_path)
        sidecar = json.loads(sidecar_path.read_text())
        data = primary.merge(
            diagnostics,
            on=["trajectory_id", "transition_index"],
            how="left",
            validate="one_to_one",
        )
        data["requested_residual"] = data["T_hp_sup_requested"] - data["T_hp_sup_mpc"]
        data["applied_residual"] = data["T_hp_sup_applied"] - data["T_hp_sup_mpc"]
        data["raw_nominal_correction"] = data["T_hp_sup_mpc"] - data["T_hp_sup_mpc_raw"]
        data["elapsed_hours"] = data["transition_index"] / 4 + 0.25
    else:
        primary, diagnostics, data, sidecar = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )
    return (
        available,
        data,
        diagnostics,
        diagnostics_path,
        primary,
        primary_path,
        sidecar,
        sidecar_path,
    )


@app.cell
def _(available, mo, primary_path):
    mo.callout(
        mo.md(
            f"**Artifact status:** {'loaded' if available else 'missing'} — `{primary_path}`"
        ),
        kind="success" if available else "warn",
    )


@app.cell
def _(available, data, mo, pd):
    if available:
        cohort = data.groupby(
            ["split", "building_family_id", "building_id", "perturbation_band"],
            as_index=False,
        ).agg(
            rows=("trajectory_id", "size"),
            trajectories=("trajectory_id", "nunique"),
            start=("timestamp_t", "min"),
            end=("timestamp_next", "max"),
            action_std_K=("applied_residual", "std"),
            room_min_C=("T_room_next", "min"),
            room_max_C=("T_room_next", "max"),
        )
        cohort_view = mo.ui.table(cohort, pagination=True, selection=None)
    else:
        cohort = pd.DataFrame()
        cohort_view = mo.md("Artifact required.")
    mo.vstack([mo.md("## Cohorts and split integrity"), cohort_view])
    return (cohort,)


@app.cell
def _(available, data, mo, pd):
    if available:
        _families = data.groupby("split")["building_family_id"].agg(
            lambda values: sorted(set(values))
        )
        integrity = pd.DataFrame(
            [
                [
                    "Expected pilot columns",
                    len(
                        primary_columns := [
                            "trajectory_id",
                            "building_id",
                            "building_family_id",
                            "split",
                            "timestamp_t",
                            "timestamp_next",
                            "T_room_next",
                            "T_hp_ret_next",
                            "T_hp_sup_applied",
                            "T_amb_t",
                            "Qdot_gains_t",
                            "T_hp_sup_mpc",
                            "T_hp_sup_requested",
                            "perturbation_band",
                            "perturbation_disposition",
                        ]
                    )
                    == 15
                    and set(primary_columns).issubset(data.columns),
                ],
                ["No null primary rows", not data[primary_columns].isna().any().any()],
                [
                    "Unique trajectory timestamps",
                    not data.duplicated(["trajectory_id", "timestamp_t"]).any(),
                ],
                [
                    "15-minute transitions",
                    data["timestamp_next"]
                    .sub(data["timestamp_t"])
                    .eq(pd.Timedelta(minutes=15))
                    .all(),
                ],
                [
                    "Train/validation family disjoint",
                    set(_families.get("train", [])).isdisjoint(
                        _families.get("validation", [])
                    ),
                ],
                [
                    "Frozen family absent",
                    not data["building_family_id"].eq("sfh_2016_now").any(),
                ],
            ],
            columns=["check", "passed"],
        )
        integrity_view = mo.ui.table(integrity, pagination=False, selection=None)
    else:
        integrity = pd.DataFrame()
        integrity_view = mo.md("Artifact required.")
    mo.vstack([mo.md("## Mechanical contract"), integrity_view])
    return (integrity,)


@app.cell
def _(available, data, mo, pd):
    if available:
        semantics = pd.DataFrame(
            [
                [
                    "Raw MPC rewritten",
                    int(data["raw_nominal_correction"].abs().gt(1e-9).sum()),
                    "Fail",
                ],
                [
                    "Largest nominal rewrite [K]",
                    float(data["raw_nominal_correction"].abs().max()),
                    "Fail",
                ],
                [
                    "Requested differs from applied",
                    int(data["T_hp_sup_requested"].ne(data["T_hp_sup_applied"]).sum()),
                    "Fail",
                ],
                [
                    "Rows labelled scaled",
                    int(data["perturbation_disposition"].eq("scaled").sum()),
                    "Diagnostic",
                ],
                [
                    "Rows labelled rejected",
                    int(data["perturbation_disposition"].eq("rejected").sum()),
                    "Diagnostic",
                ],
                ["Solver failures", int(data["solver_status"].ne(0).sum()), "Pass"],
                [
                    "Negative MPC objectives",
                    int(data["mpc_objective"].lt(0).sum()),
                    "Fail",
                ],
                [
                    "Gains above declared 8 kW",
                    int(data["Qdot_gains_t"].gt(8000).sum()),
                    "Fail",
                ],
            ],
            columns=["diagnostic", "value", "assessment"],
        )
        semantics_view = mo.ui.table(semantics, pagination=False, selection=None)
    else:
        semantics = pd.DataFrame()
        semantics_view = mo.md("Artifact required.")
    mo.vstack([mo.md("## Action semantics and solver diagnostics"), semantics_view])
    return (semantics,)


@app.cell
def _(available, data, mo, px):
    if available:
        residual = data[data["perturbation_band"] != "nominal"]
        excitation_figure = px.violin(
            residual,
            x="perturbation_band",
            y="applied_residual",
            color="perturbation_band",
            box=True,
            points=False,
            category_orders={"perturbation_band": ["small", "medium", "wide"]},
            labels={
                "applied_residual": "Applied residual [K]",
                "perturbation_band": "Band",
            },
            title="Applied excitation has distinct, symmetric amplitude bands",
        )
        excitation_figure.update_layout(height=430, showlegend=False)
        excitation_view = excitation_figure
    else:
        excitation_figure = None
        excitation_view = mo.md("Artifact required.")
    mo.vstack([mo.md("## Excitation coverage"), excitation_view])
    return (excitation_figure,)


@app.cell
def _(available, data, mo):
    options = sorted(data["trajectory_id"].unique()) if available else ["missing"]
    trajectory = mo.ui.dropdown(options, value=options[0], label="Trajectory")
    mo.output.replace(trajectory)
    return (trajectory,)


@app.cell
def _(available, data, go, make_subplots, mo, trajectory):
    if available:
        frame = data[data["trajectory_id"] == trajectory.value].sort_values(
            "timestamp_next"
        )
        trajectory_figure = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05
        )
        trajectory_figure.add_trace(
            go.Scatter(x=frame["timestamp_next"], y=frame["T_room_next"], name="Room"),
            row=1,
            col=1,
        )
        trajectory_figure.add_hline(
            y=20, line_dash="dot", line_color="#777", row=1, col=1
        )
        trajectory_figure.add_hline(
            y=26, line_dash="dot", line_color="#777", row=1, col=1
        )
        for _column, _name, _dash in (
            ("T_hp_sup_mpc_raw", "Raw MPC", "dot"),
            ("T_hp_sup_mpc", "Stored MPC", "dash"),
            ("T_hp_sup_applied", "Applied", "solid"),
        ):
            trajectory_figure.add_trace(
                go.Scatter(
                    x=frame["timestamp_next"],
                    y=frame[_column],
                    name=_name,
                    line={"dash": _dash},
                ),
                row=2,
                col=1,
            )
        trajectory_figure.add_trace(
            go.Scatter(
                x=frame["timestamp_next"],
                y=frame["applied_residual"],
                name="Applied residual",
            ),
            row=3,
            col=1,
        )
        trajectory_figure.add_trace(
            go.Scatter(x=frame["timestamp_next"], y=frame["T_amb_t"], name="Ambient"),
            row=4,
            col=1,
        )
        trajectory_figure.update_yaxes(title_text="Room [°C]", row=1, col=1)
        trajectory_figure.update_yaxes(title_text="Supply [°C]", row=2, col=1)
        trajectory_figure.update_yaxes(title_text="Residual [K]", row=3, col=1)
        trajectory_figure.update_yaxes(title_text="Ambient [°C]", row=4, col=1)
        trajectory_figure.update_layout(
            height=820, title=trajectory.value, legend_orientation="h"
        )
        trajectory_view = trajectory_figure
    else:
        frame, trajectory_figure = None, None
        trajectory_view = mo.md("Artifact required.")
    mo.vstack([mo.md("## Trajectory inspector"), trajectory_view])
    return frame, trajectory_figure


@app.cell
def _(available, data, mo, pd):
    if available:
        nominal = data[data["perturbation_band"] == "nominal"][
            [
                "building_id",
                "timestamp_next",
                "T_room_next",
                "T_hp_ret_next",
                "T_hp_sup_applied",
            ]
        ].rename(
            columns={
                "T_room_next": "T_room_nominal",
                "T_hp_ret_next": "T_hp_ret_nominal",
                "T_hp_sup_applied": "T_hp_sup_nominal",
            }
        )
        paired = data[data["perturbation_band"] != "nominal"].merge(
            nominal,
            on=["building_id", "timestamp_next"],
            how="inner",
            validate="many_to_one",
        )
        for _target in ("T_room", "T_hp_ret", "T_hp_sup"):
            paired[f"{_target}_delta"] = (
                paired[
                    f"{_target}_next" if _target != "T_hp_sup" else "T_hp_sup_applied"
                ]
                - paired[f"{_target}_nominal"]
            )
        response = paired.groupby("perturbation_band", as_index=False).agg(
            action_rms_K=("T_hp_sup_delta", lambda x: float((x.pow(2).mean()) ** 0.5)),
            room_response_rms_K=(
                "T_room_delta",
                lambda x: float((x.pow(2).mean()) ** 0.5),
            ),
            return_response_rms_K=(
                "T_hp_ret_delta",
                lambda x: float((x.pow(2).mean()) ** 0.5),
            ),
        )
        response_view = mo.ui.table(response, pagination=False, selection=None)
    else:
        paired, response = pd.DataFrame(), pd.DataFrame()
        response_view = mo.md("Artifact required.")
    mo.vstack(
        [
            mo.md("## Matched nominal response"),
            mo.md(
                "Descriptive excited-minus-nominal differences; these are not causal retained-gain estimates."
            ),
            response_view,
        ]
    )
    return paired, response


@app.cell
def _(mo, pd):
    gate = pd.DataFrame(
        [
            ["Schema and 15-minute alignment", "Pass"],
            ["Family-disjoint split", "Pass"],
            ["Nontrivial excitation", "Pass"],
            ["One runtime-parametric solver across houses", "Fail"],
            ["Raw nominal/requested/applied audit trail", "Fail"],
            ["Canonical heat-pump semantics", "Fail"],
            ["Effective mdot and complete replay metadata", "Fail"],
            ["Realized objective degradation", "Not available"],
            ["Seasonal and weather coverage", "Fail"],
            ["Retained-gain gate", "Not evaluated on this artifact"],
        ],
        columns=["criterion", "assessment"],
    )
    mo.vstack(
        [mo.md("## Pilot gate"), mo.ui.table(gate, pagination=False, selection=None)]
    )
    return (gate,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "**Review conclusion:** the artifact is useful for smoke tests and qualitative excitation analysis, but should be recollected before it is used as benchmark training data or as evidence that fine-tuning restores deployable control sensitivity."
        ),
        kind="danger",
    )


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    # Plotly switches to a WebGL renderer above ~1000 points, which renders as
    # a blank grey box wherever WebGL is unavailable. Everything here stays on
    # the SVG renderer instead (Scattergeo is SVG-based too).
    SVG = "svg"
    return Path, SVG, go, json, mo, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Benchmark source data

    Artifacts written by `data/scripts/prepare_benchmark_data.py`: TABULA building
    records, Open-Meteo reference weather and forecast runs, and Energy-Charts
    day-ahead prices.

    Reads normalized Parquet only - it never downloads. Run the acquisition
    script first:

    ```bash
    python data/scripts/prepare_benchmark_data.py --output-dir data/source
    ```
    """)


@app.cell(hide_code=True)
def _(buildings, data_dir, json, pd):
    def _locations():
        """One row per configured location, read back from the weather manifests."""
        manifests = sorted((data_dir / "manifests" / "weather_reference").glob("*.json"))
        entries = [
            entry
            for path in manifests
            for entry in json.loads(path.read_text())["request"]["locations"]
        ]
        if not entries:
            return pd.DataFrame()
        frame = pd.DataFrame(entries).drop_duplicates("location_id").reset_index(drop=True)
        # location ids follow "<country>_<city>", e.g. de_freiburg
        frame["country_code"] = frame["location_id"].str.split("_").str[0].str.upper()
        frame["city"] = (
            frame["location_id"].str.split("_").str[1].str.replace("-", " ").str.title()
        )
        if len(buildings):
            counts = (
                buildings.assign(country_code=buildings["country_code"].astype(str))
                .groupby("country_code")
                .agg(
                    variants=("building_id", "count"),
                    families=("building_family_id", "nunique"),
                )
                .reset_index()
            )
            frame = frame.merge(counts, on="country_code", how="left")
        return frame

    locations = _locations()
    return (locations,)


@app.cell(hide_code=True)
def _(go, locations):
    def _globe():
        # Scattergeo renders as SVG, so the globe works without WebGL.
        marker_size = (
            8 + locations["variants"] / 2
            if "variants" in locations
            else [12] * len(locations)
        )
        hover = (
            locations["city"]
            + ", "
            + locations["country_code"]
            + "<br>"
            + locations.get("variants", 0).astype("Int64").astype(str)
            + " building variants in "
            + locations.get("families", 0).astype("Int64").astype(str)
            + " families"
        )
        points = go.Scattergeo(
            lon=locations["longitude"],
            lat=locations["latitude"],
            text=locations["country_code"],
            hovertext=hover,
            hoverinfo="text",
            mode="markers+text",
            textposition="top center",
            textfont={"size": 11, "color": "#111827"},
            marker={
                "size": marker_size,
                "color": "#dc2626",
                "line": {"width": 1, "color": "white"},
                "opacity": 0.9,
            },
            name="benchmark locations",
        )

        base_geo = {
            "projection": {"type": "orthographic", "rotation": {"lon": 12, "lat": 45}},
            "showland": True,
            "landcolor": "#e5e7eb",
            "showocean": True,
            "oceancolor": "#dbeafe",
            "showcountries": True,
            "countrycolor": "#9ca3af",
            "coastlinecolor": "#6b7280",
            "showframe": False,
        }

        # One frame per 8 degrees of longitude - a full rotation.
        spin = list(range(12, 372, 8))
        frames = [
            go.Frame(
                name=str(lon),
                layout={"geo": {"projection": {"rotation": {"lon": lon, "lat": 45}}}},
            )
            for lon in spin
        ]

        figure = go.Figure(data=[points], frames=frames)
        figure.update_layout(
            geo=base_geo,
            height=560,
            margin={"l": 0, "r": 0, "t": 48, "b": 0},
            title="Where the benchmark buildings sit - one representative location per country",
            showlegend=False,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.05,
                    "y": 0.05,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "buttons": [
                        {
                            "label": "Spin",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 90, "redraw": True},
                                    "transition": {"duration": 0},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                        {
                            "label": "Stop",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
        )
        return figure

    _globe() if len(locations) else None


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    These are representative synthetic locations - one per country - not the
    coordinates of individual TABULA houses. Every building of a country shares
    its country's weather and price series, so the marker size is the number of
    building variants that location drives.
    """)


@app.cell(hide_code=True)
def _(Path, mo):
    try:
        _default = Path(__file__).resolve().parents[1] / "data" / "source"
    except NameError:
        _default = Path("data/source").resolve()

    data_dir_input = mo.ui.text(
        value=str(_default), label="Source data directory", full_width=True
    )
    mo.output.replace(data_dir_input)
    return (data_dir_input,)


@app.cell(hide_code=True)
def _(Path, data_dir_input):
    data_dir = Path(data_dir_input.value).expanduser()
    normalized = data_dir / "normalized"

    buildings_path = normalized / "buildings" / "tabula_sfh.parquet"
    reference_paths = sorted((normalized / "weather_reference").glob("*.parquet"))
    forecast_paths = sorted((normalized / "weather_forecasts").rglob("*.parquet"))
    price_paths = sorted((normalized / "electricity_prices").glob("*.parquet"))
    return (
        buildings_path,
        data_dir,
        forecast_paths,
        normalized,
        price_paths,
        reference_paths,
    )


@app.cell(hide_code=True)
def _(pd):
    def to_plain(frame: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas extension dtypes to numpy dtypes for plotting."""
        converted = {}
        for name, series in frame.items():
            dtype = str(series.dtype)
            if dtype in {"Float64", "Int64"}:
                converted[name] = series.astype("float64")
            elif dtype == "boolean":
                converted[name] = series.fillna(False).astype(bool)
            elif dtype == "string":
                converted[name] = series.astype(object)
        return frame.assign(**converted)

    return (to_plain,)


@app.cell(hide_code=True)
def _(
    buildings_path,
    forecast_paths,
    mo,
    normalized,
    pd,
    price_paths,
    reference_paths,
):
    inventory = pd.DataFrame(
        [
            ("buildings", 1 if buildings_path.exists() else 0),
            ("weather_reference", len(reference_paths)),
            ("weather_forecasts", len(forecast_paths)),
            ("electricity_prices", len(price_paths)),
        ],
        columns=["modality", "files"],
    )
    anything_found = int(inventory["files"].sum()) > 0

    _view = (
        mo.ui.table(inventory, pagination=False, selection=None)
        if anything_found
        else mo.callout(
            mo.md(
                f"No normalized artifacts under `{normalized}`. Run the acquisition script."
            ),
            kind="warn",
        )
    )
    mo.vstack([mo.md("## Available artifacts"), _view])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Buildings
    """)


@app.cell(hide_code=True)
def _(buildings_path, pd, to_plain):
    buildings = (
        to_plain(pd.read_parquet(buildings_path))
        if buildings_path.exists()
        else pd.DataFrame()
    )
    return (buildings,)


@app.cell(hide_code=True)
def _(buildings, mo):
    countries = sorted(buildings["country_code"].unique()) if len(buildings) else []
    country_select = mo.ui.multiselect(
        options=countries, value=countries, label="Countries"
    )
    variant_select = mo.ui.multiselect(
        options=["1", "2", "3"],
        value=["1", "2", "3"],
        label="Variants (1=existing, 2=standard, 3=ambitious)",
    )
    (
        mo.hstack([country_select, variant_select], justify="start", gap=2)
        if len(buildings)
        else None
    )
    return countries, country_select, variant_select


@app.cell(hide_code=True)
def _(buildings, countries, country_select, variant_select):
    # `or` fallbacks: on the very first render a selector's value can still be
    # empty, which would otherwise show an empty plot until the notebook is run
    # a second time.
    _countries = country_select.value or countries
    _variants = [float(v) for v in (variant_select.value or ["1", "2", "3"])]
    selected_buildings = (
        buildings[
            buildings["country_code"].isin(_countries)
            & buildings["variant_number"].isin(_variants)
        ]
        if len(buildings)
        else buildings
    )
    return (selected_buildings,)


@app.cell(hide_code=True)
def _(mo, pd, selected_buildings):
    def _cohort_summary():
        if not len(selected_buildings):
            return mo.md("*No building data loaded.*")
        cohort = (
            selected_buildings.groupby("country_code")
            .agg(
                variants=("building_id", "count"),
                families=("building_family_id", "nunique"),
                orientation_missing=("window_orientation_missing", "sum"),
                median_area_m2=("reference_area_m2", "median"),
            )
            .reset_index()
        )
        total = pd.DataFrame(
            [
                [
                    "TOTAL",
                    cohort["variants"].sum(),
                    cohort["families"].sum(),
                    cohort["orientation_missing"].sum(),
                    selected_buildings["reference_area_m2"].median(),
                ]
            ],
            columns=cohort.columns,
        )
        return mo.ui.table(
            pd.concat([cohort, total], ignore_index=True),
            pagination=False,
            selection=None,
        )

    _cohort_summary()


@app.cell(hide_code=True)
def _(SVG, px, selected_buildings):
    (
        px.scatter(
            selected_buildings,
            x="reference_area_m2",
            y="transmission_W_m2K",
            color="country_code",
            symbol="window_orientation_missing",
            hover_name="building_id",
            hover_data=[
                "building_family_id",
                "variant_number",
                "year_start",
                "year_end",
            ],
            render_mode=SVG,
            labels={
                "reference_area_m2": "Reference floor area [m²]",
                "transmission_W_m2K": "Transmission [W/(m²·K)]",
                "country_code": "Country",
                "window_orientation_missing": "Orientation missing",
            },
            title="Envelope quality vs size (lower transmission = better insulated)",
        )
        if len(selected_buildings)
        else None
    )


@app.cell(hide_code=True)
def _(mo, selected_buildings):
    (
        mo.ui.table(selected_buildings, pagination=True, page_size=10, selection=None)
        if len(selected_buildings)
        else None
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Simulation parameters

    The RC model in `i4b` is driven by lumped quantities rather than the raw
    TABULA columns. These are derived here so the spread the benchmark spans
    is visible before any mapping code is written.

    | Derived | From |
    |---|---|
    | `H_tr` [W/K] | `transmission_W_m2K` × `reference_area_m2` |
    | `H_ve` [W/K] | `ventilation_W_m2K` × `reference_area_m2` |
    | `H_tr_light` [W/K] | window 1 + window 2 + door 1 transmission |
    | `C` [Wh/K] | `thermal_capacity_Wh_m2K` × `reference_area_m2` |
    | `tau` [h] | `C / (H_tr + H_ve)` - the thermal time constant |
    | `window_floor_ratio` | total window area / floor area |

    `tau` is the one that matters most for MPC horizons: it sets how far
    ahead the thermal mass can actually be exploited.
    """)


@app.cell(hide_code=True)
def _(selected_buildings):
    def _derive(frame):
        if not len(frame):
            return frame
        window_area = frame["window_1_area_m2"] + frame["window_2_area_m2"]
        derived = frame.assign(
            H_tr_W_K=frame["transmission_W_m2K"] * frame["reference_area_m2"],
            H_ve_W_K=frame["ventilation_W_m2K"] * frame["reference_area_m2"],
            H_tr_light_W_K=(
                frame["window_1_transmission_W_K"].fillna(0)
                + frame["window_2_transmission_W_K"].fillna(0)
                + frame["door_1_transmission_W_K"].fillna(0)
            ),
            C_Wh_K=frame["thermal_capacity_Wh_m2K"] * frame["reference_area_m2"],
            window_area_m2=window_area,
            window_floor_ratio=window_area / frame["reference_area_m2"],
            volume_m3=frame["reference_area_m2"] * frame["room_height_m"],
        )
        derived["tau_h"] = derived["C_Wh_K"] / (
            derived["H_tr_W_K"] + derived["H_ve_W_K"]
        )
        derived["variant"] = derived["variant_number"].map(
            {1.0: "1 existing", 2.0: "2 standard", 3.0: "3 ambitious"}
        )
        return derived

    sim_params = _derive(selected_buildings)
    # C_Wh_K and volume_m3 are deliberately not plotted: TABULA reports
    # c_m = 45 Wh/(m^2 K) and h_room = 2.5 m for every selected variant, so both
    # are reference_area_m2 rescaled (correlation 1.000) and would suggest
    # independent variation that does not exist.
    SIM_COLUMNS = [
        "H_tr_W_K",
        "H_ve_W_K",
        "H_tr_light_W_K",
        "tau_h",
        "window_floor_ratio",
        "reference_area_m2",
    ]
    return SIM_COLUMNS, sim_params


@app.cell(hide_code=True)
def _(SIM_COLUMNS, mo, sim_params):
    (
        mo.ui.table(
            sim_params[SIM_COLUMNS]
            .describe()
            .T[["min", "25%", "50%", "75%", "max"]]
            .round(2)
            .reset_index()
            .rename(columns={"index": "parameter"}),
            pagination=False,
            selection=None,
        )
        if len(sim_params)
        else None
    )


@app.cell(hide_code=True)
def _(SIM_COLUMNS, px, sim_params):
    def _distributions():
        melted = sim_params.melt(
            id_vars=["country_code", "variant"],
            value_vars=SIM_COLUMNS,
            var_name="parameter",
            value_name="value",
        ).dropna(subset=["value"])
        figure = px.histogram(
            melted,
            x="value",
            color="variant",
            facet_col="parameter",
            facet_col_wrap=4,
            nbins=30,
            title="Distribution of simulation parameters across the cohort",
            category_orders={"variant": ["1 existing", "2 standard", "3 ambitious"]},
        )
        figure.update_xaxes(matches=None, showticklabels=True)
        figure.update_yaxes(matches=None)
        figure.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        figure.update_layout(height=560, bargap=0.05)
        return figure

    _distributions() if len(sim_params) else None


@app.cell(hide_code=True)
def _(px, sim_params):
    (
        px.box(
            sim_params.dropna(subset=["tau_h"]),
            x="country_code",
            y="tau_h",
            color="variant",
            points="all",
            category_orders={"variant": ["1 existing", "2 standard", "3 ambitious"]},
            labels={
                "country_code": "Country",
                "tau_h": "Thermal time constant tau [h]",
                "variant": "Refurbishment level",
            },
            title="Thermal time constant - how far ahead an MPC can exploit the mass",
        )
        if len(sim_params)
        else None
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Weather

    Reference weather is the Open-Meteo archive blend. Forecast runs are
    individual ECMWF IFS initializations, so the same valid time appears once
    per run at a different lead time. Both are hourly - Open-Meteo has no
    sub-hourly data for these variables, and its `minutely_15` option is
    interpolated rather than measured.
    """)


@app.cell(hide_code=True)
def _(mo, reference_paths):
    weather_select = mo.ui.dropdown(
        options={path.stem: str(path) for path in reference_paths},
        value=reference_paths[0].stem if reference_paths else None,
        label="Location and period",
    )
    overlay_forecasts = mo.ui.checkbox(value=True, label="Overlay forecast runs")
    (
        mo.hstack([weather_select, overlay_forecasts], justify="start", gap=2)
        if reference_paths
        else None
    )
    return overlay_forecasts, weather_select


@app.cell(hide_code=True)
def _(Path, forecast_paths, pd, reference_paths, to_plain, weather_select):
    _selected = weather_select.value or (
        str(reference_paths[0]) if reference_paths else None
    )
    if _selected:
        reference = to_plain(pd.read_parquet(_selected))
        location_id = reference["location_id"].iloc[0]
        matching_forecasts = [
            p for p in forecast_paths if Path(p).parent.name == location_id
        ]
    else:
        reference = pd.DataFrame()
        location_id = None
        matching_forecasts = []
    return matching_forecasts, reference


@app.cell(hide_code=True)
def _(go, matching_forecasts, overlay_forecasts, pd, reference, to_plain):
    def _weather_figure():
        figure = go.Figure()
        figure.add_scatter(
            x=reference["valid_time_utc"],
            y=reference["temperature_2m_C"],
            name="reference",
            line={"color": "#111827", "width": 1},
        )
        if overlay_forecasts.value:
            window = (
                reference["valid_time_utc"].min(),
                reference["valid_time_utc"].max(),
            )
            for path in matching_forecasts:
                run = to_plain(pd.read_parquet(path))
                run = run[run["valid_time_utc"].between(*window)]
                if run.empty:
                    continue
                figure.add_scatter(
                    x=run["valid_time_utc"],
                    y=run["temperature_2m_C"],
                    name=str(run["initialization_time_utc"].iloc[0])[:16],
                    line={"width": 2},
                    opacity=0.9,
                )
        figure.update_layout(
            title="Air temperature: reference vs archived forecast runs",
            xaxis_title="Valid time (UTC)",
            yaxis_title="Temperature [°C]",
            legend_title="Series",
            hovermode="x unified",
        )
        return figure

    _weather_figure() if len(reference) else None


@app.cell(hide_code=True)
def _(SVG, px, reference):
    def _irradiance():
        # A year of hourly irradiance is 26k points per component and unreadable
        # at full resolution, so the overview is a daily mean.
        daily = (
            reference.assign(day=reference["valid_time_utc"].dt.floor("D"))
            .groupby("day", as_index=False)[["ghi_W_m2", "dni_W_m2", "dhi_W_m2"]]
            .mean()
        )
        return px.line(
            daily.melt(
                id_vars="day",
                value_vars=["ghi_W_m2", "dni_W_m2", "dhi_W_m2"],
                var_name="component",
                value_name="irradiance",
            ),
            x="day",
            y="irradiance",
            color="component",
            render_mode=SVG,
            labels={
                "day": "Day (UTC)",
                "irradiance": "Daily mean irradiance [W/m²]",
                "component": "Component",
            },
            title="Solar irradiance, daily mean (reference)",
        )

    _irradiance() if len(reference) else None


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Electricity prices

    Native provider resolution, negative prices preserved. Cyprus has no
    supported Energy-Charts bidding zone and is absent by design.
    """)


@app.cell(hide_code=True)
def _(mo, price_paths):
    price_select = mo.ui.dropdown(
        options={path.stem: str(path) for path in price_paths},
        value=price_paths[0].stem if price_paths else None,
        label="Market and period",
    )
    mo.output.replace(price_select if price_paths else None)
    return (price_select,)


@app.cell(hide_code=True)
def _(pd, price_paths, price_select, to_plain):
    _selected = price_select.value or (str(price_paths[0]) if price_paths else None)
    prices = to_plain(pd.read_parquet(_selected)) if _selected else pd.DataFrame()
    return (prices,)


@app.cell(hide_code=True)
def _(go, prices):
    def _price_figure():
        negative = prices[prices["price_eur_per_mwh"] < 0]
        figure = go.Figure()
        figure.add_scatter(
            x=prices["delivery_start_utc"],
            y=prices["price_eur_per_mwh"],
            name="price",
            line={"color": "#2563eb", "width": 1},
        )
        if len(negative):
            figure.add_scatter(
                x=negative["delivery_start_utc"],
                y=negative["price_eur_per_mwh"],
                name=f"negative ({len(negative)})",
                mode="markers",
                marker={"color": "#dc2626", "size": 4},
            )
        figure.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        figure.update_layout(
            title=f"Day-ahead price - {prices['market_id'].iloc[0]}",
            xaxis_title="Delivery start (UTC)",
            yaxis_title="Price [EUR/MWh]",
            hovermode="x unified",
        )
        return figure

    _price_figure() if len(prices) else None


@app.cell(hide_code=True)
def _(mo, pd, price_paths, to_plain):
    def _price_coverage():
        rows = []
        for path in price_paths:
            frame = to_plain(pd.read_parquet(path))
            times = frame["delivery_start_utc"]
            gaps = times.diff().dropna()
            big = gaps[gaps > pd.Timedelta(hours=1)]
            missing_days = (
                (big.sum() - pd.Timedelta(hours=len(big))).total_seconds() / 86400
                if len(big)
                else 0.0
            )
            rows.append(
                {
                    "artifact": path.stem,
                    "rows": len(frame),
                    "from": str(times.min())[:10],
                    "to": str(times.max())[:10],
                    "gaps": len(big),
                    "missing days": round(missing_days, 1),
                }
            )
        frame = pd.DataFrame(rows)
        note = "gaps are preserved, never filled - NO1 is materially incomplete"
        return mo.vstack(
            [mo.ui.table(frame, pagination=False, selection=None), mo.md(f"*{note}*")]
        )

    _price_coverage() if price_paths else None


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Statistics
    """)


@app.cell(hide_code=True)
def _(pd, px, reference_paths, to_plain):
    def _climate():
        frames = []
        for path in reference_paths:
            frame = to_plain(pd.read_parquet(path))
            location, period = path.stem.rsplit("_period_", 1)
            frames.append(frame.assign(location=location, period=f"period_{period}"))
        if not frames:
            return None
        allw = pd.concat(frames, ignore_index=True)
        allw["month"] = (
            allw["valid_time_utc"]
            .dt.tz_localize(None)
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        monthly = allw.groupby(["location", "month"], as_index=False)[
            "temperature_2m_C"
        ].mean()
        return px.line(
            monthly,
            x="month",
            y="temperature_2m_C",
            color="location",
            render_mode="svg",
            labels={
                "month": "Month",
                "temperature_2m_C": "Mean air temperature [°C]",
                "location": "Location",
            },
            title="Monthly mean temperature - climate spread across the 7 countries",
        )

    _climate()


@app.cell(hide_code=True)
def _(pd, px, reference_paths, to_plain):
    def _heating_demand_proxy():
        rows = []
        for path in reference_paths:
            frame = to_plain(pd.read_parquet(path))
            location, period = path.stem.rsplit("_period_", 1)
            # Heating degree hours below 15 degC, the usual heating threshold.
            deficit = (15.0 - frame["temperature_2m_C"]).clip(lower=0)
            rows.append(
                {
                    "location": location,
                    "period": f"period_{period}",
                    "heating_degree_days": deficit.sum() / 24.0,
                    "mean_ghi_W_m2": frame["ghi_W_m2"].mean(),
                }
            )
        if not rows:
            return None
        return px.bar(
            pd.DataFrame(rows).sort_values("heating_degree_days"),
            x="location",
            y="heating_degree_days",
            color="period",
            barmode="group",
            labels={
                "location": "Location",
                "heating_degree_days": "Heating degree days (base 15 °C)",
                "period": "Period",
            },
            title="Heating demand proxy per location and period",
        )

    _heating_demand_proxy()


@app.cell(hide_code=True)
def _(pd, price_paths, px, to_plain):
    def _price_stats():
        frames = [to_plain(pd.read_parquet(path)) for path in price_paths]
        if not frames:
            return None
        allp = pd.concat(frames, ignore_index=True)
        allp["month"] = (
            allp["delivery_start_utc"]
            .dt.tz_localize(None)
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        monthly = allp.groupby(["market_id", "month"], as_index=False).agg(
            median_price=("price_eur_per_mwh", "median"),
        )
        return px.line(
            monthly,
            x="month",
            y="median_price",
            color="market_id",
            render_mode="svg",
            labels={
                "month": "Month",
                "median_price": "Median price [EUR/MWh]",
                "market_id": "Market",
            },
            title="Monthly median day-ahead price",
        )

    _price_stats()


@app.cell(hide_code=True)
def _(pd, price_paths, px, to_plain):
    def _price_resolution():
        rows = []
        for path in price_paths:
            frame = to_plain(pd.read_parquet(path))
            times = frame["delivery_start_utc"]
            rows.append(
                pd.DataFrame(
                    {
                        "market": frame["market_id"],
                        "month": times.dt.tz_localize(None)
                        .dt.to_period("M")
                        .dt.to_timestamp(),
                        "step_min": times.diff().dt.total_seconds().div(60),
                    }
                ).dropna()
            )
        if not rows:
            return None
        allr = pd.concat(rows, ignore_index=True)
        modal = allr.groupby(["market", "month"], as_index=False)["step_min"].agg(
            lambda s: s.mode().iloc[0]
        )
        return px.line(
            modal,
            x="month",
            y="step_min",
            color="market",
            markers=True,
            render_mode="svg",
            labels={
                "month": "Month",
                "step_min": "Native step [minutes]",
                "market": "Market",
            },
            title="Price resolution over time (day-ahead moved to 15-min MTU on 2025-10-01)",
        )

    _price_resolution()


@app.cell(hide_code=True)
def _(Path, forecast_paths, pd, px, reference_paths, to_plain):
    def _forecast_error():
        if not forecast_paths or not reference_paths:
            return None
        truth = {}
        for path in reference_paths:
            frame = to_plain(pd.read_parquet(path))
            truth.setdefault(frame["location_id"].iloc[0], []).append(
                frame.set_index("valid_time_utc")["temperature_2m_C"]
            )
        truth = {loc: pd.concat(series) for loc, series in truth.items()}

        rows = []
        for path in forecast_paths:
            run = to_plain(pd.read_parquet(Path(path)))
            location = run["location_id"].iloc[0]
            if location not in truth:
                continue
            actual = run["valid_time_utc"].map(truth[location])
            rows.append(
                pd.DataFrame(
                    {
                        "location": location,
                        "lead_hours": run["lead_hours"],
                        "abs_error": (run["temperature_2m_C"] - actual).abs(),
                    }
                ).dropna()
            )
        if not rows:
            return None
        errors = pd.concat(rows, ignore_index=True)
        by_lead = errors.groupby("lead_hours", as_index=False)["abs_error"].mean()
        return px.line(
            by_lead,
            x="lead_hours",
            y="abs_error",
            markers=True,
            render_mode="svg",
            labels={
                "lead_hours": "Forecast lead time [h]",
                "abs_error": "Mean absolute temperature error [K]",
            },
            title="ECMWF IFS temperature error vs lead time (forecast subset vs reference)",
        )

    _forecast_error()


if __name__ == "__main__":
    app.run()

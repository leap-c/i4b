"""Building the corpus: the catalog, the trajectories, and the split.

This is how the dataset is *made*. Evaluating against it needs none of this -- `i4b_bench` reads
the finished corpus and never imports from here, which is what lets the two move independently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from i4b_bench.corpus import (  # noqa: F401
    TRANSITION_COLUMNS,
    load_params,
    prepare_disturbances,
)


def _number(row: Mapping, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    return default if value is None or pd.isna(value) else float(value)


def design_mdot_hp(row: Mapping, ambient: pd.Series, *, delta_t_K: float = 15.0) -> dict:
    """Size heat-pump flow transparently and return the sizing provenance."""
    h_total = _number(row, "transmission_W_m2K") * _number(row, "reference_area_m2")
    h_total += _number(row, "ventilation_W_m2K") * _number(row, "reference_area_m2")
    design_ambient = float(np.nanpercentile(pd.to_numeric(ambient), 1))
    design_load = h_total * (20.0 - design_ambient)
    raw = design_load / (4181.0 * delta_t_K)
    value = float(np.clip(raw, 0.05, 0.50))
    return {
        "mdot_hp": value,
        "mdot_hp_raw": float(raw),
        "mdot_hp_clipped": bool(value != raw),
        "design_ambient_C": design_ambient,
        "water_delta_t_K": delta_t_K,
    }


def map_tabula_row(
    row: Mapping,
    location: Mapping,
    mdot_hp: float,
    *,
    impute_orientation: bool = True,
) -> dict:
    """Map one normalized TABULA row to an I4B building parameter dictionary."""
    area = _number(row, "reference_area_m2")
    total_window_area = _number(row, "window_1_area_m2") + _number(row, "window_2_area_m2")
    directions = {
        "north": _number(row, "window_north_area_m2"),
        "east": _number(row, "window_east_area_m2"),
        "south": _number(row, "window_south_area_m2"),
        "west": _number(row, "window_west_area_m2"),
    }
    orientation_missing = total_window_area > 0 and sum(directions.values()) == 0
    if orientation_missing and impute_orientation:
        directions = {name: total_window_area / 4.0 for name in directions}

    azimuth = {"north": 0, "east": 90, "south": 180, "west": 270}
    g_value = _number(row, "window_g_value")
    windows = [
        {
            "area": value,
            "tilt": 90,
            "azimuth": azimuth[name],
            "g_value": g_value,
            "c_frame": 0.3,
            "c_shade": 0.6,
        }
        for name, value in directions.items()
    ]
    position = {
        "lat": float(location["latitude"]),
        "long": float(location["longitude"]),
        "altitude": float(location.get("altitude", 0.0)),
        "timezone": str(location["timezone"]),
    }
    return {
        "H_ve": _number(row, "ventilation_W_m2K") * area,
        "H_tr": _number(row, "transmission_W_m2K") * area,
        "H_tr_light": sum(
            _number(row, name)
            for name in (
                "window_1_transmission_W_K",
                "window_2_transmission_W_K",
                "door_1_transmission_W_K",
            )
        ),
        "c_bldg": _number(row, "thermal_capacity_Wh_m2K"),
        "area_floor": area,
        "height_room": _number(row, "room_height_m"),
        "name": str(row["building_id"]),
        "T_offset": 0.0,
        "T_amb_lim": 20.0,
        "mdot_hp": float(mdot_hp),
        "windows": windows,
        "position": position,
    }


def make_catalog(
    buildings: pd.DataFrame,
    locations: Mapping[str, Mapping],
    ambient_by_country: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Return a flat catalog and the corresponding nested I4B parameters."""
    records = []
    params_by_id = {}
    for row in buildings.to_dict("records"):
        country = str(row["country_code"])
        sizing = design_mdot_hp(row, ambient_by_country[country])
        params = map_tabula_row(row, locations[country], sizing["mdot_hp"])
        building_id = str(row["building_id"])
        params_by_id[building_id] = params
        missing_orientation = row.get("window_orientation_missing", False)
        missing_orientation = bool(missing_orientation) if pd.notna(missing_orientation) else False
        records.append(
            {
                **row,
                **{
                    name: params[name]
                    for name in (
                        "H_ve",
                        "H_tr",
                        "H_tr_light",
                        "c_bldg",
                        "area_floor",
                        "height_room",
                    )
                },
                "mdot_hp": sizing["mdot_hp"],
                "mdot_hp_raw": sizing["mdot_hp_raw"],
                "mdot_hp_clipped": sizing["mdot_hp_clipped"],
                "window_orientation_imputed": missing_orientation,
                "window_orientation_imputation": "equal_cardinal" if missing_orientation else None,
                "mapping_provenance": json.dumps(
                    {
                        "window_orientation": "equal_cardinal"
                        if missing_orientation
                        else "source",
                        "window_defaults": {"c_frame": 0.3, "c_shade": 0.6},
                        **sizing,
                    },
                    sort_keys=True,
                ),
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
    return pd.DataFrame(records), params_by_id


def validate_buildings(
    params_by_id: Mapping[str, Mapping],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instantiate mapped 4R3C buildings and validate their discrete matrices."""
    import jax.numpy as jnp

    from i4b.core.dynamics import rhs_dispatch
    from i4b.core.integrators import linear_discretize_batch
    from i4b.models.model_buildings import Building

    if not params_by_id:
        raise ValueError("no buildings to validate")
    models = []
    for building_id, params in params_by_id.items():
        if not 0 <= params["H_tr_light"] < params["H_tr"]:
            raise ValueError(f"invalid light/heavy transmission split for {building_id}")
        model = Building(dict(params), mdot_hp=params["mdot_hp"], method="4R3C")
        thermal = np.asarray(
            [
                model.params[name]
                for name in ("H_ve", "H_tr", "C_zone", "C_wall", "C_water", "H_rad_con")
            ]
        )
        if not np.isfinite(thermal).all() or not np.greater(thermal, 0).all():
            raise ValueError(f"invalid thermal parameters for {building_id}")
        models.append(model)

    matrix_keys = (
        "H_ve",
        "H_tr",
        "C_zone",
        "C_wall",
        "C_water",
        "H_rad_con",
        "mdot_hp",
    )
    batch = {
        key: jnp.asarray([model.params[key] for model in models], dtype=jnp.float32)
        for key in matrix_keys
    }
    count = len(models)
    matrices = linear_discretize_batch(
        rhs_dispatch("4R3C"),
        jnp.zeros((count, 3), dtype=jnp.float32),
        jnp.zeros((count, 2), dtype=jnp.float32),
        batch,
        900,
    )
    ad, bd, cd = matrices
    bd = bd[..., None]
    if ad.shape != (count, 3, 3) or bd.shape != (count, 3, 1) or cd.shape != (count, 3, 2):
        raise ValueError("unexpected 4R3C discrete matrix shapes")
    if not all(np.isfinite(matrix).all() for matrix in (ad, bd, cd)):
        raise ValueError("4R3C discrete matrices must be finite")
    return ad, bd, cd


def rollout_controller(
    env, controller: Callable, *, trajectory_id: str, steps: int | None = None
) -> pd.DataFrame:
    """Roll one callable controller through an environment into canonical rows."""
    expected = ("T_room", "T_wall", "T_hp_ret")
    if tuple(env.obs_keys) != expected:
        raise ValueError(f"canonical benchmark rollouts require 4R3C states {expected}")
    observation, _ = env.reset()
    n_steps = min(len(env.p) - 1, env.max_t)
    if steps is not None:
        n_steps = min(n_steps, steps)
    rows = []
    for _ in range(n_steps):
        timestamp = env.get_cur_time()
        disturbance = env.get_cur_p()
        state = {name: float(observation[index]) for index, name in enumerate(env.obs_keys)}
        action = controller(state, env.p.iloc[env.t :])
        next_observation, _, terminated, truncated, info = env.step(env.normalize_action(action))
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "timestamp_utc": timestamp,
                **state,
                "T_hp_sup_applied": float(info["u"]),
                "T_amb": float(disturbance["T_amb"]),
                "Qdot_gains": float(disturbance["Qdot_gains"]),
            }
        )
        observation = next_observation
        if terminated or truncated:
            break
    frame = pd.DataFrame(rows, columns=TRANSITION_COLUMNS)
    frame[list(TRANSITION_COLUMNS[2:])] = frame[list(TRANSITION_COLUMNS[2:])].astype("float32")
    return frame


def aprbs(
    index: pd.DatetimeIndex,
    identity: str,
    *,
    low: float,
    high: float,
    min_hold_steps: int,
    max_hold_steps: int,
    nominal_fraction: float = 0.0,
) -> pd.Series:
    """Generate one deterministic amplitude-modulated piecewise-constant signal."""
    if min_hold_steps < 1 or max_hold_steps < min_hold_steps:
        raise ValueError("APRBS hold bounds must satisfy 1 <= min <= max")
    if not 0 <= nominal_fraction <= 1:
        raise ValueError("nominal_fraction must be between zero and one")
    seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    values = np.empty(len(index), dtype="float32")
    cursor = 0
    while cursor < len(values):
        hold = int(rng.integers(min_hold_steps, max_hold_steps + 1))
        value = 0.0 if rng.random() < nominal_fraction else float(rng.uniform(low, high))
        stop = min(cursor + hold, len(values))
        values[cursor:stop] = value
        cursor = stop
    return pd.Series(values, index=index)


def residual_controller(controller: Callable, residual: float | pd.Series) -> Callable:
    """Add a constant or timestamped residual to a physical-action controller."""

    def controlled(state, future):
        value = residual if np.isscalar(residual) else residual.loc[future.index[0]]
        return float(controller(state, future)) + float(value)

    return controlled


def safe_aprbs_controller(levels: pd.Series) -> Callable:
    """Apply absolute APRBS levels with the fixed 18-28 degC safety envelope."""
    safety_mode = None

    def controller(state, future):
        nonlocal safety_mode
        room = state["T_room"]
        if room >= 27.5:
            safety_mode = "cool"
        elif room <= 18.5:
            safety_mode = "heat"
        elif safety_mode == "cool" and room < 27.0:
            safety_mode = None
        elif safety_mode == "heat" and room > 19.0:
            safety_mode = None
        if safety_mode == "cool":
            return state["T_hp_ret"]
        if safety_mode == "heat":
            return float(levels.max())
        return float(levels.loc[future.index[0]])

    return controller


def _family_order(family_id: str) -> str:
    return hashlib.sha256(family_id.encode("utf-8")).hexdigest()


def make_split_manifests(
    trajectories: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the primary family-and-time split and the time-only ablation."""
    families = trajectories[["building_family_id", "country_code"]].drop_duplicates()
    counts = {
        "BG": (5, 1, 1),
        "CY": (2, 1, 1),
        "DE": (9, 2, 2),
        "FR": (6, 2, 2),
        "IE": (11, 2, 2),
        "NO": (6, 1, 1),
        "PL": (5, 1, 1),
    }
    family_split = {}
    for country, group in families.groupby("country_code", sort=True):
        ordered = sorted(group["building_family_id"].astype(str), key=_family_order)
        expected = counts[str(country)]
        if len(ordered) != sum(expected):
            raise ValueError(f"unexpected family count for {country}: {len(ordered)}")
        n_train, n_validation, _ = expected
        family_split.update({family: "train" for family in ordered[:n_train]})
        family_split.update(
            {family: "validation" for family in ordered[n_train : n_train + n_validation]}
        )
        family_split.update({family: "test" for family in ordered[n_train + n_validation :]})

    intervals = {
        "train": (
            pd.Timestamp("2024-04-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
        ),
        "validation": (
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-04-01", tz="UTC"),
        ),
        "test": (
            pd.Timestamp("2025-04-01", tz="UTC"),
            pd.Timestamp("2026-04-01", tz="UTC"),
        ),
    }
    primary = trajectories.copy()
    primary["split"] = primary["building_family_id"].map(family_split)
    primary = primary[
        ((primary["split"].isin(["train", "validation"])) & (primary["period_id"] == "period_a"))
        | ((primary["split"] == "test") & (primary["period_id"] == "period_b"))
    ]
    requested = primary["split"].map(intervals)
    primary["start_time_utc"] = [
        max(actual, interval[0]) for actual, interval in zip(primary["start_time_utc"], requested)
    ]
    primary["end_time_utc"] = [
        min(actual, interval[1]) for actual, interval in zip(primary["end_time_utc"], requested)
    ]
    primary = primary[primary["start_time_utc"] < primary["end_time_utc"]]

    parts = []
    for split, (start, end) in intervals.items():
        period = "period_a" if split != "test" else "period_b"
        part = trajectories[trajectories["period_id"] == period][
            ["trajectory_id", "start_time_utc", "end_time_utc"]
        ].copy()
        part["split"] = split
        part["start_time_utc"] = part["start_time_utc"].clip(lower=start)
        part["end_time_utc"] = part["end_time_utc"].clip(upper=end)
        part = part[part["start_time_utc"] < part["end_time_utc"]]
        parts.append(part)
    columns = ["trajectory_id", "split", "start_time_utc", "end_time_utc"]
    return primary[columns].reset_index(drop=True), pd.concat(parts, ignore_index=True)[columns]

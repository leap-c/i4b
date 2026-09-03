"""The compiled open-loop cases: their schema, and what makes one valid.

A case is one evaluation window, flattened into a single Parquet row: the identity of the
window, the scenario metadata a result row needs, the observation as of the anchor, and the
plans the plant was driven under. Everything a predictor is shown and everything it is scored
against is in that row, so evaluation touches neither the corpus nor the simulator.

Written once per evaluation-set release by `data/scripts/build_open_loop_cases.py`, and read
back on every evaluation through `load_cases`.

Conventions the artifact fixes
------------------------------
- `history` ends at `start_timestamp`; a row pairs a state with the input that **produced** it.
- `forecast[i]` is the interval input that, with `requested_control[i]`, produces the state
  stamped `forecast.timestamp[i]` -- one step ahead of the input. See `i4b_bench.forecast`.
- A predictor is shown `requested_control`. `applied_control` is what the actuator let through,
  kept for provenance and diagnostics, never handed over.
- A plan is identified by `plan_role`, never by its position.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .control_gain import NOMINAL_ROLE, PLAN_ROLES
from .evaluation_set import (
    CASES_FILE,
    MANIFEST_FILE,
    TIMESTEP_SECONDS,
    OpenLoopDefinition,
)
from .observation import DISTURBANCE_CHANNELS, STATE_CHANNELS, ObsView, history_channels

#: Bumped whenever the case schema or the meaning of a field changes. An evaluator refuses an
#: artifact from another version rather than reading it under the wrong contract.
SCHEMA_VERSION = 1

#: What the manifest records about semantics, so an artifact carries its own contract.
CONTROL_INPUT_SEMANTICS = "predictor receives requested controls"
TRANSITION_SEMANTICS = "x_t + u_t + d_t -> x_(t+1)"

_STAMP = pa.timestamp("s", tz="UTC")


def case_schema(view: ObsView) -> pa.Schema:
    """The Parquet schema for one view's cases.

    Built explicitly rather than inferred: pandas would type the nested columns from whatever
    happened to be in the first row's Python objects, and a channel silently arriving as
    float64, or a list of lists as a plain object column, would change the artifact without
    changing any code.

    Parameters
    ----------
    view : {"perfect", "realistic"}
        Decides the state, history and forecast channels.

    Returns
    -------
    pyarrow.Schema
    """
    channels = [pa.field(name, pa.float32()) for name in history_channels(view)]
    disturbances = [pa.field(name, pa.float32()) for name in DISTURBANCE_CHANNELS[view]]
    horizon = pa.list_(pa.float32())
    return pa.schema(
        [
            pa.field("case_id", pa.string()),
            pa.field("scenario_id", pa.string()),
            pa.field("building_id", pa.string()),
            pa.field("controller_id", pa.string()),
            pa.field("start_timestamp", _STAMP),
            pa.field("view", pa.string()),
            pa.field("timestep_seconds", pa.int32()),
            pa.field("max_context_steps", pa.int32()),
            pa.field("horizon_steps", pa.int32()),
            # what the scenario is, so a result row reads without a join back to the corpus
            pa.field("country", pa.string()),
            pa.field("period_id", pa.string()),
            pa.field("variant", pa.int32()),
            pa.field("transmission_W_m2K", pa.float64()),
            pa.field("year_start", pa.int32()),
            pa.field("year_end", pa.int32()),
            pa.field("floor_area_m2", pa.float64()),
            pa.field(
                "state",
                pa.struct([pa.field(name, pa.float32()) for name in STATE_CHANNELS[view]]),
            ),
            pa.field("history", pa.list_(pa.struct([pa.field("timestamp", _STAMP), *channels]))),
            pa.field(
                "forecast",
                pa.list_(pa.struct([pa.field("timestamp", _STAMP), *disturbances])),
            ),
            pa.field(
                "plans",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("plan_id", pa.string()),
                            pa.field("plan_role", pa.string()),
                            pa.field("requested_control", horizon),
                            pa.field("applied_control", horizon),
                            pa.field("actual_T_room", horizon),
                        ]
                    )
                ),
            ),
        ]
    )


def write_cases(path: str | Path, rows: list[dict], view: ObsView) -> pa.Table:
    """Write compiled cases to Parquet under `case_schema(view)`.

    Parameters
    ----------
    path : str or Path
        Destination file. Overwritten.
    rows : list of dict
        One dict per case, keyed by the schema's field names.
    view : {"perfect", "realistic"}
        Which schema to write under.

    Returns
    -------
    pyarrow.Table
        What was written.
    """
    table = pa.Table.from_pylist(rows, schema=case_schema(view))
    pq.write_table(table, Path(path), compression="zstd")
    return table


def load_cases(path: str | Path, view: ObsView, columns: list[str] | None = None) -> pa.Table:
    """Read cases back, in the declared schema.

    Parquet has no second-resolution timestamp -- its logical types start at milliseconds -- so
    what comes off disk is cast back to `case_schema(view)`. Every stored value is on the
    15-minute grid, so the cast is exact, and it means one schema describes the artifact rather
    than one for writing and another for reading.

    Parameters
    ----------
    path : str or Path
        The `cases.parquet` file.
    view : {"perfect", "realistic"}
        The view the file was written under; the cast fails loudly if it was not.
    columns : list of str, optional
        Only these columns. The nested ones dominate the file, so a reader that does not need
        the plans should not pay to read them.

    Returns
    -------
    pyarrow.Table
    """
    table = pq.read_table(Path(path), columns=columns)
    schema = case_schema(view)
    return table.cast(pa.schema([schema.field(name) for name in table.schema.names]))


def validate_cases(table: pa.Table, definition: OpenLoopDefinition) -> None:
    """Check a case table against the definition that produced it.

    Everything here is a property the evaluator would otherwise have to assume: the shapes it
    slices, the roles it keys on, the grid it reports timestamps from. Run on the temporary file
    read back from disk, so what is checked is what Parquet actually stored.

    Raises
    ------
    ValueError
        On the first violation, naming the case.
    """
    expected = case_schema(definition.view)
    if table.schema.remove_metadata() != expected:
        raise ValueError(
            "the case table does not match the schema for view "
            f"{definition.view!r}:\n{table.schema.remove_metadata()}\n!=\n{expected}"
        )
    if table.num_rows != len(definition.scenarios):
        raise ValueError(
            f"{table.num_rows} cases for {len(definition.scenarios)} windows in the definition"
        )

    case_ids = table.column("case_id").to_pylist()
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case ids are not unique")
    if set(case_ids) != set(definition.scenarios):
        missing = sorted(set(definition.scenarios) - set(case_ids))
        raise ValueError(f"the artifact does not cover every window, e.g. {missing[:3]}")

    horizon, context = definition.horizon_steps, definition.max_context_steps
    _check_scalar(table, "view", definition.view)
    _check_scalar(table, "timestep_seconds", TIMESTEP_SECONDS)
    _check_scalar(table, "max_context_steps", context)
    _check_scalar(table, "horizon_steps", horizon)

    _check_series(table, "history", context, case_ids)
    _check_series(table, "forecast", horizon, case_ids)

    starts = np.asarray(table.column("start_timestamp").to_numpy(zero_copy_only=False))
    step = np.timedelta64(TIMESTEP_SECONDS, "s")
    history_stamps = _stamps(table, "history", context)
    forecast_stamps = _stamps(table, "forecast", horizon)
    if not (history_stamps[:, -1] == starts).all():
        bad = case_ids[int(np.argmax(history_stamps[:, -1] != starts))]
        raise ValueError(f"{bad}: history does not end at start_timestamp")
    if not (forecast_stamps[:, 0] == starts + step).all():
        bad = case_ids[int(np.argmax(forecast_stamps[:, 0] != starts + step))]
        raise ValueError(f"{bad}: the forecast does not start one step after the anchor")

    plans = column_array(table, "plans")
    lengths = np.asarray(pc.list_value_length(plans))
    if not (lengths == len(PLAN_ROLES)).all():
        bad = case_ids[int(np.argmax(lengths != len(PLAN_ROLES)))]
        raise ValueError(f"{bad}: {lengths.min()} plans, expected {len(PLAN_ROLES)}")
    flat = plans.flatten()
    roles = np.asarray(flat.field("plan_role").to_pylist()).reshape(-1, len(PLAN_ROLES))
    if not (roles == np.asarray(PLAN_ROLES)).all():
        bad = case_ids[int(np.argmax((roles != np.asarray(PLAN_ROLES)).any(axis=1)))]
        raise ValueError(f"{bad}: plan roles are {list(roles[0])}, expected {list(PLAN_ROLES)}")
    nominal = (roles == NOMINAL_ROLE).sum(axis=1)
    if not (nominal == 1).all():
        bad = case_ids[int(np.argmax(nominal != 1))]
        raise ValueError(f"{bad}: {nominal.min()} plans tagged {NOMINAL_ROLE!r}, expected one")
    plan_ids = np.asarray(flat.field("plan_id").to_pylist()).reshape(-1, len(PLAN_ROLES))
    for row, ids in enumerate(plan_ids):
        if len(set(ids)) != len(ids):
            raise ValueError(f"{case_ids[row]}: plan ids repeat")

    for field in ("requested_control", "applied_control", "actual_T_room"):
        values = flat.field(field)
        lengths = np.asarray(pc.list_value_length(values))
        if not (lengths == horizon).all():
            bad = case_ids[int(np.argmax(lengths != horizon)) // len(PLAN_ROLES)]
            raise ValueError(f"{bad}: {field} has {lengths.min()} steps, expected {horizon}")
        _check_finite(np.asarray(values.flatten()), field, "the plans")


def sha256_file(path: str | Path) -> str:
    """The hex digest of a file, read in chunks so a corpus shard does not have to fit in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(directory: str | Path, manifest: dict[str, Any]) -> Path:
    """Write `manifest.json`. Called only once the cases beside it have validated."""
    path = Path(directory) / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(directory: str | Path) -> dict[str, Any]:
    """Read and check an evaluation set's manifest.

    Verifies the schema version and that `cases.parquet` still hashes to what the manifest
    recorded -- a half-written or hand-edited artifact is the one failure mode that would
    otherwise produce numbers rather than an error.

    Parameters
    ----------
    directory : str or Path
        The evaluation-set directory.

    Returns
    -------
    dict
        The manifest.

    Raises
    ------
    FileNotFoundError
        If the set has not been compiled.
    ValueError
        On a version or fingerprint mismatch.
    """
    directory = Path(directory)
    path = directory / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{directory.name} has no {MANIFEST_FILE}; compile it with "
            f"data/scripts/build_open_loop_cases.py"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{directory.name} was built under case schema {manifest.get('schema_version')}, "
            f"this is version {SCHEMA_VERSION}; rebuild it"
        )
    cases = directory / CASES_FILE
    if not cases.exists():
        raise FileNotFoundError(f"{directory.name} has a manifest but no {CASES_FILE}")
    digest = sha256_file(cases)
    if digest != manifest.get("cases_sha256"):
        raise ValueError(
            f"{directory.name}/{CASES_FILE} does not match its manifest "
            f"({digest[:12]} != {str(manifest.get('cases_sha256'))[:12]}); rebuild it"
        )
    return manifest


def column_array(table: pa.Table, name: str) -> pa.Array:
    """One column as a single contiguous Array, whatever chunking Parquet handed back."""
    column = table.column(name).combine_chunks()
    return column.chunk(0) if isinstance(column, pa.ChunkedArray) else column


def _check_scalar(table: pa.Table, column: str, expected) -> None:
    values = set(table.column(column).to_pylist())
    if values != {expected}:
        raise ValueError(f"{column} is {sorted(values)}, expected {expected!r} throughout")


def _check_series(table: pa.Table, column: str, length: int, case_ids: list[str]) -> None:
    """One nested series: right length everywhere, on the grid, and finite."""
    series = column_array(table, column)
    lengths = np.asarray(pc.list_value_length(series))
    if not (lengths == length).all():
        bad = case_ids[int(np.argmax(lengths != length))]
        raise ValueError(f"{bad}: {column} has {lengths.min()} rows, expected {length}")
    stamps = _stamps(table, column, length)
    step = np.timedelta64(TIMESTEP_SECONDS, "s")
    gaps = np.diff(stamps, axis=1)
    if length > 1 and not (gaps == step).all():
        bad = case_ids[int(np.argmax((gaps != step).any(axis=1)))]
        raise ValueError(f"{bad}: {column} timestamps are not contiguous on the {step} grid")
    flat = series.flatten()
    for field in flat.type:
        if field.name == "timestamp":
            continue
        _check_finite(np.asarray(flat.field(field.name)), field.name, column)


def _check_finite(values: np.ndarray, field: str, where: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{where} holds non-finite {field} values")


def _stamps(table: pa.Table, column: str, length: int) -> np.ndarray:
    """The timestamps of a nested series, as `(cases, length)` datetime64."""
    flat = column_array(table, column).flatten()
    stamps = flat.field("timestamp").to_numpy(zero_copy_only=False)
    return np.asarray(stamps).reshape(table.num_rows, length)

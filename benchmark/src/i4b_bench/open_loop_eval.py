"""Open-loop evaluation: how well does a model predict, and does it respond to the control?

A predictor gets the same observation a controller gets in the closed loop, plus candidate
control trajectories to predict under. The unit is a **case**: one building, one anchor, and the
recorded run whose history fills the context.

Cases are compiled ahead of time into `cases.parquet` (see `i4b_bench.cases`), so scoring is
five steps and no simulation: load the prepared cases, slice the history to each context length,
batch them through the predictor, check what comes back, and reduce it to metrics. Nothing here
touches the corpus, the plant, or a worker pool -- the artifact already holds every number the
plant produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .cases import column_array, load_cases, load_manifest, validate_cases
from .control_gain import NOMINAL_ROLE, gain_terms
from .dataset import EXCITATION, EXCITATION_DTYPE
from .evaluation_set import CASES_FILE, OpenLoopDefinition, load_definition, resolve_evaluation_set
from .observation import (
    DISTURBANCE_CHANNELS,
    STATE_CHANNELS,
    build_observation,
    history_channels,
)

#: The channel the metrics are computed on. A predictor may return more; omitting it is an error.
SCORED_CHANNEL = "T_room"

#: Minimum RMS room response, in K, for a case's gain to be reported. Below it the probes moved
#: the plant too little to divide by -- the pump was clipped or idle throughout.
MIN_RESPONSE_K = 1e-3


class Predictor(Protocol):
    """A batch of observations with their candidate controls, in; predictions out.

    `controls[i]` is `(plans, horizon)` and holds the **requested** supply temperatures;
    `returns[i]` maps a channel name to a `(plans, horizon)` array.
    """

    def __call__(
        self, observations: list[dict], controls: list[np.ndarray]
    ) -> list[dict[str, np.ndarray]]: ...


def eval_benchmark_open_loop(
    predictor: Predictor,
    *,
    evaluation_set: str | Path = "benchmark-v1",
    batch_size: int = 64,
) -> pd.DataFrame:
    """Score a predictor on a compiled open-loop evaluation set.

    Every case is scored at every context length the set's definition names. The official set is
    240 cases across four contexts, so 960 rows.

    Parameters
    ----------
    predictor : Predictor
        Called once per batch as ``predictor(observations, controls)``; see `Predictor`.
    evaluation_set : str or Path
        A bundled set name, e.g. ``"benchmark-v1"`` or ``"fast-eval"``, or a path to a set
        directory. Its `manifest.json` is checked against `cases.parquet` before anything runs.
    batch_size : int
        Cases per predictor call. Trades peak memory against call overhead; the results do not
        depend on it.

    Returns
    -------
    pandas.DataFrame
        One row per case and context length, with `mae_K` and `bias_K` (point accuracy on the
        plan tagged `nominal`), `response_K` (RMS movement the probes produced in the plant),
        `gain`, its two sufficient statistics, and the case's provenance. `gain` is the slope of
        the model's predicted deviation on the plant's over the probes: 1.0 moves as the plant
        does, 0.0 ignores the control, `NaN` when `response_K` fell below `MIN_RESPONSE_K`.

        Pool a set of rows by summing the statistics rather than averaging the ratios:
        ``sum(gain_cross) / sum(gain_square)``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    prepared = _PreparedSet(evaluation_set)
    rows = []
    for days, context in zip(prepared.definition.context_days, prepared.definition.context_steps):
        for start in range(0, prepared.count, batch_size):
            batch = range(start, min(start + batch_size, prepared.count))
            rows += _score(batch, predictor, context=context, days=days, prepared=prepared)
    frame = pd.DataFrame(rows)
    frame["excitation"] = frame["excitation"].astype(EXCITATION_DTYPE)
    return frame


def inspect_case(
    case_id: str,
    predictor: Predictor,
    *,
    evaluation_set: str | Path = "benchmark-v1",
    context_days: float | None = None,
) -> dict:
    """Everything one case's scoring saw, for looking at rather than aggregating.

    Reads the prepared case; it does not re-run the plant, so what comes back is exactly what
    was scored.

    Parameters
    ----------
    case_id : str
        A case in the set, e.g. ``"window001"``.
    predictor : Predictor
        Called once, on this case alone.
    evaluation_set : str or Path
        As for `eval_benchmark_open_loop`.
    context_days : float, optional
        Context length. Defaults to the set's longest.

    Returns
    -------
    dict
        ``history`` and ``forecast`` frames, the ``requested`` and ``applied`` control arrays
        `(plans, horizon)`, the plant's ``actual`` room temperatures and the model's
        ``predicted`` ones of the same shape, the ``roles`` naming each plan, the horizon
        ``timestamps``, and the scored ``row``.
    """
    prepared = _PreparedSet(evaluation_set)
    days = context_days if context_days is not None else max(prepared.definition.context_days)
    context = prepared.steps_for(days)
    case = prepared.index_of(case_id)

    observation = prepared.observation(case, context)
    requested = prepared.requested[case].astype(float)
    prediction = predictor([observation], [requested])[0]
    row = _score(
        [case], lambda _o, _c: [prediction], context=context, days=days, prepared=prepared
    )[0]
    return {
        "history": pd.DataFrame(observation["history"]).set_index("timestamp"),
        "forecast": pd.DataFrame(observation["forecast"]).set_index("timestamp"),
        "requested": requested,
        "applied": prepared.applied[case].astype(float),
        "actual": prepared.actual[case].astype(float),
        "predicted": np.asarray(prediction[SCORED_CHANNEL], dtype=float),
        "roles": list(prepared.roles[case]),
        "timestamps": pd.DatetimeIndex(observation["forecast"]["timestamp"]),
        "row": row,
    }


class _PreparedSet:
    """A compiled evaluation set, loaded and checked once per evaluation.

    Held column-wise: every case has the same history length, horizon and plan count -- the
    artifact is validated on that -- so each nested column becomes one rectangular array and a
    case is a row of it. Slicing a context is then a view rather than a rebuild, which is what
    keeps four context lengths as cheap as one.
    """

    #: Per-case scalars that travel into every result row.
    PROVENANCE_COLUMNS = (
        "case_id",
        "scenario_id",
        "building_id",
        "controller_id",
        "country",
        "period_id",
        "variant",
        "transmission_W_m2K",
        "year_start",
        "year_end",
        "floor_area_m2",
    )

    def __init__(self, evaluation_set: str | Path):
        self.directory = resolve_evaluation_set(evaluation_set)
        self.name = self.directory.name
        self.definition: OpenLoopDefinition = load_definition(self.directory)
        self.manifest = load_manifest(self.directory)
        view = self.definition.view
        self.history_channels = history_channels(view)
        self.disturbance_channels = DISTURBANCE_CHANNELS[view]

        table = load_cases(self.directory / CASES_FILE, view)
        validate_cases(table, self.definition)
        self.count = table.num_rows
        horizon = self.definition.horizon_steps
        context = self.definition.max_context_steps

        self.scalars = {name: table.column(name).to_pylist() for name in self.PROVENANCE_COLUMNS}
        self.case_ids = self.scalars["case_id"]
        self.starts = pd.to_datetime(table.column("start_timestamp").to_pylist())

        state = column_array(table, "state")
        self.state = {
            channel: np.asarray(state.field(channel)) for channel in STATE_CHANNELS[view]
        }
        self.history_stamps, self.history = _series(
            table, "history", context, self.history_channels
        )
        self.forecast_stamps, self.forecast = _series(
            table, "forecast", horizon, self.disturbance_channels
        )

        plans = column_array(table, "plans").flatten()
        probes = self.definition.probes
        self.roles = np.asarray(plans.field("plan_role").to_pylist()).reshape(self.count, probes)
        shape = (self.count, probes, horizon)
        self.requested = np.asarray(plans.field("requested_control").flatten()).reshape(shape)
        self.applied = np.asarray(plans.field("applied_control").flatten()).reshape(shape)
        self.actual = np.asarray(plans.field("actual_T_room").flatten()).reshape(shape)

        # `definition_sha256` is the manifest's -- the definition that *compiled* these cases,
        # which is what a number was produced under. The file on disk may have moved on; what
        # matters is that it still describes this artifact, and `validate_cases` above is what
        # says so (the view, the horizon, the longest context, the case ids and their count).
        self.provenance = {
            "evaluation_set": self.name,
            "split": self.definition.split,
            "view": view,
            "horizon_steps": horizon,
            "probe_count": probes,
            "probe_amplitude": self.definition.probe_amplitude,
            "corpus_manifest_sha256": self.manifest["corpus_manifest_sha256"],
            "definition_sha256": self.manifest["definition_sha256"],
        }

    def index_of(self, case_id: str) -> int:
        """Where a case sits, by id."""
        if case_id not in self.case_ids:
            raise KeyError(f"{case_id!r} is not in {self.name}")
        return self.case_ids.index(case_id)

    def observation(self, case: int, context: int) -> dict:
        """The observation for one case at one context length.

        A shorter context is exactly the tail of the stored one -- same anchor, same plans, same
        weather -- which is why the artifact stores only the longest.

        Every array is copied. The columns behind them are read-only Arrow buffers shared by all
        960 rows of a run, so handing out views would give a predictor something it cannot write
        into and could corrupt for every later case if it could. `ScenarioEnv` hands over fresh
        writable arrays, and the two loops promise the same observation, so this one does too.
        """
        history = {"timestamp": self.history_stamps[case, -context:].copy()}
        history.update(
            {name: values[case, -context:].copy() for name, values in self.history.items()}
        )
        forecast = {"timestamp": self.forecast_stamps[case].copy()}
        forecast.update({name: values[case].copy() for name, values in self.forecast.items()})
        state = {name: float(values[case]) for name, values in self.state.items()}
        return build_observation(state, history, forecast)

    def steps_for(self, days: float) -> int:
        """The context length in steps that `days` names in this definition."""
        if days not in self.definition.context_days:
            raise KeyError(
                f"{self.name} is compiled for contexts {list(self.definition.context_days)} d, "
                f"not {days}"
            )
        return self.definition.context_steps[self.definition.context_days.index(days)]


def _series(
    table, column: str, length: int, channels: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One nested series as `(timestamps, {channel: values})`, both `(cases, length)`."""
    flat = column_array(table, column).flatten()
    stamps = np.asarray(
        flat.field("timestamp").to_numpy(zero_copy_only=False), dtype="datetime64[s]"
    ).reshape(table.num_rows, length)
    values = {
        channel: np.asarray(flat.field(channel)).reshape(table.num_rows, length)
        for channel in channels
    }
    return stamps, values


def _score(
    cases: range | list[int],
    predictor: Predictor,
    *,
    context: int,
    days: float,
    prepared: _PreparedSet,
) -> list[dict]:
    """Ask the predictor about a batch of cases, and turn its answer into result rows."""
    observations = [prepared.observation(case, context) for case in cases]
    # The predictor is shown the control that was *requested*. What the actuator did with it is
    # part of the plant it is being asked to model, not something to hand it the answer to.
    controls = [prepared.requested[case] for case in cases]
    predictions = predictor(observations, controls)
    if len(predictions) != len(observations):
        raise ValueError(f"predictor returned {len(predictions)} blocks for {len(observations)}")

    rows = []
    for case, prediction in zip(cases, predictions):
        case_id = prepared.case_ids[case]
        if SCORED_CHANNEL not in prediction:
            raise ValueError(
                f"predictor returned {sorted(prediction)}, without {SCORED_CHANNEL!r}"
            )
        actual = prepared.actual[case].astype(float)
        predicted = np.asarray(prediction[SCORED_CHANNEL], dtype=float)
        if predicted.shape != actual.shape:
            raise ValueError(
                f"{case_id}: expected {actual.shape} predictions, got {predicted.shape}"
            )
        if not np.isfinite(predicted).all():
            raise ValueError(f"{case_id}: the prediction holds non-finite values")

        roles = list(prepared.roles[case])
        if roles.count(NOMINAL_ROLE) != 1:
            raise ValueError(f"{case_id}: {roles.count(NOMINAL_ROLE)} nominal plans, expected one")
        nominal = roles.index(NOMINAL_ROLE)
        error = predicted[nominal] - actual[nominal]
        cross, square = gain_terms(actual, predicted)
        response = float(np.sqrt(square / actual.size))
        spread = float(prepared.requested[case].std(axis=0).mean())
        rows.append(
            {
                **prepared.provenance,
                **{name: values[case] for name, values in prepared.scalars.items()},
                "start": str(prepared.starts[case]),
                "context_days": days,
                "excitation": EXCITATION.get(prepared.scalars["controller_id"][case]),
                # How much of the probe spread the actuator let through; 0 means the pump never
                # moved, so the case carries no control-response information.
                "realized_share": (
                    float(prepared.applied[case].std(axis=0).mean()) / spread if spread else 0.0
                ),
                "mae_K": float(np.abs(error).mean()),
                "bias_K": float(error.mean()),
                "response_K": response,
                "gain_cross": cross,
                "gain_square": square,
                "gain": cross / square if response >= MIN_RESPONSE_K else float("nan"),
            }
        )
    return rows

"""Compile an open-loop evaluation set: a definition plus the corpus, into cases.parquet.

    uv run python benchmark/data/scripts/build_open_loop_cases.py \
        benchmark/data/evaluation_sets/open_loop/benchmark-v1/definition.yaml \
        benchmark/data/evaluation_sets/open_loop/benchmark-v1

This is the only place the plant is driven for open-loop evaluation. Every window is resolved
against the corpus, seeded from its recorded controller, probed under the five plans, and
written out with everything the plant answered -- so scoring a predictor later reads one Parquet
file and nothing else.

Run once per evaluation-set version. Correctness beats speed here: the whole official set takes
about half an hour, almost all of it preparing each building's archived forecast runs, and it
produces an artifact that is then immutable for the release.

What is checked before anything is written
------------------------------------------
- every window's building exists in the corpus, exactly once;
- its named controller was actually recorded for that building;
- the whole interval -- context start through horizon end -- lies inside the declared split, on
  that trajectory rather than merely on a sibling of it;
- the trajectory joins its weather cleanly and lands on the 15-minute grid throughout
  (`load_controller_data`);
- and the finished file, read back from disk, satisfies `validate_cases`.

The manifest is written last, so a set carrying one has a case file that validated.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from i4b_bench.cases import (
    CONTROL_INPUT_SEMANTICS,
    SCHEMA_VERSION,
    TRANSITION_SEMANTICS,
    load_cases,
    sha256_file,
    validate_cases,
    write_cases,
    write_manifest,
)
from i4b_bench.control_gain import probe_plans
from i4b_bench.dataset import (
    STEP,
    load_controller_data,
    load_dataset,
    scenario_metadata,
    step_of,
    utc,
)
from i4b_bench.evaluation_set import (
    CASES_FILE,
    TIMESTEP_SECONDS,
    OpenLoopDefinition,
    Window,
    load_definition,
)
from i4b_bench.observation import DISTURBANCE_CHANNELS, history_channels
from i4b_bench.scenario_env import ScenarioEnv

#: Bumped when this script changes what it writes for an unchanged definition and corpus.
GENERATOR_VERSION = "1.0.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("definition", type=Path, help="the set's definition.yaml")
    parser.add_argument("output", type=Path, help="the evaluation-set directory to write into")
    parser.add_argument(
        "--dataset-dir", type=Path, default=None, help="corpus directory (default: the bundled one)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="compile only the first N windows, for a smoke run"
    )
    args = parser.parse_args(argv)

    definition = load_definition(args.definition)
    dataset = load_dataset(args.dataset_dir)
    windows = resolve_windows(dataset, definition)
    if args.limit is not None:
        windows = windows[: args.limit]
        definition = _restricted(definition, [case_id for case_id, _, _ in windows])

    print(f"compiling {len(windows)} case(s) from {args.definition}", flush=True)
    rows = compile_cases(dataset, definition, windows)

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / CASES_FILE
    # Write, read back, validate, and only then replace: a set is either the previous artifact
    # or a fully checked new one, never a half-written file with a manifest that vouches for it.
    temporary = destination.with_suffix(".parquet.tmp")
    write_cases(temporary, rows, definition.view)
    validate_cases(load_cases(temporary, definition.view), definition)
    os.replace(temporary, destination)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_set_name": args.output.resolve().name,
        "definition_sha256": sha256_file(args.definition),
        "corpus_manifest_sha256": sha256_file(dataset.root / "manifest.json"),
        "cases_sha256": sha256_file(destination),
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_count": len(rows),
        "view": definition.view,
        "timestep_seconds": TIMESTEP_SECONDS,
        "max_context_steps": definition.max_context_steps,
        "horizon_steps": definition.horizon_steps,
        "probe_count": definition.probes,
        "control_input_semantics": CONTROL_INPUT_SEMANTICS,
        "transition_semantics": TRANSITION_SEMANTICS,
    }
    write_manifest(args.output, manifest)
    size_mib = destination.stat().st_size / 1024**2
    print(f"wrote {len(rows)} cases to {destination} ({size_mib:.1f} MiB)")
    return 0


def resolve_windows(
    dataset, definition: OpenLoopDefinition
) -> list[tuple[str, Window, int]]:
    """Resolve every window against the corpus, or raise saying which one does not.

    Returns `(case_id, window, anchor_step)` sorted by building, so a building's archived
    forecast runs are prepared once and every window of it is compiled while they are warm.
    """
    context, horizon = definition.max_context_steps, definition.horizon_steps
    resolved = []
    for case_id, window in definition.scenarios.items():
        scenario = dataset.scenarios[dataset.scenarios["scenario_id"] == window.building]
        if len(scenario) != 1:
            raise ValueError(
                f"{case_id}: {window.building!r} matches {len(scenario)} scenarios, expected one"
            )
        trajectory_id = f"{window.building}--{window.controller}"
        recorded = dataset.trajectories[dataset.trajectories["trajectory_id"] == trajectory_id]
        if len(recorded) != 1:
            raise ValueError(
                f"{case_id}: {trajectory_id!r} matches {len(recorded)} recorded trajectories"
            )

        # The split is checked on this trajectory, not on the scenario: another controller of
        # the same building being in the split says nothing about this one.
        rows = dataset.split[dataset.split["trajectory_id"] == trajectory_id]
        rows = rows[rows["split"] == definition.split]
        if len(rows) != 1:
            raise ValueError(
                f"{case_id}: {trajectory_id!r} has {len(rows)} rows in the "
                f"{definition.split!r} split, expected one"
            )
        split_start = utc(rows.iloc[0]["start_time_utc"])
        split_end = utc(rows.iloc[0]["end_time_utc"])

        anchor = step_of(dataset, window.building, window.start)
        anchor_time = utc(window.start)
        context_start = anchor_time - context * STEP
        horizon_end = anchor_time + horizon * STEP
        if context_start < split_start:
            raise ValueError(
                f"{case_id}: a {context}-step context reaches back to {context_start}, before "
                f"the {definition.split} split opens at {split_start}"
            )
        if horizon_end > split_end:
            raise ValueError(
                f"{case_id}: a {horizon}-step horizon runs to {horizon_end}, past the end of "
                f"the {definition.split} split at {split_end}"
            )
        resolved.append((case_id, window, anchor))
    # Grouped by trajectory: a building's archived forecast runs are then prepared once, and
    # each recorded trajectory read once, however the definition happens to order its windows.
    return sorted(resolved, key=lambda item: (item[1].building, item[1].controller, item[0]))


def compile_cases(
    dataset, definition: OpenLoopDefinition, windows: list[tuple[str, Window, int]]
) -> list[dict]:
    """Drive the plant for every window and return the case rows, in definition order.

    Cases are collected in memory: the official set is ~20 MiB, and holding it costs less than
    the machinery for streaming it would.
    """
    channels = history_channels(definition.view)
    disturbances = DISTURBANCE_CHANNELS[definition.view]
    horizon, context = definition.horizon_steps, definition.max_context_steps
    # Both caches are local to this call. A memo that outlived the run could be served against a
    # corpus it was not built from.
    forecast_cache: dict = {}
    trajectories: dict[tuple[str, str], pd.DataFrame] = {}

    rows = {}
    for done, (case_id, window, anchor) in enumerate(windows, start=1):
        key = (window.building, window.controller)
        if key not in trajectories:
            # One at a time: the windows are grouped by trajectory, so the previous one is done
            # with, and a year of transitions per building adds up.
            trajectories = {key: load_controller_data(dataset, window.controller, window.building)}
        trajectory = trajectories[key]
        if anchor + horizon >= len(trajectory):
            raise ValueError(
                f"{case_id}: step {anchor} plus a {horizon}-step horizon runs past the "
                f"{len(trajectory)} recorded steps of {window.building}--{window.controller}"
            )

        env = ScenarioEnv(
            window.building,
            dataset=dataset,
            initial_controller_id=window.controller,
            max_context_length=context,
            planning_steps=horizon,
            start_step=anchor,
            use_forecast=definition.use_forecast,
            view=definition.view,
            forecast_correction=definition.forecast_correction,
            forecast_cache=forecast_cache,
        )
        observation, _ = env.reset()
        # The probes read only `T_room` and `u` out of `info`; assembling the full observation
        # on each of their `probes * horizon` steps copies the whole context every time, and
        # costs about ten times the rollout itself.
        env.build_observations = False

        baseline = trajectory["T_hp_sup_applied"].to_numpy(float)[anchor : anchor + horizon]
        plans, roles = probe_plans(
            baseline,
            definition.probe_amplitude,
            seed=_seed(case_id, window),
            hold_steps=definition.probe_hold_steps,
        )
        applied = np.empty_like(plans)
        actual = np.empty_like(plans)
        for k, plan in enumerate(plans):
            env.reset()
            for t, action in enumerate(plan):
                _, _, _, _, info = env.step(float(action))
                actual[k, t] = info["T_room"]
                # What the actuator let through: `check_hp` collapses the supply temperature
                # whenever the pump idles. Kept for provenance, never handed to a predictor.
                applied[k, t] = info["u"]

        rows[case_id] = {
            "case_id": case_id,
            "scenario_id": window.building,
            "controller_id": window.controller,
            "start_timestamp": _stamps(observation["history"]["timestamp"][-1:])[0],
            "view": definition.view,
            "timestep_seconds": TIMESTEP_SECONDS,
            "max_context_steps": context,
            "horizon_steps": horizon,
            **scenario_metadata(dataset, window.building),
            "state": {ch: float(value) for ch, value in observation["state"].items()},
            "history": _series(observation["history"], channels),
            "forecast": _series(observation["forecast"], disturbances),
            "plans": [
                {
                    "plan_id": f"{case_id}:{role}",
                    "plan_role": role,
                    "requested_control": plans[k].tolist(),
                    "applied_control": applied[k].tolist(),
                    "actual_T_room": actual[k].tolist(),
                }
                for k, role in enumerate(roles)
            ],
        }
        if done % 10 == 0 or done == len(windows):
            print(f"  {done}/{len(windows)} cases", flush=True)

    # Back into the order the definition names, so the artifact reads like the file it came from.
    return [rows[case_id] for case_id in definition.scenarios if case_id in rows]


def _series(block: dict, channels: tuple[str, ...]) -> list[dict]:
    """One observation block as a list of structs, one per timestep."""
    stamps = _stamps(block["timestamp"])
    columns = {channel: np.asarray(block[channel], dtype=float) for channel in channels}
    return [
        {"timestamp": stamps[i], **{name: float(values[i]) for name, values in columns.items()}}
        for i in range(len(stamps))
    ]


def _stamps(values) -> list[datetime]:
    """`datetime64` to timezone-aware datetimes, which is what the Arrow schema takes."""
    return list(pd.DatetimeIndex(values).tz_localize("UTC").to_pydatetime())


def _seed(case_id: str, window: Window) -> int:
    """A stable seed for a window's APRBS probe.

    Derived from what the window *is*, so the waveform does not move when a neighbouring case is
    added, removed or renumbered -- and so a rebuild reproduces the artifact exactly.
    """
    identity = f"{case_id}|{window.building}|{window.controller}|{window.start}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _restricted(definition: OpenLoopDefinition, case_ids: list[str]) -> OpenLoopDefinition:
    """The definition narrowed to `case_ids`, so `--limit` still validates what it wrote."""
    from dataclasses import replace

    keep = set(case_ids)
    return replace(
        definition,
        scenarios={name: w for name, w in definition.scenarios.items() if name in keep},
    )


if __name__ == "__main__":
    sys.exit(main())

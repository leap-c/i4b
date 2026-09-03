"""What the compiled evaluation sets promise, and what the evaluator does with them.

Open-loop scoring reads one Parquet file. That makes most of these tests cheap and corpus-free
-- they build a case by hand, or read the small `fast-eval` artifact -- and it makes the two
negative tests possible: evaluation must not reach for the corpus or the simulator, and this
asserts it by breaking both.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from i4b_bench.cases import (
    SCHEMA_VERSION,
    case_schema,
    load_cases,
    load_manifest,
    validate_cases,
    write_cases,
)
from i4b_bench.control_gain import NOMINAL_ROLE, PLAN_ROLES
from i4b_bench.evaluation_set import CASES_FILE, MANIFEST_FILE, load_definition
from i4b_bench.open_loop_eval import eval_benchmark_open_loop, inspect_case

SETS = Path(__file__).resolve().parents[1] / "data" / "evaluation_sets" / "open_loop"
FAST, CANONICAL = SETS / "fast-eval", SETS / "benchmark-v1"
LADDER = SETS / "excitation-ladder-v1"

needs_fast = pytest.mark.skipif(
    not (FAST / CASES_FILE).exists(), reason="fast-eval has not been compiled"
)
needs_canonical = pytest.mark.skipif(
    not (CANONICAL / CASES_FILE).exists(), reason="benchmark-v1 has not been compiled"
)
needs_ladder = pytest.mark.skipif(
    not (LADDER / CASES_FILE).exists(), reason="excitation-ladder-v1 has not been compiled"
)

UTC = dt.timezone.utc


def _oracle(prepared_actual):
    """A predictor that replays the plant: it should score a perfect MAE and a gain of one."""

    def predictor(observations, controls):
        return [{"T_room": prepared_actual[i]} for i in range(len(observations))]

    return predictor


@pytest.fixture(scope="module")
def fast_cases():
    definition = load_definition(FAST)
    return load_cases(FAST / CASES_FILE, definition.view), definition


# --------------------------------------------------------------------------------------------
# the artifact


def test_a_case_survives_a_parquet_round_trip(tmp_path):
    """One hand-built case, written and read back: ids, timestamps, shapes, roles, values.

    The schema is declared in Arrow rather than inferred from Python objects, so this is what
    says the declaration and the file agree -- including the second-resolution timestamps, which
    Parquet cannot store natively and `load_cases` casts back.
    """
    horizon, context = 3, 4
    anchor = dt.datetime(2025, 12, 1, tzinfo=UTC)
    step = dt.timedelta(minutes=15)
    history = [
        {
            "timestamp": anchor - (context - 1 - i) * step,
            "T_room": 20.0 + i,
            "T_hp_ret": 25.0 + i,
            "T_hp_sup_applied": 35.0 + i,
            "T_amb": 3.0 + i,
            "ghi": 10.0 * i,
            "dni": 20.0 * i,
            "dhi": 30.0 * i,
        }
        for i in range(context)
    ]
    forecast = [
        {
            "timestamp": anchor + (i + 1) * step,
            "T_amb": 4.0 + i,
            "ghi": 1.0,
            "dni": 2.0,
            "dhi": 3.0,
        }
        for i in range(horizon)
    ]
    row = {
        "case_id": "window001",
        "scenario_id": "XX.N.SFH.01.Gen.ReEx.001.001--period_b",
        "building_id": "XX.N.SFH.01.Gen.ReEx.001.001",
        "controller_id": "mpc-nominal",
        "start_timestamp": anchor,
        "view": "realistic",
        "timestep_seconds": 900,
        "max_context_steps": context,
        "horizon_steps": horizon,
        "country": "XX",
        "period_id": "period_b",
        "variant": 1,
        "transmission_W_m2K": 2.5,
        "year_start": 1919,
        "year_end": 1948,
        "floor_area_m2": 120.0,
        "state": {"T_room": 23.0, "T_hp_ret": 28.0},
        "history": history,
        "forecast": forecast,
        "plans": [
            {
                "plan_id": f"window001:{role}",
                "plan_role": role,
                "requested_control": [30.0 + k, 31.0 + k, 32.0 + k],
                "applied_control": [30.5 + k, 31.5 + k, 32.5 + k],
                "actual_T_room": [21.0 + k, 21.5 + k, 22.0 + k],
            }
            for k, role in enumerate(PLAN_ROLES)
        ],
    }
    path = tmp_path / CASES_FILE
    write_cases(path, [row], "realistic")
    table = load_cases(path, "realistic")

    assert table.schema == case_schema("realistic")
    back = table.to_pylist()[0]
    assert back["case_id"] == "window001"
    assert back["start_timestamp"] == anchor
    assert [r["timestamp"] for r in back["history"]] == [r["timestamp"] for r in history]
    assert [r["timestamp"] for r in back["forecast"]] == [r["timestamp"] for r in forecast]
    assert [p["plan_role"] for p in back["plans"]] == list(PLAN_ROLES)
    assert [p["plan_id"] for p in back["plans"]] == [p["plan_id"] for p in row["plans"]]
    assert back["plans"][2]["requested_control"] == pytest.approx([32.0, 33.0, 34.0])
    assert back["plans"][2]["applied_control"] == pytest.approx([32.5, 33.5, 34.5])
    assert back["plans"][2]["actual_T_room"] == pytest.approx([23.0, 23.5, 24.0])
    assert back["history"][-1]["T_room"] == pytest.approx(23.0)


@needs_canonical
def test_the_canonical_artifact_is_what_the_benchmark_claims():
    """240 cases, five plans each, one of them nominal, 96 ahead from 2016 of history."""
    definition = load_definition(CANONICAL)
    table = load_cases(CANONICAL / CASES_FILE, definition.view)
    validate_cases(table, definition)

    assert table.num_rows == 240
    ids = table.column("case_id").to_pylist()
    assert len(set(ids)) == 240
    assert set(table.column("horizon_steps").to_pylist()) == {96}
    assert set(table.column("max_context_steps").to_pylist()) == {2016}
    for row in table.to_pylist():
        assert len(row["history"]) == 2016
        assert len(row["forecast"]) == 96
        assert len(row["plans"]) == 5
        roles = [plan["plan_role"] for plan in row["plans"]]
        assert roles.count(NOMINAL_ROLE) == 1
        for plan in row["plans"]:
            for field in ("requested_control", "applied_control", "actual_T_room"):
                assert len(plan[field]) == 96
                assert np.isfinite(plan[field]).all()


@needs_ladder
def test_the_excitation_ladder_is_paired():
    """Every rung sees the same building at the same anchor under the same weather.

    That pairing is the ablation: an unpaired sweep would confound excitation with which
    building and which week a case landed on, and the between-building spread in gain is larger
    than the effect being measured.

    What a rung cannot share is its own history -- a different recorded controller ran a
    different year, so it arrives at the anchor in a different state and with a different plan to
    probe around. The forecast can be shared, because it depends only on the scenario and the
    decision time, and this asserts that it is: any difference between rungs is then in the past,
    not in the future they are asked about.
    """
    definition = load_definition(LADDER)
    table = load_cases(LADDER / CASES_FILE, definition.view)
    validate_cases(table, definition)

    frame = table.select(
        ["case_id", "scenario_id", "controller_id", "start_timestamp"]
    ).to_pandas()
    rungs = frame["controller_id"].unique()
    assert len(rungs) == 9, sorted(rungs)
    grouped = frame.groupby(["scenario_id", "start_timestamp"])["controller_id"]
    assert (grouped.nunique() == 9).all(), "a window is missing a rung"
    assert (grouped.count() == 9).all(), "a window repeats a rung"
    assert frame["scenario_id"].nunique() == 30
    assert table.num_rows == 270

    forecasts, histories = {}, {}
    for row in table.to_pylist():
        key = (row["scenario_id"], row["start_timestamp"])
        channels = ("T_amb", "ghi", "dni", "dhi")
        forecast = np.array([[r[c] for c in channels] for r in row["forecast"]])
        history = np.array([r["T_hp_sup_applied"] for r in row["history"]])
        if key in forecasts:
            assert forecast == pytest.approx(forecasts[key]), f"{row['case_id']}: forecast moved"
            assert not np.allclose(history, histories[key]), (
                f"{row['case_id']}: two rungs handed the same history"
            )
        else:
            forecasts[key], histories[key] = forecast, history


@needs_canonical
def test_the_canonical_artifact_stays_small():
    """Comfortably inside 50 MiB: a case is 21 days of eight channels plus five short plans."""
    assert (CANONICAL / CASES_FILE).stat().st_size < 50 * 1024**2


@needs_fast
def test_a_tampered_case_file_is_refused(tmp_path):
    """The manifest fingerprints the artifact, so a file edited underneath it does not score."""
    copy = tmp_path / "fast-eval"
    shutil.copytree(FAST, copy)
    assert load_manifest(copy)["schema_version"] == SCHEMA_VERSION  # the untouched copy is fine

    table = pq.read_table(copy / CASES_FILE)
    pq.write_table(table.slice(0, 1), copy / CASES_FILE)
    with pytest.raises(ValueError, match="does not match its manifest"):
        load_manifest(copy)


@needs_fast
def test_a_missing_manifest_is_refused(tmp_path):
    copy = tmp_path / "fast-eval"
    shutil.copytree(FAST, copy)
    (copy / MANIFEST_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="build_open_loop_cases"):
        eval_benchmark_open_loop(lambda o, c: [], evaluation_set=copy)


# --------------------------------------------------------------------------------------------
# the evaluator


@needs_fast
def test_scoring_is_independent_of_batch_size():
    """Batching is a memory knob, never a result one."""

    def predictor(observations, controls):
        return [
            {"T_room": np.full(u.shape, o["history"]["T_room"][-1])}
            for o, u in zip(observations, controls)
        ]

    one = eval_benchmark_open_loop(predictor, evaluation_set=FAST, batch_size=1)
    many = eval_benchmark_open_loop(predictor, evaluation_set=FAST, batch_size=64)
    numeric = ["mae_K", "bias_K", "response_K", "gain_cross", "gain_square", "gain"]
    assert one["case_id"].tolist() == many["case_id"].tolist()
    assert one[numeric].to_numpy() == pytest.approx(many[numeric].to_numpy(), nan_ok=True)


@needs_fast
def test_the_predictor_is_handed_the_requested_control(fast_cases):
    """Not the applied one.

    The applied supply temperature is `check_hp`'s answer to the requested one, and it can
    depend on the true future return temperature -- handing it over leaks the plant's own
    response into the question. The two differ on this corpus, so this is a real distinction and
    not a naming one.
    """
    table, definition = fast_cases
    seen = []

    def predictor(observations, controls):
        seen.extend(controls)
        return [{"T_room": np.zeros(u.shape)} for u in controls]

    eval_benchmark_open_loop(predictor, evaluation_set=FAST)
    rows = table.to_pylist()
    requested = np.array([[p["requested_control"] for p in row["plans"]] for row in rows])
    applied = np.array([[p["applied_control"] for p in row["plans"]] for row in rows])
    assert not np.allclose(requested, applied), "the corpus should clip at least one probe"
    for handed, wanted in zip(seen[: len(rows)], requested):
        assert handed == pytest.approx(wanted)


@needs_fast
def test_accuracy_is_scored_on_the_plan_tagged_nominal(fast_cases):
    """MAE and bias key on `plan_role`, never on a position in the array.

    A predictor that is exact on the nominal plan and wrong everywhere else must score zero MAE.
    The nominal plan is not in the middle of `PLAN_ROLES`, so a positional implementation fails
    this.
    """
    table, _ = fast_cases
    rows = table.to_pylist()
    nominal = {
        row["case_id"]: next(
            p for p in row["plans"] if p["plan_role"] == NOMINAL_ROLE
        )["actual_T_room"]
        for row in rows
    }
    order = [row["case_id"] for row in rows]
    index = PLAN_ROLES.index(NOMINAL_ROLE)

    def predictor(observations, controls):
        out = []
        for i, u in enumerate(controls):
            block = np.full(u.shape, 1000.0)
            block[index] = nominal[order[i]]
            out.append({"T_room": block})
        return out

    frame = eval_benchmark_open_loop(predictor, evaluation_set=FAST)
    assert frame["mae_K"].max() == pytest.approx(0.0, abs=1e-6)
    assert frame["bias_K"].abs().max() == pytest.approx(0.0, abs=1e-6)


@needs_fast
def test_gain_uses_every_probe(fast_cases):
    """Replay the plant on all five plans and gain is one; flatten any single one and it is not.

    Every probe carries information about the response, so the metric has to be sensitive to
    each of them -- a gain that only looked at, say, the two offsets would be blind to whether a
    model gets the timing right.
    """
    table, _ = fast_cases
    rows = table.to_pylist()
    actual = {
        row["case_id"]: np.array([p["actual_T_room"] for p in row["plans"]], dtype=float)
        for row in rows
    }
    order = [row["case_id"] for row in rows]

    def replay(flatten=None):
        def predictor(observations, controls):
            blocks = []
            for i in range(len(controls)):
                block = actual[order[i]].copy()
                if flatten is not None:
                    # this probe now carries no response, only the mean across the horizon
                    block[flatten] = block[flatten].mean()
                blocks.append({"T_room": block})
            return blocks

        return predictor

    exact = eval_benchmark_open_loop(replay(), evaluation_set=FAST)
    assert exact["gain"].to_numpy() == pytest.approx(1.0)
    for probe in range(len(PLAN_ROLES)):
        partial = eval_benchmark_open_loop(replay(probe), evaluation_set=FAST)
        moved = np.abs(partial["gain"].to_numpy() - 1.0)
        assert (moved > 1e-3).all(), f"gain ignored probe {probe} ({PLAN_ROLES[probe]})"


@needs_fast
def test_a_short_context_is_the_tail_of_the_long_one(fast_cases):
    """Which is why the artifact stores only the longest, and the ladder is nearly free."""
    seen = {}

    def predictor(observations, controls):
        for observation in observations:
            seen.setdefault(len(observation["history"]["T_room"]), []).append(
                observation["history"]
            )
        return [{"T_room": np.zeros(u.shape)} for u in controls]

    eval_benchmark_open_loop(predictor, evaluation_set=FAST)
    definition = load_definition(FAST)
    lengths = sorted(seen)
    assert lengths == sorted(definition.context_steps)
    longest = seen[max(lengths)][0]
    for length in lengths:
        short = seen[length][0]
        assert (short["timestamp"] == longest["timestamp"][-length:]).all()
        for channel, values in short.items():
            if channel != "timestamp":
                assert values == pytest.approx(longest[channel][-length:])


@needs_fast
@pytest.mark.parametrize(
    "broken, message",
    [
        (lambda u: {"T_room": np.full(u.shape, np.nan)}, "non-finite"),
        (lambda u: {"T_room": np.zeros((u.shape[0], u.shape[1] - 1))}, "predictions"),
        (lambda u: {"T_hp_ret": np.zeros(u.shape)}, "T_room"),
    ],
    ids=["non-finite", "wrong shape", "missing channel"],
)
def test_a_malformed_prediction_is_refused(broken, message):
    with pytest.raises(ValueError, match=message):
        eval_benchmark_open_loop(
            lambda observations, controls: [broken(u) for u in controls], evaluation_set=FAST
        )


@needs_fast
def test_evaluation_touches_neither_the_corpus_nor_the_plant(monkeypatch):
    """The whole point of compiling cases: scoring reads one Parquet file.

    Both are broken rather than counted, so the test fails on the first attempt to reach for
    either -- including one made three call levels down inside a helper.
    """
    import i4b_bench.dataset as dataset_module
    import i4b_bench.scenario_env as env_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("open-loop evaluation must not load the corpus or drive the plant")

    monkeypatch.setattr(dataset_module, "load_dataset", forbidden)
    monkeypatch.setattr(env_module.ScenarioEnv, "__init__", forbidden)

    frame = eval_benchmark_open_loop(
        lambda observations, controls: [{"T_room": np.zeros(u.shape)} for u in controls],
        evaluation_set=FAST,
    )
    assert len(frame) == 3 * len(load_definition(FAST).context_days)


@needs_fast
def test_inspection_returns_what_was_scored():
    """`inspect_case` reads the prepared case rather than re-running anything."""

    def predictor(observations, controls):
        return [
            {"T_room": np.full(u.shape, o["history"]["T_room"][-1])}
            for o, u in zip(observations, controls)
        ]

    detail = inspect_case("window001", predictor, evaluation_set=FAST, context_days=1)
    assert detail["roles"] == list(PLAN_ROLES)
    assert detail["requested"].shape == detail["applied"].shape == detail["actual"].shape
    assert len(detail["history"]) == 96
    assert len(detail["forecast"]) == 96
    assert detail["row"]["case_id"] == "window001"

    scored = eval_benchmark_open_loop(predictor, evaluation_set=FAST)
    row = scored[(scored["case_id"] == "window001") & (scored["context_days"] == 1)].iloc[0]
    assert detail["row"]["mae_K"] == pytest.approx(row["mae_K"])
    assert detail["row"]["gain"] == pytest.approx(row["gain"], nan_ok=True)

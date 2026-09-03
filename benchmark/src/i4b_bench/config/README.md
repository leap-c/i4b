# Benchmark settings

The **closed-loop** settings, one YAML file per named setting:

```
config/closed_loop/  fast_eval.yaml  benchmark.yaml
```

```python
from i4b_bench import closed_loop_setting, eval_benchmark_closed_loop

eval_benchmark_closed_loop(controller, setting=closed_loop_setting("benchmark"))
```

Open-loop sets live elsewhere. They are compiled into a case artifact rather than resolved at
run time, so a definition sits beside the artifact it produced, under
`data/evaluation_sets/open_loop/` — see the README there.

Each file has two sections. `common` states **every** field of `ClosedLoopBenchmark`, so reading
it tells you the whole setting and nothing is inherited from a default you would have to go and
look up. `scenarios` names the problem instances: an **episode** is a `building`, a `start`, an
`end` and the `controller` whose recorded run seeds the context at `start` — not the method under
test, which is the argument you pass in.

Times are written either as a plain date, meaning midnight UTC, or as a datetime naming any point
on the corpus' 15-minute grid:

```yaml
start: 2025-10-10             # midnight UTC
start: 2025-10-10 05:00:00    # an early-morning recovery
start: 2026-02-01 17:30:00    # the evening peak
```

PyYAML parses these to `datetime.date` and `datetime.datetime`; the harness resolves each against
the dataset's own clock, and rejects a time that does not land on a step boundary rather than
silently rounding it down. Time of day matters: the same building on the same date behaves very
differently anchored at an early-morning recovery than at an afternoon coast.

## What each view exposes

`view` in `common` picks how much of the plant a method may see. It is the benchmark's main
difficulty knob, and the gap between a method's two scores is the price of the missing
information.

| | `perfect` | `realistic` |
| --- | --- | --- |
| state | `T_room`, `T_wall`, `T_hp_ret` | `T_room`, `T_hp_ret` |
| disturbances | `T_amb`, `Qdot_gains` | `T_amb`, `ghi`, `dni`, `dhi` |

`perfect` is the oracle: it shows the wall node, which makes the 4R3C dynamics fully observable,
and the building-specific heat gain the simulator actually used. Neither is measurable on a real
site. `realistic` is what an installation could be built to instrument — raw weather, plus the two
temperatures a heat pump and a room sensor report — so a method there must infer the thermal mass
from its own history and the gain from the weather.

These settings live here, in the package, rather than in the dataset directory: they define what
the benchmark *measures*, which is a decision and belongs under version control. The open-loop
definitions are equally version-controlled, but sit beside the generated artifact they produce.

The two sets differ in what they cost, not in what they measure:

| set | closed loop |
| --- | --- |
| `fast_eval` | 1 building, 1 day |
| `benchmark` | 10 buildings, 14 days each |

The two loops scale different things because they cost differently. An open-loop case is a few
plant rollouts, paid once when the set is compiled, so sampling more of them is nearly free and
is the only way to get error bars on small differences between methods. A closed-loop step runs
the controller, which for a planning controller is ~2 s — a year across 30 buildings is over 500
hours. Episodes are independent, so the closed-loop set is meant to be distributed. Its ten
starts are spread through the year, since a controller that handles January is not thereby known
to handle a shoulder season.

## Scenario identifiers

Settings name their buildings outright rather than taking a count, so that adding buildings to the
corpus cannot quietly change which problems a benchmark covers. An id is a TABULA building
typology code plus the weather year — `BG.N.SFH.02.Gen.ReEx.001.001--period_b`:

| segment | example | meaning | values in this corpus |
| --- | --- | --- | --- |
| country | `BG` | ISO country code | BG, CY, DE, FR, IE, NO, PL |
| region | `N` | region within the country (**N**ational) | only `N` |
| type | `SFH` | building type (single-family house) | only `SFH` |
| period | `02` | construction-period class, see `year_start` / `year_end` | 12 classes; `01` is pre-1918 |
| construction | `Gen` | constructive type | `Gen`eric, `Tframe`, `LightFrame`, `HBlock`, `325SB` (325 mm solid brick), `Class` |
| data type | `ReEx` | **Re**al **Ex**ample rather than a synthetic average | only `ReEx` |
| building | `001` | building index within the class | 1–2 |
| variant | `001` | **refurbishment level** | 1–3 |
| period | `period_b` | which weather year | `period_a` 2024-04 → 2025-03, `period_b` 2025-04 → 2026-03 |

The variant is the segment most worth knowing: it is the renovation state, and it is visible in
the physics. The three variants of `BG.N.SFH.01.Gen.ReEx.001` have transmission
**7.04 → 2.06 → 1.80 W/m²K** — unrenovated, usual refurbishment, advanced. They are the same
house in three thermal states and behave very differently under the same weather.

That is also why `split.parquet` assigns whole *families* (`building_family_id`, everything up to
the variant) rather than individual buildings: all three refurbishment levels land on the same
side of the train/test boundary, so a model cannot have seen the insulated version of a building
it is tested on. `split` in `common` is enforced — a building from another split raises rather
than quietly producing a number nobody can trust.

A building id is therefore one house, in one thermal state, under one year of weather. So `DE x6`
in a set is six such combinations, not six distinct German houses.

## Choosing scenarios

`select_scenarios` prints a block to paste, with a header showing the country spread:

```
uv run python -m i4b_bench.select_scenarios --count 10
```

Scenarios are drawn evenly spaced over the sorted ids rather than from the front, because ids sort
by country and a prefix would be two countries rather than a sample of the corpus. Nothing writes
these files automatically: which problems a benchmark covers is a decision someone makes and
records, not something recomputed on each run.

# Evaluation settings

One file per named setting, split by loop:

```
config/open_loop/    fast.toml  benchmark.toml  full.toml
config/closed_loop/  fast.toml  benchmark.toml  full.toml
```

```python
from i4b_bench import closed_loop_setting, eval_benchmark_closed_loop

eval_benchmark_closed_loop(controller, setting=closed_loop_setting("benchmark"))
```

Each file states **every** field of its dataclass, so reading it tells you the whole setting and
nothing is inherited from a default you would have to go and look up. They live here, in the
package, rather than in the dataset directory: they define what the benchmark *measures*, which
is a decision and belongs under version control, and the dataset directory is gitignored.

The three sets differ in how much they cost, not in what they measure:

| set | open loop | closed loop |
| --- | --- | --- |
| `fast` | 1 scenario, 3 windows | 1 scenario, 1 day |
| `benchmark` | 30 scenarios, 20 windows, 4 context lengths | 10 scenarios, 14 days |
| `full` | 30 scenarios, 50 windows | 30 scenarios, a full year |

The two loops scale different things because they cost differently. An open-loop window is a few
plant rollouts, so sampling more of them is nearly free and is the only way to get error bars on
small differences between methods. A closed-loop step runs the controller, which for a planning
controller is ~2 s -- a year across 30 scenarios is over 500 hours. Scenarios are independent, so
the closed-loop sets are meant to be distributed.

## Scenario identifiers

Sets name their scenarios outright rather than taking a count, so that adding buildings to the
corpus cannot quietly change which problems a benchmark covers. An id is a TABULA building
typology code plus the weather year -- `BG.N.SFH.02.Gen.ReEx.001.001--period_b`:

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
it is tested on.

A `scenario_id` is therefore one building, in one thermal state, under one year of weather. So
`DE x6` in a set is six such combinations, not six distinct German houses.

## Choosing a set

`select_scenarios` prints a block to paste, with a header showing the country spread:

```
uv run python -m i4b_bench.select_scenarios --count 10
```

Scenarios are drawn evenly spaced over the sorted ids rather than from the front, because ids
sort by country and a prefix would be two countries rather than a sample of the corpus.

Neither script writes these files. Which scenarios a benchmark covers is a decision someone makes
and records, not something recomputed on each run.

## A note on TOML

TOML has no null. A field whose Python default is `None` is therefore written out explicitly --
`evaluation_steps = 35039` in `full` rather than an omission meaning "no limit" -- because an
omitted key would silently fall back to the dataclass default instead.

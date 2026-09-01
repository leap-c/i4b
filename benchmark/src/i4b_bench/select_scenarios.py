"""Pick a scenario set and print it, ready to paste into a config file.

A setting names its scenarios outright rather than a count, so that adding buildings to the
corpus cannot silently change which problems a benchmark runs. Choosing them is a one-off
decision, so it happens here and the result is pasted in, rather than being recomputed on every
run by a formula nobody reads.

    uv run python -m i4b_bench.select_scenarios --count 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import evaluation_scenarios, load_dataset


def select(dataset, split: str = "test", count: int | None = None) -> list[str]:
    """`count` scenarios of the split, evenly spaced over the sorted ids.

    Spaced rather than taken from the front: ids sort by country, so a prefix would be a couple
    of countries rather than a sample of the corpus.
    """
    return evaluation_scenarios(dataset, split, limit=count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=None, help="omit for the whole split")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    chosen = select(dataset, args.split, args.count)

    print(as_toml(chosen, len(evaluation_scenarios(dataset, args.split)), args.split))


def as_toml(chosen: list[str], total: int, split: str = "test") -> str:
    """The block to paste into a config file, with a header saying what it covers."""
    countries: dict[str, int] = {}
    for scenario_id in chosen:
        country = scenario_id.split(".")[0]
        countries[country] = countries.get(country, 0) + 1
    lines = [
        f"# {len(chosen)} of {total} scenarios in the {split} split",
        f"# countries: {', '.join(f'{k} x{v}' for k, v in sorted(countries.items()))}",
        "scenarios = [",
        *(f'  "{scenario_id}",' for scenario_id in chosen),
        "]",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

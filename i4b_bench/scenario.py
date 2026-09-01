"""A benchmark scenario: one building, in one refurbishment state, under one year of weather.

Each is a file under `scenarios/`, tracked in git, named by what distinguishes it -- country,
construction era, refurbishment variant, weather period -- with the facts inside rather than
encoded in the name. The file says what the problem *is*; an evaluation setting says how it is
measured, and the two are deliberately separate.

A scenario names its data in the corpus rather than containing it. The corpus is gigabytes and
gitignored; these are kilobytes and reviewable, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib

SCENARIOS = Path(__file__).parent / "scenarios"


@dataclass(frozen=True)
class Building:
    """The physics. `variant` is the refurbishment level, and `transmission` shows what it means."""

    id: str
    country: str
    location: str
    year_start: int
    year_end: int
    variant: int
    transmission_W_m2K: float
    floor_area_m2: float
    room_height_m: float
    mdot_hp: float


@dataclass(frozen=True)
class Scenario:
    """One problem instance, as read from its file."""

    name: str
    corpus_id: str
    split: str
    start: object
    end: object
    timestep_s: int
    building: Building
    comfort_band_C: tuple[float, float]
    action_limits_C: tuple[float, float]
    recorded: tuple[str, ...]

    @property
    def country(self) -> str:
        return self.building.country

    def metadata(self) -> dict:
        """The columns worth carrying into a results table, so a row explains itself."""
        return {
            "scenario": self.name,
            "country": self.building.country,
            "variant": self.building.variant,
            "transmission_W_m2K": self.building.transmission_W_m2K,
            "year_start": self.building.year_start,
            "floor_area_m2": self.building.floor_area_m2,
        }


def load(name: str, root: Path | None = None) -> Scenario:
    """Read one scenario by name, e.g. `de_1958_v1_b`."""
    path = (root or SCENARIOS) / f"{name}.toml"
    if not path.exists():
        raise KeyError(f"unknown scenario {name!r}, have {sorted(available(root))[:5]}...")
    body = tomllib.loads(path.read_text())
    return Scenario(
        name=name,
        corpus_id=body["corpus_id"],
        split=body["split"],
        start=body["period"]["start"],
        end=body["period"]["end"],
        timestep_s=body["period"]["timestep_s"],
        building=Building(**body["building"]),
        comfort_band_C=tuple(body["task"]["comfort_band_C"]),
        action_limits_C=tuple(body["task"]["action_limits_C"]),
        recorded=tuple(body["trajectories"]["recorded"]),
    )


def available(root: Path | None = None) -> list[str]:
    """Every scenario shipped with the benchmark."""
    return sorted(p.stem for p in (root or SCENARIOS).glob("*.toml"))

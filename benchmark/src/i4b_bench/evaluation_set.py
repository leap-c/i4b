"""The definition of an open-loop evaluation set, and where its artifact lives.

An evaluation set is a directory:

```
benchmark/data/evaluation_sets/open_loop/<name>/
    definition.yaml   what the set measures -- a decision, version-controlled
    cases.parquet     the compiled windows -- generated from the corpus, immutable per release
    manifest.json     what produced the artifact, and the fingerprints to check it against
```

The definition is the only hand-written half. It names the windows and fixes everything that
must not vary between two runs for their numbers to be comparable; `data/scripts/
build_open_loop_cases.py` turns it, plus the corpus, into the other two files. Evaluation reads
only the generated pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from .control_gain import PLAN_ROLES

#: The corpus' step, and so the grid every window and horizon must land on.
TIMESTEP_SECONDS = 900
_PER_HOUR = 3600 / TIMESTEP_SECONDS
_PER_DAY = 24 * _PER_HOUR

#: Where the bundled sets live. A set may equally be given as a path to any directory.
EVALUATION_SETS = Path(__file__).resolve().parents[2] / "data" / "evaluation_sets" / "open_loop"

DEFINITION_FILE = "definition.yaml"
CASES_FILE = "cases.parquet"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class Window:
    """One problem instance: a building, when it starts, and whose run fills the context."""

    building: str
    #: Where the horizon begins. A date means midnight UTC; a datetime names any time on the
    #: corpus' 15-minute grid.
    start: date | datetime
    #: The recorded run whose history seeds the context.
    controller: str


@dataclass(frozen=True)
class OpenLoopDefinition:
    """What must not vary between open-loop runs for two results to be comparable."""

    split: str
    view: str
    use_forecast: bool
    horizon_hours: float
    #: Context lengths to sweep. Every window is scored at each, and it becomes a results
    #: column. The longest is what the artifact stores; the rest are its tails.
    context_days: tuple[float, ...]
    #: How many control trajectories each window is probed with. Fixed by the plan design --
    #: written out anyway, because a reader should not have to look up a constant to know what
    #: was run.
    probes: int
    #: Peak perturbation about the recorded plan, in Kelvin.
    probe_amplitude: float
    #: Steps the APRBS probe holds each level.
    probe_hold_steps: int
    #: How far an archived forecast is pulled toward the current sensor reading, in [0, 1]. Zero
    #: here: forecast error is part of what this measures.
    forecast_correction: float
    scenarios: dict[str, Window]

    def __post_init__(self) -> None:
        if not self.context_days:
            raise ValueError("a definition must name at least one context length")
        if any(days <= 0 for days in self.context_days):
            raise ValueError(f"context lengths must be positive: {self.context_days}")
        for days in self.context_days:
            _exact_steps(days * _PER_DAY, f"context of {days} d")
        _exact_steps(self.horizon_hours * _PER_HOUR, f"horizon of {self.horizon_hours} h")
        if self.probes != len(PLAN_ROLES):
            raise ValueError(
                f"the plan design is {len(PLAN_ROLES)} probes ({', '.join(PLAN_ROLES)}), "
                f"the definition asks for {self.probes}"
            )
        if self.probes < 3 or self.probes % 2 == 0:
            raise ValueError("probes must be odd and at least three: a nominal plus signed pairs")
        if self.probe_amplitude <= 0:
            raise ValueError("probe_amplitude must be positive")
        if self.probe_hold_steps < 1:
            raise ValueError("probe_hold_steps must be at least one")
        if not 0 <= self.forecast_correction <= 1:
            raise ValueError("forecast_correction must be between zero and one")
        if not self.scenarios:
            raise ValueError("a definition must name at least one window")

    @property
    def horizon_steps(self) -> int:
        """The horizon, in corpus steps."""
        return _exact_steps(self.horizon_hours * _PER_HOUR, "horizon")

    @property
    def context_steps(self) -> tuple[int, ...]:
        """Each context length, in corpus steps, in the order the definition names them."""
        return tuple(_exact_steps(days * _PER_DAY, "context") for days in self.context_days)

    @property
    def max_context_steps(self) -> int:
        """The longest context, in steps. This is what a case stores."""
        return max(self.context_steps)


def load_definition(path: str | Path) -> OpenLoopDefinition:
    """Read and validate a `definition.yaml`.

    Everything checkable without the corpus is checked here: the context and horizon resolve to
    whole steps, the probe design matches, and no case id is defined twice. Whether a window's
    building exists, its controller was recorded, and its whole interval lies inside the split
    needs the corpus, and is checked by the case compiler.

    Parameters
    ----------
    path : str or Path
        The `definition.yaml` itself, or the set directory holding it.

    Returns
    -------
    OpenLoopDefinition
    """
    path = Path(path)
    if path.is_dir():
        path = path / DEFINITION_FILE
    body = yaml.load(path.read_text(), Loader=_StrictLoader)
    common = dict(body["common"])
    common["context_days"] = tuple(common["context_days"])
    windows = {name: Window(**entry) for name, entry in body["scenarios"].items()}
    return OpenLoopDefinition(**common, scenarios=windows)


def resolve_evaluation_set(evaluation_set: str | Path) -> Path:
    """A set's directory, from a bundled name or a path.

    Parameters
    ----------
    evaluation_set : str or Path
        A name under `EVALUATION_SETS`, e.g. ``"benchmark-v1"``, or a path to a set directory.

    Returns
    -------
    Path
        The directory. It exists and holds a `definition.yaml`.
    """
    candidate = Path(evaluation_set)
    if candidate.is_dir() and (candidate / DEFINITION_FILE).exists():
        return candidate
    bundled = EVALUATION_SETS / str(evaluation_set)
    if (bundled / DEFINITION_FILE).exists():
        return bundled
    have = sorted(p.name for p in EVALUATION_SETS.glob("*") if (p / DEFINITION_FILE).exists())
    raise KeyError(f"unknown evaluation set {str(evaluation_set)!r}, have {have}")


def _exact_steps(steps: float, what: str) -> int:
    """`steps` as an int, refusing anything that does not land on the corpus' grid."""
    rounded = round(steps)
    if abs(steps - rounded) > 1e-9 or rounded <= 0:
        raise ValueError(
            f"the {what} is {steps} steps of {TIMESTEP_SECONDS} s; it must be a positive whole "
            "number of them"
        )
    return rounded


class _StrictLoader(yaml.SafeLoader):
    """A loader that refuses duplicate mapping keys.

    PyYAML keeps the last of a repeated key, so a copy-pasted `window042:` would silently drop a
    window and leave the set one case short of what its file appears to say.
    """


def _no_duplicate_keys(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"{key!r} is defined twice")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)

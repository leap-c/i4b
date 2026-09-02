"""What a model is allowed to see, and how it is handed over.

Both evaluations build their observations through `build_observation`, so the two loops hand a
model the same thing.

A view sets how much of the plant is visible. `perfect` is the oracle -- the wall node and the
heat gain the simulator used, neither measurable on a real site, together making the dynamics
fully observable. `realistic` is what an installation could instrument: raw weather, and the
temperatures a heat pump and a room sensor report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

ObsView = Literal["perfect", "realistic"]

#: The plant's state vector, in the plant's own order. Seeding a run and reading `env.state` go
#: through this, independent of any view.
PLANT_STATE_CHANNELS = ("T_room", "T_wall", "T_hp_ret")

CONTROL_CHANNELS = ("T_hp_sup_applied",)

#: What a learned model predicts. `T_wall` stays out under both views; a method may still return
#: it, since nothing caps what the returned dict contains.
TARGET_CHANNELS = ("T_room", "T_hp_ret")

#: How much of the state each view exposes. `realistic` withholds the wall node -- nothing
#: measures a wall -- so a method there infers the thermal mass from its own history.
STATE_CHANNELS: dict[ObsView, tuple[str, ...]] = {
    "perfect": PLANT_STATE_CHANNELS,
    "realistic": ("T_room", "T_hp_ret"),
}

DISTURBANCE_CHANNELS: dict[ObsView, tuple[str, ...]] = {
    "perfect": ("T_amb", "Qdot_gains"),
    "realistic": ("T_amb", "ghi", "dni", "dhi"),
}


def history_channels(view: ObsView) -> tuple[str, ...]:
    """The channels a history row carries: state, the action that produced it, disturbances.

    Parameters
    ----------
    view : {"perfect", "realistic"}
        Which view's state and disturbance sets to combine.

    Returns
    -------
    tuple of str
        Column names, in a stable order.
    """
    return STATE_CHANNELS[view] + CONTROL_CHANNELS + DISTURBANCE_CHANNELS[view]


def build_observation(
    state: dict[str, float],
    history: dict[str, np.ndarray],
    forecast: dict[str, np.ndarray],
) -> dict:
    """Assemble the observation both evaluations hand to a model.

    Candidate future controls are not part of it -- open loop passes them alongside, so both
    loops hand over the same thing.

    Parameters
    ----------
    state : dict of str to float
        The current state, holding this view's `STATE_CHANNELS`.
    history : dict of str to numpy.ndarray
        Past rows, each action aligned onto the state it produced, plus a `timestamp` array.
        Keys are this view's `history_channels`.
    forecast : dict of str to numpy.ndarray
        Known future disturbances over the planning horizon, plus a `timestamp` array. Keys are
        this view's `DISTURBANCE_CHANNELS`.

    Returns
    -------
    dict
        ``{"state": ..., "history": ..., "forecast": ...}``.
    """
    return {"state": state, "history": history, "forecast": forecast}


def internal_gain_profile() -> Path:
    """The bundled residential internal-gain profile."""
    import i4b_data

    # i4b_data is a namespace package, so it has no __file__; __path__ works either way.
    return (
        Path(next(iter(i4b_data.__path__))).resolve()
        / "profiles"
        / "InternalGains"
        / "ResidentialDetached.csv"
    )

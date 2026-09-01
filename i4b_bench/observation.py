"""What a model is allowed to see, and how it is handed over.

The channel sets and the builder live together on purpose: the tuples say what a view exposes,
and `build_observation` hands over exactly those. Both evaluations construct observations
through this one function, so "the observation is the same in open and closed loop" is enforced
rather than promised.

Views differ only in which disturbances are visible. `perfect` shows the building-specific heat
gain the simulator actually used; `realistic` shows the raw weather a site would measure or a
forecast would deliver, leaving the model to infer the gain. Neither exposes the wall
temperature to a learned model -- nothing measures a wall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

ObsView = Literal["perfect", "realistic"]

STATE_CHANNELS = ("T_room", "T_wall", "T_hp_ret")
CONTROL_CHANNELS = ("T_hp_sup_applied",)
#: What a learned model predicts. `T_wall` is deliberately absent.
TARGET_CHANNELS = ("T_room", "T_hp_ret")

DISTURBANCE_CHANNELS: dict[ObsView, tuple[str, ...]] = {
    "perfect": ("T_amb", "Qdot_gains"),
    "realistic": ("T_amb", "ghi", "dni", "dhi"),
}


def history_channels(view: ObsView) -> tuple[str, ...]:
    """The channels a history row carries: state, the action that produced it, disturbances."""
    return STATE_CHANNELS + CONTROL_CHANNELS + DISTURBANCE_CHANNELS[view]


def build_observation(
    state: dict[str, float],
    history: dict[str, np.ndarray],
    forecast: dict[str, np.ndarray],
) -> dict:
    """Assemble the observation both evaluations hand to a model.

    `state` is the current state, `history` carries past rows with each action aligned onto the
    state it produced, and `forecast` the known future disturbances. Candidate future controls
    are *not* here: open-loop passes them alongside, so the observation stays identical between
    the two loops and reads as a description of what happened rather than of what to try.
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

"""Does the model respond to the control the way the plant does?

A counterfactual measurement, scored alongside point accuracy because a model can track room
temperature well while barely reacting to the heat pump. From one anchor, drive the plant under
several perturbed control trajectories -- same weather, same initial state, only the control
differs -- and regress the model's predicted deviation on the plant's actual deviation:

    gain = 1.0   the model moves exactly as the plant does
    gain = 0.0   the model ignores the control entirely

The probe set
-------------
Five plans, because the question behind this benchmark is whether a model is worth putting
inside an MPC, and that takes two different perturbations:

    nominal        the plan the corpus applied -- the one point accuracy is scored on
    offset_minus   nominal - amplitude, held over the whole horizon
    offset_plus    nominal + amplitude
    aprbs_minus    nominal - w, for one held pseudo-random waveform w
    aprbs_plus     nominal + w

Constant offsets ask about the steady-state response: shift the supply temperature for a day and
the room has to follow. The APRBS pair asks about timing -- a model can have the right daily gain
and still smear a two-hour step -- which is exactly what an MPC exploits when it shifts heat
around a price or comfort window. Pure offsets miss models whose dynamics are wrong but whose
integral is right; pure APRBS is more aggressive than anything an MPC would choose.

Each perturbation appears with both signs, so the deviations the gain regresses are balanced
about the nominal and a model's asymmetry does not read as a slope. The nominal plan is named,
never inferred from a position in the array.
"""

from __future__ import annotations

import numpy as np

#: The corpus's action constraints.
BOUNDS = (5.0, 65.0)

#: Steps the APRBS waveform holds each level, at 15 minutes a step. Two hours: long enough for a
#: room to start moving, short enough that a model with sluggish dynamics cannot fake it.
HOLD_STEPS = 8

#: The plans, in the order `probe_plans` returns them. `plan_role` is what identifies a plan
#: downstream; nothing may key on an index.
PLAN_ROLES = ("nominal", "offset_minus", "offset_plus", "aprbs_minus", "aprbs_plus")

#: The role whose prediction point accuracy is scored on. Exactly one plan carries it.
NOMINAL_ROLE = "nominal"


def probe_plans(
    baseline: np.ndarray,
    amplitude: float,
    *,
    seed: int,
    hold_steps: int = HOLD_STEPS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The five control trajectories around `baseline`, clipped to the actuator's range.

    Deterministic: the same `seed` and `baseline` always give the same plans, so a case artifact
    can be rebuilt and compared byte for byte.

    Parameters
    ----------
    baseline : numpy.ndarray
        The nominal plan, shape `(horizon,)`, in degrees Celsius.
    amplitude : float
        Peak perturbation about the baseline, in Kelvin. Both the offset and the APRBS levels
        stay within it.
    seed : int
        Seeds the APRBS waveform. Derive it from the case id, so a window's probes do not move
        when its neighbours in the set do.
    hold_steps : int
        Steps the waveform holds each level.

    Returns
    -------
    plans : numpy.ndarray
        Shape `(5, horizon)`, clipped to `BOUNDS`.
    roles : tuple of str
        `PLAN_ROLES`, aligned with the rows of `plans`.
    """
    if amplitude <= 0:
        raise ValueError("probe amplitude must be positive")
    if hold_steps < 1:
        raise ValueError("hold_steps must be at least one")
    baseline = np.asarray(baseline, dtype=float)
    wave = _aprbs(np.random.default_rng(seed), len(baseline), amplitude, hold_steps)
    plans = np.stack(
        [
            baseline,
            baseline - amplitude,
            baseline + amplitude,
            baseline - wave,
            baseline + wave,
        ]
    )
    return np.clip(plans, *BOUNDS), PLAN_ROLES


def _aprbs(rng: np.random.Generator, steps: int, amplitude: float, hold: int) -> np.ndarray:
    """A held pseudo-random level sequence: one uniform draw per `hold` steps."""
    levels = rng.uniform(-amplitude, amplitude, size=-(-steps // hold))
    return np.repeat(levels, hold)[:steps]


def gain_terms(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """The numerator and denominator of the slope, kept apart so windows can be pooled.

    Deviations are about the mean across probes, so whatever the probes shared -- weather, the
    starting state, any constant bias -- cancels.

    Parameters
    ----------
    actual : numpy.ndarray
        The plant's response, shape `(n_probes, horizon)`.
    predicted : numpy.ndarray
        The model's response to the same probes, same shape.

    Returns
    -------
    tuple of float
        `(cross, square)`; the gain of a pooled set is the ratio of their sums.
    """
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    plant = actual - actual.mean(axis=0)
    model = predicted - predicted.mean(axis=0)
    return float((model * plant).sum()), float((plant**2).sum())


def control_gain(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Slope of the model's predicted deviation on the plant's, over the probes.

    Parameters
    ----------
    actual : numpy.ndarray
        The plant's response, shape `(n_probes, horizon)`.
    predicted : numpy.ndarray
        The model's response to the same probes, same shape.

    Returns
    -------
    float
        1.0 moves as the plant does, 0.0 ignores the control; `NaN` when the probes did not
        move the plant.
    """
    cross, square = gain_terms(actual, predicted)
    if square <= 0:
        return float("nan")
    return cross / square

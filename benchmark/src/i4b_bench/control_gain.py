"""Does the model respond to the control the way the plant does?

A counterfactual measurement, scored alongside point accuracy because a model can track room
temperature well while barely reacting to the heat pump. From one anchor, drive the plant under
several perturbed control trajectories -- same weather, same initial state, only the control
differs -- and regress the model's predicted deviation on the plant's actual deviation:

    gain = 1.0   the model moves exactly as the plant does
    gain = 0.0   the model ignores the control entirely

Rollouts go through `ScenarioEnv`, so probes inherit the corpus' integrator, clipping and
disturbances.
"""

from __future__ import annotations

import numpy as np

#: The corpus's action constraints.
BOUNDS = (5.0, 65.0)
#: Hold length for the probe waveform, in steps -- long enough for a building to respond.
HOLD = (4, 24)


def probe_plans(
    rng: np.random.Generator,
    baseline: np.ndarray,
    amplitude: float,
    count: int,
    kind: str = "offset",
) -> np.ndarray:
    """`count` control trajectories around `baseline`, clipped to the actuator's range.

    Parameters
    ----------
    rng : numpy.random.Generator
        Used only by `kind="aprbs"`; `offset` probes are deterministic.
    baseline : numpy.ndarray
        The nominal plan, shape `(horizon,)`, in degrees Celsius.
    amplitude : float
        Peak perturbation about the baseline, in Kelvin.
    count : int
        How many probes. At least two, since the metric is a slope.
    kind : {"offset", "aprbs"}
        `offset` shifts the whole horizon by a constant, spaced evenly over
        ``[-amplitude, +amplitude]``. `aprbs` perturbs with held random steps, which excites
        more of the dynamics but mixes timing into the answer.

    Returns
    -------
    numpy.ndarray
        Shape `(count, horizon)`, clipped to `BOUNDS`.
    """
    if count < 2:
        raise ValueError("a slope needs at least two probes")
    if kind == "offset":
        deltas = np.linspace(-amplitude, amplitude, count)
        plans = baseline[None, :] + deltas[:, None]
    elif kind == "aprbs":
        plans = np.stack([baseline + _aprbs(rng, len(baseline), amplitude) for _ in range(count)])
    else:
        raise ValueError(f"unknown probe kind {kind!r}")
    return np.clip(plans, *BOUNDS)


def _aprbs(rng: np.random.Generator, steps: int, amplitude: float) -> np.ndarray:
    out, filled = [], 0
    while filled < steps:
        hold = int(rng.integers(HOLD[0], HOLD[1] + 1))
        out.append(np.full(hold, rng.uniform(-amplitude, amplitude)))
        filled += hold
    return np.concatenate(out)[:steps]


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

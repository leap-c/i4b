"""Does the model respond to the control the way the plant does?

Accuracy and control response are close to unrelated on this corpus, and only the second one
predicts closed-loop performance: a model can track room temperature well while barely reacting
to the heat pump, which makes it useless inside a controller. So the benchmark measures both.

The measurement is a counterfactual. From one anchor, drive the plant under several perturbed
control trajectories -- same weather, same initial state, only the control differs -- and ask
the model about each. Regress the model's predicted deviation on the plant's actual deviation:

    gain = 1.0   the model moves exactly as the plant does
    gain = 0.0   the model ignores the control entirely

Rollouts go through `ScenarioEnv`, so probes inherit the same integrator, the same `check_hp`
clipping and the same disturbances as the corpus and the closed loop.
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

    `offset` shifts the whole horizon by a constant, which isolates the response from the
    waveform. `aprbs` perturbs with held random steps, which excites more of the dynamics but
    mixes timing into the answer.
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
    """The numerator and denominator of the slope, so several windows can be pooled.

    Deviations are taken about the mean across probes, so whatever the probes had in common --
    weather, the starting state, any constant bias -- cancels, and only the response to
    *differences* in control survives.
    """
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    plant = actual - actual.mean(axis=0)
    model = predicted - predicted.mean(axis=0)
    return float((model * plant).sum()), float((plant**2).sum())


def control_gain(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Slope of the model's predicted deviation on the plant's, over the probes.

    Both arrays are (n_probes, horizon). Deviations are taken about the mean across probes, so
    whatever the control had in common with every probe -- weather, the starting state, any
    constant bias -- cancels, and only the response to *differences* in control is scored.
    """
    cross, square = gain_terms(actual, predicted)
    if square <= 0:
        return float("nan")  # the probes did not move the plant; nothing to be right about
    return cross / square

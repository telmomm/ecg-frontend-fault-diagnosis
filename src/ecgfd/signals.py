"""Synthetic test signals for validating the nominal circuit."""

from __future__ import annotations

import numpy as np

# (amplitude [mV], centre relative to the R peak [s], width [s]) of P, Q, R, S, T
_WAVES = (
    (0.15, -0.20, 0.025),
    (-0.10, -0.035, 0.010),
    (1.00, 0.0, 0.011),
    (-0.20, 0.035, 0.010),
    (0.30, 0.25, 0.040),
)


def synthetic_ecg(
    duration: float = 3.0, fs: float = 1000.0, heart_rate: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """Lead-I-like ECG built from Gaussian waves, 1 mV R peak. Returns (t [s], v [V])."""
    t = np.arange(round(duration * fs)) / fs
    rr = 60.0 / heart_rate
    phase = (t - 0.4) % rr
    phase = np.where(phase > rr / 2, phase - rr, phase)
    v = np.zeros_like(t)
    for amplitude, centre, width in _WAVES:
        v += amplitude * np.exp(-0.5 * ((phase - centre) / width) ** 2)
    return t, v * 1e-3

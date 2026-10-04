"""Feature sets: C1 (DC), C2 (frequency), C3 (calibration pulse), C4 (contact impedance).

The DC set C1 reads the output and the instrumentation-amplifier output, which
needs one spare ADC channel; `C1x` adds the remaining simulated nodes (the RLD
output), which would need another one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALL = "C1+C2+C3+C4"
# DC nodes of the main feature set. `ina_out` is in: it is cheap to read, exposes the
# offsets that the high-pass hides from the output, and improves localisation (E2, E8).
MAIN_DC_NODES = ("out", "ina_out")


def feature_sets(cfg: dict) -> dict[str, list[str]]:
    """Column names of each tabular feature set."""
    mcfg = cfg["measurement"]
    c1 = [f"dc_{n}" for n in mcfg["dc_nodes"] if n in MAIN_DC_NODES]
    c1x = [f"dc_{n}" for n in mcfg["dc_nodes"]]
    freqs = [f"{f:g}" for f in mcfg["ac"]["freqs"]]
    c2 = [f"{kind}_{f}" for f in freqs for kind in ("acd_mag", "acd_ph", "cmrr_db")]
    c3 = [f"pulse_{k}" for k in PULSE_FEATURES]
    c4 = [f"zlo_mag_{f:g}" for f in mcfg["lead_off"]["freqs"]]
    return {"C1": c1, "C1x": c1x, "C2": c2, "C3": c3, "C4": c4, ALL: c1 + c2 + c3 + c4}


PULSE_FEATURES = ("baseline", "peak", "end", "undershoot", "tail", "rise_time", "area")


def measurement_groups(cfg: dict) -> dict[str, tuple[list[str], float]]:
    """Self-test actions: name -> (feature columns it yields, time it takes [s]).

    One group is one thing the instrument does: read a DC level, apply one tone and
    estimate its response, or apply the calibration pulse. Used to study which
    measurements are worth taking (E8). A tone lasts `n_samples` ADC samples or two
    periods, whichever is longer; a DC reading lasts its averaging time.
    """
    mcfg = cfg["measurement"]
    fs = float(mcfg["pulse"]["fs"])
    t_dc = float(mcfg["adc"]["dc_averages"]) / fs

    def t_tone(f: float) -> float:
        return max(float(mcfg["ac"]["n_samples"]) / fs, 2.0 / f)

    groups: dict[str, tuple[list[str], float]] = {
        f"dc_{node}": ([f"dc_{node}"], t_dc) for node in mcfg["dc_nodes"]
    }
    for f in mcfg["ac"]["freqs"]:
        groups[f"diff_tone_{f:g}"] = ([f"acd_mag_{f:g}", f"acd_ph_{f:g}"], t_tone(f))
        groups[f"cm_tone_{f:g}"] = ([f"acc_mag_{f:g}"], t_tone(f))
    for f in mcfg["lead_off"]["freqs"]:
        groups[f"leadoff_tone_{f:g}"] = ([f"zlo_mag_{f:g}"], t_tone(f))
    groups["cal_pulse"] = ([f"pulse_{k}" for k in PULSE_FEATURES], float(mcfg["pulse"]["duration"]))
    return groups


def pulse_features(waveforms: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Hand-crafted descriptors of the pulse response, for tabular models."""
    p = cfg["measurement"]["pulse"]
    fs = float(p["fs"])
    i0 = round(p["delay"] * fs)
    i1 = round((p["delay"] + p["width"]) * fs)
    w = np.asarray(waveforms, dtype=float)
    baseline = w[:, : max(i0 - 1, 1)].mean(axis=1)
    x = w - baseline[:, None]
    during = x[:, i0:i1]
    peak = during.max(axis=1)
    # first sample reaching 90 % of the peak, counted from the pulse edge
    reached = during >= 0.9 * np.maximum(peak, 1e-12)[:, None]
    rise = np.where(reached.any(axis=1), reached.argmax(axis=1), during.shape[1]) / fs
    return pd.DataFrame(
        {
            "pulse_baseline": baseline,
            "pulse_peak": peak,
            "pulse_end": x[:, i1 - 1],
            "pulse_undershoot": x[:, i1:].min(axis=1),
            "pulse_tail": x[:, -1],
            "pulse_rise_time": rise,
            "pulse_area": during.sum(axis=1) / fs,
        }
    )


def add_derived(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add CMRR estimates [dB] computed from the differential and common-mode gains."""
    out = df.copy()
    for f in cfg["measurement"]["ac"]["freqs"]:
        ratio = out[f"acd_mag_{f:g}"] / np.maximum(out[f"acc_mag_{f:g}"], 1e-12)
        out[f"cmrr_db_{f:g}"] = 20.0 * np.log10(np.maximum(ratio, 1e-12))
    return out

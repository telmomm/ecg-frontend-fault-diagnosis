"""Feature sets: C1 (DC), C2 (frequency), C3 (calibration pulse), C4 (contact impedance).

"Strict" sets use only the main ADC channel; the extended DC set (`C1x`) adds
nodes that would need a spare ADC channel (see `measurement.dc_nodes` in the config).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from spicefault import Measurement

from .selftest import pulse_waveform

ALL = "C1+C2+C3+C4"


def feature_sets(cfg: dict) -> dict[str, list[str]]:
    """Column names of each tabular feature set."""
    mcfg = cfg["measurement"]
    c1 = [f"dc_{n}" for n, spec in mcfg["dc_nodes"].items() if not spec["extended"]]
    c1x = [f"dc_{n}" for n in mcfg["dc_nodes"]]
    freqs = [f"{f:g}" for f in mcfg["ac"]["freqs"]]
    c2 = [f"{kind}_{f}" for f in freqs for kind in ("acd_mag", "acd_ph", "cmrr_db")]
    c3 = [f"pulse_{k}" for k in PULSE_FEATURES]
    c4 = [f"zlo_mag_{f:g}" for f in mcfg["lead_off"]["freqs"]]
    return {"C1": c1, "C1x": c1x, "C2": c2, "C3": c3, "C4": c4, ALL: c1 + c2 + c3 + c4}


PULSE_FEATURES = ("baseline", "peak", "end", "undershoot", "tail", "rise_time", "area")


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


def pulse_measurements(cfg: dict) -> list[Measurement]:
    """The pulse descriptors of a noise-free simulation, e.g. to compute sensitivities."""
    t = pulse_waveform(cfg).times()

    def descriptor(name: str):
        def read(plot) -> float:
            samples = np.interp(t, plot["time"].real, plot["v(out)"].real)
            return pulse_features(samples[None, :], cfg)[name].iloc[0]

        return read

    names = [f"pulse_{k}" for k in PULSE_FEATURES]
    return [Measurement.custom(name, descriptor(name), "tran") for name in names]


def add_derived(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add CMRR estimates [dB] computed from the differential and common-mode gains."""
    out = df.copy()
    for f in cfg["measurement"]["ac"]["freqs"]:
        ratio = out[f"acd_mag_{f:g}"] / np.maximum(out[f"acc_mag_{f:g}"], 1e-12)
        out[f"cmrr_db_{f:g}"] = 20.0 * np.log10(np.maximum(ratio, 1e-12))
    return out

"""Simulation of the self-test measurements and of the validation transients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import CircuitInstance, Stimulus, build_netlist
from .spice import RAW_NAME, Plot, SimulationError, run_deck


@dataclass
class Measurement:
    """Noise-free result of the self-test measurements on one circuit, in service."""

    dc: dict[str, float]  # node -> voltage at the operating point
    freq: np.ndarray
    h_diff: np.ndarray  # v(out) / calibration source (complex)
    h_cm: np.ndarray  # v(out) / RLD reference source (complex)
    z_lo: np.ndarray  # v(out) / lead-off test current (complex) [V/A]
    t: np.ndarray  # ADC sample instants
    pulse: np.ndarray  # v(out) response to the calibration pulse


def expect_plots(plots: list[Plot], kinds: list[str]) -> None:
    names = [p.name for p in plots]
    if not (len(plots) == len(kinds) and all(map(str.startswith, names, kinds))):
        raise SimulationError(f"expected plots {kinds}, got {names}")


def ac_sweep(cfg: dict) -> str:
    s = cfg["measurement"]["ac"]["sweep"]
    return f"ac dec {s['points_per_decade']} {s['fstart']} {s['fstop']}"


def measure(inst: CircuitInstance, cfg: dict) -> Measurement:
    """Run the DC, AC (differential, common-mode, lead-off) and pulse analyses in one deck."""
    mcfg = cfg["measurement"]
    pulse = mcfg["pulse"]
    nodes = list(mcfg["dc_nodes"])
    fs = float(pulse["fs"])
    control = ["set appendwrite", "op", f"write {RAW_NAME} " + " ".join(f"v({n})" for n in nodes)]
    for source in ("vcal", "vcmt", "ilo"):
        control += [
            f"alter @{source}[acmag]=1",
            ac_sweep(cfg),
            f"write {RAW_NAME} v(out)",
            f"alter @{source}[acmag]=0",
        ]
    control += [
        f"tran {1 / fs} {pulse['duration']} 0 {0.25 / fs}",
        f"write {RAW_NAME} v(out)",
    ]
    stim = Stimulus(
        cal_amplitude=pulse["amplitude"],
        cal_delay=pulse["delay"],
        cal_width=pulse["width"],
        cal_edge=pulse["edge"],
    )
    plots = run_deck(build_netlist(inst, cfg, control, stim))
    expect_plots(plots, ["Operating Point", *["AC Analysis"] * 3, "Transient Analysis"])
    op, ac_d, ac_c, ac_z, tran = plots

    t = np.arange(round(pulse["duration"] * fs)) / fs
    return Measurement(
        dc={n: float(op[f"v({n})"][0].real) for n in nodes},
        freq=ac_d["frequency"].real.copy(),
        h_diff=ac_d["v(out)"].copy(),
        h_cm=ac_c["v(out)"].copy(),
        z_lo=ac_z["v(out)"].copy(),
        t=t,
        pulse=np.interp(t, tran["time"], tran["v(out)"]),
    )


def interp_response(freq: np.ndarray, h: np.ndarray, f: float) -> complex:
    """Complex response at `f`, interpolating log-magnitude and phase over log-frequency."""
    logf = np.log10(freq)
    mag = np.interp(np.log10(f), logf, np.log(np.maximum(np.abs(h), 1e-300)))
    phase = np.interp(np.log10(f), logf, np.unwrap(np.angle(h)))
    return complex(np.exp(mag) * np.exp(1j * phase))


def scalar_features(m: Measurement, cfg: dict) -> dict[str, float]:
    """Noise-free DC, frequency and lead-off features. See `features.py` for the sets."""
    mcfg = cfg["measurement"]
    feats = {f"dc_{node}": v for node, v in m.dc.items()}
    for f in mcfg["ac"]["freqs"]:
        hd = interp_response(m.freq, m.h_diff, f)
        feats[f"acd_mag_{f:g}"] = abs(hd)
        feats[f"acd_ph_{f:g}"] = float(np.degrees(np.angle(hd)))
        feats[f"acc_mag_{f:g}"] = abs(interp_response(m.freq, m.h_cm, f))
    for f in mcfg["lead_off"]["freqs"]:
        feats[f"zlo_mag_{f:g}"] = abs(interp_response(m.freq, m.z_lo, f))
    return feats


def transient(
    inst: CircuitInstance,
    cfg: dict,
    stim: Stimulus,
    duration: float,
    nodes: tuple[str, ...] = ("out",),
    tmax: float = 2.5e-4,
) -> Plot:
    """Free-form transient, used for validation (synthetic ECG, mains interference)."""
    control = [
        f"tran {tmax} {duration} 0 {tmax}",
        f"write {RAW_NAME} " + " ".join(f"v({n})" for n in nodes),
    ]
    plots = run_deck(build_netlist(inst, cfg, control, stim))
    expect_plots(plots, ["Transient Analysis"])
    return plots[0]

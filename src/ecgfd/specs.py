"""Clinical specifications of a circuit instance and its compliance (functional severity).

Specifications are properties of the instrument, so they are simulated with a
standard test network at the inputs instead of the patient's electrodes: an
electrode fault therefore leaves the circuit compliant, and its origin is recorded
by the third labelling level. Limits and test set-up live under `specs:` in the
configuration and are provisional until checked against the IEC standards.

Specifications (stored as `spec_<name>`):
- gain_error         |G / G_nominal - 1| at 10 Hz
- f_low, f_high      -3 dB frequencies relative to the 10 Hz gain [Hz]
- cmrr_db            mains-frequency source through `c_source` with the imbalance
                     network in one lead and the RLD active: source voltage over the
                     input-referred output [dB]
- noise_uvpp         input-referred noise from 0.05 to 150 Hz, 6.6 x rms [uV]
- offset_gain_error  worst gain change at 10 Hz with +-`electrode_offset` at one input
- output_offset      |V(out) - ADC mid-scale| at rest [V]
"""

from __future__ import annotations

import numpy as np

from .circuit import ELECTRODES, CircuitInstance, build_netlist, nominal_instance, spice_init
from .simulate import ac_sweep, expect_plots, interp_response
from .spice import RAW_NAME, run_deck

SPEC_NAMES = (
    "gain_error",
    "f_low",
    "f_high",
    "cmrr_db",
    "noise_uvpp",
    "offset_gain_error",
    "output_offset",
)
_NOISE_BAND = (0.05, 150.0)


def _test_setup(inst: CircuitInstance, cfg: dict) -> tuple[CircuitInstance, dict]:
    """Same circuit, standard network instead of the patient: imbalance in the LA lead."""
    network = cfg["specs"]["test_network"]
    short = {"ehc": 0.0, "rs": 1.0, "rd": 1.0, "cd": 1e-12}
    bench = inst.copy()
    bench.electrodes = {name: dict(short) for name in ELECTRODES}
    bench.electrodes["la"] = {"ehc": 0.0, **{k: float(v) for k, v in network["imbalance"].items()}}
    env = {**cfg["environment"], "c_mains_body": float(network["c_source"]), "r_iso": 1.0}
    return bench, {**cfg, "environment": env}


def corner_frequencies(freq: np.ndarray, h: np.ndarray, f_ref: float = 10.0) -> tuple[float, float]:
    """Lower and upper -3 dB frequencies relative to the gain at `f_ref`.

    NaN when the response never reaches that level around `f_ref` (dead circuit).
    """
    mag = np.abs(h) / max(abs(interp_response(freq, h, f_ref)), 1e-300)
    passband = np.flatnonzero(mag >= 1 / np.sqrt(2))
    if passband.size == 0:
        return float("nan"), float("nan")
    return float(freq[passband[0]]), float(freq[passband[-1]])


def measure_raw_specs(inst: CircuitInstance, cfg: dict) -> dict[str, float]:
    """Simulate the specification tests; `gain` is absolute (see `measure_specs`)."""
    scfg = cfg["specs"]
    offset = float(scfg["electrode_offset"])
    f_mains = float(cfg["environment"]["mains_freq"])
    ecg_on = ["alter @vecgp[acmag]=0.5", "alter @vecgn[acmag]=0.5"]
    ecg_off = ["alter @vecgp[acmag]=0", "alter @vecgn[acmag]=0"]
    control = [
        "set appendwrite",
        "op",
        f"write {RAW_NAME} v(out)",
        *ecg_on,
        ac_sweep(cfg),
        f"write {RAW_NAME} v(out)",
    ]
    for dc in (offset, -offset):
        control += [f"alter vhc_la dc={dc}", "ac lin 1 10 10", f"write {RAW_NAME} v(out)"]
    control += [
        "alter vhc_la dc=0",
        *ecg_off,
        "alter @vmains[acmag]=1",
        f"ac lin 1 {f_mains} {f_mains}",
        f"write {RAW_NAME} v(out)",
        "alter @vmains[acmag]=0",
        f"noise v(out) vecgp dec 50 {_NOISE_BAND[0]} {_NOISE_BAND[1]}",
        f"write {RAW_NAME} onoise_total",  # current plot: noise integrated over the band
    ]
    bench, bench_cfg = _test_setup(inst, cfg)
    plots = run_deck(build_netlist(bench, bench_cfg, control), spiceinit=spice_init(cfg))
    expect_plots(plots, ["Operating Point", *["AC Analysis"] * 4, "Integrated Noise"])
    op, ac, ac_pos, ac_neg, ac_cm, noise = plots

    freq, h = ac["frequency"].real, ac["v(out)"]
    gain = abs(interp_response(freq, h, 10.0))
    safe_gain = max(gain, 1e-12)
    f_low, f_high = corner_frequencies(freq, h)
    gain_mains = max(abs(interp_response(freq, h, f_mains)), 1e-12)
    cm_in = abs(ac_cm["v(out)"][0]) / gain_mains  # input-referred, per volt of source
    # ngspice names the vector "v(onoise_total)" in the raw file
    noise_rms = float(next(v for k, v in noise.vectors.items() if "onoise_total" in k)[0].real)
    adc = cfg["measurement"]["adc"]
    return {
        "gain": float(gain),
        "f_low": f_low,
        "f_high": f_high,
        "cmrr_db": float(-20 * np.log10(max(cm_in, 1e-12))),
        "noise_uvpp": 6.6e6 * noise_rms / safe_gain,
        "offset_gain_error": float(
            max(abs(abs(p["v(out)"][0]) / safe_gain - 1.0) for p in (ac_pos, ac_neg))
        ),
        "output_offset": abs(float(op["v(out)"][0].real) - 0.5 * (adc["vmin"] + adc["vmax"])),
    }


def with_nominal_gain(cfg: dict) -> dict:
    """Config with `specs.nominal_gain` filled in by simulating the nominal circuit."""
    if "nominal_gain" in cfg["specs"]:
        return cfg
    gain = measure_raw_specs(nominal_instance(cfg), cfg)["gain"]
    return {**cfg, "specs": {**cfg["specs"], "nominal_gain": gain}}


def measure_specs(inst: CircuitInstance, cfg: dict) -> dict[str, float]:
    """Specification values of one instance; `cfg` must come from `with_nominal_gain`."""
    raw = measure_raw_specs(inst, cfg)
    raw["gain_error"] = abs(raw.pop("gain") / float(cfg["specs"]["nominal_gain"]) - 1.0)
    return {name: raw[name] for name in SPEC_NAMES}


def spec_limits(cfg: dict) -> dict[str, tuple[str, float]]:
    """Specification -> ("max" | "min", limit)."""
    s = cfg["specs"]
    return {
        "gain_error": ("max", float(s["gain_error_max"])),
        "f_low": ("max", float(s["f_low_max"])),
        "f_high": ("min", float(s["f_high_min"])),
        "cmrr_db": ("min", float(s["cmrr_min_db"])),
        "noise_uvpp": ("max", float(s["noise_max_uvpp"])),
        "offset_gain_error": ("max", float(s["offset_gain_error_max"])),
        "output_offset": ("max", float(s["output_offset_max"])),
    }


def compliance(specs: dict[str, float], cfg: dict) -> dict[str, bool | str]:
    """First labelling level: `ok_<spec>`, `compliant` and the violated specifications.

    A value that cannot be computed (NaN) counts as a violation.
    """
    result: dict[str, bool | str] = {}
    violated = []
    for name, (sense, limit) in spec_limits(cfg).items():
        value = specs[name]
        ok = bool(value <= limit if sense == "max" else value >= limit)
        result[f"ok_{name}"] = ok
        if not ok:
            violated.append(name)
    result["compliant"] = not violated
    result["violated"] = ",".join(violated)
    return result

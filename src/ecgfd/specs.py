"""Clinical specifications of a circuit instance and its compliance (functional severity).

The tests follow IEC 60601-2-25:2011, clause 201.12.4 (docs/circuit.md maps each
specification to its subclause). Specifications are properties of the instrument, so they
are simulated with the standard test networks at the inputs instead of the
patient's electrodes: an electrode fault therefore leaves the circuit compliant,
and its origin is recorded by the third labelling level.

Specifications (stored as `spec_<name>`; "RTI" = referred to the input):
- gain_error         |G / G_nominal - 1| at 10 Hz
- resp_dev_lf        largest |G(f) / G(10 Hz) - 1| from 0.67 to 40 Hz
- resp_min_hf        smallest G(f) / G(10 Hz) from 40 to 150 Hz
- resp_max_hf        largest G(f) / G(10 Hz) from 40 to 500 Hz
- impulse_offset_uv  baseline shift after a 3 mV, 100 ms impulse, RTI [uV]
- impulse_slope_uvs  baseline slope after that impulse, RTI [uV/s]
- cmrr_db            20 V rms through the 100 pF divider (10 V rms unloaded) at 50 and
                     60 Hz, with 51 kOhm || 47 nF in each lead in turn, without and
                     with +-300 mV offset, RLD active: unloaded common-mode voltage
                     over the worst RTI output [dB]
- noise_uvpp         RTI noise from 0.05 to 150 Hz with 51 kOhm || 47 nF in every
                     lead, taken as 6.6 x rms [uV]
- zin_drop           worst signal loss with 620 kOhm || 4.7 nF in series with one lead
                     at 0.67 and 40 Hz, with +-300 mV offset
- offset_gain_error  worst gain change at 10 Hz with +-300 mV at one input
- input_range_mv     input amplitude that still fits the output range, given the
                     gain and the output offset [mV]
"""

from __future__ import annotations

import numpy as np

from .circuit import ELECTRODES, CircuitInstance, Stimulus, build_netlist, nominal_instance
from .simulate import ac_sweep, expect_plots, interp_response
from .spice import RAW_NAME, run_deck

# specification -> (sense of the limit, key of the limit under `specs:` in the config)
SPECS: dict[str, tuple[str, str]] = {
    "gain_error": ("max", "gain_error_max"),
    "resp_dev_lf": ("max", "resp_dev_lf_max"),
    "resp_min_hf": ("min", "resp_min_hf_min"),
    "resp_max_hf": ("max", "resp_max_hf_max"),
    "impulse_offset_uv": ("max", "impulse_offset_max_uv"),
    "impulse_slope_uvs": ("max", "impulse_slope_max_uvs"),
    "cmrr_db": ("min", "cmrr_min_db"),
    "noise_uvpp": ("max", "noise_max_uvpp"),
    "zin_drop": ("max", "zin_drop_max"),
    "offset_gain_error": ("max", "offset_gain_error_max"),
    "input_range_mv": ("min", "input_range_min_mv"),
}
SPEC_NAMES = tuple(SPECS)

_NOISE_BAND = (0.05, 150.0)
# Table 201.107: test A, tests B and C (lower limit), tests B to D (upper limit)
_LF_BAND, _HF_MIN_BAND, _HF_MAX_BAND = (0.67, 40.0), (40.0, 150.0), (40.0, 500.0)
_SHORT = {"rs": 1.0, "rd": 1.0, "cd": 1e-12}  # lead connected directly to the test source
# impulse test: start, time after the impulse where the baseline is read, end [s]
_T_START, _T_SETTLE, _T_END = 0.02, 0.05, 0.5


def _bench(inst: CircuitInstance, cfg: dict) -> tuple[CircuitInstance, dict]:
    """Same circuit on the test bench: leads shorted to the body, standard CM source."""
    source = cfg["specs"]["test_network"]["cm_source"]
    bench = inst.copy()
    bench.electrodes = {name: {"ehc": 0.0, **_SHORT} for name in ELECTRODES}
    env = {
        **cfg["environment"],
        "c_mains_body": float(source["c_series"]),
        "c_body_earth": float(source["c_shunt"]),
        "r_iso": 1.0,  # amplifier common tied to earth
    }
    return bench, {**cfg, "environment": env}


def _network(lead: str, values: dict) -> list[str]:
    """Commands that put Rd || Cd of `values` in series with a lead."""
    return [f"alter rd_{lead} = {float(values['rd'])}", f"alter cd_{lead} = {float(values['cd'])}"]


def _output_half_range(cfg: dict) -> tuple[float, float]:
    """(mid-scale, largest symmetric swing that fits both the ADC and the output stage)."""
    adc, supply, hr = cfg["measurement"]["adc"], cfg["supply"], float(cfg["opamp"]["headroom"])
    mid = 0.5 * (adc["vmin"] + adc["vmax"])
    top = min(adc["vmax"], supply["vcc"] - hr)
    bottom = max(adc["vmin"], supply["vee"] + hr)
    return mid, min(top - mid, mid - bottom)


def measure_raw_specs(inst: CircuitInstance, cfg: dict) -> dict[str, float]:
    """Simulate the specification tests; `gain` is absolute (see `measure_specs`)."""
    scfg = cfg["specs"]
    net = scfg["test_network"]
    offset = float(scfg["electrode_offset"])
    f_mains = [float(f) for f in net["cm_source"]["freqs"]]
    f_zin = [float(f) for f in net["input_impedance"]["freqs"]]
    write = f"write {RAW_NAME} v(out)"
    ecg_on = ["alter @vecgp[acmag]=0.5", "alter @vecgn[acmag]=0.5"]
    ecg_off = ["alter @vecgp[acmag]=0", "alter @vecgn[acmag]=0"]

    control = ["set appendwrite", "op", write, *ecg_on, ac_sweep(cfg), write]
    for dc in (offset, -offset):  # gain with electrode offset
        control += [f"alter vhc_la dc={dc}", "ac lin 1 10 10", write]
    # common-mode rejection: imbalance in one lead at a time, without and with offset
    control += [*ecg_off, "alter @vmains[acmag]=1"]
    for lead in ELECTRODES:
        control += _network(lead, net["imbalance"])
        for dc in (0.0, offset, -offset):
            control += [f"alter vhc_{lead} dc={dc}", f"ac lin 2 {f_mains[0]} {f_mains[1]}", write]
        control += [f"alter vhc_{lead} dc=0", *_network(lead, _SHORT)]
    control.append("alter @vmains[acmag]=0")
    # noise: the network in every lead
    for lead in ELECTRODES:
        control += _network(lead, net["imbalance"])
    control += [
        f"noise v(out) vecgp dec 50 {_NOISE_BAND[0]} {_NOISE_BAND[1]}",
        f"write {RAW_NAME} onoise_total",  # current plot: noise integrated over the band
    ]
    for lead in ELECTRODES:
        control += _network(lead, _SHORT)
    # input impedance
    control += [*_network("la", net["input_impedance"]), *ecg_on]
    for dc in (offset, -offset):
        control += [f"alter vhc_la dc={dc}", f"ac lin 2 {f_zin[0]} {f_zin[1]}", write]
    control += [
        "alter vhc_la dc=0",
        *_network("la", _SHORT),
        f"tran 2.5e-4 {_T_END} 0 2.5e-4",
        write,
    ]
    n_cm = 3 * len(ELECTRODES)
    amp, width = float(scfg["impulse"]["amplitude"]), float(scfg["impulse"]["width"])
    t_fall = _T_START + width
    impulse = (
        np.array([0.0, _T_START, _T_START + 1e-4, t_fall, t_fall + 1e-4, _T_END]),
        np.array([0.0, 0.0, amp, amp, 0.0, 0.0]),
    )
    bench, bench_cfg = _bench(inst, cfg)
    plots = run_deck(build_netlist(bench, bench_cfg, control, Stimulus(ecg=impulse)))
    expect_plots(
        plots,
        [
            "Operating Point",
            *["AC Analysis"] * (3 + n_cm),
            "Integrated Noise",
            *["AC Analysis"] * 2,
            "Transient",
        ],
    )
    op, ac, ac_pos, ac_neg = plots[:4]
    cm_plots, noise, zin_plots, tran = plots[4 : 4 + n_cm], plots[4 + n_cm], plots[-3:-1], plots[-1]

    freq, h = ac["frequency"].real, ac["v(out)"]
    gain = abs(interp_response(freq, h, 10.0))
    safe_gain = max(gain, 1e-12)
    ratio = np.abs(h) / safe_gain
    lf = (freq >= _LF_BAND[0]) & (freq <= _LF_BAND[1])
    hf_min = (freq >= _HF_MIN_BAND[0]) & (freq <= _HF_MIN_BAND[1])
    hf_max = (freq >= _HF_MAX_BAND[0]) & (freq <= _HF_MAX_BAND[1])

    # common mode: input-referred output per volt of source, against the unloaded node voltage
    source = net["cm_source"]
    unloaded = source["c_series"] / (source["c_series"] + source["c_shunt"])
    gain_mains = np.array([max(abs(interp_response(freq, h, f)), 1e-12) for f in f_mains])
    cm_in = max(float(np.max(np.abs(p["v(out)"]) / gain_mains)) for p in cm_plots)

    # ngspice names the vector "v(onoise_total)" in the raw file
    noise_rms = float(next(v for k, v in noise.vectors.items() if "onoise_total" in k)[0].real)

    direct = np.array([abs(interp_response(freq, h, f)) for f in f_zin])
    zin_drop = max(
        float(np.max(1.0 - np.abs(p["v(out)"]) / np.maximum(direct, 1e-12))) for p in zin_plots
    )

    t_read = t_fall + _T_SETTLE
    v0, v1, v2 = np.interp([0.5 * _T_START, t_read, _T_END], tran["time"], tran["v(out)"])

    mid, half_range = _output_half_range(cfg)
    out_offset = abs(float(op["v(out)"][0].real) - mid)
    return {
        "gain": float(gain),
        "resp_dev_lf": float(np.max(np.abs(ratio[lf] - 1.0))),
        "resp_min_hf": float(ratio[hf_min].min()),
        "resp_max_hf": float(ratio[hf_max].max()),
        "impulse_offset_uv": float(1e6 * abs(v1 - v0) / safe_gain),
        "impulse_slope_uvs": float(1e6 * abs(v2 - v1) / (_T_END - t_read) / safe_gain),
        "cmrr_db": float(20 * np.log10(unloaded / max(cm_in, 1e-12))),
        "noise_uvpp": 6.6e6 * noise_rms / safe_gain,
        "zin_drop": zin_drop,
        "offset_gain_error": float(
            max(abs(abs(p["v(out)"][0]) / safe_gain - 1.0) for p in (ac_pos, ac_neg))
        ),
        "input_range_mv": float(1e3 * (half_range - out_offset) / safe_gain),
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
    return {name: (sense, float(cfg["specs"][key])) for name, (sense, key) in SPECS.items()}


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

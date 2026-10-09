"""Clinical specifications of a circuit and its compliance (functional severity).

The tests follow IEC 60601-2-25:2011, clause 201.12.4 (docs/circuit.md maps each
specification to its subclause). Specifications are properties of the instrument, so they
are simulated on the test bench, with the standard test networks at the inputs
instead of the patient's electrodes: an electrode fault therefore leaves the circuit
compliant, and its origin is recorded by the `origin` tag of the fault.

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

from dataclasses import dataclass

import numpy as np
from spicefault import (
    Circuit,
    Experiment,
    Measurement,
    OperatingCondition,
    SimulationConfig,
    SimulationResult,
    Specification,
)
from spicefault.measurements import interp_response

from .population import ELECTRODES
from .selftest import OUT, ac_sweep

# specification -> (which bound is limited, key of the limit under `specs:` in the config)
SPECS: dict[str, tuple[str, str]] = {
    "spec_gain_error": ("maximum", "gain_error_max"),
    "spec_resp_dev_lf": ("maximum", "resp_dev_lf_max"),
    "spec_resp_min_hf": ("minimum", "resp_min_hf_min"),
    "spec_resp_max_hf": ("maximum", "resp_max_hf_max"),
    "spec_impulse_offset_uv": ("maximum", "impulse_offset_max_uv"),
    "spec_impulse_slope_uvs": ("maximum", "impulse_slope_max_uvs"),
    "spec_cmrr_db": ("minimum", "cmrr_min_db"),
    "spec_noise_uvpp": ("maximum", "noise_max_uvpp"),
    "spec_zin_drop": ("maximum", "zin_drop_max"),
    "spec_offset_gain_error": ("maximum", "offset_gain_error_max"),
    "spec_input_range_mv": ("minimum", "input_range_min_mv"),
}
SPEC_NAMES = tuple(SPECS)

_NOISE_BAND = (0.05, 150.0)
# Table 201.107: test A, tests B and C (lower limit), tests B to D (upper limit)
_LF_BAND, _HF_MIN_BAND, _HF_MAX_BAND = (0.67, 40.0), (40.0, 150.0), (40.0, 500.0)
_SHORT = {"rs": 1.0, "rd": 1.0, "cd": 1e-12}  # lead connected directly to the test source
# impulse test: start, edge, time after the impulse where the baseline is read, end [s]
_T_START, _T_EDGE, _T_SETTLE, _T_END = 0.02, 1e-4, 0.05, 0.5
_N_CM = 3 * len(ELECTRODES)  # common-mode runs: each lead, without and with +-offset


def _network(lead: str, values: dict) -> list[str]:
    """Commands that put Rd || Cd of `values` in series with a lead."""
    return [f"alter rd_{lead} = {float(values['rd'])}", f"alter cd_{lead} = {float(values['cd'])}"]


def _analyses(cfg: dict) -> tuple:
    """The specification tests, one after another on the same circuit."""
    net = cfg["specs"]["test_network"]
    offset = float(cfg["specs"]["electrode_offset"])
    f_mains, f_zin = net["cm_source"]["freqs"], net["input_impedance"]["freqs"]
    ecg_on = ["alter @vecgp[acmag]=0.5", "alter @vecgn[acmag]=0.5"]
    ecg_off = ["alter @vecgp[acmag]=0", "alter @vecgn[acmag]=0"]

    run: list = [("op", OUT), *ecg_on, (ac_sweep(cfg), OUT)]
    for dc in (offset, -offset):  # gain with electrode offset
        run += [f"alter vhc_la dc={dc}", ("ac lin 1 10 10", OUT)]
    # common-mode rejection: imbalance in one lead at a time, without and with offset
    run += [*ecg_off, "alter @vmains[acmag]=1"]
    for lead in ELECTRODES:
        run += _network(lead, net["imbalance"])
        for dc in (0.0, offset, -offset):
            run += [f"alter vhc_{lead} dc={dc}", (f"ac lin 2 {f_mains[0]} {f_mains[1]}", OUT)]
        run += [f"alter vhc_{lead} dc=0", *_network(lead, _SHORT)]
    run.append("alter @vmains[acmag]=0")
    # noise integrated over the band, with the network in every lead
    for lead in ELECTRODES:
        run += _network(lead, net["imbalance"])
    noise = f"noise v(out) vecgp dec 50 {_NOISE_BAND[0]} {_NOISE_BAND[1]}"
    run.append((noise, ("onoise_total",), "integrated"))
    for lead in ELECTRODES:
        run += _network(lead, _SHORT)
    # input impedance
    run += [*_network("la", net["input_impedance"]), *ecg_on]
    for dc in (offset, -offset):
        run += [f"alter vhc_la dc={dc}", (f"ac lin 2 {f_zin[0]} {f_zin[1]}", OUT)]
    # impulse response
    run += ["alter vhc_la dc=0", *_network("la", _SHORT), (f"tran 2.5e-4 {_T_END} 0 2.5e-4", OUT)]
    return tuple(run)


def _settings(cfg: dict) -> dict[tuple[str, str], float]:
    """The circuit on the bench: leads on the test source, standard common-mode source."""
    source, impulse = cfg["specs"]["test_network"]["cm_source"], cfg["specs"]["impulse"]
    settings = {
        ("Cpow", "value"): float(source["c_series"]),
        ("Cbody", "value"): float(source["c_shunt"]),
        ("Riso", "value"): 1.0,  # amplifier common tied to earth
    }
    for name in ELECTRODES:
        settings[f"Vhc_{name}", "dc"] = 0.0
        settings.update({(f"{k.capitalize()}_{name}", "value"): v for k, v in _SHORT.items()})
    for half in ("Vecgp", "Vecgn"):  # the impulse, half in each side of lead I
        settings.update(
            {
                (half, "pulse.v2"): 0.5 * float(impulse["amplitude"]),
                (half, "pulse.td"): _T_START,
                (half, "pulse.tr"): _T_EDGE,
                (half, "pulse.tf"): _T_EDGE,
                (half, "pulse.pw"): float(impulse["width"]) - _T_EDGE,
            }
        )
    return settings


def _output_half_range(circuit: Circuit, cfg: dict) -> tuple[float, float]:
    """(mid-scale, largest symmetric swing that fits both the ADC and the output stage)."""
    adc = cfg["measurement"]["adc"]
    vcc, vee = (circuit.component(name).parameters["dc"] for name in ("Vcc", "Vee"))
    stage = next(c for c in circuit.components() if c.kind == "X" and c.nodes[2] == "out")
    hr = stage.parameters["hr"]
    mid = 0.5 * (adc["vmin"] + adc["vmax"])
    top, bottom = min(adc["vmax"], vcc - hr), max(adc["vmin"], vee + hr)
    return mid, min(top - mid, mid - bottom)


@dataclass(frozen=True)
class BenchSpecs:
    """Turns the plots of the bench simulation into the specification values and the gain."""

    f_mains: tuple[float, ...]
    f_zin: tuple[float, ...]
    cm_unloaded: float  # unloaded common-mode voltage per volt of source
    impulse_width: float
    mid: float
    half_range: float
    nominal_gain: float

    def __call__(self, result: SimulationResult) -> dict[str, float]:
        plots = result.plots
        op, ac, ac_pos, ac_neg = plots[:4]
        cm_plots, noise = plots[4 : 4 + _N_CM], plots[4 + _N_CM]
        zin_plots, tran = plots[-3:-1], plots[-1]

        freq, h = ac["frequency"].real, ac["v(out)"]
        gain = abs(interp_response(freq, h, 10.0))
        safe_gain = max(gain, 1e-12)
        ratio = np.abs(h) / safe_gain
        lf = (freq >= _LF_BAND[0]) & (freq <= _LF_BAND[1])
        hf_min = (freq >= _HF_MIN_BAND[0]) & (freq <= _HF_MIN_BAND[1])
        hf_max = (freq >= _HF_MAX_BAND[0]) & (freq <= _HF_MAX_BAND[1])

        # common mode: input-referred output per volt of source
        gain_mains = np.array([max(abs(interp_response(freq, h, f)), 1e-12) for f in self.f_mains])
        cm_in = max(float(np.max(np.abs(p["v(out)"]) / gain_mains)) for p in cm_plots)

        # ngspice names the vector "v(onoise_total)" in the raw file
        noise_rms = float(next(v for k, v in noise.vectors.items() if "onoise_total" in k)[0].real)

        direct = np.array([abs(interp_response(freq, h, f)) for f in self.f_zin])
        zin_drop = max(
            float(np.max(1.0 - np.abs(p["v(out)"]) / np.maximum(direct, 1e-12))) for p in zin_plots
        )

        t_read = _T_START + self.impulse_width + _T_SETTLE
        v0, v1, v2 = np.interp([0.5 * _T_START, t_read, _T_END], tran["time"], tran["v(out)"])
        out_offset = abs(float(op["v(out)"][0].real) - self.mid)
        return {
            "gain": float(gain),
            "spec_gain_error": abs(float(gain) / self.nominal_gain - 1.0),
            "spec_resp_dev_lf": float(np.max(np.abs(ratio[lf] - 1.0))),
            "spec_resp_min_hf": float(ratio[hf_min].min()),
            "spec_resp_max_hf": float(ratio[hf_max].max()),
            "spec_impulse_offset_uv": float(1e6 * abs(v1 - v0) / safe_gain),
            "spec_impulse_slope_uvs": float(1e6 * abs(v2 - v1) / (_T_END - t_read) / safe_gain),
            "spec_cmrr_db": float(20 * np.log10(self.cm_unloaded / max(cm_in, 1e-12))),
            "spec_noise_uvpp": 6.6e6 * noise_rms / safe_gain,
            "spec_zin_drop": zin_drop,
            "spec_offset_gain_error": float(
                max(abs(abs(p["v(out)"][0]) / safe_gain - 1.0) for p in (ac_pos, ac_neg))
            ),
            "spec_input_range_mv": float(1e3 * (self.half_range - out_offset) / safe_gain),
        }


def bench(circuit: Circuit, cfg: dict) -> OperatingCondition:
    """The circuit on the test bench; `cfg` must come from `with_nominal_gain`."""
    net = cfg["specs"]["test_network"]
    source = net["cm_source"]
    mid, half_range = _output_half_range(circuit, cfg)
    specs = BenchSpecs(
        f_mains=tuple(float(f) for f in source["freqs"]),
        f_zin=tuple(float(f) for f in net["input_impedance"]["freqs"]),
        cm_unloaded=source["c_series"] / (source["c_series"] + source["c_shunt"]),
        impulse_width=float(cfg["specs"]["impulse"]["width"]),
        mid=mid,
        half_range=half_range,
        nominal_gain=float(cfg["specs"]["nominal_gain"]),
    )
    return OperatingCondition(
        "bench",
        settings=_settings(cfg),
        config=SimulationConfig(analyses=_analyses(cfg)),
        measurements=(Measurement.group(SPEC_NAMES, specs, "specifications"),),
        waveform=False,  # its transient is a step of the impulse test, not a response to keep
    )


def with_nominal_gain(circuit: Circuit, cfg: dict) -> dict:
    """Config with `specs.nominal_gain` filled in by simulating the nominal circuit."""
    if "nominal_gain" in cfg["specs"]:
        return cfg
    condition = bench(circuit, {**cfg, "specs": {**cfg["specs"], "nominal_gain": 1.0}})
    nominal = Experiment(circuit, conditions=(condition,)).nominal()["bench"].result
    if not nominal.ok:
        raise RuntimeError(f"nominal circuit: {nominal.status.value}: {nominal.message}")
    gain = condition.measurements[0].function(nominal)["gain"]
    return {**cfg, "specs": {**cfg["specs"], "nominal_gain": gain}}


def specifications(cfg: dict) -> list[Specification]:
    """The limits of the configuration, as `spicefault` specifications."""
    return [
        Specification(name, **{bound: float(cfg["specs"][key])})
        for name, (bound, key) in SPECS.items()
    ]

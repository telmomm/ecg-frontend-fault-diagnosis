"""Self-test measurements: what the instrument can measure on itself, in service.

One simulation with the patient's electrodes: operating point, three frequency
responses (calibration source, RLD reference, lead-off current) and the response to
the calibration pulse. The results are noise-free; `measurement.py` adds the
acquisition chain.
"""

from __future__ import annotations

from spicefault import Measurement, OperatingCondition, SimulationConfig, Waveform

OUT = ("v(out)",)
# position of each frequency response among the plots of the simulation
AC_DIFF, AC_CM, AC_LEAD_OFF = 1, 2, 3


def ac_sweep(cfg: dict) -> str:
    s = cfg["measurement"]["ac"]["sweep"]
    return f"ac dec {s['points_per_decade']} {s['fstart']} {s['fstop']}"


def pulse_waveform(cfg: dict) -> Waveform:
    """Calibration-pulse response at the ADC sample instants."""
    pulse = cfg["measurement"]["pulse"]
    return Waveform("v(out)", float(pulse["fs"]), float(pulse["duration"]))


def service(cfg: dict) -> OperatingCondition:
    """The circuit in service, running its self-test."""
    m = cfg["measurement"]
    pulse, fs = m["pulse"], float(m["pulse"]["fs"])
    analyses: list = [("op", tuple(f"v({node})" for node in m["dc_nodes"]))]
    for source in ("vcal", "vcmt", "ilo"):
        analyses += [f"alter @{source}[acmag]=1", (ac_sweep(cfg), OUT), f"alter @{source}[acmag]=0"]
    analyses.append((f"tran {1 / fs} {pulse['duration']} 0 {0.25 / fs}", OUT))

    measurements = [Measurement.value(f"v({node})", name=f"dc_{node}") for node in m["dc_nodes"]]
    for f in m["ac"]["freqs"]:
        measurements += [
            Measurement.magnitude("v(out)", f, AC_DIFF, f"acd_mag_{f:g}"),
            Measurement.phase("v(out)", f, AC_DIFF, f"acd_ph_{f:g}"),
            Measurement.magnitude("v(out)", f, AC_CM, f"acc_mag_{f:g}"),
        ]
    measurements += [
        Measurement.magnitude("v(out)", f, AC_LEAD_OFF, f"zlo_mag_{f:g}")
        for f in m["lead_off"]["freqs"]
    ]
    return OperatingCondition(
        "service",
        settings={
            ("Vcal", "pulse.v2"): float(pulse["amplitude"]),
            ("Vcal", "pulse.td"): float(pulse["delay"]),
            ("Vcal", "pulse.tr"): float(pulse["edge"]),
            ("Vcal", "pulse.tf"): float(pulse["edge"]),
            ("Vcal", "pulse.pw"): float(pulse["width"]),
        },
        config=SimulationConfig(analyses=tuple(analyses)),
        measurements=tuple(measurements),
    )

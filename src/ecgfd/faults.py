"""Fault catalogue: the single faults of the study, as `spicefault` faults.

Every fault is applied on top of a Monte Carlo draw, so the remaining components
keep their manufacturing spread. A fault only says what was injected; whether the
circuit still meets its specifications is decided separately (see `specs.py`).

Two tags go with each fault into the dataset: `component`, the part to locate
(designator, or electrode(s) joined by "+"), and `origin`, where the defect is
(`circuit` or `electrode`; empty for the healthy circuit).
"""

from __future__ import annotations

from spicefault import Circuit, Fault
from spicefault.faults import (
    CompositeFault,
    OpenCircuit,
    ParametricFault,
    SeriesResistanceFault,
    ShortCircuit,
)

from .circuit import ELECTRODES, front_end

TAGS = ("component", "origin")


def fault_catalogue(circuit: Circuit, cfg: dict) -> list[Fault]:
    """All single-fault conditions of a circuit, in a fixed order (no healthy)."""
    f = cfg["faults"]
    passives, opamps, inas = front_end(circuit)
    faults: list[Fault] = []

    def parametric(kind, component, instance, parameter, values, rule):
        tags = {"component": component, "origin": "circuit"}
        faults.extend(
            ParametricFault(instance, parameter, fault_type=kind, tags=tags, **{rule: float(v)})
            for v in values
        )

    for name in passives:
        tags = {"component": name, "origin": "circuit"}
        faults.append(OpenCircuit(name, r_open=f["r_open"], tags=tags))
        faults.append(ShortCircuit(name, r_short=f["r_short"], tags=tags))
        parametric("parametric", name, name, "value", f["parametric"], "deviation")
        if name.startswith("C"):  # ageing: capacitance loss together with series resistance
            faults += [
                CompositeFault(
                    "cap_degradation",
                    [
                        ParametricFault(name, deviation=-float(d["c_loss"])),
                        SeriesResistanceFault(name, d["esr"]),
                    ],
                    magnitude=float(d["c_loss"]),
                    unit="capacitance loss",
                    tags=tags,
                )
                for d in f["cap_degradation"]
            ]
    for x in opamps:
        parametric("opamp_vos", x[1:], x, "vos", f["opamp"]["vos"], "fault_value")
        parametric("opamp_aol", x[1:], x, "aol", f["opamp"]["aol_factor"], "factor")
    for x in inas:
        parametric("ina_vos", x[1:], x, "vos", f["ina"]["vos"], "fault_value")
        parametric("ina_cmrr", x[1:], x, "cmrr_db", f["ina"]["cmrr_db"], "fault_value")
        parametric("ina_gain", x[1:], x, "gerr", f["ina"]["gain_error"], "fault_value")

    for name in ELECTRODES:
        faults.append(
            ParametricFault(
                f"Rs_{name}",
                fault_value=f["electrode"]["r_off"],
                fault_type="electrode_off",
                tags={"component": name, "origin": "electrode"},
            )
        )
    # gel drying: Rd * k, Cd / k. On one electrode it is an imbalance; "la+ra" is balanced
    for target in (*ELECTRODES, "la+ra"):
        faults += [
            CompositeFault(
                "electrode_high_z",
                [
                    part
                    for name in target.split("+")
                    for part in (
                        ParametricFault(f"Rd_{name}", factor=float(k)),
                        ParametricFault(f"Cd_{name}", divisor=float(k)),
                    )
                ],
                magnitude=float(k),
                unit="factor",
                tags={"component": target, "origin": "electrode"},
            )
            for k in f["electrode"]["high_z_factor"]
        ]
    return faults

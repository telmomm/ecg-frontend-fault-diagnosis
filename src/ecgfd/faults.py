"""Fault catalogue: the single faults of the study, as a `spicefault` fault universe.

Every fault is applied on top of a Monte Carlo draw, so the remaining components
keep their manufacturing spread. A fault only says what was injected; whether the
circuit still meets its specifications is decided separately (see `specs.py`).

Two tags go with each fault into the dataset: `component`, the part to locate
(designator, or electrode(s) joined by "+"), and `origin`, where the defect is
(`circuit` or `electrode`; empty for the healthy circuit).
"""

from __future__ import annotations

from spicefault import Circuit, Component, Fault
from spicefault.faults import (
    CompositeFault,
    FaultRule,
    FaultUniverse,
    ParametricFault,
    SeriesResistanceFault,
    open_rule,
    parametric_rule,
    short_rule,
)

from .circuit import ELECTRODES, front_end

TAGS = ("component", "origin")


def _high_z(electrodes: list[str], factor: float) -> Fault:
    """Gel drying of the named electrodes: Rd * factor, Cd / factor."""
    parts = [
        part
        for name in electrodes
        for part in (
            ParametricFault(f"Rd_{name}", factor=factor),
            ParametricFault(f"Cd_{name}", divisor=factor),
        )
    ]
    return CompositeFault("electrode_high_z", parts, magnitude=factor, unit="factor")


def fault_universe(circuit: Circuit, cfg: dict) -> FaultUniverse:
    """The faults of one component or one electrode, by rules; see its `coverage_matrix`."""
    f = cfg["faults"]
    passives = tuple(front_end(circuit)[0])

    def ageing(c: Component) -> list[Fault]:
        """Capacitance loss together with series resistance."""
        return [
            CompositeFault(
                "cap_degradation",
                [
                    ParametricFault(c.name, deviation=-float(d["c_loss"])),
                    SeriesResistanceFault(c.name, d["esr"]),
                ],
                magnitude=float(d["c_loss"]),
                unit="capacitance loss",
            )
            for d in f["cap_degradation"]
        ]

    def high_z(c: Component) -> list[Fault]:
        return [_high_z([c.name[3:]], float(k)) for k in f["electrode"]["high_z_factor"]]

    part = {"components": passives, "tags": lambda c: {"component": c.name, "origin": "circuit"}}
    amplifier = {"kinds": "X", "tags": lambda c: {"component": c.name[1:], "origin": "circuit"}}
    opamp, ina = {**amplifier, "models": ("opamp",)}, {**amplifier, "models": ("ina",)}
    electrode = {"tags": lambda c: {"component": c.name[3:], "origin": "electrode"}}
    rules = [
        open_rule("RC", f["r_open"], **part),
        short_rule("RC", f["r_short"], **part),
        parametric_rule(f["parametric"], "RC", **part),
        FaultRule("cap_degradation", ageing, "C", **part),
        parametric_rule(parameter="vos", fault_values=f["opamp"]["vos"], fault_type="opamp_vos",
                        **opamp),
        parametric_rule(parameter="aol", factors=f["opamp"]["aol_factor"], fault_type="opamp_aol",
                        **opamp),
        parametric_rule(parameter="vos", fault_values=f["ina"]["vos"], fault_type="ina_vos", **ina),
        parametric_rule(parameter="cmrr_db", fault_values=f["ina"]["cmrr_db"],
                        fault_type="ina_cmrr", **ina),
        parametric_rule(parameter="gerr", fault_values=f["ina"]["gain_error"],
                        fault_type="ina_gain", **ina),
        parametric_rule(fault_values=[f["electrode"]["r_off"]], fault_type="electrode_off",
                        components=tuple(f"Rs_{name}" for name in ELECTRODES), **electrode),
        FaultRule("electrode_high_z", high_z,
                  components=tuple(f"Rd_{name}" for name in ELECTRODES), **electrode),
    ]  # fmt: skip
    return FaultUniverse(circuit, rules)


def fault_catalogue(circuit: Circuit, cfg: dict) -> list[Fault]:
    """All single-fault conditions of a circuit, in a fixed order (no healthy).

    The universe, plus the drying of both measuring electrodes at once: on one
    electrode it is an imbalance, on "la+ra" it is balanced. The random stream of a
    case depends on the place of its fault here: circuit faults first, in netlist
    order, then detached electrodes, then dried ones.
    """
    balanced = [
        _high_z(["la", "ra"], float(k)).with_tags({"component": "la+ra", "origin": "electrode"})
        for k in cfg["faults"]["electrode"]["high_z_factor"]
    ]
    single = fault_universe(circuit, cfg).selected()
    place = {"circuit": 0, "electrode_off": 1, "electrode_high_z": 2}
    single = sorted(single, key=lambda f: place.get(f.tags["origin"], place.get(f.fault_type)))
    return [*single, *balanced]

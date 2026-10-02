"""Fault model: catalogue of single faults and their injection into a circuit instance.

Every fault is applied on top of a Monte Carlo draw, so the remaining components
keep their manufacturing spread. A fault only says what was injected; whether the
circuit still meets its specifications is decided separately (see `specs.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .circuit import CircuitInstance, get_circuit

CIRCUIT_KINDS = (
    "open",
    "short",
    "parametric",
    "cap_degradation",
    "opamp_vos",
    "opamp_aol",
    "ina_vos",
    "ina_cmrr",
    "ina_gain",
)
ELECTRODE_KINDS = ("electrode_off", "electrode_high_z")
HARD_KINDS = ("open", "short", "electrode_off")


@dataclass(frozen=True)
class Fault:
    kind: str  # "healthy" or one of CIRCUIT_KINDS / ELECTRODE_KINDS
    target: str = ""  # component designator, or electrode(s) joined by "+"
    level: float = 0.0  # relative deviation, capacitance loss, offset [V], dB or factor

    @property
    def id(self) -> str:
        if self.kind == "healthy":
            return "healthy"
        if self.kind in HARD_KINDS:
            return f"{self.target}:{self.kind}"
        return f"{self.target}:{self.kind}:{self.level:+g}"

    @property
    def origin(self) -> str:
        """Third labelling level: where the injected defect is."""
        if self.kind == "healthy":
            return "none"
        return "electrode" if self.kind in ELECTRODE_KINDS else "circuit"

    def apply(self, inst: CircuitInstance, cfg: dict) -> CircuitInstance:
        """Return a copy of `inst` with this fault injected."""
        out = inst.copy()
        fcfg = cfg["faults"]
        if self.kind == "healthy":
            return out
        if self.kind == "open":
            out.series_r[self.target] = float(fcfg["r_open"])
        elif self.kind == "short":
            out.parallel_r[self.target] = float(fcfg["r_short"])
        elif self.kind == "parametric":
            out.passives[self.target] *= 1.0 + self.level
        elif self.kind == "cap_degradation":
            esr = next(d["esr"] for d in fcfg["cap_degradation"] if d["c_loss"] == self.level)
            out.passives[self.target] *= 1.0 - self.level
            out.series_r[self.target] = float(esr)
        elif self.kind == "opamp_vos":
            out.opamps[self.target]["vos"] = self.level
        elif self.kind == "opamp_aol":
            out.opamps[self.target]["aol"] *= self.level
        elif self.kind == "ina_vos":
            out.inas[self.target]["vos"] = self.level
        elif self.kind == "ina_cmrr":
            out.inas[self.target]["cmrr_db"] = self.level
        elif self.kind == "ina_gain":
            out.inas[self.target]["gain_error"] = self.level
        elif self.kind == "electrode_off":
            out.electrodes[self.target]["rs"] = float(fcfg["electrode"]["r_off"])
        elif self.kind == "electrode_high_z":
            for name in self.target.split("+"):
                out.electrodes[name]["rd"] *= self.level
                out.electrodes[name]["cd"] /= self.level
        else:
            raise ValueError(f"unknown fault kind: {self.kind}")
        return out


HEALTHY = Fault("healthy")


def fault_catalogue(cfg: dict) -> list[Fault]:
    """All single-fault conditions of the selected circuit, in a fixed order (no healthy)."""
    fcfg = cfg["faults"]
    circuit = get_circuit(cfg)
    faults: list[Fault] = []
    for p in circuit.passives:
        faults.append(Fault("open", p.name))
        faults.append(Fault("short", p.name))
        faults += [Fault("parametric", p.name, float(d)) for d in fcfg["parametric"]]
        if p.kind == "C":
            faults += [
                Fault("cap_degradation", p.name, float(d["c_loss"]))
                for d in fcfg["cap_degradation"]
            ]
    for u in circuit.opamps:
        faults += [Fault("opamp_vos", u.name, float(v)) for v in fcfg["opamp"]["vos"]]
        faults += [Fault("opamp_aol", u.name, float(k)) for k in fcfg["opamp"]["aol_factor"]]
    for u in circuit.inas:
        faults += [Fault("ina_vos", u.name, float(v)) for v in fcfg["ina"]["vos"]]
        faults += [Fault("ina_cmrr", u.name, float(v)) for v in fcfg["ina"]["cmrr_db"]]
        faults += [Fault("ina_gain", u.name, float(v)) for v in fcfg["ina"]["gain_error"]]
    for name in ("la", "ra", "rl"):
        faults.append(Fault("electrode_off", name))
    # single-electrode high impedance is an imbalance; "la+ra" is the balanced case
    for target in ("la", "ra", "rl", "la+ra"):
        faults += [
            Fault("electrode_high_z", target, float(k)) for k in fcfg["electrode"]["high_z_factor"]
        ]
    return faults

"""The healthy population: normal variation of the circuits and of their electrodes.

The circuits themselves are the netlists in `circuits/` (`integrated.cir`, the main
one, and `reference.cir`; see docs/circuit.md). This module loads them, says which of
their components make the front-end, and defines how healthy units differ from the
nominal netlist: component tolerances, amplifier offsets and gains, and the
skin-electrode interfaces. The spreads are in the configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from spicefault import Circuit, VariationSet
from spicefault.netlist import Netlist
from spicefault.variation import (
    CustomVariation,
    Draw,
    JointVariation,
    ToleranceVariation,
    instance_tolerances,
    log_uniform_factor,
    tolerances,
)

from .config import REPO_ROOT

CIRCUITS = ("integrated", "reference")
ELECTRODES = ("la", "ra", "rl")


def load_circuit(cfg: dict) -> Circuit:
    return Circuit.from_netlist(REPO_ROOT / "circuits" / f"{cfg['circuit']}.cir")


def front_end(circuit: Circuit) -> tuple[list[str], list[str], list[str]]:
    """(passives, op-amps, instrumentation amplifiers) of the front-end, in netlist order.

    The passives are the numbered resistors and capacitors (R1, C3, ...); the
    electrodes and the patient coupling have other names.
    """
    components = circuit.components()
    passives = [c.name for c in components if re.fullmatch(r"[RC]\d+", c.name)]
    opamps, inas = ([c.name for c in components if c.model == model] for model in ("opamp", "ina"))
    return passives, opamps, inas


def _random_sign(rng: np.random.Generator, nominal: float) -> float:
    return 1.0 if rng.uniform() < 0.5 else -1.0


@dataclass(frozen=True)
class ElectrodeSampler:
    """Draws the three skin-electrode interfaces; `cfg` is the `electrodes` block.

    A family is drawn, then an electrode type of that family. The three electrodes
    are of that type, but their parameters are drawn independently around its
    medians, so contact-impedance imbalance is part of the normal variation.
    """

    cfg: dict

    def __call__(self, rng: np.random.Generator, netlist: Netlist) -> Draw:
        names = list(self.cfg["mix"])
        family = str(rng.choice(names, p=[float(self.cfg["mix"][n]) for n in names]))
        kind = str(rng.choice(list(self.cfg["families"][family])))
        medians = self.cfg["families"][family][kind]
        values = {}
        for name in ELECTRODES:
            for key in ("rs", "rd", "cd"):
                factor = log_uniform_factor(rng, self.cfg["spread"])
                values[f"{key.capitalize()}_{name}", "value"] = float(medians[key]) * factor
            ehc = netlist.value(f"Vhc_{name}", "dc")
            values[f"Vhc_{name}", "dc"] = ehc + float(self.cfg["ehc_abs"]) * rng.uniform(-1, 1)
        return Draw(values, {"electrode_type": family, "electrode_kind": kind})


def population(circuit: Circuit, cfg: dict) -> VariationSet:
    """Healthy circuits and electrodes. The draw order is fixed, so a seed defines a circuit."""
    tol = cfg["tolerances"]
    dist = tol["distribution"]
    passives, _, inas = front_end(circuit)
    by_kind = {"R": float(tol["resistor"]), "C": float(tol["capacitor"])}
    opamp = {"vos": (cfg["opamp"]["vos_max"], "absolute"), "aol": cfg["opamp"]["aol_rel"]}
    variations = [
        *tolerances(circuit, by_kind, dist, tol.get("overrides"), passives),
        *instance_tolerances(circuit, "opamp", opamp, dist),
    ]
    for name in inas:
        a = cfg["ina"]
        variations += [
            ToleranceVariation(name, float(a["vos_max"]), dist, "vos", relative=False),
            ToleranceVariation(name, float(a["cmrr_db_tol"]), dist, "cmrr_db", relative=False),
            # the common-mode error of a real part can have either polarity
            CustomVariation(name, _random_sign, "cmrr_sign", "+1 or -1, equally likely"),
            ToleranceVariation(name, float(a["gain_error_max"]), dist, "gerr", relative=False),
        ]
    targets = [
        (f"{prefix}_{name}", parameter)
        for name in ELECTRODES
        for prefix, parameter in (("Rs", "value"), ("Rd", "value"), ("Cd", "value"), ("Vhc", "dc"))
    ]
    variations.append(
        JointVariation(
            "electrodes",
            tuple(targets),
            ElectrodeSampler(cfg["electrodes"]),
            "family, electrode type, then each parameter log-uniform around its median",
        )
    )
    return VariationSet(variations)

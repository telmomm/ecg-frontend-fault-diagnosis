"""Monte Carlo sampling of healthy circuits and electrodes."""

from __future__ import annotations

import numpy as np

from .circuit import ELECTRODES, CircuitInstance, get_circuit, nominal_instance


def _unit(rng: np.random.Generator, distribution: str) -> float:
    """Draw a deviation in [-1, 1], in units of the tolerance."""
    if distribution == "uniform":
        return float(rng.uniform(-1.0, 1.0))
    if distribution == "truncnorm":
        while True:
            x = rng.normal(0.0, 1.0 / 3.0)
            if abs(x) <= 1.0:
                return float(x)
    raise ValueError(f"unknown tolerance distribution: {distribution}")


def passive_tolerance(name: str, cfg: dict) -> float:
    tol = cfg["tolerances"]
    if name in tol.get("overrides", {}):
        return float(tol["overrides"][name])
    return float(tol["resistor"] if name.startswith("R") else tol["capacitor"])


def sample_electrodes(cfg: dict, rng: np.random.Generator) -> tuple[str, dict[str, dict]]:
    """Draw an electrode family and, independently, the parameters of each electrode.

    Independent draws make contact-impedance imbalance part of the normal variation.
    """
    ecfg = cfg["electrodes"]
    names = list(ecfg["mix"])
    family = str(rng.choice(names, p=[float(ecfg["mix"][n]) for n in names]))
    ranges = ecfg["families"][family]
    electrodes = {}
    for name in ELECTRODES:
        e = {}
        for key in ("rs", "rd", "cd"):
            lo, hi = (float(x) for x in ranges[key])
            e[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        e["ehc"] = float(ecfg["nominal"]["ehc"]) + float(ecfg["ehc_abs"]) * rng.uniform(-1, 1)
        electrodes[name] = e
    return family, electrodes


def sample_instance(cfg: dict, rng: np.random.Generator) -> CircuitInstance:
    """Draw one healthy circuit. The draw order is fixed, so a seed defines the circuit."""
    dist = cfg["tolerances"]["distribution"]
    circuit = get_circuit(cfg)
    inst = nominal_instance(cfg)

    for p in circuit.passives:
        inst.passives[p.name] = p.value * (1.0 + passive_tolerance(p.name, cfg) * _unit(rng, dist))

    ocfg = cfg["opamp"]
    for u in circuit.opamps:
        o = inst.opamps[u.name]
        o["vos"] = float(ocfg["vos_max"]) * _unit(rng, dist)
        o["aol"] *= 1.0 + float(ocfg["aol_rel"]) * _unit(rng, dist)

    for u in circuit.inas:
        a, icfg = inst.inas[u.name], cfg["ina"]
        a["vos"] = float(icfg["vos_max"]) * _unit(rng, dist)
        a["cmrr_db"] += float(icfg["cmrr_db_tol"]) * _unit(rng, dist)
        # the common-mode error of a real part can have either polarity
        a["cmrr_sign"] = 1.0 if rng.uniform() < 0.5 else -1.0
        a["gain_error"] = float(icfg["gain_error_max"]) * _unit(rng, dist)

    inst.electrode_type, inst.electrodes = sample_electrodes(cfg, rng)
    return inst


def instance_parameters(inst: CircuitInstance) -> dict[str, float]:
    """Flat record of the realised values, stored with every simulation."""
    row = {f"p_{name}": value for name, value in inst.passives.items()}
    for name, o in inst.opamps.items():
        row[f"p_{name}_vos"] = o["vos"]
        row[f"p_{name}_aol"] = o["aol"]
    for name, a in inst.inas.items():
        for key in ("vos", "cmrr_db", "cmrr_sign", "gain_error"):
            row[f"p_{name}_{key}"] = a[key]
    for name, e in inst.electrodes.items():
        for key in ("ehc", "rs", "rd", "cd"):
            row[f"p_{name}_{key}"] = e[key]
    return row

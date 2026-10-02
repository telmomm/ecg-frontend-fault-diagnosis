"""The circuits under study: topology, nominal values and netlist generation.

Two single-lead ECG front-ends (lead I = LA - RA) sharing the same signal chain
(input network, instrumentation amplifier, driven right leg, 0.04 Hz high-pass,
gain stage, Sallen-Key low-pass):

- `integrated` (main): INA333 instrumentation amplifier (behavioural model built
  from its data sheet) surrounded by a discrete network, single 3.3 V supply with
  a mid-supply reference;
- `reference`: discrete three-op-amp instrumentation amplifier on +-5 V, comparable
  with the benchmark circuits of the fault-diagnosis literature.

See docs/circuit.md for the design rationale. A `CircuitInstance` holds one
concrete realisation (after Monte Carlo sampling and fault injection);
`build_netlist` turns it into SPICE.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Passive:
    name: str
    kind: str  # "R" or "C"
    n1: str
    n2: str
    value: float
    stage: str
    role: str


@dataclass(frozen=True)
class OpAmp:
    name: str
    inp: str
    inn: str
    out: str
    stage: str
    role: str


@dataclass(frozen=True)
class Ina:
    """Integrated instrumentation amplifier with an external gain resistor (rgp-rgn)."""

    name: str
    inp: str
    inn: str
    rgp: str
    rgn: str
    out: str
    stage: str
    role: str


@dataclass(frozen=True)
class Circuit:
    name: str
    ref: str  # node the signal path is referenced to ("0" or the mid-supply reference)
    passives: tuple[Passive, ...]
    opamps: tuple[OpAmp, ...]
    inas: tuple[Ina, ...] = ()

    def passive(self, name: str) -> Passive:
        return next(p for p in self.passives if p.name == name)


INTEGRATED = Circuit(
    name="integrated",
    ref="vref",
    passives=(
        # Input network
        Passive("R1", "R", "cal_la", "inp", 10e3, "input", "LA series protection resistor"),
        Passive("R2", "R", "lead_ra", "inn", 10e3, "input", "RA series protection resistor"),
        Passive("R3", "R", "inp", "vref", 10e6, "input", "LA input bias resistor"),
        Passive("R4", "R", "inn", "vref", 10e6, "input", "RA input bias resistor"),
        Passive("C1", "C", "inp", "0", 100e-12, "input", "LA common-mode RFI capacitor"),
        Passive("C2", "C", "inn", "0", 100e-12, "input", "RA common-mode RFI capacitor"),
        Passive("C3", "C", "inp", "inn", 1e-9, "input", "differential RFI capacitor"),
        # External gain resistor of the INA333, split to sense the common mode. G = 4.03,
        # low enough for a +-300 mV electrode offset to fit in the 3.3 V supply
        Passive("R5", "R", "rgp", "mid", 16.5e3, "ina", "gain resistor, LA half"),
        Passive("R6", "R", "mid", "rgn", 16.5e3, "ina", "gain resistor, RA half"),
        # Driven right leg
        Passive("R7", "R", "cmbuf", "cm", 10e3, "rld", "RLD integrator input resistor"),
        Passive("R8", "R", "cm", "rld_out", 10e6, "rld", "RLD integrator DC feedback"),
        Passive("C4", "C", "cm", "rld_out", 10e-9, "rld", "RLD integrator capacitor"),
        Passive("R9", "R", "rld_out", "lead_rl", 100e3, "rld", "RLD output current limiter"),
        # High-pass (0.041 Hz) and gain stage (G = 1 + R12 / R11 = 69)
        Passive("C5", "C", "ina_out", "hp", 1e-6, "hpf", "high-pass capacitor"),
        Passive("R10", "R", "hp", "vref", 3.9e6, "hpf", "high-pass resistor"),
        Passive("R11", "R", "gfb", "vref", 1e3, "gain", "gain stage, reference resistor"),
        Passive("R12", "R", "gout", "gfb", 68e3, "gain", "gain stage, feedback resistor"),
        # Sallen-Key low-pass, 2nd-order Butterworth, 185 Hz
        Passive("R13", "R", "gout", "sk1", 10e3, "lpf", "Sallen-Key input resistor"),
        Passive("R14", "R", "sk1", "sk2", 10e3, "lpf", "Sallen-Key second resistor"),
        Passive("C6", "C", "sk1", "out", 120e-9, "lpf", "Sallen-Key feedback capacitor"),
        Passive("C7", "C", "sk2", "0", 62e-9, "lpf", "Sallen-Key shunt capacitor"),
        # Mid-supply reference
        Passive("R15", "R", "vcc", "vdiv", 100e3, "ref", "reference divider, upper resistor"),
        Passive("R16", "R", "vdiv", "0", 100e3, "ref", "reference divider, lower resistor"),
        Passive("C8", "C", "vdiv", "0", 1e-6, "ref", "reference bypass capacitor"),
    ),
    opamps=(
        OpAmp("U2", "mid", "cmbuf", "cmbuf", "rld", "common-mode buffer"),
        OpAmp("U3", "cmref", "cm", "rld_out", "rld", "driven-right-leg integrator"),
        OpAmp("U4", "hp", "gfb", "gout", "gain", "gain stage"),
        OpAmp("U5", "sk2", "out", "out", "lpf", "Sallen-Key buffer"),
        OpAmp("U6", "vdiv", "vref", "vref", "ref", "reference buffer"),
    ),
    inas=(Ina("U1", "inp", "inn", "rgp", "rgn", "ina_out", "ina", "instrumentation amplifier"),),
)

REFERENCE = Circuit(
    name="reference",
    ref="0",
    passives=(
        # Input network
        Passive("R1", "R", "cal_la", "inp", 10e3, "input", "LA series protection / RFI resistor"),
        Passive("R2", "R", "lead_ra", "inn", 10e3, "input", "RA series protection / RFI resistor"),
        Passive("R3", "R", "inp", "0", 10e6, "input", "LA input bias resistor"),
        Passive("R4", "R", "inn", "0", 10e6, "input", "RA input bias resistor"),
        Passive("C1", "C", "inp", "0", 1e-9, "input", "LA RFI capacitor"),
        Passive("C2", "C", "inn", "0", 1e-9, "input", "RA RFI capacitor"),
        # Instrumentation amplifier, first stage (G = 1 + (R5 + R6) / R7 = 10.36)
        Passive("R5", "R", "o1", "fb1", 22e3, "ina", "U1 feedback resistor"),
        Passive("R6", "R", "o2", "fb2", 22e3, "ina", "U2 feedback resistor"),
        Passive("R7", "R", "fb1", "fb2", 4.7e3, "ina", "gain-setting resistor"),
        # Instrumentation amplifier, difference stage (G = 1)
        Passive("R8", "R", "o2", "dn", 10e3, "ina", "difference amp, inverting input"),
        Passive("R9", "R", "dn", "ina_out", 10e3, "ina", "difference amp, feedback"),
        Passive("R10", "R", "o1", "dp", 10e3, "ina", "difference amp, non-inverting input"),
        Passive("R11", "R", "dp", "0", 10e3, "ina", "difference amp, reference"),
        # Driven right leg
        Passive("R12", "R", "o1", "cm", 22e3, "rld", "common-mode sense (U1 side)"),
        Passive("R13", "R", "o2", "cm", 22e3, "rld", "common-mode sense (U2 side)"),
        Passive("R14", "R", "cm", "rld_out", 10e6, "rld", "RLD integrator DC feedback"),
        Passive("C3", "C", "cm", "rld_out", 10e-9, "rld", "RLD integrator capacitor"),
        Passive("R15", "R", "rld_out", "lead_rl", 100e3, "rld", "RLD output current limiter"),
        # High-pass (0.041 Hz) and gain stage (G = 1 + R18 / R17 = 40)
        Passive("C4", "C", "ina_out", "hp", 1e-6, "hpf", "high-pass capacitor"),
        Passive("R16", "R", "hp", "0", 3.9e6, "hpf", "high-pass resistor"),
        Passive("R17", "R", "gfb", "0", 1e3, "gain", "gain stage, ground resistor"),
        Passive("R18", "R", "gout", "gfb", 39e3, "gain", "gain stage, feedback resistor"),
        # Sallen-Key low-pass, 2nd-order Butterworth, 185 Hz
        Passive("R19", "R", "gout", "sk1", 10e3, "lpf", "Sallen-Key input resistor"),
        Passive("R20", "R", "sk1", "sk2", 10e3, "lpf", "Sallen-Key second resistor"),
        Passive("C5", "C", "sk1", "out", 120e-9, "lpf", "Sallen-Key feedback capacitor"),
        Passive("C6", "C", "sk2", "0", 62e-9, "lpf", "Sallen-Key shunt capacitor"),
    ),
    opamps=(
        OpAmp("U1", "inp", "fb1", "o1", "ina", "INA input buffer (LA)"),
        OpAmp("U2", "inn", "fb2", "o2", "ina", "INA input buffer (RA)"),
        OpAmp("U3", "dp", "dn", "ina_out", "ina", "INA difference amplifier"),
        OpAmp("U4", "cmref", "cm", "rld_out", "rld", "driven-right-leg integrator"),
        OpAmp("U5", "hp", "gfb", "gout", "gain", "gain stage"),
        OpAmp("U6", "sk2", "out", "out", "lpf", "Sallen-Key buffer"),
    ),
)

CIRCUITS = {c.name: c for c in (INTEGRATED, REFERENCE)}

# electrode name -> (body-side node, lead-side node)
ELECTRODES: dict[str, tuple[str, str]] = {
    "la": ("body_la", "lead_la"),
    "ra": ("body_ra", "lead_ra"),
    "rl": ("body", "lead_rl"),
}

_OPAMP_KEYS = ("aol", "gbw", "rout", "en", "headroom")

# 4kT at 27 degC, used to turn a resistor into a white-noise voltage source
_FOUR_KT = 1.65763e-20

# Single-pole behavioural op-amp: offset, white input noise, finite Aol/GBW,
# output clamped `hr` volts from the rails (clamping the gain node avoids windup).
# The INA is the classic three-amplifier structure with ideal internal resistors;
# offset and finite CMRR are an error voltage in series with the + input.
SUBCIRCUITS = f"""\
.subckt opamp inp inn out vcc vee aol=2e5 gbw=3e6 vos=0 rout=50 en=18n hr=1.5
Vos p1 inp dc {{vos}}
Rn nz 0 {{en*en/{_FOUR_KT}}}
En p1 p2 nz 0 1
Rin p2 inn 1e12
Gm 0 n1 p2 inn 1m
R1 n1 0 {{aol*1k}}
C1 n1 0 {{1m/(6.283185307*gbw)}}
Bcl n1 0 I = v(n1) > v(vcc)-{{hr}} ? v(n1)-v(vcc)+{{hr}} :
+ (v(n1) < v(vee)+{{hr}} ? v(n1)-v(vee)-{{hr}} : 0)
Eo n2 0 n1 0 1
Ro n2 out {{rout}}
.ends opamp

.subckt ina inp inn rgp rgn ref out vcc vee rfb=50k rdiff=150k vos=0 cmrr=1e5 gerr=0
+ aol=1e6 gbw=1e6 rout=50 en=30n hr=0.05
Berr pe inp V = {{vos}} + (0.5*(v(inp)+v(inn)) - v(ref))/{{cmrr}}
XA1 pe rgp o1 vcc vee opamp aol={{aol}} gbw={{gbw}} rout={{rout}} en={{en}} hr={{hr}}
XA2 inn rgn o2 vcc vee opamp aol={{aol}} gbw={{gbw}} rout={{rout}} en={{en}} hr={{hr}}
Rf1 o1 rgp {{rfb*(1+gerr)}}
Rf2 o2 rgn {{rfb*(1+gerr)}}
Rd1 o2 dn {{rdiff}}
Rd2 dn out {{rdiff}}
Rd3 o1 dp {{rdiff}}
Rd4 dp ref {{rdiff}}
XA3 dp dn out vcc vee opamp aol={{aol}} gbw={{gbw}} rout={{rout}} en={{en}} hr={{hr}}
.ends ina"""


@dataclass
class CircuitInstance:
    """One realisation of a circuit: component values plus injected defects."""

    circuit: str
    passives: dict[str, float]
    opamps: dict[str, dict[str, float]]  # U2 -> {aol, gbw, vos, rout, en, headroom}
    inas: dict[str, dict[str, float]]  # U1 -> {rfb, vos, cmrr_db, cmrr_sign, gain_error}
    electrodes: dict[str, dict[str, float]]  # la -> {ehc, rs, rd, cd}
    electrode_type: str = "nominal"
    series_r: dict[str, float] = field(default_factory=dict)  # opens, capacitor ESR
    parallel_r: dict[str, float] = field(default_factory=dict)  # shorts

    def copy(self) -> CircuitInstance:
        return copy.deepcopy(self)


def get_circuit(cfg: dict) -> Circuit:
    return CIRCUITS[cfg["circuit"]]



def nominal_instance(cfg: dict) -> CircuitInstance:
    circuit = get_circuit(cfg)
    opamp = {k: float(cfg["opamp"][k]) for k in _OPAMP_KEYS}
    ina = cfg.get("ina", {"rfb": 0.0, "cmrr_db": 0.0})
    return CircuitInstance(
        circuit=circuit.name,
        passives={p.name: p.value for p in circuit.passives},
        opamps={u.name: {**opamp, "vos": 0.0} for u in circuit.opamps},
        inas={
            u.name: {
                "rfb": float(ina["rfb"]),
                "cmrr_db": float(ina["cmrr_db"]),
                "cmrr_sign": 1.0,
                "vos": 0.0,
                "gain_error": 0.0,
            }
            for u in circuit.inas
        },
        electrodes={name: dict(cfg["electrodes"]["nominal"]) for name in ELECTRODES},
    )


@dataclass
class Stimulus:
    """Sources applied during a transient. Everything is off by default."""

    cal_amplitude: float = 0.0  # calibration pulse in series with the LA lead [V]
    cal_delay: float = 0.1
    cal_width: float = 0.2
    cal_edge: float = 1e-4
    ecg: tuple[np.ndarray, np.ndarray] | None = None  # (t [s], lead-I voltage [V])
    mains_vpeak: float = 0.0  # mains voltage coupled capacitively to the body [V]


def _pwl(t: np.ndarray, v: np.ndarray) -> str:
    pairs = [f"{ti:.6g} {vi:.6g}" for ti, vi in zip(t, v, strict=True)]
    lines = [" ".join(pairs[i : i + 8]) for i in range(0, len(pairs), 8)]
    return "pwl(\n+ " + "\n+ ".join(lines) + ")"


def build_netlist(
    inst: CircuitInstance,
    cfg: dict,
    control: list[str],
    stim: Stimulus | None = None,
    title: str = "ECG front-end",
) -> str:
    """Return a complete ngspice deck running the `control` commands.

    Sources whose AC magnitude the control block may `alter`: Vcal (differential,
    in series with the LA lead), Vcmt (RLD reference), Ilo (differential current
    into the inputs), Vecgp/Vecgn (ECG in the body) and Vmains.
    """
    stim = stim or Stimulus()
    circuit = CIRCUITS[inst.circuit]
    env = cfg["environment"]
    lines = [f"* {title} ({circuit.name})", SUBCIRCUITS, ""]

    lines += [
        "* supplies",
        f"Vcc vcc 0 dc {cfg['supply']['vcc']}",
        f"Vee vee 0 dc {cfg['supply']['vee']}",
        "",
        "* patient and mains coupling (node 0 is the isolated amplifier common)",
        f"Vmains mains earth dc 0 ac 0 sin(0 {stim.mains_vpeak} {env['mains_freq']})",
        f"Cpow mains body {env['c_mains_body']}",
        f"Cbody body earth {env['c_body_earth']}",
        f"Ciso earth 0 {env['c_iso']}",
        f"Riso earth 0 {env['r_iso']}",
    ]
    half_ecg = "dc 0 ac 0"
    if stim.ecg is not None:
        half_ecg += " " + _pwl(stim.ecg[0], 0.5 * np.asarray(stim.ecg[1]))
    lines += [f"Vecgp body_la body {half_ecg}", f"Vecgn body body_ra {half_ecg}", ""]

    lines.append("* skin-electrode interfaces")
    for name, (body, lead) in ELECTRODES.items():
        e = inst.electrodes[name]
        lines += [
            f"Vhc_{name} {body} {name}_a dc {e['ehc']}",
            f"Rs_{name} {name}_a {name}_b {e['rs']}",
            f"Rd_{name} {name}_b {lead} {e['rd']}",
            f"Cd_{name} {name}_b {lead} {e['cd']}",
        ]

    lines += [
        "",
        "* internal test sources: calibration pulse, RLD reference, lead-off current",
        f"Vcal cal_la lead_la dc 0 ac 0 pulse(0 {stim.cal_amplitude} {stim.cal_delay} "
        f"{stim.cal_edge} {stim.cal_edge} {stim.cal_width} 1e3)",
        f"Vcmt cmref {circuit.ref} dc 0 ac 0",
        "Ilo inn inp dc 0 ac 0",
        "",
        "* front-end",
    ]
    for p in circuit.passives:
        n2 = p.n2
        if p.name in inst.series_r:
            n2 = f"{p.name}_x"
            lines.append(f"Rser_{p.name} {n2} {p.n2} {inst.series_r[p.name]}")
        lines.append(f"{p.name} {p.n1} {n2} {inst.passives[p.name]}")
        if p.name in inst.parallel_r:
            lines.append(f"Rpar_{p.name} {p.n1} {p.n2} {inst.parallel_r[p.name]}")
    for u in circuit.opamps:
        o = inst.opamps[u.name]
        lines.append(
            f"X{u.name} {u.inp} {u.inn} {u.out} vcc vee opamp aol={o['aol']} gbw={o['gbw']} "
            f"vos={o['vos']} rout={o['rout']} en={o['en']} hr={o['headroom']}"
        )
    for u in circuit.inas:
        a, o = inst.inas[u.name], cfg["ina"]
        lines.append(
            f"X{u.name} {u.inp} {u.inn} {u.rgp} {u.rgn} {circuit.ref} {u.out} vcc vee ina "
            f"rfb={a['rfb']} rdiff={o['rdiff']} vos={a['vos']} "
            f"cmrr={a['cmrr_sign'] * 10 ** (a['cmrr_db'] / 20)} gerr={a['gain_error']}"
            f"\n+ aol={o['aol']} gbw={o['gbw']} rout={o['rout']} en={o['en']} hr={o['headroom']}"
        )

    lines += ["", ".control", "set noaskquit", *control, ".endc", ".end", ""]
    return "\n".join(lines)

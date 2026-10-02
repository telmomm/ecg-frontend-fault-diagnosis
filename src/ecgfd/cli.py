"""Command-line entry point: `ecgfd [--config FILE] [--circuit NAME] <command>`."""

from __future__ import annotations

import argparse
import os
from collections import Counter

from .circuit import CIRCUITS, build_netlist, get_circuit, nominal_instance
from .config import DEFAULT_CONFIG, load_config
from .dataset import build_tasks, generate
from .faults import fault_catalogue
from .simulate import measure, scalar_features
from .specs import compliance, measure_specs, spec_limits, with_nominal_gain


def _cmd_netlist(cfg: dict, args: argparse.Namespace) -> None:
    print(build_netlist(nominal_instance(cfg), cfg, control=["op"], title="ECG front-end, nominal"))


def _cmd_components(cfg: dict, args: argparse.Namespace) -> None:
    circuit = get_circuit(cfg)
    for p in circuit.passives:
        print(f"{p.name:4s} {p.value:10.3g}  {p.n1:>8s} - {p.n2:<8s} [{p.stage}] {p.role}")
    for u in circuit.inas:
        print(f"{u.name:4s} {'':10s}  +{u.inp} -{u.inn} out={u.out} [{u.stage}] {u.role}")
    for u in circuit.opamps:
        print(f"{u.name:4s} {'':10s}  +{u.inp} -{u.inn} out={u.out} [{u.stage}] {u.role}")


def _cmd_nominal(cfg: dict, args: argparse.Namespace) -> None:
    cfg = with_nominal_gain(cfg)
    inst = nominal_instance(cfg)
    m = measure(inst, cfg)
    print("self-test features (noise-free)")
    for name, value in scalar_features(m, cfg).items():
        print(f"  {name:18s} {value: .6g}")
    specs = measure_specs(inst, cfg)
    labels = compliance(specs, cfg)
    print(f"specifications (nominal gain {cfg['specs']['nominal_gain']:.4g})")
    for name, (sense, limit) in spec_limits(cfg).items():
        verdict = "ok" if labels[f"ok_{name}"] else "FAIL"
        print(f"  {name:18s} {specs[name]: .6g}  ({sense} {limit:g})  {verdict}")


def _cmd_faults(cfg: dict, args: argparse.Namespace) -> None:
    faults = fault_catalogue(cfg)
    if args.list:
        for f in faults:
            print(f.id)
    for kind, n in Counter(f.kind for f in faults).items():
        print(f"{kind:18s} {n}")
    print(f"{'conditions':18s} {len(faults)} (+ healthy)")
    print(f"{'simulations':18s} {len(build_tasks(cfg))}")


def _cmd_generate(cfg: dict, args: argparse.Namespace) -> None:
    out = generate(cfg, args.out, jobs=args.jobs)
    print(f"dataset written to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ecgfd", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML study configuration")
    parser.add_argument("--circuit", choices=sorted(CIRCUITS), help="override the config's circuit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("netlist", help="print the nominal netlist").set_defaults(func=_cmd_netlist)
    sub.add_parser("components", help="list the components").set_defaults(func=_cmd_components)
    sub.add_parser("nominal", help="simulate the nominal circuit").set_defaults(func=_cmd_nominal)

    p = sub.add_parser("faults", help="summarise the fault catalogue")
    p.add_argument("--list", action="store_true", help="print every condition id")
    p.set_defaults(func=_cmd_faults)

    p = sub.add_parser("generate", help="simulate the dataset of one circuit")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel ngspice processes")
    p.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    args.func(load_config(args.config, args.circuit), args)


if __name__ == "__main__":
    main()

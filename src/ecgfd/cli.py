"""Command-line entry point: `ecgfd [--config FILE] [--circuit NAME] <command>`."""

from __future__ import annotations

import argparse
import os
from collections import Counter

from .circuit import CIRCUITS, load_circuit
from .config import DEFAULT_CONFIG, load_config
from .dataset import experiment, generate, relabel
from .faults import fault_universe
from .specs import specifications, with_nominal_gain


def _cmd_netlist(cfg: dict, args: argparse.Namespace) -> None:
    print(load_circuit(cfg).to_netlist())


def _cmd_nominal(cfg: dict, args: argparse.Namespace) -> None:
    cfg = with_nominal_gain(load_circuit(cfg), cfg)
    nominal = experiment(cfg, healthy_only=True).nominal()
    print("self-test features (noise-free)")
    for name, value in nominal["service"].measurements.items():
        print(f"  {name:18s} {value: .6g}")
    specs = nominal["bench"].measurements
    print(f"specifications (nominal gain {cfg['specs']['nominal_gain']:.4g})")
    for spec in specifications(cfg):
        limit = f"max {spec.maximum:g}" if spec.minimum is None else f"min {spec.minimum:g}"
        name, value = spec.name.removeprefix("spec_"), specs[spec.name]
        print(f"  {name:18s} {value: .6g}  ({limit})  {'ok' if spec.met(value) else 'FAIL'}")


def _cmd_faults(cfg: dict, args: argparse.Namespace) -> None:
    study = experiment(cfg)
    if args.list:
        for fault in study.faults:
            print(fault.fault_id)
    if args.coverage:  # faults per component and type; the balanced electrode pair is apart
        matrix = fault_universe(study.circuit, cfg).coverage_matrix()
        print(matrix.fillna(0).astype(int).to_string())
    for kind, n in Counter(fault.fault_type for fault in study.faults).items():
        print(f"{kind:18s} {n}")
    print(f"{'faults':18s} {len(study.faults)} (+ healthy)")
    print(f"{'cases':18s} {len(study.plan()) // len(study.conditions)}")


def _cmd_generate(cfg: dict, args: argparse.Namespace) -> None:
    out = generate(cfg, args.out, jobs=args.jobs)
    print(f"dataset written to {out}")


def _cmd_relabel(cfg: dict, args: argparse.Namespace) -> None:
    cases = relabel(args.data, cfg["specs"])
    print(f"{int(cases['compliant'].sum())} of {len(cases)} cases compliant with the new limits")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ecgfd", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML study configuration")
    parser.add_argument("--circuit", choices=CIRCUITS, help="override the config's circuit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("netlist", help="print the nominal netlist").set_defaults(func=_cmd_netlist)
    sub.add_parser("nominal", help="simulate the nominal circuit").set_defaults(func=_cmd_nominal)

    p = sub.add_parser("faults", help="summarise the fault catalogue")
    p.add_argument("--list", action="store_true", help="print every fault id")
    p.add_argument("--coverage", action="store_true", help="print components x fault types")
    p.set_defaults(func=_cmd_faults)

    p = sub.add_parser("generate", help="simulate the dataset of one circuit")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel ngspice processes")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("relabel", help="recompute compliance with the limits of --config")
    p.add_argument("--data", required=True, help="dataset folder")
    p.set_defaults(func=_cmd_relabel)

    args = parser.parse_args(argv)
    args.func(load_config(args.config, args.circuit), args)


if __name__ == "__main__":
    main()

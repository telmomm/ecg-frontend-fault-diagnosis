"""E1 - Validation of the nominal circuit and of its spread without faults.

Checks every specification for the nominal circuit and for a Monte Carlo
population of healthy circuits (yield), and draws the frequency response, the
calibration-pulse response and a synthetic ECG through the chain.

    python experiments/e1_nominal_validation.py [--circuit reference] [--n-mc 200]
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spicefault import SimulationConfig, Simulator

from _common import SERIES, parser, results_dir, save_json, set_style
from ecgfd.circuit import load_circuit
from ecgfd.config import load_config
from ecgfd.dataset import experiment, observe
from ecgfd.selftest import AC_DIFF, OUT, pulse_waveform
from ecgfd.signals import synthetic_ecg
from ecgfd.specs import SPEC_NAMES, compliance, spec_limits, with_nominal_gain


def specs_table(samples: list[dict]) -> pd.DataFrame:
    """Specification values (columns named as the specifications) and their labels."""
    specs = pd.DataFrame(samples)[[f"spec_{name}" for name in SPEC_NAMES]]
    return specs.rename(columns=lambda c: c.removeprefix("spec_")), specs


def ecg_response(circuit, t: np.ndarray, v: np.ndarray):
    """Transient of the nominal circuit with the lead-I voltage `v(t)` in the body."""
    pairs = [f"{ti:.6g} {vi:.6g}" for ti, vi in zip(t, 0.5 * v, strict=True)]  # half per side
    lines = [" ".join(pairs[i : i + 8]) for i in range(0, len(pairs), 8)]
    pwl = "pwl(\n+ " + "\n+ ".join(lines) + ")"
    netlist = re.sub(r"(?m)^(Vecg[pn] .*?) pulse\(.*$", rf"\1 {pwl}", circuit.to_netlist())
    config = SimulationConfig(analyses=((f"tran 2.5e-4 {t[-1]} 0 2.5e-4", OUT),))
    return Simulator().run(netlist, config).plot()


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--n-mc", type=int, default=200, help="healthy Monte Carlo circuits")
    args = p.parse_args()
    cfg = load_config(args.config, args.circuit)
    circuit = load_circuit(cfg)
    cfg = with_nominal_gain(circuit, cfg)
    out = results_dir("e1", cfg["circuit"])
    set_style()
    waveform = pulse_waveform(cfg)

    # --- nominal circuit --------------------------------------------------------
    observed = observe(cfg)
    in_service = observed["service"]["result"]
    freq, h_diff = in_service.plot(AC_DIFF)["frequency"].real, in_service.plot(AC_DIFF)["v(out)"]
    nominal, nominal_specs = specs_table([observed["bench"]])
    nominal = nominal.iloc[0]

    # --- Monte Carlo, healthy ---------------------------------------------------
    samples = experiment(cfg, healthy_only=True, healthy_samples=args.n_mc).run(args.jobs).samples
    service, bench = samples[0::2], samples[1::2]  # two conditions per drawn circuit
    mc, mc_specs = specs_table([s.measurements for s in bench])
    mc = pd.concat([mc, compliance(mc_specs, cfg)], axis=1)
    mc["electrode_type"] = [s.labels["electrode_type"] for s in bench]
    mags = np.array([np.abs(s.result.plot(AC_DIFF)["v(out)"]) for s in service])
    pulses = np.array([waveform(s.result) for s in service])
    mc.to_csv(out / "monte_carlo.csv", index=False)

    limits = spec_limits(cfg)
    names = list(limits)
    summary = pd.DataFrame(
        {
            "limit": {name: f"{sense} {limit:g}" for name, (sense, limit) in limits.items()},
            "nominal": nominal,
            "mc_min": mc[names].min(),
            "mc_median": mc[names].median(),
            "mc_max": mc[names].max(),
            "mc_pass_rate": mc[[f"ok_{name}" for name in names]].mean().set_axis(names),
        }
    )
    summary.to_csv(out / "summary.csv")
    verdict = {
        "circuit": cfg["circuit"],
        "nominal_gain": cfg["specs"]["nominal_gain"],
        "nominal_compliant": bool(compliance(nominal_specs, cfg)["compliant"].iloc[0]),
        "n_mc": args.n_mc,
        "mc_yield": float(mc["compliant"].mean()),
    }
    save_json(verdict, out / "verdict.json")
    print(summary.to_string(float_format=lambda x: f"{x:.4g}"))
    print(verdict)

    # --- figures (in service: patient electrodes of both families) ---------------
    label = f"Healthy Monte Carlo range (n = {args.n_mc})"
    peak = np.abs(h_diff).max()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.fill_between(freq, mags.min(axis=0), mags.max(axis=0), color=SERIES[0], alpha=0.25,
                    linewidth=0, label=label)
    ax.loglog(freq, np.abs(h_diff), label="Nominal")
    ax.set(xlabel="Frequency (Hz)", ylabel="Differential gain (V/V)", xlim=(0.05, 1e3),
           ylim=(peak / 100, peak * 2), title="Frequency response, calibration input to output")
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(out / "frequency_response.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    t = waveform.times()
    ax.fill_between(t, pulses.min(axis=0), pulses.max(axis=0), color=SERIES[0], alpha=0.25,
                    linewidth=0, label=label)
    ax.plot(t, waveform(in_service), label="Nominal")
    ax.set(xlabel="Time (s)", ylabel="Output (V)", title="Response to the 1 mV calibration pulse")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "pulse_response.png")

    t_ecg, v_ecg = synthetic_ecg(duration=3.0)
    plot = ecg_response(circuit, t_ecg, v_ecg)
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.0), sharex=True)
    axes[0].plot(t_ecg, 1e3 * v_ecg)
    axes[0].set(ylabel="Input (mV)", title="Synthetic ECG through the nominal front-end")
    axes[1].plot(plot["time"], plot["v(out)"])
    axes[1].set(xlabel="Time (s)", ylabel="Output (V)")
    fig.tight_layout()
    fig.savefig(out / "synthetic_ecg.png")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

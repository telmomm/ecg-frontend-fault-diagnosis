"""E1 - Validation of the nominal circuit and of its spread without faults.

Checks every specification for the nominal circuit and for a Monte Carlo
population of healthy circuits (yield), and draws the frequency response, the
calibration-pulse response and a synthetic ECG through the chain.

    python experiments/e1_nominal_validation.py [--circuit reference] [--n-mc 200]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import SERIES, parser, results_dir, save_json, set_style
from ecgfd.circuit import Stimulus, nominal_instance
from ecgfd.config import load_config
from ecgfd.dataset import sample_rng
from ecgfd.sampling import sample_instance
from ecgfd.signals import synthetic_ecg
from ecgfd.simulate import measure, transient
from ecgfd.specs import compliance, measure_specs, spec_limits, with_nominal_gain


def _mc_worker(replica: int, cfg: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    inst = sample_instance(cfg, sample_rng(cfg, 0, replica))
    specs = measure_specs(inst, cfg)
    m = measure(inst, cfg)
    row = {**specs, **compliance(specs, cfg), "electrode_type": inst.electrode_type}
    return row, np.abs(m.h_diff), m.pulse


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--n-mc", type=int, default=200, help="healthy Monte Carlo circuits")
    args = p.parse_args()
    cfg = with_nominal_gain(load_config(args.config, args.circuit))
    out = results_dir("e1", cfg["circuit"])
    set_style()

    # --- nominal circuit --------------------------------------------------------
    inst = nominal_instance(cfg)
    m = measure(inst, cfg)
    nominal = measure_specs(inst, cfg)

    # --- Monte Carlo, healthy ---------------------------------------------------
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(partial(_mc_worker, cfg=cfg), range(args.n_mc), chunksize=4))
    mc = pd.DataFrame([r[0] for r in results])
    mags = np.array([r[1] for r in results])
    pulses = np.array([r[2] for r in results])
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
        "nominal_compliant": bool(compliance(nominal, cfg)["compliant"]),
        "n_mc": args.n_mc,
        "mc_yield": float(mc["compliant"].mean()),
    }
    save_json(verdict, out / "verdict.json")
    print(summary.to_string(float_format=lambda x: f"{x:.4g}"))
    print(verdict)

    # --- figures (in service: patient electrodes of both families) ---------------
    label = f"Healthy Monte Carlo range (n = {args.n_mc})"
    peak = np.abs(m.h_diff).max()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.fill_between(m.freq, mags.min(axis=0), mags.max(axis=0), color=SERIES[0], alpha=0.25,
                    linewidth=0, label=label)
    ax.loglog(m.freq, np.abs(m.h_diff), label="Nominal")
    ax.set(xlabel="Frequency (Hz)", ylabel="Differential gain (V/V)", xlim=(0.05, 1e3),
           ylim=(peak / 100, peak * 2), title="Frequency response, calibration input to output")
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(out / "frequency_response.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.fill_between(m.t, pulses.min(axis=0), pulses.max(axis=0), color=SERIES[0], alpha=0.25,
                    linewidth=0, label=label)
    ax.plot(m.t, m.pulse, label="Nominal")
    ax.set(xlabel="Time (s)", ylabel="Output (V)", title="Response to the 1 mV calibration pulse")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "pulse_response.png")

    t_ecg, v_ecg = synthetic_ecg(duration=3.0)
    plot = transient(inst, cfg, Stimulus(ecg=(t_ecg, v_ecg)), duration=3.0)
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

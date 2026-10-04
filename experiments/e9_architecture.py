"""E9 - Effect of the architecture on testability (H5).

Compares the main circuit (integrated INA + discrete network) with the reference
circuit (discrete three-op-amp amplifier) on the non-compliant cases:

- from E2: components located without ambiguity, testability rank;
- multivariate class separability of the components (centroid distance over
  within-class spread), computed here;
- from E5, if it has been run: best macro F1 of the localisation baseline.

Run E2 (and optionally E5) on both datasets first.

    python experiments/e9_architecture.py --data-dir data/v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from _common import SERIES, load_measured, output_name, results_root, set_style
from ecgfd.config import REPO_ROOT
from ecgfd.evaluation import centroid_separability
from ecgfd.features import feature_sets

CIRCUITS = ("integrated", "reference")


def comparison_figure(table: pd.DataFrame, path) -> None:
    sets = list(dict.fromkeys(table["feature_set"]))
    y = np.arange(len(sets))
    height = 0.38
    panels = (
        ("unambiguous_share", "Components located without ambiguity (E2)", "{:.0%}"),
        ("separability", "Class separability, centroid / spread", "{:.2f}"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, (column, title, fmt) in zip(axes, panels, strict=True):
        for k, circuit in enumerate(CIRCUITS):
            rows = table[table["circuit"] == circuit].set_index("feature_set").reindex(sets)
            values = rows[column].to_numpy()
            ax.barh(y + (k - 0.5) * height, values, height=height * 0.92,
                    color=SERIES[k], label=circuit)
            for yi, v in zip(y + (k - 0.5) * height, values, strict=True):
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7, color="#52514e")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
        if column != "unambiguous_share":
            ax.margins(x=0.18)
    axes[0].set_yticks(y, sets)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.1)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8)
    fig.suptitle("Non-compliant cases: integrated against reference circuit", fontsize=10, x=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "v1"),
                   help="folder holding one dataset per circuit")
    p.add_argument("--electrode-kinds", nargs="*", help="same filter as used in E2/E5")
    p.add_argument("--tag", help="same tag as used in E2/E5")
    args = p.parse_args()

    rows = []
    for circuit in CIRCUITS:
        name = output_name(circuit, args)
        e2 = results_root(args) / "e2" / name / "summary.csv"
        if not e2.exists():
            raise SystemExit(f"{e2} not found: run E2 on the {circuit} dataset first")
        summary = pd.read_csv(e2).set_index("feature_set")
        e5_path = results_root(args) / "e5" / name / "localisation.csv"
        e5 = None
        source = e5_path.with_name("source.json")
        data_path = str(Path(f"{args.data_dir}/{circuit}").resolve())
        same_data = source.exists() and json.loads(source.read_text())["data"] == data_path
        if e5_path.exists() and same_data:
            scores = pd.read_csv(e5_path, header=[0, 1])
            sets = scores.iloc[:, 0]  # first column: feature set
            e5 = scores[("f1_macro", "mean")].groupby(sets).max()
        elif e5_path.exists():
            print(f"ignoring {e5_path}: it was not run on {data_path}")

        measured, _, cfg = load_measured(f"{args.data_dir}/{circuit}", args.electrode_kinds)
        nc = measured[~measured["compliant"].astype(bool)]
        labels = nc["target"].to_numpy()
        for set_name, features in feature_sets(cfg).items():
            s = summary.loc[set_name]
            rows.append(
                {
                    "circuit": circuit,
                    "feature_set": set_name,
                    "n_components": int(s["n_components"]),
                    "unambiguous_share": s["n_unambiguous_components"] / s["n_components"],
                    "mean_confusable": s["mean_confusable"],
                    "testability_rank": int(s["testability_rank"]),
                    "separability": centroid_separability(nc[features].to_numpy(), labels),
                    "e5_best_f1_macro": np.nan if e5 is None else e5.get(set_name, np.nan),
                }
            )

    table = pd.DataFrame(rows)
    out = results_root(args) / "e9" / (args.tag or ("subset" if args.electrode_kinds else "all"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    set_style()
    comparison_figure(table, out / "comparison.png")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

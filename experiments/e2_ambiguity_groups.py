"""E2 - Testability: sensitivities, detectability and ambiguity groups (H3).

Model-free map of what each feature set can see and distinguish, in three parts:

1. Small deviations: sensitivity of every feature to a 1 % change of every passive,
   in units of the healthy spread; testability rank and collinear components for a
   reference deviation of 10 %.
2. Detectability: conditions whose Monte Carlo cloud cannot be told apart from the
   healthy one, and how many of those are out of specification (built-in escapes);
   plus a limit test on the healthy envelope as the classical reference.
3. Localisation: on the non-compliant cases only, which components can be mistaken
   for which (the classes E5 has to locate), and the resulting ambiguity groups.

Run it on the dataset of each circuit, then E9 compares them (H5).

    python experiments/e2_ambiguity_groups.py --data data/v1/integrated [--threshold 3]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from _common import SERIES, load_measured, parser, results_dir, save_json, set_style
from ecgfd.ambiguity import (
    ambiguity_groups,
    collinear_groups,
    component_groups,
    confusable_components,
    envelope_detection,
    fault_dictionary,
    normalised_sensitivity,
    sensitivity_matrix,
    separability,
    testability_rank,
)
from ecgfd.features import feature_sets

# sequential single-hue ramp for magnitudes
BLUES = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])


def sensitivity_figure(z: pd.DataFrame, path, circuit: str) -> None:
    magnitude = np.log10(np.clip(np.abs(z.to_numpy()), 1e-2, 1e1))
    fig, ax = plt.subplots(figsize=(0.32 * z.shape[1] + 2.5, 0.22 * z.shape[0] + 1.6))
    image = ax.imshow(magnitude, cmap=BLUES, vmin=-2, vmax=1, aspect="auto")
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=90)
    ax.set_yticks(range(z.shape[0]), z.index)
    ax.grid(False)
    ax.set_title(f"Sensitivity to a 1 % change, {circuit} circuit")
    bar = fig.colorbar(image, ax=ax, shrink=0.6)
    bar.set_label("log10 |change| in healthy standard deviations")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def localisation_figure(summary: pd.DataFrame, path, circuit: str) -> None:
    share = summary["n_unambiguous_components"] / summary["n_components"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.barh(summary["feature_set"], share, color=SERIES[0], height=0.6)
    for y, (value, n) in enumerate(zip(share, summary["n_components"], strict=True)):
        ax.text(value + 0.01, y, f"{value:.0%} of {n}", va="center", fontsize=8, color="#52514e")
    ax.set(xlim=(0, 1.15), xlabel="Components located without ambiguity",
           title=f"Non-compliant cases, {circuit} circuit")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = parser(__doc__.splitlines()[0], dataset=True)
    p.add_argument("--threshold", type=float, default=3.0, help="d' below which conditions merge")
    p.add_argument("--min-cases", type=int, default=20,
                   help="non-compliant cases a condition needs to enter the localisation map")
    args = p.parse_args()

    measured, _, cfg = load_measured(args.data, args.electrode_kinds)
    measured["compliant"] = measured["compliant"].astype(bool)
    circuit = cfg["circuit"]
    out = results_dir("e2", circuit, args)
    set_style()
    sets = feature_sets(cfg)
    healthy_spread = measured.loc[measured["kind"] == "healthy"].std(numeric_only=True)
    # floor the spread at 10 % of the healthy one, so saturated features stay comparable
    floor = 0.1 * healthy_spread

    # 1. small deviations -------------------------------------------------------------
    sensitivity = sensitivity_matrix(cfg)
    z = normalised_sensitivity(sensitivity, healthy_spread)
    sensitivity.to_csv(out / "sensitivity.csv")
    z.to_csv(out / "sensitivity_normalised.csv")
    sensitivity_figure(z, out / "sensitivity.png", circuit)

    # 2 and 3. simulated faults ---------------------------------------------------------
    non_compliant_share = 1.0 - measured.groupby("condition")["compliant"].mean()
    nc = measured[~measured["compliant"]]
    counts = nc["condition"].value_counts()
    nc = nc[nc["condition"].isin(counts[counts >= args.min_cases].index)]
    component_of = dict(zip(nc["condition"], nc["target"], strict=False))

    rows, groups_out = [], {}
    for name, features in sets.items():
        rows_z = z.loc[z.index.intersection(features)]
        collinear, insensitive = collinear_groups(rows_z, threshold=args.threshold)

        mean, std = fault_dictionary(measured, features)
        d = separability(mean, std, floor)
        undetectable = sorted(d.index[d.loc["healthy"] < args.threshold].drop("healthy"))
        escapes = [c for c in undetectable if non_compliant_share[c] >= 0.5]
        detected, false_alarm = envelope_detection(measured, features)
        missed = detected.index[detected < 0.5]
        missed_escapes = [c for c in missed if non_compliant_share[c] >= 0.5]

        mean_nc, std_nc = fault_dictionary(nc, features)
        d_nc = separability(mean_nc, std_nc, floor)
        partners = confusable_components(d_nc, component_of, args.threshold)
        components = component_groups(d_nc, component_of, args.threshold)

        groups_out[name] = {
            "undetectable": undetectable,
            "undetectable_mostly_non_compliant": escapes,
            "limit_test_missed_mostly_non_compliant": sorted(missed_escapes),
            "condition_groups_non_compliant": ambiguity_groups(d_nc, args.threshold),
            "confusable_components": partners,
            "component_groups": components,
            "collinear_components": collinear,
            "insensitive_components": insensitive,
        }
        mean.to_csv(out / f"dictionary_{name}.csv")
        rows.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "testability_rank": testability_rank(rows_z, threshold=args.threshold),
                "n_insensitive": len(insensitive),
                "n_collinear_groups": sum(len(g) > 1 for g in collinear),
                "n_conditions": len(mean) - 1,
                "n_undetectable": len(undetectable),
                "n_undetectable_mostly_non_compliant": len(escapes),
                "limit_test_false_alarm": false_alarm,
                "n_limit_test_missed": len(missed),
                "n_limit_test_missed_mostly_non_compliant": len(missed_escapes),
                "n_localised_conditions": len(mean_nc),
                "n_components": len(partners),
                "n_unambiguous_components": sum(not v for v in partners.values()),
                "mean_confusable": float(np.mean([len(v) for v in partners.values()])),
                "n_component_groups": len(components),
                "largest_component_group": max(len(g) for g in components),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    save_json({"threshold": args.threshold, "min_cases": args.min_cases, **groups_out},
              out / "groups.json")
    localisation_figure(summary, out / "localisation.png", circuit)
    print(summary.to_string(index=False))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

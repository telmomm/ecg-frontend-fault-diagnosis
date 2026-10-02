"""E2 - Testability: sensitivities, fault dictionary and ambiguity groups (H3).

Model-free map of what each feature set can distinguish, before training anything.
Run it on the dataset of each circuit to compare architectures (E9, H5).

    python experiments/e2_ambiguity_groups.py --data data/v1/integrated [--threshold 3]
"""

from __future__ import annotations

import pandas as pd

from _common import load_measured, parser, results_dir, save_json
from ecgfd.ambiguity import ambiguity_groups, fault_dictionary, sensitivity_matrix, separability
from ecgfd.features import feature_sets


def main() -> None:
    p = parser(__doc__.splitlines()[0], dataset=True)
    p.add_argument("--threshold", type=float, default=3.0, help="d' below which conditions merge")
    args = p.parse_args()

    measured, _, cfg = load_measured(args.data)
    out = results_dir("e2", cfg["circuit"])
    sensitivity_matrix(cfg).to_csv(out / "sensitivity.csv")

    rows = []
    for name, features in feature_sets(cfg).items():
        mean, std = fault_dictionary(measured, features)
        # floor the spread at 10 % of the healthy one, so saturated features stay comparable
        floor = 0.1 * std.loc["healthy"]
        d = separability(mean, std, floor)
        groups = ambiguity_groups(d, args.threshold)
        # judged directly against healthy, not through chains of similar faults
        undetectable = sorted(d.index[(d.loc["healthy"] < args.threshold)].drop("healthy"))
        save_json(
            {"threshold": args.threshold, "undetectable": undetectable, "groups": groups},
            out / f"groups_{name}.json",
        )
        mean.to_csv(out / f"dictionary_{name}.csv")
        rows.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "n_conditions": len(mean),
                "n_groups": len(groups),
                "n_singletons": sum(len(g) == 1 for g in groups),
                "n_undetectable": len(undetectable),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

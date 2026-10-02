"""E4 - Functional severity against percentage severity (H2).

1. How many injected faults leave the circuit within specifications, by fault type
   and magnitude.
2. What a detector trained on "a component is out of tolerance" (percentage
   severity) costs when judged by what matters, compliance: escapes and false
   rejects, against the same model trained on compliance (functional severity).

    python experiments/e4_severity.py --data data/v1/integrated
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from _common import load_measured, parser, results_dir
from ecgfd.evaluation import decision_report, replica_split
from ecgfd.features import ALL, feature_sets


def main() -> None:
    args = parser(__doc__.splitlines()[0], dataset=True).parse_args()
    measured, _, cfg = load_measured(args.data)
    out = results_dir("e4", cfg["circuit"])
    seed = int(cfg["seed"])

    injected = measured[measured["origin"] == "circuit"]
    by_fault = (
        injected.groupby(["kind", "level"])["compliant"]
        .agg(still_compliant="mean", n="size")
        .reset_index()
    )
    by_fault.to_csv(out / "compliance_by_fault.csv", index=False)
    print(by_fault.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    labels = {
        "percentage": (measured["origin"] == "circuit").to_numpy(),
        "functional": ~measured["compliant"].to_numpy(dtype=bool),
    }
    truth = labels["functional"]
    train, test = replica_split(measured, test_fraction=0.3, seed=seed)
    x = measured[feature_sets(cfg)[ALL]].to_numpy()
    rows = []
    for name, y in labels.items():
        model = HistGradientBoostingClassifier(random_state=seed).fit(x[train], y[train])
        row = {"trained_on": name, "bad_fraction_in_labels": float(y.mean())}
        row.update(decision_report(truth[test], model.predict(x[test])))
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "label_comparison.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

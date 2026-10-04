"""E4 - Functional severity against percentage severity (H2).

1. How many injected faults leave the circuit within specifications, by fault type
   and magnitude.
2. What a detector trained on "a component is out of tolerance" (percentage
   severity) costs when judged by what matters, compliance: escapes and false
   rejects, against the same model trained on compliance (functional severity).
3. Which injected faults the percentage-trained detector rejects although the
   circuit is compliant (the false alarms that functional labelling removes).

    python experiments/e4_severity.py --data data/v1/integrated
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from _common import feature_matrix, load_measured, parser, results_dir, split, summarise
from ecgfd.evaluation import decision_report
from ecgfd.features import ALL, feature_sets


def main() -> None:
    args = parser(__doc__.splitlines()[0], dataset=True).parse_args()
    measured, _, cfg = load_measured(args.data, args.electrode_kinds)
    measured["compliant"] = measured["compliant"].astype(bool)
    out = results_dir("e4", cfg["circuit"], args)
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
        "functional": ~measured["compliant"].to_numpy(),
    }
    truth = labels["functional"]
    x = feature_matrix(measured, feature_sets(cfg)[ALL], args)
    rows, false_rejects = [], []
    for repeat in range(args.repeats):
        train, val, test = split(measured, repeat, seed)
        fit_rows = np.concatenate([train, val])
        for name, y in labels.items():
            # balanced weights: otherwise the majority label wins wherever classes overlap
            model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
            model.fit(x[fit_rows], y[fit_rows])
            pred = model.predict(x[test]).astype(bool)
            row = {"trained_on": name, "repeat": repeat,
                   "bad_fraction_in_labels": float(y.mean())}
            row.update(decision_report(truth[test], pred))
            rows.append(row)
            rejected_good = measured.iloc[test][pred & ~truth[test]]
            false_rejects.append(
                rejected_good.groupby("kind").size().rename("cases").reset_index()
                .assign(trained_on=name, repeat=repeat)
            )

    rows = pd.DataFrame(rows)
    rows.to_csv(out / "label_comparison_repeats.csv", index=False)
    comparison = summarise(rows, ["trained_on"])
    comparison.to_csv(out / "label_comparison.csv", index=False)
    by_kind = (
        pd.concat(false_rejects)
        .groupby(["trained_on", "kind"])["cases"].mean()
        .rename("false_rejects_per_split").reset_index()
    )
    by_kind.to_csv(out / "false_rejects_by_kind.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(by_kind.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

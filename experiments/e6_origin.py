"""E6 - Electrode degradation against circuit fault (H4).

Three-class problem on every case: no defect, defect in the circuit, defect in the
electrodes. Baseline with untuned classifiers and a single split.

    python experiments/e6_origin.py --data data/v1/integrated
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from _common import compare_classifiers, load_measured, parser, results_dir
from ecgfd.evaluation import replica_split
from ecgfd.features import ALL, feature_sets

CLASSES = ["none", "circuit", "electrode"]


def main() -> None:
    args = parser(__doc__.splitlines()[0], dataset=True).parse_args()
    measured, _, cfg = load_measured(args.data)
    out = results_dir("e6", cfg["circuit"])
    seed = int(cfg["seed"])

    y = measured["origin"].map(CLASSES.index).to_numpy()
    train, test = replica_split(measured, test_fraction=0.3, seed=seed)
    results = compare_classifiers(measured, y, train, test, cfg, args.models)
    results.to_csv(out / "baseline.csv", index=False)

    x = measured[feature_sets(cfg)[ALL]].to_numpy()
    model = HistGradientBoostingClassifier(random_state=seed).fit(x[train], y[train])
    matrix = confusion_matrix(y[test], model.predict(x[test]), labels=range(len(CLASSES)))
    matrix = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).rename_axis("true")
    matrix.to_csv(out / "confusion_gradient_boosting.csv")
    print(matrix.to_string())
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

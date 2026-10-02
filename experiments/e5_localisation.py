"""E5 - Locating the cause when the circuit is out of specification (H3).

Classifiers trained and tested on non-compliant cases only, with the failed
component as the class. Untuned defaults and a single split: this is the baseline;
merging classes by ambiguity group (from E2) and tuning on a validation split are
still to be added.

    python experiments/e5_localisation.py --data data/v1/integrated [--cnn]
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from _common import compare_classifiers, load_measured, parser, results_dir
from ecgfd.evaluation import localisation_report, replica_split


def main() -> None:
    p = parser(__doc__.splitlines()[0], dataset=True)
    p.add_argument("--cnn", action="store_true", help="also train the 1D CNN (needs torch)")
    args = p.parse_args()
    measured, wav, cfg = load_measured(args.data)
    out = results_dir("e5", cfg["circuit"])
    seed = int(cfg["seed"])

    keep = ~measured["compliant"].to_numpy(dtype=bool)
    measured, wav = measured[keep].reset_index(drop=True), wav[keep]
    y = LabelEncoder().fit_transform(measured["target"])
    train, test = replica_split(measured, test_fraction=0.3, seed=seed)
    print(f"{len(measured)} non-compliant cases, {y.max() + 1} components")

    results = compare_classifiers(measured, y, train, test, cfg, args.models)

    if args.cnn:
        from ecgfd.models import cnn1d

        # the CNN needs its own validation split, carved out of the training rows
        sub_train, sub_val = replica_split(measured.iloc[train], test_fraction=0.2, seed=seed + 1)
        tr, va = train[sub_train], train[sub_val]
        model, _ = cnn1d.fit(wav[tr], y[tr], wav[va], y[va], int(y.max()) + 1, seed=seed)
        row = {"feature_set": "C3 waveform", "model": "cnn1d"}
        row.update(localisation_report(y[test], cnn1d.predict(model, wav[test])))
        print(row)
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    results.to_csv(out / "baseline.csv", index=False)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

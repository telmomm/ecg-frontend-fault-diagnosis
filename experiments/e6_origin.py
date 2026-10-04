"""E6 - Electrode degradation against circuit fault (H4).

Three-class problem on every case, defined by what the instrument has to do:

- `none`: nothing to act on (healthy circuit, or a deviation that leaves it compliant);
- `circuit`: the circuit is out of specification;
- `electrode`: an electrode is detached or degraded, or the series protection
  resistor of a lead is open. The last one is a circuit fault, but it is
  electrically the same as a detached electrode and calls for the same action
  (check the contact and the lead), so both are one class.

With `--injected-origin` the classes follow what was injected instead, so every
circuit fault counts as `circuit` even when it is harmless; that version mixes this
question with the one of E4. The classes are unbalanced, so the models weight them
inversely to their frequency where they can, and the scores to read are the balanced
accuracy, the macro F1 and the recall of each class.

    python experiments/e6_origin.py --data data/v1/integrated
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from _common import load_measured, parser, results_dir, run_classifiers, summarise
from ecgfd.features import ALL
from ecgfd.models.classical import FAST_MODELS

CLASSES = ["none", "circuit", "electrode"]
LEAD_RESISTORS = ("R1", "R2")  # series protection resistors of LA and RA, in both circuits


def main() -> None:
    p = parser(__doc__.splitlines()[0], dataset=True)
    p.add_argument("--injected-origin", action="store_true",
                   help="label by the injected fault instead of by what has to be acted on")
    args = p.parse_args()
    if args.injected_origin:
        args.tag = "injected" if args.tag is None else f"{args.tag}-injected"
    measured, _, cfg = load_measured(args.data, args.electrode_kinds)
    out = results_dir("e6", cfg["circuit"], args)

    if not args.injected_origin:
        harmless = (measured["origin"] == "circuit") & measured["compliant"].astype(bool)
        measured["origin"] = measured["origin"].mask(harmless, "none")
        open_lead = (measured["kind"] == "open") & measured["target"].isin(LEAD_RESISTORS)
        measured["origin"] = measured["origin"].mask(open_lead, "electrode")
    y = measured["origin"].map(CLASSES.index).to_numpy()
    print(measured["origin"].value_counts().to_string())
    rows, fitted = run_classifiers(
        measured, y, cfg, args, default_models=FAST_MODELS, balanced=True
    )
    rows.to_csv(out / "origin_repeats.csv", index=False)
    summary = summarise(rows, ["feature_set", "model"])
    summary.to_csv(out / "origin.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    on_all = summary[summary["feature_set"] == ALL]
    best = on_all.loc[on_all[("f1_macro", "mean")].idxmax(), ("model", "")]
    model, x, test = fitted[(ALL, best)]
    matrix = confusion_matrix(y[test], model.predict(x[test]), labels=range(len(CLASSES)))
    table = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).rename_axis("true")
    table["recall"] = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    table.to_csv(out / "confusion_best.csv")
    print(f"best on {ALL}: {best}")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    # which electrode faults are taken for something else, and which circuit faults for electrode
    cases = measured.iloc[test].assign(predicted=[CLASSES[k] for k in model.predict(x[test])])
    by_kind = (
        cases.groupby(["origin", "kind"])["predicted"].value_counts(normalize=True)
        .rename("share").reset_index()
    )
    by_kind.to_csv(out / "predicted_by_kind.csv", index=False)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

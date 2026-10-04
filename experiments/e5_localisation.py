"""E5 - Locating the cause when the circuit is out of specification (H3).

Classifiers trained and tested on non-compliant cases only, at two levels:

- by ambiguity group (main result): components that the sensitivities predict to be
  indistinguishable are one class, known before any model is trained;
- by component: the finer question, which E7 shows to rely on memorised fault
  magnitudes inside a group.

Hyper-parameters are tuned on a validation split and the scores are averaged over
repetitions. For the best component-level model on the full feature set it also
writes the recall of every component next to its group and to the number of
components E2 predicted it could be mistaken for.

    python experiments/e5_localisation.py --data data/v1/integrated [--cnn]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from _common import (
    ambiguity_group_of,
    load_measured,
    parser,
    results_dir,
    results_root,
    run_classifiers,
    save_json,
    split,
    summarise,
)
from ecgfd.features import ALL


def e2_confusable(circuit: str, args) -> dict[str, list[str]] | None:
    """Confusable components predicted by E2 for the same data selection, if available."""
    tag = args.tag or ("subset" if args.electrode_kinds else None)
    name = circuit if tag is None else f"{circuit}-{tag}"
    path = results_root(args) / "e2" / name / "groups.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())[ALL]["confusable_components"]


def main() -> None:
    p = parser(__doc__.splitlines()[0], dataset=True)
    p.add_argument("--cnn", action="store_true", help="also train the 1D CNN (needs torch)")
    args = p.parse_args()
    measured, wav, cfg = load_measured(args.data, args.electrode_kinds)
    out = results_dir("e5", cfg["circuit"], args)
    seed = int(cfg["seed"])

    full = measured
    keep = ~measured["compliant"].to_numpy(dtype=bool)
    measured, wav = measured[keep].reset_index(drop=True), wav[keep]
    encoder = LabelEncoder().fit(measured["target"])
    y = encoder.transform(measured["target"])
    print(f"{len(measured)} non-compliant cases, {len(encoder.classes_)} components")

    group_of = ambiguity_group_of(full, cfg)
    groups = measured["target"].map(group_of)
    print(f"{groups.nunique()} ambiguity groups:",
          sorted(g for g in groups.unique() if "+" in g))
    y_group = LabelEncoder().fit_transform(groups)
    group_rows, _ = run_classifiers(measured, y_group, cfg, args, top_k=3)

    if args.cnn:  # on the raw pulse response, for the main (group) result
        from ecgfd.models import cnn1d

        adc = cfg["measurement"]["adc"]
        offset, scale = 0.5 * (adc["vmin"] + adc["vmax"]), 0.5 * (adc["vmax"] - adc["vmin"])
        for repeat in range(args.repeats):
            train, val, test = split(measured, repeat, seed)
            model, _ = cnn1d.fit(wav[train], y_group[train], wav[val], y_group[val],
                                 int(y_group.max()) + 1, offset=offset, scale=scale, seed=seed)
            pred = cnn1d.predict(model, wav[test], offset, scale)
            row = {
                "feature_set": "C3 waveform", "model": "cnn1d", "repeat": repeat,
                "accuracy": float(np.mean(pred == y_group[test])),
                "f1_macro": float(
                    f1_score(y_group[test], pred, average="macro", zero_division=0)
                ),
            }
            print(row, flush=True)
            group_rows = pd.concat([group_rows, pd.DataFrame([row])], ignore_index=True)

    group_rows.to_csv(out / "localisation_by_group_repeats.csv", index=False)
    by_group = summarise(group_rows, ["feature_set", "model"])
    by_group.to_csv(out / "localisation_by_group.csv", index=False)
    print(by_group.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    rows, fitted = run_classifiers(measured, y, cfg, args, top_k=3)
    rows.to_csv(out / "localisation_repeats.csv", index=False)
    summary = summarise(rows, ["feature_set", "model"])
    summary.to_csv(out / "localisation.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # best model on the full feature set: confusion and per-component recall
    on_all = summary[summary["feature_set"] == ALL]
    best = on_all.loc[on_all[("f1_macro", "mean")].idxmax(), ("model", "")]
    model, x, test = fitted[(ALL, best)]
    matrix = confusion_matrix(y[test], model.predict(x[test]), labels=range(len(encoder.classes_)))
    names = list(encoder.classes_)
    pd.DataFrame(matrix, index=names, columns=names).rename_axis("true").to_csv(
        out / "confusion_best.csv"
    )
    cases = matrix.sum(axis=1)
    per_component = pd.DataFrame(
        {
            "group": [group_of[n] for n in names],
            "cases": cases,
            "recall": np.diag(matrix) / np.maximum(cases, 1),
        },
        index=pd.Index(names, name="component"),
    )
    info = {"best_model": best, "data": str(Path(args.data).resolve()), "n_cases": len(measured)}
    confusable = e2_confusable(cfg["circuit"], args)
    if confusable is not None:
        per_component["e2_confusable"] = [len(confusable.get(n, [])) for n in names]
        rho, pvalue = stats.spearmanr(per_component["e2_confusable"], per_component["recall"])
        info.update({"spearman_recall_vs_e2_confusable": float(rho), "p_value": float(pvalue)})
        print(f"recall against E2 confusable components: rho = {rho:.2f} (p = {pvalue:.3g})")
    per_component.to_csv(out / "recall_by_component.csv")
    save_json(info, out / "source.json")
    print(per_component.sort_values("recall").to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

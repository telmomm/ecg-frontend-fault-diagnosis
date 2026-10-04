"""E7 - Generalisation and robustness outside the training conditions.

Two tasks on the full feature set, with the models that did best in phase 6: the
pass/fail decision (gradient boosting, balanced classes) and the localisation of
the faulty component on non-compliant cases (random forest). Three questions:

1. Unseen magnitudes: parametric faults of some magnitudes are kept out of training
   and the models are scored on them, against the same models trained with every
   magnitude (in-distribution reference).
2. Noise and quantisation: ADC noise and resolution are swept, with the models
   trained either at the swept setting (matched) or at the default one (mismatched).
3. Other tolerances (optional, `--shift-data`): models trained on the main dataset
   are scored on datasets generated with other tolerance settings and seeds.
4. Common-mode test tone: its amplitude and duration are raised to see whether the
   faults of the driven-right-leg stage, which the default tone does not see, become
   detectable.

    python experiments/e7_robustness.py --data data/v1/integrated \\
        --shift-data data/shift/truncnorm/integrated data/shift/tolerance/integrated
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from _common import (
    SERIES,
    ambiguity_group_of,
    parser,
    results_dir,
    set_style,
    split,
    summarise,
)
from ecgfd.circuit import get_circuit
from ecgfd.dataset import load_dataset
from ecgfd.evaluation import decision_report, magnitude_split
from ecgfd.features import ALL, feature_sets
from ecgfd.measurement import apply_measurement_model

# held-out parametric magnitudes (both signs)
SCENARIOS = {
    "unseen_small (5 %, 10 %)": [0.05, 0.1],
    "unseen_middle (20 %)": [0.2],
    "unseen_large (50 %)": [0.5],
}
# (amplitude at the RLD reference [V], ADC samples per tone); the default is (0.1, 1000)
CM_TONES = ((0.1, 1000), (0.5, 1000), (1.0, 1000), (1.0, 10000))
NOISE_RMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)  # ADC noise [V]; the default is 1e-3
ADC_BITS = (8, 10, 12, 16)  # the default is 12


def measure(df, wav, cfg, seed, adc=None, kinds=None, ac=None):
    """Features as measured with the given ADC and tone settings; optional electrode filter."""
    if kinds:
        keep = df["electrode_kind"].isin(kinds).to_numpy()
        df, wav = df[keep].reset_index(drop=True), wav[keep]
    measured, _ = apply_measurement_model(df, wav, cfg, np.random.default_rng(seed), adc, ac)
    measured["compliant"] = measured["compliant"].astype(bool)
    return measured


class Models:
    """Pass/fail classifier and component locator, trained on the given rows."""

    group_of: dict[str, str] = {}  # target -> a-priori ambiguity group, set in main()

    def __init__(self, measured: pd.DataFrame, rows: np.ndarray, features: list[str], seed: int):
        self.features = features
        train = measured.iloc[rows]
        self.decision = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
        self.decision.fit(train[features].to_numpy(), ~train["compliant"].to_numpy())
        bad = train[~train["compliant"]]
        self.locator = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        self.locator.fit(bad[features].to_numpy(), bad["target"].to_numpy())

    def score(self, cases: pd.DataFrame) -> dict[str, float]:
        """Decision quality on `cases`, and localisation on their non-compliant part."""
        is_bad = ~cases["compliant"].to_numpy()
        pred = self.decision.predict(cases[self.features].to_numpy()).astype(bool)
        out = {"n_cases": len(cases), **decision_report(is_bad, pred)}
        if 0 < is_bad.sum() < len(cases):
            out["balanced_accuracy"] = float(balanced_accuracy_score(is_bad, pred))
        bad = cases[is_bad]
        if len(bad):
            located = self.locator.predict(bad[self.features].to_numpy())
            truth = bad["target"].to_numpy()
            out["localisation_accuracy"] = float(np.mean(located == truth))
            out["localisation_f1_macro"] = float(
                f1_score(truth, located, average="macro", zero_division=0)
            )
            # same predictions, counted as right when they fall in the true ambiguity group
            group = np.vectorize(lambda name: self.group_of.get(name, name))
            out["group_accuracy"] = float(np.mean(group(located) == group(truth)))
            out["group_f1_macro"] = float(
                f1_score(group(truth), group(located), average="macro", zero_division=0)
            )
        return out


def unseen_magnitudes(measured, features, seed, repeats) -> pd.DataFrame:
    rows = []
    parametric = (measured["kind"] == "parametric").to_numpy()
    level = measured["level"].abs().to_numpy()
    for repeat in range(repeats):
        train, val, test = split(measured, repeat, seed)
        seen = Models(measured, np.concatenate([train, val]), features, seed)
        for name, held in SCENARIOS.items():
            signed = [s * v for v in held for s in (1, -1)]
            train_h, test_h = magnitude_split(measured, signed, seed=seed + repeat)
            unseen = Models(measured, train_h, features, seed)
            target = parametric & np.isin(level, held)
            for label, model, test_rows in (("seen", seen, test), ("unseen", unseen, test_h)):
                cases = measured.iloc[test_rows[target[test_rows]]]
                rows.append({"scenario": name, "training": label, "repeat": repeat,
                             **model.score(cases)})
    return pd.DataFrame(rows)


def noise_and_quantisation(df, wav, cfg, features, seed, repeats, kinds) -> pd.DataFrame:
    settings = [("noise_rms", v, {"noise_rms": v}) for v in NOISE_RMS]
    settings += [("bits", v, {"bits": v}) for v in ADC_BITS]
    rows = []
    for repeat in range(repeats):
        default = measure(df, wav, cfg, seed + repeat, kinds=kinds)
        train, val, test = split(default, repeat, seed)
        fit_rows = np.concatenate([train, val])
        mismatched = Models(default, fit_rows, features, seed)
        for parameter, value, adc in settings:
            swept = measure(df, wav, cfg, seed + repeat, adc, kinds)
            matched = Models(swept, fit_rows, features, seed)
            for label, model in (("matched", matched), ("mismatched", mismatched)):
                rows.append({"parameter": parameter, "value": value, "training": label,
                             "repeat": repeat, **model.score(swept.iloc[test])})
        print(f"noise and quantisation: repeat {repeat} done", flush=True)
    return pd.DataFrame(rows)


def common_mode_tone(df, wav, cfg, features, seed, repeats, kinds) -> pd.DataFrame:
    """Decision quality overall and on the faults of the driven-right-leg stage."""
    circuit = get_circuit(cfg)
    rld = {c.name for c in (*circuit.passives, *circuit.opamps) if c.stage == "rld"}
    rows = []
    for repeat in range(repeats):
        for amplitude, n_samples in CM_TONES:
            ac = {"cm_amplitude": amplitude, "n_samples": n_samples}
            measured = measure(df, wav, cfg, seed + repeat, kinds=kinds, ac=ac)
            train, val, test = split(measured, repeat, seed)
            model = Models(measured, np.concatenate([train, val]), features, seed)
            cases = measured.iloc[test]
            on_rld = cases[cases["target"].isin(rld) & ~cases["compliant"]]
            pred = model.decision.predict(on_rld[features].to_numpy()).astype(bool)
            rows.append({
                "cm_amplitude": amplitude, "n_samples": n_samples, "repeat": repeat,
                **{k: v for k, v in model.score(cases).items() if not k.startswith("local")
                   and not k.startswith("group")},
                "rld_non_compliant_cases": len(on_rld),
                "rld_escape_rate": float(np.mean(~pred)) if len(on_rld) else np.nan,
            })
    return pd.DataFrame(rows)


def other_tolerances(measured, features, seed, repeats, paths, kinds) -> pd.DataFrame:
    rows = []
    for repeat in range(repeats):
        train, val, test = split(measured, repeat, seed)
        model = Models(measured, np.concatenate([train, val]), features, seed)
        rows.append({"test_data": "same distribution", "repeat": repeat,
                     **model.score(measured.iloc[test])})
        for path in paths:
            df, wav, cfg = load_dataset(path)
            shifted = measure(df, wav, cfg, seed + repeat, kinds=kinds)
            name = f"{cfg['tolerances']['distribution']}, R {cfg['tolerances']['resistor']:.0%}, " \
                   f"C {cfg['tolerances']['capacitor']:.0%}"
            rows.append({"test_data": name, "repeat": repeat, **model.score(shifted)})
    return pd.DataFrame(rows)


def noise_figure(summary: pd.DataFrame, path, circuit: str) -> None:
    panels = (("balanced_accuracy", "Pass/fail, balanced accuracy"),
              ("localisation_f1_macro", "Localisation, macro F1"))
    noise = summary[summary["parameter"] == "noise_rms"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for k, training in enumerate(("matched", "mismatched")):
            rows = noise[noise["training"] == training].sort_values("value")
            x = 1e3 * rows["value"].to_numpy(dtype=float)
            ax.fill_between(x, rows[(metric, "ci_low")].to_numpy(dtype=float),
                            rows[(metric, "ci_high")].to_numpy(dtype=float),
                            color=SERIES[k], alpha=0.2, linewidth=0)
            ax.plot(x, rows[(metric, "mean")].to_numpy(dtype=float), color=SERIES[k],
                    marker="o", markersize=4, label=f"trained {training}")
        ax.set(xscale="log", xlabel="ADC noise (mV rms)", title=title)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Robustness to measurement noise, {circuit} circuit", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = parser(__doc__.splitlines()[0], dataset=True)
    p.add_argument("--shift-data", nargs="*", default=[],
                   help="datasets of the same circuit generated with other tolerance settings")
    args = p.parse_args()
    df, wav, cfg = load_dataset(args.data)
    seed = int(cfg["seed"])
    out = results_dir("e7", cfg["circuit"], args)
    features = feature_sets(cfg)[ALL]
    measured = measure(df, wav, cfg, seed, kinds=args.electrode_kinds)
    Models.group_of = ambiguity_group_of(measured, cfg)
    set_style()

    def report(rows: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
        rows.to_csv(out / f"{name}_repeats.csv", index=False)
        summary = summarise(rows, keys)
        summary.to_csv(out / f"{name}.csv", index=False)
        means = summary[[c for c in summary.columns if c[1] in ("", "mean")]]
        print(f"--- {name}")
        print(means.droplevel(1, axis=1).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return summary

    report(unseen_magnitudes(measured, features, seed, args.repeats),
           ["scenario", "training"], "unseen_magnitudes")
    sweep = report(
        noise_and_quantisation(df, wav, cfg, features, seed, args.repeats, args.electrode_kinds),
        ["parameter", "value", "training"], "noise_quantisation",
    )
    noise_figure(sweep, out / "noise.png", cfg["circuit"])
    report(common_mode_tone(df, wav, cfg, features, seed, args.repeats, args.electrode_kinds),
           ["cm_amplitude", "n_samples"], "common_mode_tone")
    if args.shift_data:
        report(
            other_tolerances(measured, features, seed, args.repeats, args.shift_data,
                             args.electrode_kinds),
            ["test_data"], "other_tolerances",
        )
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

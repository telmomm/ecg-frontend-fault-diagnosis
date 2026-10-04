"""E8 - Minimal set of self-test measurements: cost against performance.

A measurement is one action of the instrument (a DC reading, one tone, the
calibration pulse; see `ecgfd.features.measurement_groups`). Starting from none,
the measurement that most improves the validation score is added at each step
(greedy forward selection), for two tasks: the pass/fail decision (balanced
accuracy) and the localisation of the component on non-compliant cases (macro F1).
The test score along the path gives the cost-performance curve; repeating it over
splits shows how stable the choice is.

    python experiments/e8_minimal_tests.py --data data/v1/integrated [--max-steps 8]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from _common import SERIES, load_measured, parser, results_dir, set_style, split
from ecgfd.features import MAIN_DC_NODES, measurement_groups

TOLERANCE = 0.01  # a set is "enough" when its score is within this of the full set


def decision_task(measured: pd.DataFrame, seed: int):
    y = ~measured["compliant"].to_numpy(dtype=bool)
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    return np.arange(len(measured)), y, model, balanced_accuracy_score


def localisation_task(measured: pd.DataFrame, seed: int):
    rows = np.flatnonzero(~measured["compliant"].to_numpy(dtype=bool))
    y = measured["target"].to_numpy()
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)

    def macro_f1(truth, pred):
        return f1_score(truth, pred, average="macro", zero_division=0)

    return rows, y, model, macro_f1


TASKS = {"decision": decision_task, "localisation": localisation_task}


def greedy_path(measured, groups, task, repeat, seed, max_steps) -> list[dict]:
    """Forward selection on validation; returns one record per step, plus the full set."""
    rows, y, model, metric = TASKS[task](measured, seed)
    train, val, test = (np.intersect1d(part, rows) for part in split(measured, repeat, seed))
    fit_rows = np.concatenate([train, val])

    def score(columns: list[str], fit: np.ndarray, evaluate: np.ndarray) -> float:
        x = measured[columns].to_numpy()
        return float(metric(y[evaluate], model.fit(x[fit], y[fit]).predict(x[evaluate])))

    chosen: list[str] = []
    columns: list[str] = []
    path = []
    for step in range(1, min(max_steps, len(groups)) + 1):
        candidates = {
            name: score(columns + groups[name][0], train, val)
            for name in groups
            if name not in chosen
        }
        best = max(candidates, key=candidates.get)
        chosen.append(best)
        columns += groups[best][0]
        path.append(
            {
                "task": task, "repeat": repeat, "step": step, "added": best,
                "n_features": len(columns),
                "time_s": sum(groups[name][1] for name in chosen),
                "validation_score": candidates[best],
                "test_score": score(columns, fit_rows, test),
            }
        )
        print(f"{task} repeat {repeat} step {step}: + {best:18s} "
              f"test {path[-1]['test_score']:.3f}", flush=True)
    everything = [c for cols, _ in groups.values() for c in cols]
    path.append(
        {
            "task": task, "repeat": repeat, "step": len(groups), "added": "(all measurements)",
            "n_features": len(everything),
            "time_s": sum(t for _, t in groups.values()),
            "validation_score": np.nan,
            "test_score": score(everything, fit_rows, test),
        }
    )
    return path


def curve_figure(paths: pd.DataFrame, n_groups: int, path, circuit: str) -> None:
    titles = {"decision": "Pass/fail, balanced accuracy", "localisation": "Localisation, macro F1"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, (task, title) in zip(axes, titles.items(), strict=True):
        steps = paths[(paths["task"] == task) & (paths["step"] < n_groups)]
        stats = steps.groupby("step")["test_score"].agg(["mean", "min", "max"])
        full = paths[(paths["task"] == task) & (paths["step"] == n_groups)]["test_score"].mean()
        ax.fill_between(stats.index, stats["min"], stats["max"], color=SERIES[0], alpha=0.2,
                        linewidth=0)
        ax.plot(stats.index, stats["mean"], color=SERIES[0], marker="o", markersize=4,
                label="greedy selection (range over splits)")
        ax.axhline(full, color=SERIES[1], linewidth=1.2, linestyle="--",
                   label=f"all {n_groups} measurements")
        ax.set(xlabel="Number of measurements", title=title)
        ax.set_xticks(stats.index)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Cost against performance, {circuit} circuit", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = parser(__doc__.splitlines()[0], dataset=True)
    p.add_argument("--max-steps", type=int, default=8, help="measurements added greedily")
    p.add_argument("--strict", action="store_true",
                   help="leave out the DC nodes that are not in the main feature set")
    args = p.parse_args()
    if args.strict:
        args.tag = "strict" if args.tag is None else f"{args.tag}-strict"
    measured, _, cfg = load_measured(args.data, args.electrode_kinds)
    out = results_dir("e8", cfg["circuit"], args)
    seed = int(cfg["seed"])
    groups = measurement_groups(cfg)
    if args.strict:
        extra = {f"dc_{n}" for n in cfg["measurement"]["dc_nodes"] if n not in MAIN_DC_NODES}
        groups = {name: g for name, g in groups.items() if name not in extra}

    records = []
    for task in TASKS:
        for repeat in range(args.repeats):
            records += greedy_path(measured, groups, task, repeat, seed, args.max_steps)
    paths = pd.DataFrame(records)
    paths.to_csv(out / "paths.csv", index=False)

    # how often each measurement enters, and at which step on average
    picks = paths[paths["step"] < len(groups)]
    stability = (
        picks.groupby(["task", "added"])["step"]
        .agg(times_selected="size", mean_step="mean")
        .sort_values(["task", "times_selected", "mean_step"], ascending=[True, False, True])
        .reset_index()
    )
    stability.to_csv(out / "selection_stability.csv", index=False)

    # smallest set within TOLERANCE of the full set, per task and split
    rows = []
    for (task, repeat), run in paths.groupby(["task", "repeat"]):
        full = run.loc[run["step"] == len(groups), "test_score"].iloc[0]
        steps = run[run["step"] < len(groups)]
        enough = steps[steps["test_score"] >= full - TOLERANCE]
        first = enough.iloc[0] if len(enough) else None
        rows.append(
            {
                "task": task, "repeat": repeat, "full_score": full,
                "full_time_s": run.loc[run["step"] == len(groups), "time_s"].iloc[0],
                "n_measurements": np.nan if first is None else int(first["step"]),
                "time_s": np.nan if first is None else first["time_s"],
                "score": np.nan if first is None else first["test_score"],
                "measurements": "" if first is None else
                " + ".join(steps[steps["step"] <= first["step"]]["added"]),
            }
        )
    minimal = pd.DataFrame(rows)
    minimal.to_csv(out / "minimal_sets.csv", index=False)

    set_style()
    curve_figure(paths, len(groups), out / "cost_performance.png", cfg["circuit"])
    print(stability.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(minimal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

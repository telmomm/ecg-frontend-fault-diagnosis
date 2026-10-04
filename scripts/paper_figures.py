"""Figures of the manuscript, drawn from the tables that the experiments wrote.

Nothing is simulated or trained here: every number comes from `results/` (run the
experiments first, `make all-experiments`) and the schematic from `docs/figures/`
(`python scripts/draw_schematics.py`). Output: vector PDF files in `manuscript/figures/`.

    python scripts/paper_figures.py [--results results] [--out manuscript/figures]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# names used in the manuscript: A is the realistic circuit, B the benchmark-like one
CIRCUITS = {"integrated": "Circuit A", "reference": "Circuit B"}
COLOUR = {"integrated": "#0072B2", "reference": "#D55E00"}
TITLES = {
    "integrated": "Circuit A (integrated amplifier)",
    "reference": "Circuit B (discrete amplifier)",
}
INK, MUTED, GRID = "#111111", "#666666", "#dddddd"
COLUMN, PAGE = 3.45, 7.1  # widths of one column and of the page in IEEEtran [in]
ALL = "C1+C2+C3+C4"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "text.color": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def summary(path: Path) -> pd.DataFrame:
    """Table written by `summarise`: two header rows (metric, mean / ci_low / ci_high)."""
    df = pd.read_csv(path, header=[0, 1])
    df.columns = [a if b.startswith("Unnamed") else f"{a}_{b}" for a, b in df.columns]
    return df


def compliance_by_magnitude(results: Path, out: Path) -> None:
    """Share of the circuits with one component off nominal that still comply."""
    fig, ax = plt.subplots(figsize=(COLUMN, 1.75))
    marker = {"integrated": "o", "reference": "s"}
    for circuit, label in CIRCUITS.items():
        df = pd.read_csv(results / "e4" / circuit / "compliance_by_fault.csv")
        df = df[df["kind"] == "parametric"].sort_values("level")
        x = np.arange(len(df))
        ax.plot(x, 100 * df["still_compliant"], marker=marker[circuit], ms=4, lw=1.2,
                color=COLOUR[circuit], label=label)
    ax.set_xticks(x, [f"{100 * v:+.0f}" for v in df["level"]])
    ax.set_xlabel("Deviation of one component from nominal (%)")
    ax.set_ylabel("Circuits still compliant (%)")
    ax.set_ylim(40, 100)
    ax.legend(loc="lower center", ncol=2)
    fig.savefig(out / "compliance_by_magnitude.pdf")
    plt.close(fig)


METHODS = {
    "limit_test": ("Limit test", "X"),
    "regression": ("Specification regression", "^"),
    "classifier": ("Classifier", "o"),
    "classifier+threshold": ("Classifier, 1 % escape target", "s"),
}


def operating_points(results: Path, out: Path) -> None:
    """Escapes against false rejects of each decision method, per circuit."""
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.5), sharex=True, sharey=True)
    for ax, circuit in zip(axes, CIRCUITS, strict=True):
        for tag, filled in (("", True), ("-gel-solid", False)):
            df = summary(results / "e3" / f"{circuit}{tag}" / "pass_fail.csv")
            df = df[df["feature_set"] == ALL].set_index("method")
            for method, (_, marker) in METHODS.items():
                row = 100 * df.loc[method].drop("feature_set").astype(float)
                x, y = row["false_reject_rate_mean"], row["escape_rate_mean"]
                ax.errorbar(
                    x, y,
                    xerr=[[x - max(row["false_reject_rate_ci_low"], 0.05)],
                          [row["false_reject_rate_ci_high"] - x]],
                    yerr=[[y - max(row["escape_rate_ci_low"], 0.05)],
                          [row["escape_rate_ci_high"] - y]],
                    fmt=marker, ms=5.5, mew=1.1, elinewidth=0.7, capsize=0,
                    color=COLOUR[circuit], mfc=COLOUR[circuit] if filled else "white",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.4, 100)
        ax.set_ylim(0.6, 50)
        ticks = [1, 3, 10, 30]
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
        ax.minorticks_off()
        ax.set_title(TITLES[circuit])
        ax.set_xlabel("False rejects (% of compliant circuits)")
    axes[0].set_ylabel("Escapes (% of non-compliant circuits)")
    handles = [
        plt.Line2D([], [], ls="", marker=m, color=INK, mfc=INK, ms=5.5, label=label)
        for label, m in METHODS.values()
    ]
    handles += [
        plt.Line2D([], [], ls="", marker="o", color=INK, mfc=INK, ms=5.5,
                   label="All electrode types"),
        plt.Line2D([], [], ls="", marker="o", color=INK, mfc="white", ms=5.5, mew=1.1,
                   label="Without porous dry electrodes"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17),
               columnspacing=1.5, handletextpad=0.3)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out / "operating_points_full.pdf")
    plt.close(fig)


def recall_by_component(results: Path, out: Path) -> None:
    """Recall of every component by the best component-level model, by ambiguity group."""
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 3.9))
    for ax, circuit in zip(axes, CIRCUITS, strict=True):
        df = pd.read_csv(results / "e5" / circuit / "recall_by_component.csv")
        df["grouped"] = df["group"].str.contains(r"\+")
        df["letter"] = df["component"].str[0]
        df["number"] = df["component"].str[1:].astype(int)
        df["size"] = df.groupby("group")["component"].transform("size")
        df["first"] = df.groupby("group")["number"].transform("min")
        df.loc[~df["grouped"], "first"] = 0  # lone components: by letter, then number
        df = df.sort_values(["grouped", "size", "first", "letter", "number"]).reset_index(drop=True)
        x = np.arange(len(df))
        single = ~df["grouped"].to_numpy()
        ax.bar(x[single], 100 * df["recall"][single], width=0.7, color=COLOUR[circuit])
        ax.bar(x[~single], 100 * df["recall"][~single], width=0.7, color="white",
               edgecolor=COLOUR[circuit], hatch="////", linewidth=0.8)
        # bracket under each a-priori group
        for _group, rows in df[df["grouped"]].groupby("group", sort=False):
            ax.plot([rows.index[0] - 0.35, rows.index[-1] + 0.35], [104, 104], color=INK, lw=0.9,
                    clip_on=False)
        ax.set_xticks(x, df["component"], rotation=90)
        ax.set_xlim(-0.7, len(df) - 0.3)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Recall (%)")
        ax.set_title(TITLES[circuit], pad=9)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MUTED, label="Component on its own"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, hatch="////",
                      label="Member of a predicted group (bracket)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06),
               columnspacing=1.2)
    fig.subplots_adjust(hspace=0.62)
    fig.savefig(out / "recall_by_component.pdf")
    plt.close(fig)


SCENARIOS = {
    "unseen_small (5 %, 10 %)": "5 and 10 %",
    "unseen_middle (20 %)": "20 %",
    "unseen_large (50 %)": "50 %",
}


def unseen_magnitudes(results: Path, out: Path) -> None:
    """Localisation accuracy on fault magnitudes seen or not seen in training."""
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.2), sharey=True)
    width = 0.2
    bars = (  # (column, training, offset, filled)
        ("localisation_accuracy_mean", "seen", -1.5, True),
        ("localisation_accuracy_mean", "unseen", -0.5, False),
        ("group_accuracy_mean", "seen", 0.5, True),
        ("group_accuracy_mean", "unseen", 1.5, False),
    )
    for ax, circuit in zip(axes, CIRCUITS, strict=True):
        df = summary(results / "e7" / circuit / "unseen_magnitudes.csv")
        x = np.arange(len(SCENARIOS))
        for column, training, offset, filled in bars:
            rows = df[df["training"] == training].set_index("scenario").loc[list(SCENARIOS)]
            values = 100 * rows[column].astype(float).to_numpy()
            by_group = column.startswith("group")
            ax.bar(
                x + offset * width, values, width=width * 0.9,
                color=(COLOUR[circuit] if by_group else MUTED) if filled else "white",
                edgecolor=COLOUR[circuit] if by_group else MUTED,
                hatch=None if filled else "////", linewidth=0.8,
            )
            for xi, v in zip(x + offset * width, values, strict=True):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x, [f"Held out: {v}" for v in SCENARIOS.values()])
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
        ax.set_title(TITLES[circuit])
    axes[0].set_ylabel("Correctly located (%)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MUTED, label="Component, magnitude in training"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, hatch="////",
                      label="Component, magnitude held out"),
        plt.Rectangle((0, 0), 1, 1, color=INK, label="Group, magnitude in training"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK, hatch="////",
                      label="Group, magnitude held out"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12),
               columnspacing=1.2)
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(out / "unseen_magnitudes_full.pdf")
    plt.close(fig)


TASKS = {
    "decision": "Compliance decision\n(balanced accuracy, %)",
    "localisation": "Component location\n(macro F1, %)",
}


def cost_performance(results: Path, out: Path) -> None:
    """Score against self-test time as measurements are added one at a time."""
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.2))
    style = {"integrated": ("o", "-"), "reference": ("s", "-")}
    for ax, (task, ylabel) in zip(axes, TASKS.items(), strict=True):
        for circuit, label in CIRCUITS.items():
            df = pd.read_csv(results / "e8" / circuit / "paths.csv")
            df = df[df["task"] == task]
            full = df[df["added"] == "(all measurements)"]
            path = df[df["added"] != "(all measurements)"]
            marker, line = style[circuit]
            for k, (_, rows) in enumerate(path.groupby("repeat")):
                ax.plot(rows["time_s"], 100 * rows["test_score"], line, marker=marker, ms=2.8,
                        lw=0.8, color=COLOUR[circuit], alpha=0.85,
                        label=f"{label}, greedy selection" if k == 0 else None)
            ax.axhline(100 * full["test_score"].mean(), color=COLOUR[circuit], lw=0.9, ls="--",
                       label=f"{label}, all 16 measurements (97 s)")
        ax.set_xscale("log")
        ax.set_xlim(0.8, 60)
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.minorticks_off()
        ax.set_xlabel("Duration of the self-test (s)")
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(86, 98)
    axes[1].set_ylim(35, 80)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2),
               columnspacing=1.2)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out / "cost_performance_full.pdf")
    plt.close(fig)


# --- single-column versions for the main text (the page-wide ones go to the supplement) ---


def operating_points_compact(results: Path, out: Path) -> None:
    """Both circuits on one pair of axes: color is the circuit, marker the method."""
    fig, ax = plt.subplots(figsize=(COLUMN, 2.5))
    for circuit in CIRCUITS:
        for tag, filled in (("", True), ("-gel-solid", False)):
            df = summary(results / "e3" / f"{circuit}{tag}" / "pass_fail.csv")
            df = df[df["feature_set"] == ALL].set_index("method")
            for method, (_, marker) in METHODS.items():
                row = 100 * df.loc[method].drop("feature_set").astype(float)
                ax.plot(row["false_reject_rate_mean"], row["escape_rate_mean"], marker, ms=5,
                        mew=1.1, color=COLOUR[circuit],
                        mfc=COLOUR[circuit] if filled else "white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 100)
    ax.set_ylim(0.6, 50)
    ax.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
    ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_xlabel("False rejects (% of compliant circuits)")
    ax.set_ylabel("Escapes (% of non-compliant)")
    handles = [plt.Line2D([], [], ls="", marker="o", color=COLOUR[c], ms=5, label=label)
               for c, label in CIRCUITS.items()]
    handles += [plt.Line2D([], [], ls="", marker=m, color=INK, ms=5, label=label)
                for label, m in METHODS.values()]
    handles += [
        plt.Line2D([], [], ls="", marker="o", color=INK, ms=5, label="All electrode types"),
        plt.Line2D([], [], ls="", marker="o", color=INK, mfc="white", ms=5, mew=1.1,
                   label="Without porous dry electrodes"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.33),
               columnspacing=1.0, handletextpad=0.2)
    fig.savefig(out / "operating_points.pdf")
    plt.close(fig)


def unseen_magnitudes_compact(results: Path, out: Path) -> None:
    """Held-out magnitudes only: located component against located group, both circuits."""
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    width = 0.2
    x = np.arange(len(SCENARIOS))
    offset = -1.5
    for circuit in CIRCUITS:
        df = summary(results / "e7" / circuit / "unseen_magnitudes.csv")
        rows = df[df["training"] == "unseen"].set_index("scenario").loc[list(SCENARIOS)]
        columns = (("localisation_accuracy_mean", False), ("group_accuracy_mean", True))
        for column, filled in columns:
            values = 100 * rows[column].astype(float).to_numpy()
            ax.bar(x + offset * width, values, width=width * 0.9,
                   color=COLOUR[circuit] if filled else "white", edgecolor=COLOUR[circuit],
                   hatch=None if filled else "////", linewidth=0.8)
            for xi, v in zip(x + offset * width, values, strict=True):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
            offset += 1.0
    ax.set_xticks(x, [f"Held out: {v}" for v in SCENARIOS.values()])
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Correctly located (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[c], label=label)
               for c, label in CIRCUITS.items()]
    handles += [
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK, hatch="////", label="Component"),
        plt.Rectangle((0, 0), 1, 1, color=INK, label="Ambiguity group"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12),
               columnspacing=1.0, handletextpad=0.4)
    fig.savefig(out / "unseen_magnitudes.pdf")
    plt.close(fig)


def cost_performance_compact(results: Path, out: Path) -> None:
    """Compliance decision only."""
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    for circuit, label in CIRCUITS.items():
        df = pd.read_csv(results / "e8" / circuit / "paths.csv")
        df = df[df["task"] == "decision"]
        full = df[df["added"] == "(all measurements)"]
        path = df[df["added"] != "(all measurements)"]
        marker = "o" if circuit == "integrated" else "s"
        for k, (_, rows) in enumerate(path.groupby("repeat")):
            ax.plot(rows["time_s"], 100 * rows["test_score"], marker=marker, ms=2.8, lw=0.8,
                    color=COLOUR[circuit], alpha=0.85, label=label if k == 0 else None)
        ax.axhline(100 * full["test_score"].mean(), color=COLOUR[circuit], lw=0.9, ls="--")
    ax.plot([], [], color=INK, lw=0.9, ls="--", label="All measurements (97 s)")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 60)
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(86, 98)
    ax.set_xlabel("Duration of the self-test (s)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.legend(loc="lower right")
    fig.savefig(out / "cost_performance.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=str(REPO_ROOT / "results"))
    p.add_argument("--out", default=str(REPO_ROOT / "manuscript" / "figures"))
    args = p.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_style()
    compliance_by_magnitude(results, out)
    operating_points(results, out)
    recall_by_component(results, out)
    unseen_magnitudes(results, out)
    cost_performance(results, out)
    operating_points_compact(results, out)
    unseen_magnitudes_compact(results, out)
    cost_performance_compact(results, out)
    for circuit in CIRCUITS:
        source = REPO_ROOT / "docs" / "figures" / f"schematic_{circuit}.pdf"
        if source.exists():
            shutil.copy(source, out / source.name)
        else:
            print(f"missing {source}: run scripts/draw_schematics.py")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()

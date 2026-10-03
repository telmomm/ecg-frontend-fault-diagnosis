"""Shared helpers for the experiment scripts: paths, arguments, data and figure style."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecgfd.circuit import CIRCUITS  # noqa: E402
from ecgfd.config import DEFAULT_CONFIG, REPO_ROOT  # noqa: E402
from ecgfd.dataset import load_dataset  # noqa: E402
from ecgfd.evaluation import localisation_report  # noqa: E402
from ecgfd.features import feature_sets  # noqa: E402
from ecgfd.measurement import apply_measurement_model  # noqa: E402
from ecgfd.models.classical import classifiers  # noqa: E402

# Categorical colours, assigned in this fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "figure.dpi": 150,
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": "#52514e",
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 1.6,
            "axes.prop_cycle": plt.cycler(color=SERIES),
            "legend.frameon": False,
        }
    )


def parser(description: str, dataset: bool = False) -> argparse.ArgumentParser:
    """Arguments of a script working on a dataset folder, or on a configuration."""
    p = argparse.ArgumentParser(description=description)
    if dataset:
        default = REPO_ROOT / "data" / "smoke" / "integrated"
        p.add_argument("--data", default=str(default), help="dataset folder (one circuit)")
        p.add_argument("--models", nargs="*", help="subset of models to run")
        p.add_argument(
            "--electrode-kinds",
            nargs="*",
            help="keep only cases with these electrode types (column electrode_kind)",
        )
        p.add_argument("--tag", help="results folder suffix ('subset' if --electrode-kinds is set)")
    else:
        p.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML study configuration")
        p.add_argument("--circuit", choices=sorted(CIRCUITS), help="override the config's circuit")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    return p


def output_name(circuit: str, args: argparse.Namespace | None = None) -> str:
    """Results subfolder: the circuit, plus a tag when the data were filtered."""
    tag = getattr(args, "tag", None)
    if tag is None and getattr(args, "electrode_kinds", None):
        tag = "subset"
    return circuit if tag is None else f"{circuit}-{tag}"


def results_dir(experiment: str, circuit: str, args: argparse.Namespace | None = None) -> Path:
    path = REPO_ROOT / "results" / experiment / output_name(circuit, args)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2))


def load_measured(
    path: str, electrode_kinds: list[str] | None = None
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Dataset as the instrument would measure it: default ADC noise and quantisation.

    `electrode_kinds` keeps only the cases with those electrode types, e.g. to study
    the circuit without porous dry electrodes without simulating again.
    """
    df, waveforms, cfg = load_dataset(path)
    if electrode_kinds:
        unknown = set(electrode_kinds) - set(df["electrode_kind"])
        if unknown:
            raise ValueError(f"unknown electrode types: {sorted(unknown)}")
        keep = df["electrode_kind"].isin(electrode_kinds).to_numpy()
        df, waveforms = df[keep].reset_index(drop=True), waveforms[keep]
    rng = np.random.default_rng(int(cfg["seed"]))
    measured, wav = apply_measurement_model(df, waveforms, cfg, rng)
    return measured, wav, cfg


def compare_classifiers(
    measured: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    cfg: dict,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Accuracy and macro F1 of every reference classifier on every feature set."""
    rows = []
    for set_name, features in feature_sets(cfg).items():
        x = measured[features].to_numpy()
        for model_name, model in classifiers(int(cfg["seed"])).items():
            if models and model_name not in models:
                continue
            model.fit(x[train], y[train])
            row = {"feature_set": set_name, "model": model_name}
            row.update(localisation_report(y[test], model.predict(x[test])))
            rows.append(row)
            print(row, flush=True)
    return pd.DataFrame(rows)

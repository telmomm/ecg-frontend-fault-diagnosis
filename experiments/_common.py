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
from scipy import stats  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    top_k_accuracy_score,
)

from ecgfd.ambiguity import (  # noqa: E402
    a_priori_groups,
    normalised_sensitivity,
    sensitivity_matrix,
)
from ecgfd.circuit import CIRCUITS  # noqa: E402
from ecgfd.config import DEFAULT_CONFIG, REPO_ROOT  # noqa: E402
from ecgfd.dataset import load_dataset  # noqa: E402
from ecgfd.evaluation import replica_split  # noqa: E402
from ecgfd.features import feature_sets  # noqa: E402
from ecgfd.measurement import apply_measurement_model  # noqa: E402
from ecgfd.models.classical import classifiers, tune  # noqa: E402

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
        p.add_argument("--repeats", type=int, default=3, help="train/validation/test repetitions")
        p.add_argument(
            "--known-electrode",
            action="store_true",
            help="give the electrode type to the models as an extra input",
        )
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
    if getattr(args, "known_electrode", False):
        tag = "known-electrode" if tag is None else f"{tag}-known-electrode"
    return circuit if tag is None else f"{circuit}-{tag}"


def results_root(args: argparse.Namespace | None = None) -> Path:
    """`results/`, or `results/smoke/` when the data are the smoke datasets.

    Keeps trial runs on the tiny datasets from overwriting the results of the study.
    """
    data = getattr(args, "data", None)
    release = Path(data).parent.name if data else Path(getattr(args, "data_dir", "") or "").name
    root = REPO_ROOT / "results"
    return root / "smoke" if release == "smoke" else root


def results_dir(experiment: str, circuit: str, args: argparse.Namespace | None = None) -> Path:
    path = results_root(args) / experiment / output_name(circuit, args)
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


def ambiguity_group_of(measured: pd.DataFrame, cfg: dict) -> dict[str, str]:
    """Target -> a-priori ambiguity group, from the sensitivities of the nominal circuit.

    Passives with parallel sensitivity vectors share a group ("R5+R6+R11+R12"); every
    other target (op-amps, the INA, lone passives) is its own group. Nothing here
    looks at the fault data, so the groups are known before any model is trained.
    """
    healthy_spread = measured.loc[measured["kind"] == "healthy"].std(numeric_only=True)
    z = normalised_sensitivity(sensitivity_matrix(cfg), healthy_spread)
    groups = a_priori_groups(z)
    return {target: groups.get(target, target) for target in measured["target"].unique()}


def split(measured: pd.DataFrame, repeat: int, seed: int) -> tuple[np.ndarray, ...]:
    """(train, validation, test) row indices of one repetition: 52.5 % / 17.5 % / 30 %.

    Every condition appears in the three parts. Hyper-parameters, guard bands and
    thresholds are chosen on the validation part; the test part is only scored.
    """
    rest, test = replica_split(measured, test_fraction=0.3, seed=seed + repeat)
    sub_train, sub_val = replica_split(measured.iloc[rest], 0.25, seed=seed + 1000 + repeat)
    return rest[sub_train], rest[sub_val], test


def feature_matrix(measured: pd.DataFrame, features: list[str], args) -> np.ndarray:
    """Features as an array; with --known-electrode, plus the one-hot electrode type."""
    x = measured[features].to_numpy(dtype=float)
    if getattr(args, "known_electrode", False):
        kinds = pd.get_dummies(measured["electrode_kind"]).to_numpy(dtype=float)
        x = np.hstack([x, kinds])
    return x


def mean_ci(values: pd.Series) -> pd.Series:
    """Mean and 95 % confidence interval of the mean over repetitions (Student's t)."""
    v = values.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return pd.Series({"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan})
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return pd.Series({"mean": v.mean(), "ci_low": v.mean() - half, "ci_high": v.mean() + half})


def summarise(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per `keys` with mean and confidence interval of every metric column."""
    metrics = [c for c in rows.columns if c not in keys and c != "repeat"]
    parts = {m: rows.groupby(keys, sort=False)[m].apply(mean_ci).unstack() for m in metrics}
    return pd.concat(parts, axis=1).reset_index()


def classification_scores(model, x: np.ndarray, y: np.ndarray, top_k: int = 0) -> dict:
    """Accuracy, balanced accuracy and macro F1; top-k accuracy if the model gives scores."""
    pred = model.predict(x)
    scores = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    if top_k and hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        scores[f"top{top_k}_accuracy"] = float(
            top_k_accuracy_score(y, proba, k=top_k, labels=model.classes_)
        )
    return scores


def run_classifiers(
    measured: pd.DataFrame,
    y: np.ndarray,
    cfg: dict,
    args: argparse.Namespace,
    default_models: tuple[str, ...] | None = None,
    balanced: bool = False,
    top_k: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Tune and score every classifier on every feature set over `args.repeats` splits.

    Hyper-parameters are tuned on the validation part of the first repetition and
    reused in the others; each repetition refits on train + validation and scores
    the test part. Returns (one row per repetition, last fitted model per
    (feature set, model) together with its test indices).
    """
    seed = int(cfg["seed"])
    wanted = args.models or default_models
    splits = [split(measured, r, seed) for r in range(args.repeats)]
    rows, fitted = [], {}
    for set_name, features in feature_sets(cfg).items():
        x = feature_matrix(measured, features, args)
        for model_name, (model, grid) in classifiers(seed, balanced).items():
            if wanted and model_name not in wanted:
                continue
            train, val, _ = splits[0]
            params = tune(model, grid, x[train], y[train], x[val], y[val])
            model.set_params(**params)
            for repeat, (train, val, test) in enumerate(splits):
                fit_rows = np.concatenate([train, val])
                model.fit(x[fit_rows], y[fit_rows])
                row = {"feature_set": set_name, "model": model_name, "repeat": repeat}
                row.update(classification_scores(model, x[test], y[test], top_k))
                rows.append(row)
            fitted[(set_name, model_name)] = (model, x, test)
            print(f"{set_name:12s} {model_name:18s} f1_macro {row['f1_macro']:.3f}  {params}",
                  flush=True)
    return pd.DataFrame(rows), fitted

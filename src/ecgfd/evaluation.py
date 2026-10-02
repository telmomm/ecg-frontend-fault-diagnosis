"""Metrics and leakage-free data splits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def false_reject_rate(is_bad: np.ndarray, pred_bad: np.ndarray) -> float:
    """Fraction of good circuits rejected (false alarms)."""
    good = ~np.asarray(is_bad, dtype=bool)
    return float(np.asarray(pred_bad, dtype=bool)[good].mean()) if good.any() else np.nan


def escape_rate(is_bad: np.ndarray, pred_bad: np.ndarray) -> float:
    """Fraction of bad circuits accepted: the critical figure for a medical device."""
    bad = np.asarray(is_bad, dtype=bool)
    return float((~np.asarray(pred_bad, dtype=bool))[bad].mean()) if bad.any() else np.nan


def decision_report(is_bad: np.ndarray, pred_bad: np.ndarray) -> dict[str, float]:
    """Pass/fail decision quality; "bad" is non-compliant (or faulty, for comparison)."""
    return {
        "escape_rate": escape_rate(is_bad, pred_bad),
        "false_reject_rate": false_reject_rate(is_bad, pred_bad),
        "accuracy": float(accuracy_score(is_bad, pred_bad)),
    }


def localisation_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def escape_rate_by(
    df: pd.DataFrame, is_bad: np.ndarray, pred_bad: np.ndarray, by=("kind", "level")
) -> pd.DataFrame:
    """Escape rate broken down by fault type and magnitude (bad rows only)."""
    bad = np.asarray(is_bad, dtype=bool)
    sub = df.loc[bad, list(by)].copy()
    sub["escaped"] = ~np.asarray(pred_bad, dtype=bool)[bad]
    return sub.groupby(list(by))["escaped"].agg(rate="mean", n="size").reset_index()


def centroid_separability(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean distance between class centroids over mean within-class spread (E9).

    Features are standardised first. Larger means classes are easier to separate.
    """
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)
    classes = np.unique(labels)
    centroids = np.array([x[labels == c].mean(axis=0) for c in classes])
    within = np.mean([np.linalg.norm(x[labels == c] - centroids[i], axis=1).mean()
                      for i, c in enumerate(classes)])
    dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    between = dist[np.triu_indices(len(classes), k=1)].mean()
    return float(between / max(within, 1e-12))


def bootstrap_ci(
    metric: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of `metric(*arrays)` over samples."""
    rng = np.random.default_rng(seed)
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric(*(a[idx] for a in arrays)))
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def replica_split(
    df: pd.DataFrame, test_fraction: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split stratified by condition: every condition appears in train and test.

    It does not test generalisation to unseen magnitudes (see `magnitude_split`).
    """
    rng = np.random.default_rng(seed)
    test = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("condition").indices.items():
        idx = rng.permutation(idx)
        test[idx[: max(1, round(test_fraction * len(idx)))]] = True
    return np.flatnonzero(~test), np.flatnonzero(test)


def magnitude_split(
    df: pd.DataFrame, test_levels: list[float], test_fraction: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole parametric magnitudes (E7).

    Parametric faults with `level` in `test_levels` go entirely to the test set and
    the remaining parametric faults entirely to training; other rows (healthy, hard
    faults, ...) are split by replica as in `replica_split`.
    """
    train, test = replica_split(df, test_fraction, seed)
    is_test = np.zeros(len(df), dtype=bool)
    is_test[test] = True
    parametric = (df["kind"] == "parametric").to_numpy()
    held_out = parametric & np.isin(df["level"].to_numpy(), test_levels)
    is_test[parametric] = held_out[parametric]
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)

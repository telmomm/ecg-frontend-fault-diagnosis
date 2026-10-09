"""Decision and localisation metrics.

Leakage-free splits and bootstrap intervals come from `spicefault`
(`spicefault.dataset.split_by_replica`, `spicefault.reliability.bootstrap_interval`).
"""

from __future__ import annotations

import numpy as np
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

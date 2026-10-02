"""Fault dictionary, sensitivities and ambiguity groups (experiment E2).

First-pass, model-free analysis of what a feature set can distinguish before any
classifier is trained. Two conditions are called indistinguishable when no single
feature separates their Monte Carlo clouds by more than `threshold` pooled standard
deviations. This univariate criterion is conservative (a multivariate classifier
may still separate them) and is meant as a map, not as a final verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .circuit import get_circuit, nominal_instance
from .simulate import measure, scalar_features


def fault_dictionary(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and standard deviation of every feature per condition."""
    grouped = df.groupby("condition", sort=False)[features]
    return grouped.mean(), grouped.std(ddof=0)


def separability(
    mean: pd.DataFrame, std: pd.DataFrame, floor: pd.Series | None = None
) -> pd.DataFrame:
    """Pairwise d' between conditions: max over features of |mu_i - mu_j| / pooled sigma.

    `floor` is a per-feature lower bound of sigma (e.g. the measurement noise), which
    keeps numerically identical clouds from looking infinitely separable.
    """
    mu, var = mean.to_numpy(), std.to_numpy() ** 2
    if floor is not None:
        var = np.maximum(var, floor.reindex(mean.columns).to_numpy() ** 2)
    diff = np.abs(mu[:, None, :] - mu[None, :, :])
    pooled = np.sqrt(0.5 * (var[:, None, :] + var[None, :, :]))
    d = np.max(diff / np.maximum(pooled, 1e-300), axis=2)
    return pd.DataFrame(d, index=mean.index, columns=mean.index)


def ambiguity_groups(d: pd.DataFrame, threshold: float = 3.0) -> list[list[str]]:
    """Connected components of the graph linking conditions with d' < threshold.

    Groups are returned largest first. Linking is transitive, so a chain of similar
    conditions merges into one group; to list undetectable faults compare each
    condition with "healthy" directly (`d.loc["healthy"] < threshold`).
    """
    names = list(d.index)
    parent = list(range(len(names)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    close = d.to_numpy() < threshold
    for i, j in zip(*np.nonzero(np.triu(close, k=1)), strict=True):
        parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        groups.setdefault(find(i), []).append(name)
    return sorted(groups.values(), key=len, reverse=True)


def sensitivity_matrix(cfg: dict, rel_step: float = 0.01) -> pd.DataFrame:
    """Normalised sensitivities of the scalar features to each passive, at nominal.

    Entry [feature, component] is the change of the feature for a +1 % change of the
    component, by central differences. Columns that are proportional to each other
    reveal components that cannot be told apart by small deviations.
    """
    base = nominal_instance(cfg)
    columns = {}
    for p in get_circuit(cfg).passives:
        feats = []
        for sign in (+1.0, -1.0):
            inst = base.copy()
            inst.passives[p.name] *= 1.0 + sign * rel_step
            feats.append(pd.Series(scalar_features(measure(inst, cfg), cfg)))
        columns[p.name] = (feats[0] - feats[1]) / (2.0 * rel_step) * 0.01
    return pd.DataFrame(columns)

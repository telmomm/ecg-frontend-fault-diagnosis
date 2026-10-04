"""Testability analysis (experiments E2 and E9): sensitivities, detectability, ambiguity.

Model-free analysis of what a feature set can distinguish before any classifier is
trained, in two complementary views:

- **Small deviations** (`sensitivity_matrix`): how much each feature moves for a 1 %
  change of each component, in units of the healthy spread. A deviation is called
  visible when it moves the features by at least `threshold` healthy standard
  deviations; components whose sensitivity vectors are parallel cannot be told apart
  by small deviations, and the rank of the matrix bounds how many can
  (`testability_rank`, `collinear_groups`).
- **Simulated faults** (`fault_dictionary`, `separability`): two conditions are called
  indistinguishable when no single feature separates their Monte Carlo clouds by more
  than `threshold` pooled spreads. Centres and spreads are robust (median and
  IQR / 1.349), because faults that saturate the output give heavy-tailed clouds
  whose mean and standard deviation hide an obvious shift. This univariate criterion
  is conservative (a multivariate classifier may still separate them); it is a map,
  not a verdict. `envelope_detection` is the classical limit-test counterpart.
  `confusable_components` and `component_groups` lift the condition graph to
  components, which is the level E5 has to locate.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .circuit import get_circuit, nominal_instance
from .features import pulse_features
from .simulate import measure, scalar_features


def _groups(names: list[str], pairs: Iterable[tuple[int, int]]) -> list[list[str]]:
    """Connected components of the graph on `names` with edges `pairs`, largest first."""
    parent = list(range(len(names)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        groups.setdefault(find(i), []).append(name)
    return sorted(groups.values(), key=len, reverse=True)


# --- simulated faults ---------------------------------------------------------------


def fault_dictionary(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust centre and spread of every feature per condition: median and IQR / 1.349.

    The spread equals the standard deviation for a normal cloud and is not inflated
    by the few extreme values of a saturating fault.
    """
    grouped = df.groupby("condition", sort=False)[features]
    spread = (grouped.quantile(0.75) - grouped.quantile(0.25)) / 1.349
    return grouped.median(), spread


def envelope_limits(
    healthy: pd.DataFrame, features: list[str], coverage: float = 0.99
) -> tuple[pd.Series, pd.Series]:
    """Per-feature limits of a limit test, from healthy cases.

    Each feature gets the central interval of the healthy cases, widened so that the
    union of all features keeps about `coverage` of them (Bonferroni).
    """
    tail = (1.0 - coverage) / (2 * len(features))
    return healthy[features].quantile(tail), healthy[features].quantile(1.0 - tail)


def envelope_flags(df: pd.DataFrame, limits: tuple[pd.Series, pd.Series]) -> np.ndarray:
    """True for the cases with any feature outside its limits."""
    lo, hi = limits
    return ((df[lo.index] < lo) | (df[hi.index] > hi)).any(axis=1).to_numpy()


def envelope_detection(
    df: pd.DataFrame, features: list[str], coverage: float = 0.99
) -> tuple[pd.Series, float]:
    """Limit test: (fraction of each condition's cases flagged, false-alarm rate)."""
    limits = envelope_limits(df[df["kind"] == "healthy"], features, coverage)
    flagged = pd.Series(envelope_flags(df, limits), index=df.index)
    rate = flagged.groupby(df["condition"]).mean()
    return rate.drop("healthy"), float(rate["healthy"])


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

    Linking is transitive, so a chain of similar conditions merges into one group; to
    list undetectable faults compare each condition with "healthy" directly
    (`d.loc["healthy"] < threshold`).
    """
    close = np.triu(d.to_numpy() < threshold, k=1)
    return _groups(list(d.index), zip(*np.nonzero(close), strict=True))


def confusable_components(
    d: pd.DataFrame, component_of: dict[str, str], threshold: float = 3.0
) -> dict[str, list[str]]:
    """For each component, the other components with a condition closer than `threshold`.

    Not transitive: it answers "a case of this component could be mistaken for which
    others?". A component with an empty list can be located without ambiguity.
    """
    conditions = list(d.index)
    partners: dict[str, set[str]] = {component_of[c]: set() for c in conditions}
    close = np.triu(d.to_numpy() < threshold, k=1)
    for i, j in zip(*np.nonzero(close), strict=True):
        a, b = component_of[conditions[i]], component_of[conditions[j]]
        if a != b:
            partners[a].add(b)
            partners[b].add(a)
    return {name: sorted(others) for name, others in sorted(partners.items())}


def component_groups(
    d: pd.DataFrame, component_of: dict[str, str], threshold: float = 3.0
) -> list[list[str]]:
    """Transitive closure of `confusable_components`: groups of mutually linked components.

    `component_of` maps each condition of `d` to its component (the `target` column).
    Chains of similar components merge into one group, so this is an upper bound of
    the ambiguity; `confusable_components` gives the direct relations.
    """
    conditions = list(d.index)
    components = sorted({component_of[c] for c in conditions})
    index = {name: k for k, name in enumerate(components)}
    close = np.triu(d.to_numpy() < threshold, k=1)
    pairs = [
        (index[component_of[conditions[i]]], index[component_of[conditions[j]]])
        for i, j in zip(*np.nonzero(close), strict=True)
    ]
    return _groups(components, pairs)


# --- small deviations ---------------------------------------------------------------


def sensitivity_matrix(cfg: dict, rel_step: float = 0.01) -> pd.DataFrame:
    """Sensitivities of the features to each passive, at the nominal circuit.

    Entry [feature, component] is the change of the feature for a +1 % change of the
    component, by central differences. Rows are the noise-free scalar features and
    the pulse descriptors (set C3).
    """
    base = nominal_instance(cfg)

    def features(inst) -> pd.Series:
        m = measure(inst, cfg)
        pulse = pulse_features(m.pulse[None, :], cfg).iloc[0]
        return pd.concat([pd.Series(scalar_features(m, cfg)), pulse])

    columns = {}
    for p in get_circuit(cfg).passives:
        plus, minus = base.copy(), base.copy()
        plus.passives[p.name] *= 1.0 + rel_step
        minus.passives[p.name] *= 1.0 - rel_step
        columns[p.name] = (features(plus) - features(minus)) / (2.0 * rel_step) * 0.01
    return pd.DataFrame(columns)


def normalised_sensitivity(sensitivity: pd.DataFrame, spread: pd.Series) -> pd.DataFrame:
    """Sensitivity in units of the healthy spread of each feature (z per 1 % change).

    A value of 1 means that a 1 % deviation of the component moves the feature by one
    standard deviation of the healthy population. Features without spread are dropped.
    """
    spread = spread.reindex(sensitivity.index)
    keep = spread.notna() & (spread > 0)
    return sensitivity[keep].div(spread[keep], axis=0)


def testability_rank(z: pd.DataFrame, deviation_pct: float = 10.0, threshold: float = 3.0) -> int:
    """Number of independent directions along which a small deviation is visible.

    Counts the singular values s of the normalised sensitivity matrix (z per 1 %) with
    s * `deviation_pct` >= `threshold`: directions along which a deviation of that size
    moves the features by at least `threshold` healthy standard deviations. It bounds
    how many components can be told apart by deviations of that size.
    """
    if z.empty:
        return 0
    singular = np.linalg.svd(z.to_numpy(), compute_uv=False)
    return int(np.sum(singular * deviation_pct >= threshold))


def a_priori_groups(
    z: pd.DataFrame, cos_threshold: float = 0.99, min_relative_norm: float = 0.01
) -> dict[str, str]:
    """Component -> ambiguity group, predicted from the sensitivities alone.

    Components whose sensitivity vectors are parallel form a group, named after its
    members ("R5+R6"); every other component is its own group. This is about which
    components act on the measurements in the same way, whatever the size of the
    effect, so no visibility threshold is applied. Components whose sensitivity is
    below `min_relative_norm` of the largest one are left alone: the direction of a
    nearly null vector is numerical noise.
    """
    norms = np.linalg.norm(z.to_numpy(), axis=0)
    floor = min_relative_norm * norms.max()
    keep = [c for c, n in zip(z.columns, norms, strict=True) if n >= floor]
    v = z[keep].to_numpy() / norms[[z.columns.get_loc(c) for c in keep]]
    close = np.triu(np.abs(v.T @ v) >= cos_threshold, k=1)
    mapping = {name: name for name in z.columns}
    for members in _groups(keep, zip(*np.nonzero(close), strict=True)):
        if len(members) > 1:
            label = "+".join(sorted(members, key=lambda n: (n[0], int(n[1:]))))
            mapping.update({name: label for name in members})
    return mapping


def collinear_groups(
    z: pd.DataFrame,
    cos_threshold: float = 0.99,
    deviation_pct: float = 10.0,
    threshold: float = 3.0,
) -> tuple[list[list[str]], list[str]]:
    """(groups of components with parallel sensitivity vectors, insensitive components).

    A component is insensitive when a deviation of `deviation_pct` moves the features
    by less than `threshold` healthy standard deviations (Euclidean norm). The rest are
    linked when the absolute cosine of their sensitivity vectors reaches
    `cos_threshold`, since then a small deviation of one can be mimicked by the other.
    """
    min_norm = threshold / deviation_pct
    norms = np.linalg.norm(z.to_numpy(), axis=0)
    sensitive = [c for c, n in zip(z.columns, norms, strict=True) if n >= min_norm]
    insensitive = [c for c, n in zip(z.columns, norms, strict=True) if n < min_norm]
    v = z[sensitive].to_numpy()
    v = v / np.maximum(np.linalg.norm(v, axis=0), 1e-300)
    cos = np.abs(v.T @ v)
    close = np.triu(cos >= cos_threshold, k=1)
    return _groups(sensitive, zip(*np.nonzero(close), strict=True)), insensitive

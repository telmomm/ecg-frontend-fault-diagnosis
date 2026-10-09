"""E3 - Predicting the specifications from self-test measurements (H1).

Alternate-test style: one regressor per specification, then a pass/fail decision by
comparing the predictions with the limits. A direct compliant/non-compliant
classifier is the reference. Baseline with untuned gradient boosting and a single
split; guard bands on the predicted values are the next step to trade escapes
against false rejects.

    python experiments/e3_spec_prediction.py --data data/v1/integrated
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from spicefault.dataset import split_by_replica
from spicefault.reliability import bootstrap_interval

from _common import load_measured, parser, results_dir
from ecgfd.evaluation import decision_report, escape_rate
from ecgfd.features import feature_sets
from ecgfd.specs import specifications

CLIP = 3.0  # targets are clipped at CLIP x limit, and min specs also at limit / CLIP


def clipped_target(values: pd.Series, upper: bool, limit: float) -> np.ndarray:
    """Bound the regression target: far beyond the limit, only "fails badly" matters.

    `upper` says that the limit is a maximum. Specifications that could not be
    computed (NaN, dead circuit) take the worst value.
    """
    if upper:
        return values.fillna(np.inf).clip(upper=CLIP * limit).to_numpy()
    return values.fillna(-np.inf).clip(lower=limit / CLIP, upper=CLIP * limit).to_numpy()


def main() -> None:
    args = parser(__doc__.splitlines()[0], dataset=True).parse_args()
    measured, _, cfg = load_measured(args.data, args.electrode_kinds)
    out = results_dir("e3", cfg["circuit"], args)
    seed = int(cfg["seed"])
    train, test = split_by_replica(measured, test_fraction=0.3, seed=seed)
    limits = specifications(cfg)
    is_bad = ~measured["compliant"].to_numpy(dtype=bool)

    errors, decisions = [], []
    for set_name, features in feature_sets(cfg).items():
        x = measured[features].to_numpy()
        pred_bad = np.zeros(len(test), dtype=bool)
        for spec in limits:
            upper = spec.minimum is None
            limit = spec.maximum if upper else spec.minimum
            y = clipped_target(measured[spec.name], upper, limit)
            model = HistGradientBoostingRegressor(random_state=seed).fit(x[train], y[train])
            pred = model.predict(x[test])
            pred_bad |= pred > limit if upper else pred < limit
            rmse = float(np.sqrt(np.mean((pred - y[test]) ** 2)))
            name = spec.name.removeprefix("spec_")
            errors.append({"feature_set": set_name, "spec": name, "rmse_over_limit": rmse / limit})

        direct = HistGradientBoostingClassifier(random_state=seed).fit(x[train], is_bad[train])
        methods = {"regression": pred_bad, "classifier": direct.predict(x[test])}
        for method, prediction in methods.items():
            row = {"feature_set": set_name, "method": method}
            row.update(decision_report(is_bad[test], prediction))
            truth, n = is_bad[test], len(test)

            def resampled(rng, truth=truth, prediction=prediction, n=n) -> float:
                rows = rng.integers(0, n, n)
                return escape_rate(truth[rows], prediction[rows])

            lo, hi = bootstrap_interval(resampled, seed=seed)
            row.update({"escape_ci_low": lo, "escape_ci_high": hi})
            decisions.append(row)

    errors = pd.DataFrame(errors)
    errors = errors.pivot(index="spec", columns="feature_set", values="rmse_over_limit")
    decisions = pd.DataFrame(decisions)
    errors.to_csv(out / "spec_rmse_over_limit.csv")
    decisions.to_csv(out / "pass_fail.csv", index=False)
    print(errors.to_string(float_format=lambda v: f"{v:.3f}"))
    print(decisions.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

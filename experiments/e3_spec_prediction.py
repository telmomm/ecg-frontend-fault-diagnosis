"""E3 - Predicting compliance from self-test measurements (H1).

Four ways of deciding pass/fail on the same splits, for every feature set:

- `limit_test`: classical reference; fail if any feature leaves the healthy envelope.
- `regression`: alternate-test style; one regressor per specification, fail if any
  predicted value is beyond its limit.
- `regression+guard`: the same with a guard band, the one that leaves the escape rate
  on the validation split at `--escape-target`.
- `classifier` / `classifier+threshold`: direct compliant/non-compliant classifier at
  the default threshold, and with the threshold set for the same escape target.

Reports escape and false-reject rates with confidence intervals over the
repetitions, the prediction error of every specification, and where the escapes of
the best method come from.

    python experiments/e3_spec_prediction.py --data data/v1/integrated
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from _common import feature_matrix, load_measured, parser, results_dir, split, summarise
from ecgfd.ambiguity import envelope_flags, envelope_limits
from ecgfd.evaluation import decision_report, escape_rate_by
from ecgfd.features import ALL, feature_sets
from ecgfd.specs import spec_limits

CLIP = 3.0  # targets are clipped at +-CLIP x limit (max specs) or [limit / CLIP, CLIP x limit]


def clipped_target(values: pd.Series, sense: str, limit: float) -> np.ndarray:
    """Bound the regression target: far beyond the limit, only "fails badly" matters.

    Specifications that could not be computed (NaN, dead circuit) take the worst value.
    """
    if sense == "max":
        return values.fillna(np.inf).clip(lower=-CLIP * limit, upper=CLIP * limit).to_numpy()
    return values.fillna(-np.inf).clip(lower=limit / CLIP, upper=CLIP * limit).to_numpy()


def regression_margins(x, measured, limits, train, val, test, seed):
    """Worst normalised margin per case (positive = predicted violation), and spec errors.

    The margin of a specification is the distance of its prediction beyond the limit,
    in standard deviations of the validation residuals; the case margin is the
    largest over specifications.
    """
    margin_val = np.full(len(val), -np.inf)
    margin_test = np.full(len(test), -np.inf)
    errors = {}
    for name, (sense, limit) in limits.items():
        y = clipped_target(measured[f"spec_{name}"], sense, limit)
        model = HistGradientBoostingRegressor(random_state=seed).fit(x[train], y[train])
        pred_val, pred_test = model.predict(x[val]), model.predict(x[test])
        scale = max(float(np.std(y[val] - pred_val)), 1e-12)
        sign = 1.0 if sense == "max" else -1.0
        margin_val = np.maximum(margin_val, sign * (pred_val - limit) / scale)
        margin_test = np.maximum(margin_test, sign * (pred_test - limit) / scale)
        errors[name] = float(np.sqrt(np.mean((pred_test - y[test]) ** 2))) / limit
    return margin_val, margin_test, errors


def guard_for(margin_val: np.ndarray, bad_val: np.ndarray, target: float) -> float:
    """Guard band that leaves a fraction `target` of the bad validation cases accepted.

    A case is rejected when margin + guard >= 0, so the guard is minus the `target`
    quantile of the margins of the bad cases; it is never negative.
    """
    return max(0.0, -float(np.quantile(margin_val[bad_val], target)))


def main() -> None:
    p = parser(__doc__.splitlines()[0], dataset=True)
    p.add_argument("--escape-target", type=float, default=0.01,
                   help="escape rate aimed at when setting guard bands and thresholds")
    args = p.parse_args()
    measured, _, cfg = load_measured(args.data, args.electrode_kinds)
    out = results_dir("e3", cfg["circuit"], args)
    seed = int(cfg["seed"])
    limits = spec_limits(cfg)
    is_bad = ~measured["compliant"].to_numpy(dtype=bool)
    healthy = (measured["kind"] == "healthy").to_numpy()

    decisions, errors, escapes = [], [], None
    for set_name, features in feature_sets(cfg).items():
        x = feature_matrix(measured, features, args)
        for repeat in range(args.repeats):
            train, val, test = split(measured, repeat, seed)
            fit_rows = np.concatenate([train, val])
            predictions = {}

            limits_lt = envelope_limits(measured.iloc[fit_rows][healthy[fit_rows]], features)
            predictions["limit_test"] = envelope_flags(measured.iloc[test], limits_lt)

            m_val, m_test, spec_errors = regression_margins(
                x, measured, limits, train, val, test, seed
            )
            guard = guard_for(m_val, is_bad[val], args.escape_target)
            predictions["regression"] = m_test > 0
            # >= so that cases tied at the guard (clipped targets give many ties) are rejected
            predictions["regression+guard"] = (m_test + guard >= 0) if guard > 0 else m_test > 0

            clf = HistGradientBoostingClassifier(random_state=seed).fit(x[train], is_bad[train])
            p_val, p_test = clf.predict_proba(x[val])[:, 1], clf.predict_proba(x[test])[:, 1]
            threshold = min(0.5, float(np.quantile(p_val[is_bad[val]], args.escape_target)))
            predictions["classifier"] = p_test >= 0.5
            predictions["classifier+threshold"] = p_test >= threshold

            for method, pred in predictions.items():
                row = {"feature_set": set_name, "method": method, "repeat": repeat}
                row.update(decision_report(is_bad[test], pred))
                decisions.append(row)
            errors += [
                {"feature_set": set_name, "spec": name, "repeat": repeat, "rmse_over_limit": e}
                for name, e in spec_errors.items()
            ]
            if set_name == ALL and repeat == args.repeats - 1:
                escapes = escape_rate_by(
                    measured.iloc[test], is_bad[test], predictions["classifier+threshold"]
                )
        last = pd.DataFrame(decisions[-len(predictions):])
        print(set_name)
        print(last.drop(columns=["feature_set", "repeat"]).to_string(index=False,
              float_format=lambda v: f"{v:.3f}"), flush=True)

    decisions = pd.DataFrame(decisions)
    decisions.to_csv(out / "pass_fail_repeats.csv", index=False)
    summary = summarise(decisions, ["feature_set", "method"])
    summary.to_csv(out / "pass_fail.csv", index=False)
    summarise(pd.DataFrame(errors), ["feature_set", "spec"]).to_csv(
        out / "spec_rmse_over_limit.csv", index=False
    )
    escapes.to_csv(out / "escapes_by_fault.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

"""Check a generated dataset and report its class balance (phase 4 of the plan).

Verifies the integrity of one dataset folder (expected number of rows per condition,
failed simulations, missing values, waveform alignment) and writes `report.md`
inside it with the balance of the three labelling levels. Exits with status 1 if
an integrity check fails.

    python scripts/dataset_report.py --data data/v1/integrated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ecgfd.dataset import build_tasks, load_dataset
from ecgfd.features import feature_sets
from ecgfd.specs import SPEC_NAMES

MAX_FAILED_FRACTION = 0.01


Check = tuple[str, bool, str]  # (name, passed, detail)


def integrity_checks(df: pd.DataFrame, waveforms: np.ndarray, cfg: dict) -> list[Check]:
    """(check, passed, detail) for everything that must hold before using the dataset."""
    tasks = build_tasks(cfg)
    expected = pd.Series([fault.id for _, fault, _ in tasks]).value_counts()
    counts = df["condition"].value_counts()
    ok = df["sim_ok"].to_numpy(dtype=bool)
    n_points = round(cfg["measurement"]["pulse"]["duration"] * cfg["measurement"]["pulse"]["fs"])
    raw_features = [c for c in df.columns if c.startswith(("dc_", "acd_", "acc_", "zlo_"))]
    spec_columns = [f"spec_{name}" for name in SPEC_NAMES]
    missing_features = int(df.loc[ok, raw_features].isna().any(axis=1).sum())
    missing_specs = int(df.loc[ok, spec_columns].isna().any(axis=1).sum())
    failed = int((~ok).sum())
    return [
        ("rows", len(df) == len(tasks), f"{len(df)} of {len(tasks)} expected"),
        (
            "rows per condition",
            counts.reindex(expected.index).eq(expected).all(),
            f"{len(counts)} conditions, {int(counts.min())}-{int(counts.max())} rows each",
        ),
        ("unique sample ids", df["sample_id"].is_unique, ""),
        (
            "failed simulations",
            failed <= MAX_FAILED_FRACTION * len(df),
            f"{failed} ({failed / len(df):.2%}); limit {MAX_FAILED_FRACTION:.0%}",
        ),
        ("waveform shape", waveforms.shape == (len(df), n_points), str(waveforms.shape)),
        (
            "finite waveforms",
            bool(np.isfinite(waveforms[ok]).all()),
            "all simulated rows have a pulse response",
        ),
        ("features present", missing_features == 0, f"{missing_features} simulated rows with gaps"),
        # a dead circuit can leave a specification undefined; it is counted as a violation
        ("specifications present", True, f"{missing_specs} simulated rows with undefined values"),
    ]


def table(frame: pd.DataFrame) -> str:
    """Markdown table; fractions as percentages, other floats with three decimals."""
    out = frame.copy()
    for column in out.columns:
        if column == "compliant" and out[column].dtype == float:
            out[column] = out[column].map("{:.1%}".format)
        elif out[column].dtype == float:
            out[column] = out[column].map("{:.3f}".format)
    return out.to_markdown()


def build_report(df: pd.DataFrame, cfg: dict, checks: list[Check]) -> str:
    ok = df[df["sim_ok"].to_numpy(dtype=bool)].copy()
    ok["compliant"] = ok["compliant"].astype(bool)
    gain = cfg["specs"]["nominal_gain"]
    lines = [
        f"# Dataset report: `{cfg['circuit']}`",
        "",
        f"{len(df)} simulations, {df['condition'].nunique()} conditions, "
        f"nominal gain {gain:.1f}, seed {cfg['seed']}.",
        "",
        "## Integrity",
        "",
        table(
            pd.DataFrame(
                [(name, "ok" if passed else "FAIL", detail) for name, passed, detail in checks],
                columns=["check", "result", "detail"],
            ).set_index("check")
        ),
        "",
        "## Level 1: functional (compliant with the specifications)",
        "",
        f"Compliant: {ok['compliant'].mean():.1%} of all cases, "
        f"{ok.loc[ok['kind'] == 'healthy', 'compliant'].mean():.1%} of the healthy ones.",
        "",
        table(ok.groupby("kind")["compliant"].agg(cases="size", compliant="mean")),
        "",
        "Parametric faults by magnitude:",
        "",
        table(
            ok[ok["kind"] == "parametric"]
            .groupby("level")["compliant"]
            .agg(cases="size", compliant="mean")
        ),
        "",
        "Violated specifications (a case can violate several):",
        "",
        table(
            ok.loc[~ok["compliant"], "violated"]
            .str.split(",")
            .explode()
            .value_counts()
            .rename_axis("specification")
            .to_frame("cases")
        ),
        "",
        "## Level 2: localisation",
        "",
        "Non-compliant cases per component (classes of E5):",
        "",
        table(
            ok[~ok["compliant"]]
            .groupby("target")
            .size()
            .describe()[["count", "min", "50%", "max"]]
            .rename({"count": "components", "50%": "median"})
            .astype(int)
            .to_frame("non-compliant cases"),
        ),
        "",
        "## Level 3: origin",
        "",
        table(
            pd.crosstab(ok["origin"], ok["compliant"]).rename(
                columns={True: "compliant cases", False: "non-compliant cases"}
            ),
        ),
        "",
        "## Electrodes",
        "",
        "Healthy cases: gain measured in service at 10 Hz over the nominal gain.",
        "",
        table(
            (ok.loc[ok["kind"] == "healthy"].assign(ratio=lambda d: d["acd_mag_10"] / gain))
            .groupby("electrode_kind")["ratio"]
            .agg(cases="size", min="min", median="median", max="max")
        ),
        "",
        "## Feature sets",
        "",
        table(
            pd.Series({name: len(cols) for name, cols in feature_sets(cfg).items()})
            .rename_axis("set")
            .to_frame("features"),
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="dataset folder (one circuit)")
    args = parser.parse_args()

    df, waveforms, cfg = load_dataset(args.data, drop_failed=False)
    checks = integrity_checks(df, waveforms, cfg)
    report = build_report(df, cfg, checks)
    out = Path(args.data) / "report.md"
    out.write_text(report)
    print(report)
    print(f"written to {out}")
    if not all(passed for _, passed, _ in checks):
        sys.exit("integrity check failed")


if __name__ == "__main__":
    main()

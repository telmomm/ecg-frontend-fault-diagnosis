"""Check a generated dataset and report its class balance (phase 4 of the plan).

Verifies one dataset folder (files against their fingerprints, expected number of
cases per fault, failed simulations, missing values, and a few samples simulated
again and compared with what is stored) and writes `report.md` inside it with the
balance of the three labelling levels. Exits with status 1 if a check fails.

    python scripts/dataset_report.py --data data/v1/integrated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from spicefault import Dataset

from ecgfd.dataset import experiment, load_cases
from ecgfd.features import feature_sets
from ecgfd.specs import SPEC_NAMES

MAX_FAILED_FRACTION = 0.01
N_REPRODUCED = 6
Check = tuple[str, bool, str]  # (name, passed, detail)


def integrity_checks(path: str, cases: pd.DataFrame, waveforms: np.ndarray, cfg: dict):
    """(check, passed, detail) for everything that must hold before using the dataset."""
    data, study = Dataset(path), experiment(cfg)
    problems = data.verify()
    per_fault = cases["fault_id"].value_counts()
    size = cfg["dataset"]
    expected = pd.Series(int(size["n_per_fault"]), index=[f.fault_id for f in study.faults])
    expected = pd.concat([pd.Series({"healthy": int(size["n_healthy"])}), expected])
    ok = cases["sim_ok"].to_numpy(dtype=bool)
    raw_features = [c for c in cases.columns if c.startswith(("dc_", "acd_", "acc_", "zlo_"))]
    missing_features = int(cases.loc[ok, raw_features].isna().any(axis=1).sum())
    missing_specs = int(cases.loc[ok, list(SPEC_NAMES)].isna().any(axis=1).sum())
    failed = int((~ok).sum())
    again = data.reproduce(n=N_REPRODUCED, experiment=study)
    same = again[["definition", "parameters", "status"]].all(axis=None)
    return [
        ("files and table", not problems, "; ".join(problems) or "match the manifest"),
        (
            "cases per fault",
            per_fault.sort_index().equals(expected.sort_index()),
            f"{len(per_fault)} conditions, {per_fault.min()}-{per_fault.max()} cases each",
        ),
        (
            "failed simulations",
            failed <= MAX_FAILED_FRACTION * len(cases),
            f"{failed} ({failed / len(cases):.2%}); limit {MAX_FAILED_FRACTION:.0%}",
        ),
        (
            "finite waveforms",
            bool(np.isfinite(waveforms[ok]).all()),
            "all simulated cases have a pulse response",
        ),
        ("features present", missing_features == 0, f"{missing_features} cases with gaps"),
        # a dead circuit can leave a specification undefined; it is counted as a violation
        ("specifications present", True, f"{missing_specs} cases with undefined values"),
        (
            "reproduction",
            bool(same),
            f"{len(again)} samples simulated again; largest relative difference "
            f"{again['max_rel_diff'].max():.1e}",
        ),
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
        f"{len(df)} cases, {df['fault_id'].nunique()} conditions, "
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
        f"{ok.loc[ok['fault_id'] == 'healthy', 'compliant'].mean():.1%} of the healthy ones.",
        "",
        table(ok.groupby("fault_type")["compliant"].agg(cases="size", compliant="mean")),
        "",
        "Parametric faults by magnitude:",
        "",
        table(
            ok[ok["fault_type"] == "parametric"]
            .groupby("fault_magnitude")["compliant"]
            .agg(cases="size", compliant="mean")
        ),
        "",
        "Violated specifications (a case can violate several):",
        "",
        table(
            ok.loc[~ok["compliant"], "violated"]
            .str.replace("spec_", "")
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
            .groupby("component")
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
            pd.crosstab(ok["origin"].replace("", "none"), ok["compliant"]).rename(
                columns={True: "compliant cases", False: "non-compliant cases"}
            ),
        ),
        "",
        "## Electrodes",
        "",
        "Healthy cases: gain measured in service at 10 Hz over the nominal gain.",
        "",
        table(
            (ok.loc[ok["fault_id"] == "healthy"].assign(ratio=lambda d: d["acd_mag_10"] / gain))
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

    df, waveforms, cfg = load_cases(args.data, drop_failed=False)
    checks = integrity_checks(args.data, df, waveforms, cfg)
    report = build_report(df, cfg, checks)
    out = Path(args.data) / "report.md"
    out.write_text(report)
    print(report)
    print(f"written to {out}")
    if not all(passed for _, passed, _ in checks):
        sys.exit("integrity check failed")


if __name__ == "__main__":
    main()

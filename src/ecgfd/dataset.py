"""Dataset generation (Monte Carlo + fault injection) and loading, on `spicefault`.

A dataset is the folder of a `spicefault.FaultCampaign`: `samples.parquet`,
`waveforms.npy`, `circuit.cir`, `metadata.json` and `manifest.json` (docs/dataset.md).
Every case (a drawn circuit, with or without a fault) gives two rows, one per
operating condition: `service`, with the self-test measurements and the
calibration-pulse response, and `bench`, with the specifications. `load_cases`
joins them into one row per case.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from spicefault import Dataset, Experiment, Fault, FaultCampaign
from spicefault.dataset import load_dataset

from . import __version__
from .circuit import load_circuit, population
from .config import REPO_ROOT
from .faults import TAGS, fault_catalogue
from .selftest import pulse_waveform, service
from .specs import bench, compliance, with_nominal_gain

CASE = ["fault_index", "replica"]  # a drawn circuit: the same under both conditions


def experiment(cfg: dict, healthy_only: bool = False, **kwargs) -> Experiment:
    """The study as a `spicefault` experiment: population, faults, service and bench.

    With `healthy_only` no fault is injected. Other arguments go to `Experiment`,
    e.g. `variations=()` for the nominal circuit or `faults=[...]` for chosen ones.
    """
    circuit = load_circuit(cfg)
    cfg = with_nominal_gain(circuit, cfg)
    settings = {
        "faults": () if healthy_only else fault_catalogue(circuit, cfg),
        "variations": population(circuit, cfg),
        "conditions": (service(cfg), bench(circuit, cfg)),
        "waveform": pulse_waveform(cfg),
        "samples": int(cfg["dataset"]["n_per_fault"]),
        "healthy_samples": int(cfg["dataset"]["n_healthy"]),
        "seed": int(cfg["seed"]),
        **kwargs,
    }
    return Experiment(circuit, **settings)


def observe(cfg: dict, fault: Fault | None = None) -> dict[str, dict]:
    """The nominal circuit, or the nominal circuit with one fault, under both conditions.

    Returns `{"service": {...}, "bench": {...}}`, each with its measurements and, under
    `result`, the simulation itself.
    """
    faults = {"healthy_only": True} if fault is None else {"faults": [fault], "samples": 1}
    study = experiment(cfg, variations=(), healthy_samples=int(fault is None), **faults)
    return {
        s.condition: {**{k: v for k, v in s.measurements.items() if v == v}, "result": s.result}
        for s in study.run()
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def generate(cfg: dict, out_dir: str | Path, jobs: int = 1, **kwargs) -> Path:
    """Simulate every case of `cfg` (one circuit), write the dataset and label it.

    An interrupted run on the same folder and configuration resumes after the last
    complete chunk. Other arguments go to `FaultCampaign.run`.
    """
    study = experiment(cfg)
    cfg = with_nominal_gain(study.circuit, cfg)
    metadata = {"ecgfd_version": __version__, "git_commit": _git_commit(), "config": cfg}
    campaign = FaultCampaign.from_experiment(study, out_dir, tag_columns=TAGS, metadata=metadata)
    campaign.run(workers=jobs, **kwargs)
    label(out_dir)
    return Path(out_dir)


def label(path: str | Path, specs_cfg: dict | None = None) -> pd.DataFrame:
    """Write the compliance labels of a dataset; returns its cases.

    With `specs_cfg` (the `specs` block of a configuration) the labels are recomputed
    with new limits from the stored `spec_*` values, so nothing is simulated. This is
    not valid if the test set-up changed, because that changes the values themselves.
    """
    data = Dataset(path)
    cfg = data.manifest.user["config"]
    note = "limits of the generating configuration"
    if specs_cfg is not None:
        old = cfg["specs"]
        for key in ("test_network", "impulse", "electrode_offset"):
            if specs_cfg[key] != old[key]:
                raise ValueError(f"specs.{key} differs from the dataset: it must be re-simulated")
        cfg["specs"] = {**specs_cfg, "nominal_gain": old["nominal_gain"]}
        note = "relabelled with new limits (see user.config.specs)"
    on_bench = data.samples[data.samples["condition"] == "bench"]
    labels = pd.concat([on_bench[CASE], compliance(on_bench, cfg)], axis=1)
    frame = data.samples[CASE].merge(labels, on=CASE, how="left").drop(columns=CASE)
    data.update_columns(frame.set_index(data.samples.index), note)
    return load_cases(path)[0]


def load_cases(
    path: str | Path, drop_failed: bool = True
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return (cases, pulse responses, config used to generate them), one row per case.

    A case is the `service` row of a drawn circuit with the specifications of its
    `bench` row; it is failed if either simulation failed.
    """
    df, waveforms, manifest = load_dataset(path, drop_failed=False)
    in_service = (df["condition"] == "service").to_numpy()
    cases = df[in_service].reset_index(drop=True)
    on_bench = df[~in_service].reset_index(drop=True)
    if not cases[CASE].equals(on_bench[CASE]):
        raise ValueError(f"{path}: the service and bench rows do not pair up")
    specs = [c for c in df.columns if c.startswith("spec_")]
    cases[specs] = on_bench[specs]
    cases["sim_ok"] &= on_bench["sim_ok"]
    waveforms = waveforms[in_service]
    if drop_failed:
        keep = cases["sim_ok"].to_numpy()
        cases, waveforms = cases[keep].reset_index(drop=True), waveforms[keep]
    return cases, waveforms, manifest["user"]["config"]

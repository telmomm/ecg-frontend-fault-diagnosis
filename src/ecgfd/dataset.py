"""Dataset generation (Monte Carlo + fault injection) and loading, on `spicefault`.

A dataset is the folder of a `spicefault.FaultCampaign`: `samples.parquet`,
`waveforms.npy`, `circuit.cir`, `metadata.json` and `manifest.json` (docs/dataset.md).
Every case (a drawn circuit, with or without a fault) gives two rows, one per
operating condition: `service`, with the self-test measurements and the
calibration-pulse response, and `bench`, with the specifications. `load_cases`
returns one row per case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from spicefault import Dataset, Experiment, FaultCampaign

from . import __version__
from .circuit import load_circuit, population
from .faults import TAGS, fault_catalogue
from .selftest import pulse_waveform, service
from .specs import bench, specifications, with_nominal_gain


def experiment(cfg: dict, healthy_only: bool = False, **kwargs) -> Experiment:
    """The study as a `spicefault` experiment: population, faults, service and bench.

    With `healthy_only` no fault is injected. Other arguments go to `Experiment`.
    Its `nominal()` gives the nominal circuit, alone or with one fault.
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


def generate(cfg: dict, out_dir: str | Path, jobs: int = 1, **kwargs) -> Path:
    """Simulate every case of `cfg` (one circuit), write the dataset and label it.

    An interrupted run on the same folder and configuration resumes after the last
    complete chunk. Other arguments go to `FaultCampaign.run`.
    """
    study = experiment(cfg)
    cfg = with_nominal_gain(study.circuit, cfg)
    metadata = {"ecgfd_version": __version__, "config": cfg}
    campaign = FaultCampaign.from_experiment(study, out_dir, tag_columns=TAGS, metadata=metadata)
    campaign.run(workers=jobs, **kwargs)
    Dataset(out_dir).label(specifications(cfg), "limits of the generating configuration")
    return Path(out_dir)


def relabel(path: str | Path, specs_cfg: dict) -> pd.DataFrame:
    """Label a dataset again with the limits of `specs_cfg`; returns its cases.

    `specs_cfg` is the `specs` block of a configuration. The stored `spec_*` values are
    reused, so nothing is simulated. This is not valid if the test set-up changed,
    because that changes the values themselves.
    """
    data = Dataset(path)
    cfg = data.manifest.user["config"]
    for key in ("test_network", "impulse", "electrode_offset"):
        if specs_cfg[key] != cfg["specs"][key]:
            raise ValueError(f"specs.{key} differs from the dataset: it must be re-simulated")
    cfg["specs"] = {**specs_cfg, "nominal_gain": cfg["specs"]["nominal_gain"]}
    data.label(specifications(cfg), "relabelled with new limits")  # also writes the manifest
    return load_cases(path)[0]


def load_cases(
    path: str | Path, drop_failed: bool = True
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return (cases, pulse responses, config used to generate them), one row per case.

    A case is a drawn circuit with its self-test measurements in service and its
    specifications on the bench; it is failed if either simulation failed.
    """
    data = Dataset(path)
    cases, waveforms = data.cases()
    if drop_failed:
        keep = cases["sim_ok"].to_numpy()
        cases, waveforms = cases[keep].reset_index(drop=True), waveforms[keep]
    return cases, waveforms, data.manifest.user["config"]

"""Dataset generation (Monte Carlo + fault injection) and loading.

A dataset is a directory with:
- samples.parquet  one row per simulation: what was injected (`condition`, `kind`,
                   `target`, `level`), the three labelling levels (`compliant` and
                   `violated`; `target`; `origin`), the specification values
                   (`spec_*`, `ok_*`), the realised component values (`p_*`) and the
                   noise-free scalar features;
- waveforms.npy    float32 [n_samples, n_points] calibration-pulse responses, row-aligned;
- manifest.json    configuration, software versions and counts.
"""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .faults import HEALTHY, Fault, fault_catalogue
from .sampling import instance_parameters, sample_instance
from .simulate import measure, scalar_features
from .specs import compliance, measure_specs, with_nominal_gain
from .spice import SimulationError, ngspice_version

_CFG: dict = {}


def sample_rng(cfg: dict, condition_index: int, replica: int) -> np.random.Generator:
    """Independent stream per simulation, whatever the execution order."""
    seq = np.random.SeedSequence(int(cfg["seed"]), spawn_key=(condition_index, replica))
    return np.random.default_rng(seq)


def build_tasks(cfg: dict) -> list[tuple[int, Fault, int]]:
    """(condition index, fault, replica) for every simulation; index 0 is healthy."""
    d = cfg["dataset"]
    tasks = [(0, HEALTHY, r) for r in range(int(d["n_healthy"]))]
    for i, fault in enumerate(fault_catalogue(cfg), start=1):
        tasks += [(i, fault, r) for r in range(int(d["n_per_fault"]))]
    return tasks


def simulate_task(task: tuple[int, Fault, int], cfg: dict) -> tuple[dict, np.ndarray | None]:
    index, fault, replica = task
    inst = fault.apply(sample_instance(cfg, sample_rng(cfg, index, replica)), cfg)
    row = {
        "condition": fault.id,
        "condition_index": index,
        "kind": fault.kind,
        "target": fault.target,
        "level": fault.level,
        "origin": fault.origin,
        "is_faulty": fault.kind != "healthy",
        "electrode_type": inst.electrode_type,
        "replica": replica,
        "sim_ok": True,
        **instance_parameters(inst),
    }
    try:
        m = measure(inst, cfg)
        specs = measure_specs(inst, cfg)
    except SimulationError:
        row["sim_ok"] = False
        return row, None
    row.update(scalar_features(m, cfg))
    row.update({f"spec_{name}": value for name, value in specs.items()})
    row.update(compliance(specs, cfg))
    return row, m.pulse.astype(np.float32)


def _init_worker(cfg: dict) -> None:
    global _CFG
    _CFG = cfg


def _worker(task: tuple[int, Fault, int]) -> tuple[dict, np.ndarray | None]:
    return simulate_task(task, _CFG)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def generate(cfg: dict, out_dir: str | Path, jobs: int = 1, progress: bool = True) -> Path:
    """Simulate every task of `cfg` (one circuit) and write the dataset to `out_dir`."""
    cfg = with_nominal_gain(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(cfg)
    n_points = round(cfg["measurement"]["pulse"]["duration"] * cfg["measurement"]["pulse"]["fs"])
    rows: list[dict] = []
    waveforms = np.full((len(tasks), n_points), np.nan, dtype=np.float32)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, (row, wave) in enumerate(pool.map(_worker, tasks, chunksize=16)):
            rows.append(row)
            if wave is not None:
                waveforms[i] = wave
            if progress and (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{i + 1}/{len(tasks)} simulations ({rate:.0f}/s)", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "sample_id", np.arange(len(df)))
    df.to_parquet(out_dir / "samples.parquet", index=False)
    np.save(out_dir / "waveforms.npy", waveforms)
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ecgfd_version": __version__,
        "git_commit": _git_commit(),
        "ngspice": ngspice_version(),
        "circuit": cfg["circuit"],
        "n_samples": len(df),
        "n_conditions": int(df["condition"].nunique()),
        "n_failed": int((~df["sim_ok"]).sum()),
        "n_compliant": int((df["compliant"] == True).sum()),  # noqa: E712 (NaN when failed)
        "elapsed_s": round(time.time() - t0, 1),
        "config": cfg,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def load_dataset(
    path: str | Path, drop_failed: bool = True
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return (samples, waveforms, config used to generate them)."""
    path = Path(path)
    df = pd.read_parquet(path / "samples.parquet")
    waveforms = np.load(path / "waveforms.npy")
    cfg = json.loads((path / "manifest.json").read_text())["config"]
    if drop_failed:
        keep = df["sim_ok"].to_numpy()
        df, waveforms = df[keep].reset_index(drop=True), waveforms[keep]
    return df, waveforms, cfg

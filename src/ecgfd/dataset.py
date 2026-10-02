"""Dataset generation (Monte Carlo + fault injection) and loading.

A dataset is a directory with:
- samples.parquet  one row per simulation: what was injected (`condition`, `kind`,
                   `target`, `level`), the three labelling levels (`compliant` and
                   `violated`; `target`; `origin`), the specification values
                   (`spec_*`, `ok_*`), the realised component values (`p_*`) and the
                   noise-free scalar features;
- waveforms.npy    float32 [n_samples, n_points] calibration-pulse responses, row-aligned;
- manifest.json    configuration, software versions and counts.

While a generation is running, finished chunks are kept under `parts/`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
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
from .specs import SPEC_NAMES, compliance, measure_specs, with_nominal_gain
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
        "electrode_kind": inst.electrode_kind,
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


def _config_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def simulate_parts(
    tasks: list[tuple[int, Fault, int]],
    cfg: dict,
    parts_dir: Path,
    jobs: int = 1,
    chunk: int = 2000,
    progress: bool = True,
) -> list[Path]:
    """Simulate `tasks` in chunks, one pair of files per chunk; finished chunks are kept.

    This is what makes a long generation resumable: a chunk is written only once it
    is complete, and a chunk already on disk is not simulated again.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_points = round(cfg["measurement"]["pulse"]["duration"] * cfg["measurement"]["pulse"]["fs"])
    stems = [parts_dir / f"part_{start:07d}" for start in range(0, len(tasks), chunk)]
    done, t0 = 0, time.time()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(cfg,)) as pool:
        for stem, start in zip(stems, range(0, len(tasks), chunk), strict=True):
            if stem.with_suffix(".parquet").exists():
                continue
            subset = tasks[start : start + chunk]
            rows = []
            waveforms = np.full((len(subset), n_points), np.nan, dtype=np.float32)
            for i, (row, wave) in enumerate(pool.map(_worker, subset, chunksize=16)):
                rows.append({"sample_id": start + i, **row})
                if wave is not None:
                    waveforms[i] = wave
                done += 1
                if progress and done % 500 == 0:
                    rate = done / (time.time() - t0)
                    print(f"{start + i + 1}/{len(tasks)} simulations ({rate:.0f}/s)", flush=True)
            np.save(stem.with_suffix(".npy"), waveforms)
            # the table is written last and renamed, so its presence marks a complete chunk
            tmp = stem.with_suffix(".tmp")
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            tmp.rename(stem.with_suffix(".parquet"))
    return stems


def generate(
    cfg: dict, out_dir: str | Path, jobs: int = 1, progress: bool = True, chunk: int = 2000
) -> Path:
    """Simulate every task of `cfg` (one circuit) and write the dataset to `out_dir`.

    If a previous run on the same folder and configuration was interrupted, it
    resumes after the last complete chunk.
    """
    out_dir = Path(out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    key_file = parts_dir / "config.sha256"
    key = _config_key({**cfg, "chunk": chunk})
    if key_file.exists() and key_file.read_text() != key:
        raise ValueError(
            f"{parts_dir} holds a partial run made with another configuration; "
            "delete it or choose another output folder"
        )
    key_file.write_text(key)

    cfg = with_nominal_gain(cfg)
    t0 = time.time()
    stems = simulate_parts(build_tasks(cfg), cfg, parts_dir, jobs, chunk, progress)
    df = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    waveforms = np.concatenate([np.load(s.with_suffix(".npy")) for s in stems])

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
        "elapsed_last_run_s": round(time.time() - t0, 1),
        "config": cfg,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(parts_dir)
    return out_dir


def relabel(path: str | Path, specs_cfg: dict) -> pd.DataFrame:
    """Recompute the compliance labels of a dataset with new specification limits.

    Only the limits are taken from `specs_cfg` (the `specs` block of a configuration);
    the stored `spec_*` values are reused, so nothing is simulated. This is not valid
    if the test set-up changed, because that changes the values themselves.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    cfg = manifest["config"]
    old = cfg["specs"]
    for key in ("test_network", "impulse", "electrode_offset"):
        if specs_cfg[key] != old[key]:
            raise ValueError(f"specs.{key} differs from the dataset: it must be re-simulated")
    cfg["specs"] = {**specs_cfg, "nominal_gain": old["nominal_gain"]}

    df = pd.read_parquet(path / "samples.parquet")
    ok = df["sim_ok"].to_numpy()
    labels = pd.DataFrame(
        [
            compliance({name: row[f"spec_{name}"] for name in SPEC_NAMES}, cfg)
            for _, row in df[ok].iterrows()
        ],
        index=df.index[ok],
    )
    for column in labels:
        df.loc[ok, column] = labels[column]
    df.to_parquet(path / "samples.parquet", index=False)
    manifest["n_compliant"] = int(labels["compliant"].sum())
    manifest["relabelled"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return df


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

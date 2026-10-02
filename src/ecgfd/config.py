"""Loading of the YAML study configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent is not None:
        cfg = _deep_merge(_read(path.parent / parent), cfg)
    return cfg


def load_config(path: str | Path = DEFAULT_CONFIG, circuit: str | None = None) -> dict[str, Any]:
    """Read a config file and resolve it for one circuit.

    `extends:` is resolved relative to the file. The block `circuits.<name>` of the
    selected circuit (argument, else the `circuit` key) is merged over the common
    settings, so the rest of the code sees a flat configuration.
    """
    cfg = _read(Path(path))
    variants = cfg.pop("circuits")
    name = circuit or cfg["circuit"]
    if name not in variants:
        raise ValueError(f"unknown circuit {name!r}; available: {sorted(variants)}")
    cfg = _deep_merge(cfg, variants[name])
    cfg["circuit"] = name
    return cfg

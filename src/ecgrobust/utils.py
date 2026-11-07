"""Shared helpers: configuration, seeding, paths, artifact I/O."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

LOG = logging.getLogger("ecgrobust")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- config
def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config; ``extends: other.yaml`` merges onto a base file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("extends", None)
    if base:
        cfg = deep_update(load_config(path.parent / base), cfg)
    cfg["_config_path"] = str(path)
    return cfg


# --------------------------------------------------------------------------- seeding
def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**32)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- paths
@dataclass
class Paths:
    root: Path
    processed: Path

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Paths":
        p = cls(Path(cfg["output_dir"]), Path(cfg["paths"]["processed_dir"]))
        for d in (p.root, p.processed, p.features, p.models, p.predictions, p.tables, p.figures):
            d.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def splits(self) -> Path:
        return self.root / "splits.json"

    @property
    def test_lock(self) -> Path:
        return self.root / "TEST_SET_UNLOCKED.json"

    def dataset(self, name: str) -> Path:
        return self.processed / f"{name}.pkl"


# --------------------------------------------------------------------------- I/O
def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def environment_manifest(cfg: dict) -> dict:
    """Versions of everything that can change a number in the paper."""
    import importlib

    pkgs = {}
    for name in ["numpy", "scipy", "pandas", "sklearn", "lightgbm", "neurokit2", "wfdb", "torch", "shap"]:
        try:
            pkgs[name] = importlib.import_module(name).__version__
        except Exception:  # pragma: no cover
            pkgs[name] = None
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": pkgs,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }

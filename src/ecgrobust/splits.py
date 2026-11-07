"""The internal hold-out split and the test-set lock.

The split is made once, before any modelling, and written to ``splits.json``
together with a hash of the test ids. Training code asserts that no test id
reaches it. Predictions on the test set are refused until ``unlock-test`` has
been run, which records when the lock was lifted and which model artifacts
existed at that moment. That record shows the models were frozen before the
test set was used.
"""
from __future__ import annotations

import hashlib
import logging
import time

from sklearn.model_selection import train_test_split

from .data import load_dataset
from .utils import Paths, load_json, save_json

LOG = logging.getLogger(__name__)


def _hash_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def make_split(cfg: dict) -> dict:
    paths = Paths.from_cfg(cfg)
    if paths.splits.exists():
        LOG.info("Split already exists at %s; refusing to overwrite.", paths.splits)
        return load_json(paths.splits)
    meta = load_dataset(cfg, "internal")["meta"]
    s = cfg["split"]
    ids = meta.index.to_numpy(dtype=object)
    strata = meta[s["stratify_on"]].to_numpy(dtype=object)
    train, test = train_test_split(ids, test_size=s["test_fraction"], stratify=strata, random_state=s["seed"])
    out = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": s["seed"],
        "test_fraction": s["test_fraction"],
        "stratify_on": s["stratify_on"],
        "train": sorted(map(str, train)),
        "test": sorted(map(str, test)),
        "test_sha256": _hash_ids(map(str, test)),
        "class_counts_train": meta.loc[train, "label4"].value_counts().to_dict(),
        "class_counts_test": meta.loc[test, "label4"].value_counts().to_dict(),
    }
    save_json(out, paths.splits)
    LOG.info("Split: %d train / %d test. Test classes %s", len(train), len(test), out["class_counts_test"])
    return out


def load_split(cfg: dict) -> dict:
    sp = load_json(Paths.from_cfg(cfg).splits)
    assert _hash_ids(sp["test"]) == sp["test_sha256"], "splits.json test ids were modified"
    assert not set(sp["train"]) & set(sp["test"])
    return sp


def assert_no_test_leak(cfg: dict, ids) -> None:
    leaked = set(map(str, ids)) & set(load_split(cfg)["test"])
    if leaked:
        raise RuntimeError(f"{len(leaked)} internal test records reached training code")


def unlock_test(cfg: dict) -> None:
    paths = Paths.from_cfg(cfg)
    artifacts = sorted(str(p.relative_to(paths.root)) for p in paths.models.rglob("*") if p.is_file())
    save_json({"unlocked": time.strftime("%Y-%m-%dT%H:%M:%S"), "frozen_model_artifacts": artifacts}, paths.test_lock)
    LOG.info("Internal test set unlocked; %d frozen model artifacts recorded.", len(artifacts))


def require_unlocked(cfg: dict) -> None:
    if not Paths.from_cfg(cfg).test_lock.exists():
        raise RuntimeError(
            "The internal test set is still locked. Finish all model development, then run "
            "`python -m ecgrobust unlock-test` once before evaluating."
        )

"""Pipeline stages: features -> train -> (unlock test) -> evaluate.

Every stage is resumable: finished artifacts are skipped on re-run.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import models as M
from .data import LABELS4, load_dataset
from .features import extract_many
from .noise import (Condition, NoiseBank, augmentation_spec, corrupt_for_test, corrupt_for_train,
                    evaluation_conditions)
from .splits import assert_no_test_leak, load_split, require_unlocked
from .utils import Paths, environment_manifest, load_pickle, save_json, save_pickle, set_global_seed

LOG = logging.getLogger(__name__)


def _bank(cfg: dict) -> NoiseBank:
    return NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])


def binary_ids(meta: pd.DataFrame, ids) -> list[str]:
    """Binary task: AF vs not-AF; the noisy '~' class is dropped."""
    return [r for r in ids if meta.at[r, "label4"] != "~"]


def feature_path(paths: Paths, key: str):
    return paths.features / f"{key}.parquet"


# =========================================================================== features
def run_features(cfg: dict) -> None:
    paths = Paths.from_cfg(cfg)
    fs, fcfg = cfg["fs"], cfg["features"]
    internal, external = load_dataset(cfg, "internal"), load_dataset(cfg, "external")
    split, bank = load_split(cfg), _bank(cfg)
    data = {"internal": internal, "external": external}

    # clean features (+ reference R-peaks used to score detection under noise)
    for name, ds in data.items():
        out, ref = feature_path(paths, f"{name}__clean"), paths.features / f"{name}__refpeaks.pkl"
        if out.exists() and ref.exists():
            continue
        df = extract_many(ds["signals"], fs, fcfg, return_peaks=True, desc=f"{name} clean")
        save_pickle(df.pop("_peaks").to_dict(), ref)
        df.to_parquet(out)

    # noisy copies of the training set (feature-model augmentation)
    out = feature_path(paths, "internal__train_aug")
    if cfg["augmentation"]["enabled"] and not out.exists():
        tr = binary_ids(internal["meta"], split["train"])
        spec = augmentation_spec(tr, cfg)
        sigs = {f'{s["record_id"]}#aug{s["copy"]}': corrupt_for_train(internal["signals"][s["record_id"]], s, bank)
                for s in spec}
        df = extract_many(sigs, fs, fcfg, desc="train augmentation")
        df["group"] = [s["record_id"] for s in spec]
        df["noise"] = [s["noise"] for s in spec]
        df["snr_db"] = [s["snr_db"] for s in spec]
        df.to_parquet(out)

    # corrupted evaluation sets (labels are not touched here)
    for cond in evaluation_conditions(cfg):
        out = feature_path(paths, cond.key)
        if cond.is_clean or out.exists():
            continue
        ds = data[cond.dataset]
        ids = split["test"] if cond.dataset == "internal" else list(ds["meta"].index)
        ref = load_pickle(paths.features / f"{cond.dataset}__refpeaks.pkl")
        sigs = {r: corrupt_for_test(ds["signals"][r], r, cond, bank, cfg["noise"]["seed"]) for r in ids}
        extract_many(sigs, fs, fcfg, ref_peaks={r: ref[r] for r in ids}, desc=cond.key).to_parquet(out)


# =========================================================================== train
def _model_path(paths: Paths, task: str, variant: str, model: str, seed: int):
    return paths.models / task / variant / f"{model}_seed{seed}.pkl"


def training_jobs(cfg: dict):
    """Cheap models first; the CNN last, starting with the run that tunes its hyper-parameters."""
    variants = ["standard"] + (["augmented"] if cfg["augmentation"]["enabled"] else [])
    jobs = []
    for variant in variants:
        for model in cfg["models"]["binary"]:
            for seed in cfg["seeds"]:
                jobs.append(("binary", variant, model, seed))
    if cfg["fourclass"]["enabled"]:
        for model in cfg["fourclass"]["models"]:
            seeds = cfg["fourclass"].get("cnn_seeds", cfg["fourclass"]["seeds"]) if model == "cnn" else cfg["fourclass"]["seeds"]
            for seed in seeds:
                jobs.append(("fourclass", "standard", model, seed))
    yield from sorted(jobs, key=lambda j: (j[2] == "cnn", j[0] != "binary"))


def _cnn_fixed_hp(cfg: dict, paths: Paths, task: str, variant: str, seed: int) -> dict | None:
    """With ``tune_once``, the CNN grid is searched by 5-fold CV only for the binary task,
    standard training, first seed; every other CNN run reuses the selected settings."""
    if not cfg["models"]["cnn"].get("tune_once"):
        return None
    first = cfg["seeds"][0]
    if (task, variant, seed) == ("binary", "standard", first):
        return None
    return load_pickle(_model_path(paths, "binary", "standard", "cnn", first))["hp"]


def run_train(cfg: dict, only: list[str] | None = None) -> None:
    paths = Paths.from_cfg(cfg)
    save_json(environment_manifest(cfg), paths.root / "environment_train.json")
    internal, split = load_dataset(cfg, "internal"), load_split(cfg)
    meta = internal["meta"]
    assert_no_test_leak(cfg, split["train"])
    clean = pd.read_parquet(feature_path(paths, "internal__clean"))
    aug_path = feature_path(paths, "internal__train_aug")
    aug = pd.read_parquet(aug_path) if aug_path.exists() else None
    bank = _bank(cfg)
    aug_spec_by_rec: dict = {}
    if cfg["augmentation"]["enabled"]:
        for s in augmentation_spec(binary_ids(meta, split["train"]), cfg):
            aug_spec_by_rec.setdefault(s["record_id"], []).append(s)

    for task, variant, model, seed in training_jobs(cfg):
        if only and model not in only:
            continue
        path = _model_path(paths, task, variant, model, seed)
        if path.exists():
            continue
        set_global_seed(seed)
        if task == "binary":
            ids, n_classes = binary_ids(meta, split["train"]), 2
            y = meta.loc[ids, "af"].to_numpy(int)
        else:
            ids, n_classes = list(split["train"]), 4
            y = meta.loc[ids, "label4"].map({l: i for i, l in enumerate(LABELS4)}).to_numpy(int)
        assert_no_test_leak(cfg, ids)
        is_aug_variant = variant == "augmented"
        LOG.info("== train %s / %s / %s / seed %d (n=%d)", task, variant, model, seed, len(ids))
        if model == "cnn":
            b = M.fit_cnn([internal["signals"][r] for r in ids], ids, y, cfg, seed, n_classes,
                          bank if is_aug_variant else None, aug_spec_by_rec if is_aug_variant else None,
                          fixed_hp=_cnn_fixed_hp(cfg, paths, task, variant, seed),
                          cache_dir=paths.root / "cnn_checkpoints" / f"{task}_{variant}_seed{seed}")
        else:
            X, groups, yy, is_aug = clean.loc[ids], np.asarray(ids), y, None
            if is_aug_variant:
                A = aug[aug.group.isin(ids)]
                X = pd.concat([X, A[X.columns]])
                groups = np.r_[groups, A.group.to_numpy()]
                yy = np.r_[y, meta.loc[A.group, "af"].to_numpy(int)]
                is_aug = np.r_[np.zeros(len(ids), bool), np.ones(len(A), bool)]
            b = M.fit_tabular(model, X, yy, groups, cfg, seed, n_classes, is_aug)
        b.update(model=model, task=task, variant=variant, seed=seed)
        # quality distribution on clean training data -> fixed abstention thresholds
        b["train_quality"] = clean.loc[ids, cfg["evaluation"]["quality_score"]].to_numpy(float)
        save_pickle(b, path)


# =========================================================================== evaluate
def _cond_from_key(key: str) -> Condition:
    parts = key.split("__")
    if parts[1] == "clean":
        return Condition(parts[0])
    return Condition(parts[0], parts[1], float(parts[2].replace("dB", "")), int(parts[3][1:]))


def run_evaluate(cfg: dict) -> None:
    require_unlocked(cfg)
    paths = Paths.from_cfg(cfg)
    save_json(environment_manifest(cfg), paths.root / "environment_evaluate.json")
    data = {"internal": load_dataset(cfg, "internal"), "external": load_dataset(cfg, "external")}
    split, bank = load_split(cfg), _bank(cfg)
    conds = evaluation_conditions(cfg)
    qcols = ["q_avgqrs", "q_zhao", "q_ksqi", "q_psqi", "q_bassqi", "rr_n_beats"]

    jobs: dict = {}
    for task, variant, model, seed in training_jobs(cfg):
        jobs.setdefault((task, variant, model), []).append(seed)

    for (task, variant, model), seeds in jobs.items():
        out = paths.predictions / f"{task}__{variant}__{model}.parquet"
        if out.exists():
            continue
        bundles = [load_pickle(_model_path(paths, task, variant, model, s)) for s in seeds]
        cnn_models = [M.load_cnn(b, cfg) for b in bundles] if model == "cnn" else [None] * len(bundles)
        rows = []
        for cond in conds:
            if task == "fourclass" and cond.key != "internal__clean":
                continue
            ds = data[cond.dataset]
            meta = ds["meta"]
            if cond.dataset == "internal":
                ids = binary_ids(meta, split["test"]) if task == "binary" else list(split["test"])
            else:
                ids = list(meta.index)
            feats = pd.read_parquet(feature_path(paths, cond.key)).loc[ids]
            sigs = None
            if model == "cnn":
                sigs = [corrupt_for_test(ds["signals"][r], r, cond, bank, cfg["noise"]["seed"]) for r in ids]
            for b, cm in zip(bundles, cnn_models):
                raw = M.raw_scores(b, X=feats, signals=sigs, cfg=cfg, cnn_model=cm)
                df = pd.DataFrame({"record_id": ids, "patient_id": meta.loc[ids, "patient_id"].to_numpy()})
                df["task"], df["variant"], df["model"], df["seed"] = task, variant, model, b["seed"]
                df["condition"], df["dataset"], df["noise"] = cond.key, cond.dataset, cond.noise
                df["snr_db"], df["realization"] = cond.snr_db, cond.realization
                if task == "binary":
                    df["y"] = meta.loc[ids, "af"].to_numpy(int)
                    df["raw"] = raw
                    df["prob"] = b["calibrator"].transform(raw)
                    df["threshold"] = b["threshold"]
                else:
                    df["y"] = meta.loc[ids, "label4"].map({l: i for i, l in enumerate(LABELS4)}).to_numpy(int)
                    for k in range(raw.shape[1]):
                        df[f"p{k}"] = raw[:, k]
                    df["pred"] = raw.argmax(1)
                for c in qcols:
                    df[c] = feats[c].to_numpy(float)
                rows.append(df)
            LOG.info("evaluated %s/%s/%s on %s", task, variant, model, cond.key)
        pd.concat(rows, ignore_index=True).to_parquet(out)

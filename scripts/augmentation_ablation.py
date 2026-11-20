"""Does the benefit of noise augmentation depend on which noise, and how much, is used for training?

    python scripts/augmentation_ablation.py --config configs/local.yaml features
    python scripts/augmentation_ablation.py --config configs/local.yaml train-tabular
    python scripts/augmentation_ablation.py --config configs/local.yaml train-cnn [--cnn-seeds 0 1 2]
    python scripts/augmentation_ablation.py --config configs/local.yaml evaluate

Augmentation configurations (one noisy copy per training recording, training half of each NSTDB record only):

  full          all three noise types, SNR uniform in [-6, 24] dB (the main study's configuration, refitted here)
  no_em         baseline wander and muscle artifact only (electrode motion held out)
  no_ma         baseline wander and electrode motion only (muscle artifact held out)
  no_bw         muscle artifact and electrode motion only (baseline wander held out)
  mild          all three types, SNR uniform in [6, 24] dB
  severe        all three types, SNR uniform in [-6, 6] dB

Logistic regression and gradient boosting are refitted for every configuration and seed with the
hyper-parameters already selected for the main study's augmented model of the same seed, so that the
comparison is about the training noise and not about a new hyper-parameter search. Calibration and the
decision threshold come from out-of-fold training predictions, exactly as in the main study. The CNN is
retrained only for "no_em" (seeds 0-2 by default) because each fit takes about half an hour on a CPU.

Every model is evaluated on the internal test set (clean and the 18 noise conditions, realization 0) and on
PTB-XL (clean and every external noise condition whose features exist).

Outputs: outputs/revision/models/augablation/..., outputs/revision/features/train_aug__<config>.parquet,
outputs/revision/predictions/augablation__<config>__<model>.parquet
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402
import sklearn  # noqa: E402

from ecgrobust import models as M  # noqa: E402
from ecgrobust.data import load_dataset  # noqa: E402
from ecgrobust.features import FEATURES, extract_many  # noqa: E402
from ecgrobust.noise import NoiseBank, augmentation_spec, corrupt_for_test, corrupt_for_train  # noqa: E402
from ecgrobust.pipeline import _cond_from_key, binary_ids, feature_path  # noqa: E402
from ecgrobust.splits import assert_no_test_leak, load_split  # noqa: E402
from ecgrobust.utils import (Paths, load_config, load_pickle, save_pickle, set_global_seed, setup_logging,  # noqa: E402
                             single_instance)

if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) < (1, 8):
    LogisticRegression.multi_class = "deprecated"

LOG = logging.getLogger("augmentation_ablation")
CONFIGS = {
    "full": {"types": ["bw", "ma", "em"], "snr_range": [-6, 24]},
    "no_em": {"types": ["bw", "ma"], "snr_range": [-6, 24]},
    "no_ma": {"types": ["bw", "em"], "snr_range": [-6, 24]},
    "no_bw": {"types": ["ma", "em"], "snr_range": [-6, 24]},
    "mild": {"types": ["bw", "ma", "em"], "snr_range": [6, 24]},
    "severe": {"types": ["bw", "ma", "em"], "snr_range": [-6, 6]},
}


def cfg_for(cfg: dict, name: str) -> dict:
    c = copy.deepcopy(cfg)
    c["noise"]["types"] = CONFIGS[name]["types"]
    c["augmentation"]["snr_range"] = CONFIGS[name]["snr_range"]
    if name != "full":   # a different deterministic table for each configuration
        c["augmentation"]["seed"] = cfg["augmentation"]["seed"] + 1000 + list(CONFIGS).index(name)
    return c


class Ctx:
    def __init__(self, cfg: dict):
        self.cfg, self.paths = cfg, Paths.from_cfg(cfg)
        self.rev = self.paths.root / "revision"
        for d in ["features", "predictions", "models/augablation"]:
            (self.rev / d).mkdir(parents=True, exist_ok=True)
        self.internal = load_dataset(cfg, "internal")
        self.meta = self.internal["meta"]
        self.split = load_split(cfg)
        self.train = binary_ids(self.meta, self.split["train"])
        assert_no_test_leak(cfg, self.train)
        self.bank = NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])

    def aug_features(self, name: str) -> pd.DataFrame:
        if name == "full":
            return pd.read_parquet(feature_path(self.paths, "internal__train_aug"))
        f = self.rev / "features" / f"train_aug__{name}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        c = cfg_for(self.cfg, name)
        spec = augmentation_spec(self.train, c)
        sigs = {f'{s["record_id"]}#aug{s["copy"]}': corrupt_for_train(self.internal["signals"][s["record_id"]], s, self.bank)
                for s in spec}
        df = extract_many(sigs, self.cfg["fs"], self.cfg["features"], desc=f"train augmentation {name}")
        df["group"] = [s["record_id"] for s in spec]
        df["noise"] = [s["noise"] for s in spec]
        df["snr_db"] = [s["snr_db"] for s in spec]
        df.to_parquet(f)
        return df

    def model_path(self, name: str, model: str, seed: int) -> Path:
        return self.rev / "models" / "augablation" / name / f"{model}_seed{seed}.pkl"


def fit_fixed(model: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, is_aug: np.ndarray, params: dict,
              cfg: dict, seed: int) -> dict:
    """Refit with given hyper-parameters; out-of-fold predictions for calibration and threshold."""
    X = X[FEATURES]
    est = M._logreg(seed) if model == "logreg" else M._lgbm(seed, 2)
    est.set_params(**params)
    if model == "logreg" and tuple(int(v) for v in sklearn.__version__.split(".")[:2]) < (1, 8):
        # scikit-learn < 1.8 ignores l1_ratio unless the penalty is elastic net (1.8+ uses l1_ratio directly);
        # l1_ratio 0 or 1 then gives exactly the L2 or L1 penalty selected for the main study
        est.set_params(clf__penalty="elasticnet")
    splits = M._cv_splits(y, groups, cfg["cv_folds"], seed)
    oof = cross_val_predict(clone(est), X, y, cv=splits, method="predict_proba", n_jobs=-1)[:, 1]
    est.fit(X, y)
    b = {"estimator": est, "best_params": params, "cv_score": np.nan}
    return M._finish(b, oof, y, X.index, "prob", 2, is_aug)


def train_tabular(ctx: Ctx) -> None:
    clean = pd.read_parquet(feature_path(ctx.paths, "internal__clean"))
    y0 = ctx.meta.loc[ctx.train, "af"].to_numpy(int)
    for name in CONFIGS:
        A = ctx.aug_features(name)
        A = A[A.group.isin(ctx.train)]
        X = pd.concat([clean.loc[ctx.train], A[clean.columns.intersection(A.columns)]])
        groups = np.r_[np.asarray(ctx.train), A.group.to_numpy()]
        y = np.r_[y0, ctx.meta.loc[A.group, "af"].to_numpy(int)]
        is_aug = np.r_[np.zeros(len(ctx.train), bool), np.ones(len(A), bool)]
        for model in ["logreg", "lgbm"]:
            for seed in ctx.cfg["seeds"]:
                out = ctx.model_path(name, model, seed)
                if out.exists():
                    continue
                set_global_seed(seed)
                ref = load_pickle(ctx.paths.models / "binary" / "augmented" / f"{model}_seed{seed}.pkl")
                b = fit_fixed(model, X, y, groups, is_aug, ref["best_params"], ctx.cfg, seed)
                b.update(model=model, task="binary", variant=f"aug_{name}", seed=seed)
                out.parent.mkdir(parents=True, exist_ok=True)
                save_pickle(b, out)
                LOG.info("trained %s %s seed %d (threshold %.3f)", name, model, seed, b["threshold"])


def train_cnn(ctx: Ctx, seeds: list[int], name: str = "no_em") -> None:
    c = cfg_for(ctx.cfg, name)
    y = ctx.meta.loc[ctx.train, "af"].to_numpy(int)
    spec_by = {}
    for s in augmentation_spec(ctx.train, c):
        spec_by.setdefault(s["record_id"], []).append(s)
    for seed in seeds:
        out = ctx.model_path(name, "cnn", seed)
        if out.exists():
            continue
        set_global_seed(seed)
        hp = load_pickle(ctx.paths.models / "binary" / "standard" / f"cnn_seed{ctx.cfg['seeds'][0]}.pkl")["hp"]
        b = M.fit_cnn([ctx.internal["signals"][r] for r in ctx.train], ctx.train, y, c, seed, 2, ctx.bank, spec_by,
                      fixed_hp=hp, cache_dir=ctx.rev / "cnn_checkpoints" / f"augablation_{name}_seed{seed}")
        b.update(model="cnn", task="binary", variant=f"aug_{name}", seed=seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(b, out)
        LOG.info("trained CNN %s seed %d", name, seed)


def evaluate(ctx: Ctx) -> None:
    ext = load_dataset(ctx.cfg, "external")
    test = binary_ids(ctx.meta, ctx.split["test"])
    keys = sorted(p.stem for p in ctx.paths.features.glob("internal__*.parquet")
                  if p.stem not in ("internal__train_aug",) and "refpeaks" not in p.stem)
    keys += sorted(p.stem for p in ctx.paths.features.glob("external__*.parquet") if "refpeaks" not in p.stem)
    for name in CONFIGS:
        for model in ["logreg", "lgbm", "cnn"]:
            paths = sorted((ctx.rev / "models" / "augablation" / name).glob(f"{model}_seed*.pkl"))
            if not paths:
                continue
            out = ctx.rev / "predictions" / f"augablation__{name}__{model}.parquet"
            old = pd.read_parquet(out) if out.exists() else None
            done = set(old.condition) if old is not None else set()
            parts = [] if old is None else [old]
            bundles = [load_pickle(p) for p in paths]
            cms = [M.load_cnn(b, ctx.cfg) for b in bundles] if model == "cnn" else [None] * len(bundles)
            for key in keys:
                if key in done:
                    continue
                ds, ids = (ctx.internal, test) if key.startswith("internal") else (ext, list(ext["meta"].index))
                feats = pd.read_parquet(feature_path(ctx.paths, key)).loc[ids]
                sigs = None
                if model == "cnn":
                    cond = _cond_from_key(key)
                    sigs = [corrupt_for_test(ds["signals"][r], r, cond, ctx.bank, ctx.cfg["noise"]["seed"]) for r in ids]
                for b, cm in zip(bundles, cms):
                    raw = M.raw_scores(b, X=feats, signals=sigs, cfg=ctx.cfg, cnn_model=cm)
                    df = pd.DataFrame({"record_id": ids, "patient_id": ds["meta"].loc[ids, "patient_id"].to_numpy(),
                                       "y": ds["meta"].loc[ids, "af"].to_numpy(int)})
                    df["config"], df["model"], df["seed"], df["condition"] = name, model, b["seed"], key
                    df["prob"], df["threshold"] = b["calibrator"].transform(raw), b["threshold"]
                    parts.append(df)
                LOG.info("evaluated %s %s on %s", name, model, key)
                pd.concat(parts, ignore_index=True).to_parquet(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("what", choices=["features", "train-tabular", "train-cnn", "evaluate", "all-tabular"])
    ap.add_argument("--cnn-seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    setup_logging()
    ctx = Ctx(load_config(a.config))
    single_instance(ctx.rev / "augablation.lock")
    if a.what in ("features", "all-tabular"):
        for name in CONFIGS:
            ctx.aug_features(name)
    if a.what in ("train-tabular", "all-tabular"):
        train_tabular(ctx)
    if a.what == "train-cnn":
        train_cnn(ctx, a.cnn_seeds)
    if a.what in ("evaluate", "all-tabular"):
        evaluate(ctx)


if __name__ == "__main__":
    main()

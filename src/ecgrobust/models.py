"""The four models behind one interface.

Every trained model is stored as a *bundle* (a dict) holding the fitted
estimator, its tuned hyper-parameters, its out-of-fold (OOF) training
predictions, a Platt calibrator fitted on those OOF predictions, and a decision
threshold that maximizes F1 on the calibrated OOF predictions. Nothing in a
bundle is derived from test data.

Platt scaling is monotone, so it changes neither AUROC nor AUPRC; it only undoes
the probability distortion introduced by class weighting, which matters for the
calibration analysis and makes thresholds comparable across models.
"""
from __future__ import annotations

import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import cnn as cnnlib
from .features import FEATURES
from .metrics import average_precision, best_f1_threshold, challenge_score
from .noise import NoiseBank, corrupt_for_train
from .utils import load_pickle, save_pickle, stable_seed

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- calibration
class Platt:
    """Logistic recalibration on logit(p) (probabilistic models) or a z-scored score (rule)."""

    def __init__(self, kind: str = "prob"):
        self.kind = kind

    def _z(self, raw):
        raw = np.asarray(raw, float)
        if self.kind == "prob":
            p = np.clip(raw, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))[:, None]
        return ((raw - self.mu_) / self.sd_)[:, None]

    def fit(self, raw, y):
        raw = np.asarray(raw, float)
        self.mu_, self.sd_ = float(np.mean(raw)), float(np.std(raw) or 1.0)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(raw), y)
        return self

    def transform(self, raw):
        return self.lr_.predict_proba(self._z(raw))[:, 1]


# --------------------------------------------------------------------------- rule baseline
class RuleModel(BaseEstimator, ClassifierMixin):
    """Flag AF when one beat-irregularity measure exceeds a threshold.

    Its score is the raw feature value; the threshold is chosen on the training
    data by the same max-F1 rule used for every other model. Recordings where
    beat detection failed get the training median (i.e. no evidence either way).
    """

    def __init__(self, feature: str = "rr_rmssd_norm"):
        self.feature = feature

    def fit(self, X, y):
        self.fill_ = float(np.nanmedian(X[self.feature]))
        self.classes_ = np.array([0, 1])
        return self

    def score_raw(self, X):
        return X[self.feature].fillna(self.fill_).to_numpy(float)


# --------------------------------------------------------------------------- estimators
def _logreg(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", class_weight="balanced", max_iter=5000, random_state=seed)),
    ])


def _logreg_grid(cfg: dict) -> dict:
    l1 = {"l1": 1.0, "l2": 0.0}
    return {"clf__C": list(cfg["models"]["logreg"]["C"]),
            "clf__l1_ratio": [l1[p] for p in cfg["models"]["logreg"]["penalty"]]}


def _lgbm(seed: int, n_classes: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(objective="binary" if n_classes == 2 else "multiclass", class_weight="balanced",
                          subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1)


LGBM_SPACE = {
    "n_estimators": randint(100, 800),
    "learning_rate": loguniform(0.01, 0.2),
    "num_leaves": randint(7, 64),
    "min_child_samples": randint(5, 80),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": loguniform(1e-3, 10),
}


def _scorer(n_classes: int):
    if n_classes == 2:
        return "average_precision"
    return make_scorer(lambda y, p: challenge_score(y, p)["score"])


def _cv_splits(y, groups, k, seed):
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


# --------------------------------------------------------------------------- tabular fit
def fit_tabular(name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: dict, seed: int,
                n_classes: int, is_aug: np.ndarray | None = None) -> dict:
    X = X[FEATURES]
    splits = _cv_splits(y, groups, cfg["cv_folds"], seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        if name == "rule":
            est = RuleModel(cfg["models"]["rule"]["feature"]).fit(X, y)
            raw = est.score_raw(X)  # nothing is learned beyond a median fill, so in-sample == OOF
            return _finish({"estimator": est, "best_params": {"feature": est.feature}, "cv_score": average_precision(y, raw)},
                           raw, y, X.index, "score", n_classes, is_aug)
        if name == "logreg":
            search = GridSearchCV(_logreg(seed), _logreg_grid(cfg), scoring=_scorer(n_classes), cv=splits, n_jobs=-1)
        elif name == "lgbm":
            search = RandomizedSearchCV(_lgbm(seed, n_classes), LGBM_SPACE, n_iter=cfg["models"]["lgbm"]["n_iter"],
                                        scoring=_scorer(n_classes), cv=splits, n_jobs=-1, random_state=seed)
        else:
            raise ValueError(name)
        search.fit(X, y)
        oof = cross_val_predict(clone(search.best_estimator_), X, y, cv=splits, method="predict_proba", n_jobs=-1)
    raw = oof[:, 1] if n_classes == 2 else oof
    LOG.info("%-6s seed=%d best=%s cv=%.4f", name, seed, search.best_params_, search.best_score_)
    return _finish({"estimator": search.best_estimator_, "best_params": search.best_params_,
                    "cv_score": float(search.best_score_)}, raw, y, X.index, "prob", n_classes, is_aug)


def _finish(b: dict, raw, y, index, kind: str, n_classes: int, is_aug) -> dict:
    oof = pd.DataFrame({"y": y}, index=index)
    if is_aug is not None:
        oof["is_aug"] = is_aug
    if n_classes == 2:
        cal = Platt(kind).fit(raw, y)
        prob = cal.transform(raw)
        oof["raw"], oof["prob"] = raw, prob
        b.update(calibrator=cal, threshold=best_f1_threshold(y, prob))
    else:
        for k in range(n_classes):
            oof[f"p{k}"] = raw[:, k]
        b.update(calibrator=None, threshold=None)
    b.update(oof=oof, raw_kind=kind, n_classes=n_classes)
    return b


# --------------------------------------------------------------------------- CNN fit
def fit_cnn(signals: list, ids: list, y: np.ndarray, cfg: dict, seed: int, n_classes: int,
            bank: NoiseBank | None, aug_spec: dict | None, fixed_hp: dict | None = None,
            cache_dir=None) -> dict:
    """Hyper-parameters by k-fold CV, then a final fit on the training data.

    Calibration source (``models.cnn.calibration``):
    * ``oof``     - out-of-fold predictions from the CV models (as for the feature models);
    * ``holdout`` - the final fit's stratified inner validation split (also used for early
      stopping). Used when per-seed CV is computationally prohibitive; CV is then run only
      to choose hyper-parameters (``fixed_hp`` is None).

    With augmentation, calibration predictions are also made on the deterministic noisy
    copies used by the feature models, so every augmented model is calibrated on the same
    clean+noisy mixture.
    """
    fs, ccfg = cfg["fs"], cfg["models"]["cnn"]
    calib = ccfg.get("calibration", "oof")
    n_out = 1 if n_classes == 2 else n_classes
    g = ccfg["grid"]
    grid = [fixed_hp] if fixed_hp else [dict(zip(g, v)) for v in itertools.product(*g.values())]
    splits = list(StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    pos = {r: i for i, r in enumerate(ids)}

    def aug_preds(model, rec_idx):
        keys = [k for i in rec_idx for k in aug_spec.get(ids[i], [])]
        if not keys:
            return {}
        sig = [corrupt_for_train(signals[pos[s["record_id"]]], s, bank) for s in keys]
        return {f'{s["record_id"]}#aug{s["copy"]}': (p, y[pos[s["record_id"]]])
                for s, p in zip(keys, cnnlib.predict(model, sig, fs, ccfg, n_out))}

    best = None
    if calib == "oof" or len(grid) > 1:
        for hp in grid:
            oof = np.zeros(len(y)) if n_out == 1 else np.zeros((len(y), n_out))
            oof_aug = {}
            for f, (tr, va) in enumerate(splits):
                # per-fold checkpoint: a CV fold can take an hour on CPU, so a restart resumes here
                # the file name encodes the hyper-parameters and every other training setting, so a
                # config change cannot silently reuse a stale fold
                fp = stable_seed(sorted(hp.items()), sorted((k, str(v)) for k, v in ccfg.items() if k != "grid"),
                                 cfg["augmentation"], cfg["cv_folds"])
                ck = None if cache_dir is None else Path(cache_dir) / f"cfg_{fp}_fold{f}.pkl"
                if ck is not None and ck.exists():
                    saved = load_pickle(ck)
                    oof[va], fold_aug = saved["oof"], saved["aug"]
                    LOG.info("cnn    seed=%d hp=%s fold %d loaded from checkpoint", seed, hp, f)
                else:
                    m, _ = cnnlib.train([signals[i] for i in tr], y[tr], fs, cfg, hp, stable_seed(seed, f), n_classes, bank)
                    oof[va] = cnnlib.predict(m, [signals[i] for i in va], fs, ccfg, n_out)
                    fold_aug = aug_preds(m, va) if aug_spec is not None and calib == "oof" else {}
                    if ck is not None:
                        save_pickle({"oof": oof[va], "aug": fold_aug}, ck)
                        LOG.info("cnn    seed=%d hp=%s fold %d done (checkpointed)", seed, hp, f)
                oof_aug.update(fold_aug)
            score = average_precision(y, oof) if n_out == 1 else challenge_score(y, oof.argmax(1))["score"]
            LOG.info("cnn    seed=%d hp=%s cv=%.4f", seed, hp, score)
            if best is None or score > best[0]:
                best = (score, hp, oof, oof_aug)
    else:
        best = (np.nan, grid[0], None, {})
    score, hp, oof, oof_aug = best
    model, info = cnnlib.train(signals, y, fs, cfg, hp, seed, n_classes, bank)
    if calib == "oof":
        index, yy, raw = list(ids), list(y), list(oof)
    else:
        va = info["va_idx"]
        index, yy, raw = [ids[i] for i in va], list(y[va]), list(info["va_pred"])
        if aug_spec is not None:
            oof_aug = aug_preds(model, va)
        if not np.isfinite(score):
            score = info["inner_val"]
    is_aug = [False] * len(index)
    for k, (p, yk) in oof_aug.items():
        index.append(k)
        yy.append(yk)
        raw.append(p)
        is_aug.append(True)
    raw = np.asarray(raw)
    LOG.info("cnn    seed=%d final: best_epoch=%d inner_val=%.4f calibration=%s (n=%d)", seed, info["best_epoch"],
             info["inner_val"], calib, len(index))
    b = {"estimator": None, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "hp": hp,
         "best_params": hp, "cv_score": float(score), "calibration_source": calib,
         "final_fit": {k: v for k, v in info.items() if k in ("best_epoch", "inner_val")},
         "history": info["history"]}
    return _finish(b, raw, np.asarray(yy), pd.Index(index, name="record_id"), "prob", n_classes,
                   np.asarray(is_aug) if aug_spec is not None else None)


def load_cnn(bundle: dict, cfg: dict):
    n_out = 1 if bundle["n_classes"] == 2 else bundle["n_classes"]
    m = cnnlib.build(n_out, cfg, bundle["hp"])
    m.load_state_dict(bundle["state_dict"])
    m.eval()
    return m


# --------------------------------------------------------------------------- scoring
def raw_scores(bundle: dict, X: pd.DataFrame | None = None, signals: list | None = None, cfg: dict | None = None,
               cnn_model=None) -> np.ndarray:
    """Uncalibrated scores: (n,) for binary, (n, k) for multiclass."""
    if bundle["model"] == "cnn":
        n_out = 1 if bundle["n_classes"] == 2 else bundle["n_classes"]
        return cnnlib.predict(cnn_model or load_cnn(bundle, cfg), signals, cfg["fs"], cfg["models"]["cnn"], n_out)
    X = X[FEATURES]
    if bundle["model"] == "rule":
        return bundle["estimator"].score_raw(X)
    p = bundle["estimator"].predict_proba(X)
    return p[:, 1] if bundle["n_classes"] == 2 else p

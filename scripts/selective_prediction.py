"""Selective prediction: does quality-based abstention fail only for one quality index, or generally?

    python scripts/selective_prediction.py --config configs/local.yaml

Quality measures (higher = better quality), all computed without labels:

  avgqrs     NeuroKit2 average-QRS index (the main study's measure)
  zhao       NeuroKit2 fuzzy index of Zhao and Zhang (2018), three levels
  ksqi       kurtosis of the cleaned signal (Li et al 2008)
  psqi       relative power in the QRS band, 5-15 Hz over 5-40 Hz (Li et al 2008)
  bassqi     1 - relative power below 1 Hz (Li et al 2008)
  bsqi       beat-detection agreement: F1 between the NeuroKit2 and Hamilton (2002) detectors on the same signal
  learned    a multi-feature quality model: logistic regression on the indices above (plus the beat count per
             second, the valid-RR fraction and QRS template correlation), trained on the CinC 2017 *development*
             recordings to separate the expert-labelled noisy class from the others; score = 1 - P(noisy)

Analyses (standard training, gradient boosting and the CNN, means over five seeds)
  1. coverage-risk: false-positive rate, false-negative rate and the AF rejection ratio (share of AF recordings
     withheld / share of non-AF recordings withheld) as the lowest-quality recordings are withheld, for the
     internal test set clean and under EM 0 dB, EM -6 dB and MA -6 dB;
  2. fixed source thresholds: each measure's threshold is its 5th percentile on clean CinC training data; the
     share withheld is reported internally and on PTB-XL (clean, EM -6 dB, MA -6 dB), overall and by sex and age;
  3. target-domain recalibration: PTB-XL patients are split in half (seeded); the threshold is the 5th
     percentile on clean recordings of one half and is evaluated on the other half only.

Outputs: outputs/tables/selective_coverage_risk.csv, outputs/tables/selective_thresholds.csv,
outputs/tables/selective_subgroups.csv, outputs/revision/quality_scores.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrobust.data import load_dataset  # noqa: E402
from ecgrobust.features import match_beats  # noqa: E402
from ecgrobust.noise import NoiseBank, corrupt_for_test  # noqa: E402
from ecgrobust.pipeline import _cond_from_key, binary_ids, feature_path  # noqa: E402
from ecgrobust.splits import load_split  # noqa: E402
from ecgrobust.utils import Paths, load_config, setup_logging  # noqa: E402

LOG = logging.getLogger("selective_prediction")
MEASURES = ["avgqrs", "zhao", "ksqi", "psqi", "bassqi", "bsqi", "learned"]
INTERNAL = ["internal__clean", "internal__em__+0dB__r0", "internal__em__-6dB__r0", "internal__ma__-6dB__r0"]
EXTERNAL = ["external__clean", "external__em__-6dB__r0", "external__ma__-6dB__r0"]
COVERAGES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
LEARNED_INPUTS = ["q_avgqrs", "q_zhao", "q_ksqi", "q_psqi", "q_bassqi", "beats_per_s", "rr_valid_frac", "qrs_corr_mean"]


def two_detector_agreement(sig: np.ndarray, fs: int) -> float:
    import warnings

    import neurokit2 as nk

    warnings.filterwarnings("ignore")
    try:
        try:
            sig, _ = nk.ecg_invert(sig, sampling_rate=fs)
        except Exception:
            pass
        a = nk.ecg_peaks(nk.ecg_clean(sig, sampling_rate=fs), sampling_rate=fs, correct_artifacts=False)[1]["ECG_R_Peaks"]
        b = nk.ecg_peaks(nk.ecg_clean(sig, sampling_rate=fs, method="hamilton2002"), sampling_rate=fs,
                         method="hamilton2002", correct_artifacts=False)[1]["ECG_R_Peaks"]
        tp, fp, fn = match_beats(np.asarray(a), np.asarray(b), round(0.15 * fs))
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    except Exception:
        return 0.0


def prep(df: pd.DataFrame, dur) -> pd.DataFrame:
    df = df.copy()
    df["beats_per_s"] = df.rr_n_beats / np.asarray(dur, float)
    return df


def fit_learned_quality(cfg: dict, paths: Paths, internal: dict | None = None):
    """Logistic regression separating the expert-labelled noisy CinC development recordings from the rest."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    internal = internal or load_dataset(cfg, "internal")
    meta_i = internal["meta"]
    clean_i = prep(pd.read_parquet(feature_path(paths, "internal__clean")), meta_i.duration_s)
    tr = list(load_split(cfg)["train"])
    noisy = (meta_i.loc[tr, "label4"] == "~").astype(int).to_numpy()
    learned = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                            LogisticRegression(class_weight="balanced", max_iter=2000))
    learned.fit(clean_i.loc[tr, LEARNED_INPUTS], noisy)
    LOG.info("learned quality model: trained on %d development recordings (%d labelled noisy)", len(tr), noisy.sum())
    return learned


def quality_table(cfg: dict, paths: Paths) -> pd.DataFrame:
    """One row per (condition, record) with every quality measure."""
    from joblib import Parallel, delayed

    out = paths.root / "revision" / "quality_scores.parquet"
    if out.exists():
        return pd.read_parquet(out)
    fs = cfg["fs"]
    internal, external = load_dataset(cfg, "internal"), load_dataset(cfg, "external")
    split = load_split(cfg)
    bank = NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])
    meta_i = internal["meta"]
    learned = fit_learned_quality(cfg, paths, internal)

    rows = []
    test_b = binary_ids(meta_i, split["test"])
    train_b = binary_ids(meta_i, split["train"])
    jobs = [("internal__train_clean", internal, train_b)] + [(k, internal, test_b) for k in INTERNAL] + \
           [(k, external, list(external["meta"].index)) for k in EXTERNAL]
    for key, ds, ids in jobs:
        fkey = "internal__clean" if key == "internal__train_clean" else key
        f = prep(pd.read_parquet(feature_path(paths, fkey)).loc[ids], ds["meta"].loc[ids, "duration_s"])
        cond = _cond_from_key(fkey)
        sigs = [corrupt_for_test(ds["signals"][r], r, cond, bank, cfg["noise"]["seed"]) for r in ids]
        bsqi = Parallel(n_jobs=cfg["features"]["n_jobs"], batch_size=32)(delayed(two_detector_agreement)(s, fs) for s in sigs)
        q = pd.DataFrame({"condition": key, "record_id": ids, "avgqrs": f.q_avgqrs.to_numpy(), "zhao": f.q_zhao.to_numpy(),
                          "ksqi": f.q_ksqi.to_numpy(), "psqi": f.q_psqi.to_numpy(), "bassqi": f.q_bassqi.to_numpy(),
                          "bsqi": np.asarray(bsqi, float),
                          "learned": 1 - learned.predict_proba(f[LEARNED_INPUTS])[:, 1]})
        rows.append(q)
        LOG.info("quality measures for %s (n=%d)", key, len(ids))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out)
    return df


def rank_quality(q: np.ndarray, seed: int = 0) -> np.ndarray:
    """Quality with missing values treated as the lowest and ties broken at random (reproducibly)."""
    q = np.where(np.isfinite(q), q, -np.inf)
    jitter = np.random.default_rng(seed).random(len(q)) * 1e-9
    return q + jitter


def preds(paths: Paths, model: str, key: str) -> pd.DataFrame:
    if key.startswith("external") and key != "external__clean":
        f = paths.predictions / f"external_noise__standard__{model}.parquet"
    else:
        f = paths.predictions / f"binary__standard__{model}.parquet"
    return pd.read_parquet(f, columns=["record_id", "y", "seed", "prob", "threshold", "condition"],
                           filters=[("condition", "==", key)])


def rates(y, pos):
    n_af, n_non = (y == 1).sum(), (y == 0).sum()
    fp, fn = ((pos == 1) & (y == 0)).sum(), ((pos == 0) & (y == 1)).sum()
    tp = ((pos == 1) & (y == 1)).sum()
    return {"fpr": fp / n_non if n_non else np.nan, "fnr": fn / n_af if n_af else np.nan,
            "sensitivity": tp / n_af if n_af else np.nan, "specificity": 1 - fp / n_non if n_non else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan}


def coverage_risk(q: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    rows = []
    for model in ["lgbm", "cnn"]:
        for key in INTERNAL + EXTERNAL:
            p = preds(paths, model, key)
            qq = q[q.condition == key].set_index("record_id")
            for meas in MEASURES:
                score = pd.Series(rank_quality(qq[meas].to_numpy(float)), index=qq.index)
                for cov in COVERAGES:
                    keep_n = int(round(cov * len(score)))
                    keep = set(score.sort_values(ascending=False).index[:keep_n])
                    per = []
                    for s, g in p.groupby("seed"):
                        m = g.record_id.isin(keep).to_numpy()
                        y = g.y.to_numpy()
                        r = rates(y[m], (g.prob.to_numpy() >= g.threshold.to_numpy())[m].astype(int))
                        rej_af = 1 - m[y == 1].mean()
                        rej_non = 1 - m[y == 0].mean()
                        r["af_rejection_ratio"] = rej_af / rej_non if rej_non > 0 else np.nan
                        r["withheld_af"], r["withheld_non_af"] = rej_af, rej_non
                        per.append(r)
                    rows.append({"model": model, "condition": key, "measure": meas, "coverage": cov,
                                 **{k: float(np.nanmean([d[k] for d in per])) for k in per[0]}})
    return pd.DataFrame(rows)


def thresholds(q: pd.DataFrame, cfg: dict, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    ext_meta = load_dataset(cfg, "external")["meta"]
    pid = ext_meta.patient_id
    rng = np.random.default_rng(2024)
    pats = pid.unique()
    half_a = set(rng.choice(pats, len(pats) // 2, replace=False))
    in_a = pid.isin(half_a)
    rows, sub = [], []
    for meas in MEASURES:
        src = np.nanpercentile(q[q.condition == "internal__train_clean"][meas].to_numpy(float), 5)
        ext_clean = q[q.condition == "external__clean"].set_index("record_id")[meas]
        tgt = np.nanpercentile(ext_clean[in_a.reindex(ext_clean.index).to_numpy()].to_numpy(float), 5)
        for key in INTERNAL + EXTERNAL:
            qq = q[q.condition == key].set_index("record_id")[meas]
            for policy, thr in [("source (CinC training, 5th percentile)", src),
                                ("target (PTB-XL half A, 5th percentile)", tgt)]:
                if policy.startswith("target") and not key.startswith("external"):
                    continue
                ids = qq.index if policy.startswith("source") else qq.index[~in_a.reindex(qq.index).to_numpy()]
                keep = (qq.loc[ids].fillna(-np.inf) >= thr)
                for model in ["lgbm", "cnn"]:
                    p = preds(paths, model, key)
                    p = p[p.record_id.isin(ids)]
                    per = []
                    for s, g in p.groupby("seed"):
                        m = keep.reindex(g.record_id).to_numpy()
                        y = g.y.to_numpy()
                        r = rates(y[m], (g.prob.to_numpy() >= g.threshold.to_numpy())[m].astype(int))
                        r["withheld_all"], r["withheld_af"], r["withheld_non_af"] = 1 - m.mean(), 1 - m[y == 1].mean(), 1 - m[y == 0].mean()
                        per.append(r)
                    rows.append({"measure": meas, "policy": policy, "threshold": thr, "condition": key, "model": model,
                                 "n": len(ids), **{k: float(np.nanmean([d[k] for d in per])) for k in per[0]}})
                if key.startswith("external"):
                    meta = ext_meta.loc[ids]
                    groups = {"female": meta.sex == 1, "male": meta.sex == 0, "age < 65": meta.age < 65,
                              "age >= 65": meta.age >= 65}
                    for gname, gm in groups.items():
                        gm = gm.to_numpy()
                        sub.append({"measure": meas, "policy": policy, "condition": key, "subgroup": gname,
                                    "n": int(gm.sum()), "withheld": float(1 - keep.to_numpy()[gm].mean())})
    return pd.DataFrame(rows), pd.DataFrame(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    paths = Paths.from_cfg(cfg)
    (paths.root / "revision").mkdir(exist_ok=True)
    q = quality_table(cfg, paths)
    coverage_risk(q, paths).to_csv(paths.tables / "selective_coverage_risk.csv", index=False)
    thr, sub = thresholds(q, cfg, paths)
    thr.to_csv(paths.tables / "selective_thresholds.csv", index=False)
    sub.to_csv(paths.tables / "selective_subgroups.csv", index=False)
    LOG.info("wrote selective_* tables")


if __name__ == "__main__":
    main()

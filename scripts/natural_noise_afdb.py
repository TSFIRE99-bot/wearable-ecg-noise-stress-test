"""Naturally noisy recordings: the trained models on the MIT-BIH Atrial Fibrillation Database (afdb).

    python scripts/natural_noise_afdb.py --config configs/local.yaml

The main study adds recorded noise to clean signals. This script instead evaluates the unchanged models (all
five seeds, standard and augmented training) on ambulatory Holter ECGs whose noise is real and intrinsic to the
recording, with expert rhythm annotations:

* afdb: 23 two-lead Holter recordings of about 10 h (250 Hz) from 23 patients with paroxysmal AF; the first
  lead (ECG1) is used, resampled to 300 Hz;
* non-overlapping 30 s windows that lie entirely within one annotated rhythm episode: AF ("(AFIB") is positive;
  normal rhythm ("(N"), atrial flutter ("(AFL") and junctional rhythm ("(J") are negative; mixed windows are
  excluded;
* windows are grouped into fifths by the learned quality score (trained on the CinC 2017 development recordings
  only, see selective_prediction.py), so the noisiest fifth can be compared with the cleanest. Beat-detection
  agreement between two detectors is reported as a second, independent noise indicator;
* as a bridge to the main study, the cleanest fifth is also corrupted with the study's additive electrode-motion
  and muscle-artifact noise at 0 and -6 dB.

Uncertainty: 95% confidence intervals from a bootstrap over the 23 patients (recordings).

Outputs: outputs/revision/features/afdb__*.parquet, outputs/revision/predictions/afdb__<variant>__<model>.parquet,
outputs/tables/afdb_windows.csv, outputs/tables/afdb_by_quality.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ecgrobust import models as M  # noqa: E402
from ecgrobust.data import load_dataset  # noqa: E402
from ecgrobust.features import extract_many  # noqa: E402
from ecgrobust.metrics import binary_metrics  # noqa: E402
from ecgrobust.noise import Condition, NoiseBank, corrupt_for_test  # noqa: E402
from ecgrobust.utils import Paths, load_config, load_pickle, setup_logging, single_instance  # noqa: E402
from selective_prediction import LEARNED_INPUTS, fit_learned_quality, prep, two_detector_agreement  # noqa: E402

import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) < (1, 8):
    LogisticRegression.multi_class = "deprecated"

LOG = logging.getLogger("natural_noise_afdb")
MODELS, VARIANTS = ["rule", "logreg", "lgbm", "cnn"], ["standard", "augmented"]
WIN_S = 30


def load_windows(afdb: Path, fs: int) -> tuple[dict, pd.DataFrame]:
    import wfdb

    sigs, rows = {}, []
    for hea in sorted(afdb.glob("*.hea")):
        rec = hea.stem
        if not (afdb / f"{rec}.dat").exists():
            continue
        r = wfdb.rdrecord(str(afdb / rec))
        a = wfdb.rdann(str(afdb / rec), "atr")
        x = resample_poly(np.nan_to_num(r.p_signal[:, 0].astype(np.float64)), 6, 5).astype(np.float32)
        sc = fs / r.fs
        bounds = [(round(s * sc), aux.strip().rstrip("\x00")) for s, aux in zip(a.sample, a.aux_note) if aux.strip()]
        bounds.append((len(x), None))
        n = WIN_S * fs
        for (start, rh), (end, _) in zip(bounds[:-1], bounds[1:]):
            k = 0
            while start + (k + 1) * n <= end:
                lo = start + k * n
                wid = f"afdb_{rec}_{lo:08d}"
                sigs[wid] = x[lo:lo + n]
                rows.append({"record_id": wid, "patient_id": rec, "rhythm": rh, "af": int(rh == "(AFIB")})
                k += 1
    meta = pd.DataFrame(rows).set_index("record_id")
    meta = meta[meta.rhythm.isin(["(AFIB", "(N", "(AFL", "(J"])]
    return {k: sigs[k] for k in meta.index}, meta


def metrics_by(df: pd.DataFrame, group: str, B: int = 1000, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pats = np.array(sorted(df.patient_id.unique()))
    picks = [rng.choice(pats, len(pats)) for _ in range(B)]
    out = []
    for (variant, model, g), d in df.groupby(["variant", "model", group]):
        by_p = {p: s for p, s in d.groupby("patient_id")}

        def est(sub):
            per = [binary_metrics(s.y.to_numpy(), s.prob.to_numpy(), s.threshold.iloc[0]) for _, s in sub.groupby("seed")]
            return {k: float(np.nanmean([m[k] for m in per])) for k in ["sensitivity", "specificity", "ppv", "auroc", "f1"]}

        point = est(d)
        reps = []
        for pk in picks:
            sub = pd.concat([by_p[p] for p in pk if p in by_p])
            reps.append(est(sub))
        row = {"variant": variant, "model": model, group: g, "n_windows": int(d.record_id.nunique()),
               "n_af": int(d.drop_duplicates("record_id").y.sum()), "n_patients": int(d.patient_id.nunique())}
        for k, v in point.items():
            vals = np.array([r[k] for r in reps], float)
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = v, *np.nanpercentile(vals, [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out)


def main():
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--afdb", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    paths = Paths.from_cfg(cfg)
    rev = paths.root / "revision"
    single_instance(rev / "afdb.lock")
    for d in ["features", "predictions"]:
        (rev / d).mkdir(parents=True, exist_ok=True)
    fs = cfg["fs"]
    sigs, meta = load_windows(Path(a.afdb or Path(cfg["paths"]["raw_dir"]) / "afdb"), fs)
    ids = list(meta.index)
    LOG.info("afdb: %d windows from %d patients; rhythms %s", len(ids), meta.patient_id.nunique(),
             meta.rhythm.value_counts().to_dict())

    f_clean = rev / "features" / "afdb__natural.parquet"
    if f_clean.exists():
        feats = pd.read_parquet(f_clean)
    else:
        feats = extract_many(sigs, fs, cfg["features"], return_peaks=True, desc="afdb natural")
        peaks = feats.pop("_peaks")
        feats["_peaks_n"] = peaks.apply(len)
        pd.to_pickle(peaks.to_dict(), rev / "features" / "afdb__natural_peaks.pkl")
        feats.to_parquet(f_clean)
    q = rev / "features" / "afdb__quality.parquet"
    if q.exists():
        qual = pd.read_parquet(q)
    else:
        learned = fit_learned_quality(cfg, paths)
        f2 = prep(feats.loc[ids], np.full(len(ids), WIN_S))
        bsqi = Parallel(n_jobs=cfg["features"]["n_jobs"], batch_size=32)(delayed(two_detector_agreement)(sigs[r], fs) for r in ids)
        qual = pd.DataFrame({"learned": 1 - learned.predict_proba(f2[LEARNED_INPUTS])[:, 1], "bsqi": bsqi,
                             "avgqrs": f2.q_avgqrs.to_numpy()}, index=pd.Index(ids, name="record_id"))
        qual.to_parquet(q)
    meta = meta.join(qual)
    meta["quality_fifth"] = pd.qcut(meta.learned.rank(method="first"), 5, labels=["Q1 (noisiest)", "Q2", "Q3", "Q4", "Q5 (cleanest)"])
    meta.to_csv(paths.tables / "afdb_windows.csv")

    # additive noise on the cleanest fifth
    clean_ids = list(meta.index[meta.quality_fifth == "Q5 (cleanest)"])
    bank = NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])
    ref = pd.read_pickle(rev / "features" / "afdb__natural_peaks.pkl")
    conds = {"natural": (None, ids)}
    for n in ("em", "ma"):
        for s in (0.0, -6.0):
            conds[f"additive_{n}_{s:+g}dB"] = (Condition("afdb", n, s, 0), clean_ids)

    for variant in VARIANTS:
        for model in MODELS:
            out = rev / "predictions" / f"afdb__{variant}__{model}.parquet"
            if out.exists():
                continue
            bs = [load_pickle(paths.models / "binary" / variant / f"{model}_seed{s}.pkl") for s in cfg["seeds"]]
            cms = [M.load_cnn(b, cfg) for b in bs] if model == "cnn" else [None] * len(bs)
            parts = []
            for name, (cond, cids) in conds.items():
                if cond is None:
                    s_used, f_used = sigs, feats
                else:
                    s_used = {r: corrupt_for_test(sigs[r], r, cond, bank, cfg["noise"]["seed"]) for r in cids}
                    ff = rev / "features" / f"afdb__{name}.parquet"
                    if ff.exists():
                        f_used = pd.read_parquet(ff)
                    else:
                        f_used = extract_many(s_used, fs, cfg["features"], ref_peaks={r: ref[r] for r in cids}, desc=f"afdb {name}")
                        f_used.to_parquet(ff)
                X = f_used.loc[cids]
                sl = [s_used[r] for r in cids] if model == "cnn" else None
                for b, cm in zip(bs, cms):
                    raw = M.raw_scores(b, X=X, signals=sl, cfg=cfg, cnn_model=cm)
                    df = pd.DataFrame({"record_id": cids, "patient_id": meta.loc[cids, "patient_id"].to_numpy(),
                                       "y": meta.loc[cids, "af"].to_numpy(int)})
                    df["variant"], df["model"], df["seed"], df["condition"] = variant, model, b["seed"], name
                    df["prob"], df["threshold"] = b["calibrator"].transform(raw), b["threshold"]
                    parts.append(df)
                LOG.info("afdb: scored %s/%s on %s", variant, model, name)
            pd.concat(parts, ignore_index=True).to_parquet(out)

    allp = pd.concat([pd.read_parquet(f) for f in sorted((rev / "predictions").glob("afdb__*.parquet"))], ignore_index=True)
    allp = allp.join(meta[["quality_fifth", "rhythm"]], on="record_id")
    nat = allp[allp.condition == "natural"]
    t1 = metrics_by(nat, "quality_fifth", a.bootstrap)
    t1.insert(2, "condition", "natural")
    cl = allp[allp.record_id.isin(clean_ids)].copy()
    t2 = metrics_by(cl, "condition", a.bootstrap)
    t2 = t2.rename(columns={"condition": "quality_fifth"})
    t2.insert(2, "condition", t2.pop("quality_fifth"))
    t2["quality_fifth"] = "Q5 (cleanest)"
    pd.concat([t1, t2], ignore_index=True).to_csv(paths.tables / "afdb_by_quality.csv", index=False)
    LOG.info("wrote afdb_by_quality.csv")


if __name__ == "__main__":
    main()

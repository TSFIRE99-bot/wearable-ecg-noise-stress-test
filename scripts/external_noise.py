"""Secondary analysis: does the noise failure pattern replicate on the external test set?

    python scripts/external_noise.py --config configs/local.yaml --check
    python scripts/external_noise.py --config configs/local.yaml

The main study corrupted only the internal test set (evaluation.external_noise: false, since
the full grid would need about 22k x 18 feature extractions). This script corrupts PTB-XL lead I
with the same NSTDB test-partition noise, the same seeding and the same feature pipeline, but
only for the conditions named in CONDITIONS, and scores the already-trained models (standard and
augmented, all five seeds). Nothing is retrained and no threshold or calibrator is changed.

--check   recompute a sample of stored features and predictions and stop; use it to confirm that
          the current Python environment reproduces the saved results before trusting new ones.

Outputs
  outputs/features/external__<noise>__<snr>dB__r0.parquet           features of the corrupted records
  outputs/predictions/external_noise__<variant>__<model>.parquet    per-seed probabilities
  outputs/tables/external_noise.csv                                  metrics with patient-level 95% CIs
  outputs/tables/external_noise_paired.csv                           paired changes vs clean and augmentation effects
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

from ecgrobust import models as M  # noqa: E402
from ecgrobust.data import load_dataset  # noqa: E402
from ecgrobust.features import extract_many  # noqa: E402
from ecgrobust.metrics import BINARY_METRICS, bootstrap_indices, paired_difference, summarize_condition  # noqa: E402
from ecgrobust.noise import Condition, NoiseBank, corrupt_for_test  # noqa: E402
from ecgrobust.pipeline import binary_ids, feature_path  # noqa: E402
from ecgrobust.splits import load_split  # noqa: E402
from ecgrobust.utils import Paths, load_config, load_pickle, setup_logging  # noqa: E402

LOG = logging.getLogger("external_noise")

# Models were pickled with scikit-learn 1.9, which dropped LogisticRegression.multi_class. If an
# older scikit-learn reads them (e.g. a different Python environment), give the attribute its
# binary default. --check confirms the recomputed probabilities match the stored ones.
import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) < (1, 8):
    LogisticRegression.multi_class = "deprecated"
CONDITIONS = [("em", 6.0), ("em", 0.0), ("em", -6.0), ("ma", -6.0)]
MODELS, VARIANTS = ["rule", "logreg", "lgbm", "cnn"], ["standard", "augmented"]
REPORT = ["auroc", "auprc", "f1", "sensitivity", "specificity", "ppv"]


def bundles(paths: Paths, cfg: dict, variant: str, model: str) -> list[dict]:
    return [load_pickle(paths.models / "binary" / variant / f"{model}_seed{s}.pkl") for s in cfg["seeds"]]


def check(cfg: dict, paths: Paths, bank: NoiseBank, n: int = 60) -> bool:
    """Recompute stored features and predictions for a sample; report the largest differences."""
    ok = True
    internal, split = load_dataset(cfg, "internal"), load_split(cfg)
    ids = binary_ids(internal["meta"], split["test"])[:n]
    cond = Condition("internal", "em", -6.0, 0)
    ref = load_pickle(paths.features / "internal__refpeaks.pkl")
    sigs = {r: corrupt_for_test(internal["signals"][r], r, cond, bank, cfg["noise"]["seed"]) for r in ids}
    new = extract_many(sigs, cfg["fs"], cfg["features"], ref_peaks={r: ref[r] for r in ids}, desc="check")
    old = pd.read_parquet(feature_path(paths, cond.key)).loc[ids, new.columns]
    num = new.select_dtypes("number").columns
    diff = (new[num] - old[num]).abs().max().sort_values(ascending=False)
    LOG.info("features: largest absolute differences on %d records\n%s", n, diff.head(8).to_string())
    ok &= bool(np.nanmax(diff.to_numpy()) < 1e-6)

    feats = pd.read_parquet(feature_path(paths, "external__clean"))
    ext = load_dataset(cfg, "external")
    for model in MODELS:
        b = bundles(paths, cfg, "standard", model)[0]
        stored = pd.read_parquet(paths.predictions / f"binary__standard__{model}.parquet",
                                 filters=[("condition", "==", "external__clean"), ("seed", "==", b["seed"])])
        stored = stored.set_index("record_id")
        rid = list(stored.index[:400] if model == "cnn" else stored.index)
        cm = M.load_cnn(b, cfg) if model == "cnn" else None
        sig = [ext["signals"][r] for r in rid] if model == "cnn" else None
        prob = b["calibrator"].transform(M.raw_scores(b, X=feats.loc[rid], signals=sig, cfg=cfg, cnn_model=cm))
        d = float(np.max(np.abs(prob - stored.loc[rid, "prob"].to_numpy())))
        LOG.info("predictions %-7s seed %d: max |prob difference| = %.2e over %d records", model, b["seed"], d, len(rid))
        ok &= d < 1e-5
    LOG.info("CHECK %s", "PASSED" if ok else "FAILED")
    return ok


def run_features(cfg: dict, paths: Paths, bank: NoiseBank, ext: dict) -> None:
    ids = list(ext["meta"].index)
    ref = load_pickle(paths.features / "external__refpeaks.pkl")
    for noise, snr in CONDITIONS:
        cond = Condition("external", noise, snr, 0)
        out = feature_path(paths, cond.key)
        if out.exists():
            continue
        sigs = {r: corrupt_for_test(ext["signals"][r], r, cond, bank, cfg["noise"]["seed"]) for r in ids}
        extract_many(sigs, cfg["fs"], cfg["features"], ref_peaks={r: ref[r] for r in ids}, desc=cond.key).to_parquet(out)


def run_predictions(cfg: dict, paths: Paths, bank: NoiseBank, ext: dict) -> None:
    meta, ids = ext["meta"], list(ext["meta"].index)
    for variant in VARIANTS:
        for model in MODELS:
            out = paths.predictions / f"external_noise__{variant}__{model}.parquet"
            if out.exists():
                continue
            bs = bundles(paths, cfg, variant, model)
            cms = [M.load_cnn(b, cfg) for b in bs] if model == "cnn" else [None] * len(bs)
            rows = []
            for noise, snr in CONDITIONS:
                cond = Condition("external", noise, snr, 0)
                feats = pd.read_parquet(feature_path(paths, cond.key)).loc[ids]
                sigs = ([corrupt_for_test(ext["signals"][r], r, cond, bank, cfg["noise"]["seed"]) for r in ids]
                        if model == "cnn" else None)
                for b, cm in zip(bs, cms):
                    raw = M.raw_scores(b, X=feats, signals=sigs, cfg=cfg, cnn_model=cm)
                    df = pd.DataFrame({"record_id": ids, "patient_id": meta.loc[ids, "patient_id"].to_numpy(),
                                       "y": meta.loc[ids, "af"].to_numpy(int)})
                    df["variant"], df["model"], df["seed"] = variant, model, b["seed"]
                    df["condition"], df["noise"], df["snr_db"] = cond.key, noise, snr
                    df["prob"], df["threshold"] = b["calibrator"].transform(raw), b["threshold"]
                    df["q_avgqrs"], df["rr_n_beats"] = feats["q_avgqrs"].to_numpy(float), feats["rr_n_beats"].to_numpy(float)
                    rows.append(df)
                LOG.info("scored %s/%s on %s", variant, model, cond.key)
            pd.concat(rows, ignore_index=True).to_parquet(out)


def matrix(df: pd.DataFrame, ids: list[str], seeds: list[int]):
    """(seeds, n) probabilities and (seeds,) thresholds in a fixed record order."""
    P = np.vstack([df[df.seed == s].set_index("record_id").loc[ids, "prob"].to_numpy() for s in seeds])
    thr = np.array([df[df.seed == s].threshold.iloc[0] for s in seeds])
    return P, thr


def run_tables(cfg: dict, paths: Paths, ext: dict) -> None:
    meta, ids, seeds = ext["meta"], list(ext["meta"].index), cfg["seeds"]
    y = meta.loc[ids, "af"].to_numpy(int)
    boots = bootstrap_indices(len(ids), cfg["evaluation"]["bootstrap"], cfg["evaluation"]["bootstrap_seed"],
                              meta.loc[ids, "patient_id"].to_numpy())
    summ, rows = {}, []
    for variant in VARIANTS:
        for model in MODELS:
            clean = pd.read_parquet(paths.predictions / f"binary__{variant}__{model}.parquet",
                                    filters=[("condition", "==", "external__clean")])
            noisy = pd.read_parquet(paths.predictions / f"external_noise__{variant}__{model}.parquet")
            for key, df in [("external__clean", clean)] + list(noisy.groupby("condition")):
                P, thr = matrix(df, ids, seeds)
                s = summarize_condition(y, P, thr, boots)
                summ[(variant, model, key)] = s
                row = {"variant": variant, "model": model, "condition": key}
                for k in REPORT:
                    row[k], row[f"{k}_sd"], row[f"{k}_lo"], row[f"{k}_hi"] = s[k], s[f"{k}_sd"], s[f"{k}_lo"], s[f"{k}_hi"]
                rows.append(row)
                LOG.info("%s %s %s AUROC %.3f sens %.3f spec %.3f", variant, model, key, s["auroc"], s["sensitivity"], s["specificity"])
    pd.DataFrame(rows).to_csv(paths.tables / "external_noise.csv", index=False)

    j = {k: i for i, k in enumerate(BINARY_METRICS)}
    pr = []
    conds = [Condition("external", n, s, 0).key for n, s in CONDITIONS]
    for variant in VARIANTS:
        for model in MODELS:
            for c in conds:   # noisy minus clean, same model
                a, b = summ[(variant, model, c)], summ[(variant, model, "external__clean")]
                for k in REPORT:
                    d = paired_difference(a["_reps"][:, j[k]], b["_reps"][:, j[k]], a[k], b[k])
                    pr.append({"comparison": "noisy - clean", "variant": variant, "model": model, "condition": c, "metric": k, **d})
    for model in MODELS:
        for c in ["external__clean"] + conds:   # augmented minus standard
            a, b = summ[("augmented", model, c)], summ[("standard", model, c)]
            for k in REPORT:
                d = paired_difference(a["_reps"][:, j[k]], b["_reps"][:, j[k]], a[k], b[k])
                pr.append({"comparison": "augmented - standard", "variant": "", "model": model, "condition": c, "metric": k, **d})
    for variant in VARIANTS:
        for c in ["external__clean"] + conds:   # CNN minus gradient boosting
            a, b = summ[(variant, "cnn", c)], summ[(variant, "lgbm", c)]
            for k in REPORT:
                d = paired_difference(a["_reps"][:, j[k]], b["_reps"][:, j[k]], a[k], b[k])
                pr.append({"comparison": "cnn - lgbm", "variant": variant, "model": "", "condition": c, "metric": k, **d})
    pd.DataFrame(pr).to_csv(paths.tables / "external_noise_paired.csv", index=False)
    LOG.info("wrote external_noise.csv and external_noise_paired.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    paths = Paths.from_cfg(cfg)
    bank = NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])
    if a.check:
        sys.exit(0 if check(cfg, paths, bank) else 1)
    ext = load_dataset(cfg, "external")
    run_features(cfg, paths, bank, ext)
    run_predictions(cfg, paths, bank, ext)
    run_tables(cfg, paths, ext)


if __name__ == "__main__":
    main()

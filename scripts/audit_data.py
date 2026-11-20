"""Data, label and leakage audit (no model is retrained).

    python scripts/audit_data.py --config configs/local.yaml

Checks
  1. Exact and near-duplicate signals inside CinC 2017, and whether any cross the train/test split.
  2. Leakage paths in every stored model bundle: out-of-fold (calibration/threshold) predictions,
     stored training quality and augmentation copies must come from training records only.
  3. Noise separation: training-augmentation and test-corruption segments come from disjoint halves of
     each NSTDB record.
  4. PTB-XL label composition: AF prevalence, AF with co-coded flutter, and the non-AF rhythm and ectopy
     categories, with the share of each category that every model calls AF (clean and under electrode
     motion at -6 dB). This shows whether atrial flutter or frequent ectopy drive the external false alarms.
  5. A label sensitivity analysis: external performance with atrial-flutter records excluded.

Outputs
  outputs/tables/audit_duplicates.csv, outputs/tables/external_by_rhythm.csv,
  outputs/tables/external_label_sensitivity.csv, docs/audit_report.md
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrobust.data import load_dataset  # noqa: E402
from ecgrobust.metrics import auroc, binary_metrics  # noqa: E402
from ecgrobust.pipeline import binary_ids, feature_path  # noqa: E402
from ecgrobust.splits import load_split  # noqa: E402
from ecgrobust.utils import Paths, load_config, load_pickle, setup_logging  # noqa: E402

MODELS, VARIANTS = ["rule", "logreg", "lgbm", "cnn"], ["standard", "augmented"]
RHYTHM_CODES = ["SR", "AFIB", "AFLT", "STACH", "SBRAD", "SARRH", "SVARR", "SVTAC", "PSVT", "BIGU", "TRIGU", "PACE"]
ECTOPY_CODES = ["PVC", "PAC"]


def sig_hash(x: np.ndarray, n: int | None = None) -> str:
    x = np.round(np.asarray(x[:n] if n else x, float), 4)
    return hashlib.sha1(x.tobytes()).hexdigest()


def duplicates(internal: dict, split: dict) -> tuple[pd.DataFrame, dict]:
    test = set(split["test"])
    rows = []
    for rid, x in internal["signals"].items():
        rows.append({"record_id": rid, "full": sig_hash(x), "first3s": sig_hash(x, 900), "n": len(x),
                     "set": "test" if rid in test else "train"})
    d = pd.DataFrame(rows)
    out = {}
    for key in ["full", "first3s"]:
        g = d.groupby(key)
        dup = g.filter(lambda s: len(s) > 1)
        cross = g.filter(lambda s: s["set"].nunique() > 1)
        out[key] = {"groups": int(dup[key].nunique()), "records": int(len(dup)),
                    "cross_split_groups": int(cross[key].nunique()), "cross_split_records": int(len(cross))}
    dup_rows = d[d.duplicated("first3s", keep=False)].sort_values("first3s")
    return dup_rows, out


def bundle_leaks(cfg: dict, paths: Paths, split: dict) -> list[str]:
    test, notes = set(split["test"]), []
    for f in sorted(paths.models.rglob("*.pkl")):
        b = load_pickle(f)
        oof_ids = {i.split("#aug")[0] for i in map(str, b["oof"].index)}
        bad = oof_ids & test
        notes.append(f"{f.relative_to(paths.models)}: {len(b['oof'])} calibration rows, test ids among them: {len(bad)}")
        if bad:
            raise SystemExit(f"LEAK: {f} calibrated on {len(bad)} test records")
    aug = pd.read_parquet(feature_path(paths, "internal__train_aug"))
    n_bad = int(aug.group.isin(test).sum())
    notes.append(f"feature-model augmentation copies: {len(aug)}, made from test records: {n_bad}")
    if n_bad:
        raise SystemExit("LEAK: augmentation copies made from test records")
    return notes


def noise_separation(cfg: dict) -> str:
    nst = load_dataset(cfg, "nstdb")
    f = cfg["noise"]["train_fraction"]
    parts = []
    for k, arr in nst.items():
        cut = int(arr.shape[1] * f)
        parts.append(f"{k}: samples 0-{cut - 1} for training augmentation, {cut}-{arr.shape[1] - 1} for test corruption")
    return "; ".join(parts)


def predictions(paths: Paths, variant: str, model: str, condition: str) -> pd.DataFrame:
    if condition == "external__clean":
        return pd.read_parquet(paths.predictions / f"binary__{variant}__{model}.parquet",
                               filters=[("condition", "==", condition)])
    return pd.read_parquet(paths.predictions / f"external_noise__{variant}__{model}.parquet",
                           filters=[("condition", "==", condition)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    paths = Paths.from_cfg(cfg)
    split = load_split(cfg)
    internal = load_dataset(cfg, "internal")
    lines = ["# Data, label and leakage audit", "",
             "Produced by `scripts/audit_data.py`; nothing in this audit changes a model or a result.", ""]

    # 1. duplicates
    dup_rows, dup = duplicates(internal, split)
    dup_rows.to_csv(paths.tables / "audit_duplicates.csv", index=False)
    lines += ["## 1. Duplicate recordings in CinC 2017", "",
              f"- Identical signals: {dup['full']['groups']} groups ({dup['full']['records']} recordings); "
              f"groups crossing the train/test split: {dup['full']['cross_split_groups']}.",
              f"- Identical first 3 s (near-duplicates): {dup['first3s']['groups']} groups ({dup['first3s']['records']} "
              f"recordings); crossing the split: {dup['first3s']['cross_split_groups']} groups "
              f"({dup['first3s']['cross_split_records']} recordings).",
              "- CinC 2017 (AliveCor, 300 Hz) and PTB-XL (clinical 12-lead, 500 Hz) come from different devices and "
              "institutions, so no recording can appear in both.", ""]

    # 2. bundles
    lines += ["## 2. Leakage paths in the stored models", ""]
    lines += [f"- {n}" for n in bundle_leaks(cfg, paths, split)]
    lines += ["", "Other checks, by construction in the code:",
              "- The test split is stored with a SHA-256 hash and re-verified on every load; training code raises "
              "an error if a test identifier reaches it.",
              "- Logistic regression imputes and scales inside the cross-validated pipeline (training folds only); "
              "LightGBM uses no population statistics; the rule's missing-value fill is the training median.",
              "- The CNN z-scores each 9 s window on its own (no population statistics) and draws its inner "
              "validation split by recording, so windows of one recording never sit on both sides.",
              "- Thresholds and Platt calibrators are fitted on the calibration rows listed above, all from "
              "training records.", ""]

    # 3. noise
    lines += ["## 3. Noise separation", "", f"- {noise_separation(cfg)}.", ""]

    # 4. PTB-XL composition and false alarms by rhythm
    db = pd.read_csv(Path(cfg["data"]["ptbxl"]["dir"]) / "ptbxl_database.csv", index_col="ecg_id")
    db["scp"] = db.scp_codes.apply(ast.literal_eval)
    db.index = [f"ptbxl_{i:05d}" for i in db.index]
    ext = load_dataset(cfg, "external")["meta"]
    db = db.loc[ext.index]
    codes = {c: db.scp.apply(lambda d, c=c: c in d) for c in RHYTHM_CODES + ECTOPY_CODES}
    lik = db.scp.apply(lambda d: d.get("AFIB"))
    lines += ["## 4. PTB-XL labels", "",
              f"- AF (statement AFIB): {int(codes['AFIB'].sum())} of {len(db)} records ({codes['AFIB'].mean():.1%}). "
              f"AFIB likelihood values: {lik.dropna().value_counts().to_dict()} (PTB-XL gives rhythm statements "
              "likelihood 0 by convention, so every AFIB statement was counted).",
              f"- AFIB and AFLT both coded: {int((codes['AFIB'] & codes['AFLT']).sum())} records (labelled AF). "
              f"AFLT without AFIB: {int((codes['AFLT'] & ~codes['AFIB']).sum())} records (labelled non-AF, as "
              "atrial flutter is 'other rhythm' in CinC 2017).", ""]
    rows = []
    for cond in ["external__clean", "external__em__-6dB__r0"]:
        for variant in VARIANTS:
            for model in MODELS:
                try:
                    p = predictions(paths, variant, model, cond)
                except Exception:
                    continue
                p["pos"] = (p.prob >= p.threshold).astype(float)
                called = p.groupby("record_id").pos.mean()   # share of seeds calling AF
                for cat in ["AFIB"] + [c for c in RHYTHM_CODES + ECTOPY_CODES if c != "AFIB"]:
                    m = codes[cat] if cat == "AFIB" else (codes[cat] & ~codes["AFIB"])
                    ids = m[m].index
                    if len(ids) == 0:
                        continue
                    rows.append({"condition": cond, "variant": variant, "model": model, "category": cat,
                                 "n": len(ids), "share_called_af": float(called.loc[ids].mean())})
    byr = pd.DataFrame(rows)
    byr.to_csv(paths.tables / "external_by_rhythm.csv", index=False)
    piv = byr[(byr.variant == "standard")].pivot_table(index=["category", "n"], columns=["condition", "model"],
                                                        values="share_called_af").round(3)
    lines += ["Share of recordings called AF (mean over five seeds), standard training. Categories other than "
              "AFIB exclude records that also carry AFIB:", "", piv.to_markdown(), ""]

    # 5. label sensitivity: drop AFLT records
    rows = []
    keep = ~codes["AFLT"]
    for variant in VARIANTS:
        for model in MODELS:
            for cond in ["external__clean", "external__em__-6dB__r0"]:
                try:
                    p = predictions(paths, variant, model, cond)
                except Exception:
                    continue
                for label, ids in [("all records", ext.index), ("atrial flutter excluded", keep[keep].index)]:
                    q = p[p.record_id.isin(ids)]
                    per = [binary_metrics(g.y.to_numpy(), g.prob.to_numpy(), g.threshold.iloc[0]) for _, g in q.groupby("seed")]
                    rows.append({"variant": variant, "model": model, "condition": cond, "set": label, "n": int(q.record_id.nunique()),
                                 **{k: float(np.mean([d[k] for d in per])) for k in ["auroc", "sensitivity", "specificity", "ppv"]}})
    sens = pd.DataFrame(rows)
    sens.to_csv(paths.tables / "external_label_sensitivity.csv", index=False)
    lines += ["## 5. Label sensitivity: atrial flutter excluded", "",
              sens[sens.variant == "standard"].round(3).to_markdown(index=False), ""]
    out = ROOT / "docs" / "audit_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

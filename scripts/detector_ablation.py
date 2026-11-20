"""Is the false-alarm failure of the beat-based models specific to the NeuroKit2 R-peak detector?

    python scripts/detector_ablation.py --config configs/local.yaml

For each alternative detector (Hamilton 2002, Elgendi et al 2010, Pan and Tompkins 1985, all as implemented in
NeuroKit2) the whole feature pipeline is rerun with that detector: features of the clean CinC training
recordings, and of the internal test set clean and under EM 0 dB, EM -6 dB and MA -6 dB (realization 0, the same
noise as the main study). The rule, logistic regression and gradient boosting are then refitted on the
detector's own training features (standard training, five seeds, with the hyper-parameters selected for the
main study's model of the same seed) and evaluated on its test features. NeuroKit2 rows are the main study's
models, for reference.

Outputs: outputs/revision/features/detector__<detector>__<key>.parquet,
outputs/revision/predictions/detector_ablation.parquet, outputs/tables/detector_ablation.csv
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
sys.path.insert(0, str(ROOT / "scripts"))

from ecgrobust import models as M  # noqa: E402
from ecgrobust.data import load_dataset  # noqa: E402
from ecgrobust.features import extract_many  # noqa: E402
from ecgrobust.metrics import binary_metrics  # noqa: E402
from ecgrobust.noise import NoiseBank, corrupt_for_test  # noqa: E402
from ecgrobust.pipeline import _cond_from_key, binary_ids, feature_path  # noqa: E402
from ecgrobust.splits import assert_no_test_leak, load_split  # noqa: E402
from ecgrobust.utils import Paths, load_config, load_pickle, set_global_seed, setup_logging  # noqa: E402
from augmentation_ablation import fit_fixed  # noqa: E402

LOG = logging.getLogger("detector_ablation")
DETECTORS = ["hamilton2002", "elgendi2010", "pantompkins1985"]
TEST_KEYS = ["internal__clean", "internal__em__+0dB__r0", "internal__em__-6dB__r0", "internal__ma__-6dB__r0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    paths = Paths.from_cfg(cfg)
    rev = paths.root / "revision"
    for d in ["features", "predictions"]:
        (rev / d).mkdir(parents=True, exist_ok=True)
    internal = load_dataset(cfg, "internal")
    meta, split = internal["meta"], load_split(cfg)
    train, test = binary_ids(meta, split["train"]), binary_ids(meta, split["test"])
    assert_no_test_leak(cfg, train)
    bank = NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])
    ref = load_pickle(paths.features / "internal__refpeaks.pkl")

    def feats(det: str, key: str, ids: list[str]) -> pd.DataFrame:
        f = rev / "features" / f"detector__{det}__{key}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        cond = _cond_from_key(key.replace("_train", ""))
        sigs = {r: corrupt_for_test(internal["signals"][r], r, cond, bank, cfg["noise"]["seed"]) for r in ids}
        df = extract_many(sigs, cfg["fs"], cfg["features"], ref_peaks=None if cond.is_clean else {r: ref[r] for r in ids},
                          desc=f"{det} {key}", detector=det)
        df.to_parquet(f)
        return df

    rows = []
    y_tr = meta.loc[train, "af"].to_numpy(int)
    for det in DETECTORS:
        Xtr = feats(det, "internal__clean_train", train).loc[train]
        tests = {k: feats(det, k, test).loc[test] for k in TEST_KEYS}
        for model in ["rule", "logreg", "lgbm"]:
            for seed in cfg["seeds"]:
                set_global_seed(seed)
                if model == "rule":
                    est = M.RuleModel(cfg["models"]["rule"]["feature"]).fit(Xtr, y_tr)
                    b = M._finish({"estimator": est, "best_params": {}, "cv_score": np.nan}, est.score_raw(Xtr), y_tr,
                                  Xtr.index, "score", 2, None)
                else:
                    params = load_pickle(paths.models / "binary" / "standard" / f"{model}_seed{seed}.pkl")["best_params"]
                    b = fit_fixed(model, Xtr, y_tr, np.asarray(train), None, params, cfg, seed)
                b.update(model=model, n_classes=2)
                for key, X in tests.items():
                    raw = M.raw_scores(b, X=X)
                    rows.append(pd.DataFrame({"record_id": test, "y": meta.loc[test, "af"].to_numpy(int),
                                              "detector": det, "model": model, "seed": seed, "condition": key,
                                              "prob": b["calibrator"].transform(raw), "threshold": b["threshold"],
                                              "beat_fp": X.get("beat_fp", pd.Series(np.nan, index=X.index)).to_numpy(float),
                                              "beat_fn": X.get("beat_fn", pd.Series(np.nan, index=X.index)).to_numpy(float)}))
            LOG.info("detector %s: %s done", det, model)
    # NeuroKit2 reference rows from the main study's predictions
    for model in ["rule", "logreg", "lgbm"]:
        p = pd.read_parquet(paths.predictions / f"binary__standard__{model}.parquet",
                            columns=["record_id", "y", "seed", "condition", "prob", "threshold"],
                            filters=[("condition", "in", TEST_KEYS)])
        p["detector"], p["model"] = "neurokit", model
        rows.append(p)
    pred = pd.concat(rows, ignore_index=True)
    pred.to_parquet(rev / "predictions" / "detector_ablation.parquet")
    out = []
    for (det, model, key), d in pred.groupby(["detector", "model", "condition"]):
        per = [binary_metrics(g.y.to_numpy(), g.prob.to_numpy(), g.threshold.iloc[0]) for _, g in d.groupby("seed")]
        out.append({"detector": det, "model": model, "condition": key,
                    **{k: float(np.mean([m[k] for m in per])) for k in ["auroc", "sensitivity", "specificity", "ppv", "f1"]}})
    pd.DataFrame(out).to_csv(paths.tables / "detector_ablation.csv", index=False)
    LOG.info("wrote detector_ablation.csv")


if __name__ == "__main__":
    main()

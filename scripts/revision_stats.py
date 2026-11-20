"""Tables for the revision analyses, built from stored predictions (nothing is re-run here).

    python scripts/revision_stats.py --config configs/local.yaml

Uncertainty
-----------
At a fixed threshold, the mean over seeds (and noise realizations) of sensitivity equals the mean over AF records
of each record's *call rate*, the share of its seed x realization predictions that call AF; the same holds for
specificity, and PPV is a ratio of such sums. Confidence intervals therefore come from resampling independent
units with all their seeds, realizations and paired model predictions kept together: recordings for CinC 2017
(which has no patient identifiers) and patients for PTB-XL and the MIT-BIH databases. This keeps repeated
measurements of the same recording (several noise levels, realizations and models) inside one resampling unit.

Sections (each runs only when its inputs exist)
  variance        noise-realization vs training-seed vs test-sample variability (realizations 0-4)
  primary         six designated primary comparisons, Holm-adjusted (see PRIMARY); internal ones on fresh draws 1-4
  oracle          feature models with clean-signal R peaks vs their normal pipeline
  detection       false AF calls vs spurious / missed beats, and feature drift under electrode motion
  cnn             CNN probability shift, threshold diagnostics and noise-specific thresholds
  regional        noise confined to QRS complexes or to the rest of the waveform
  altnoise        alternative noise generators (QRS-amplitude scaling, bursts)
  operating       confusion matrices, NPV, a sensitivity-target threshold, and prevalence-adjusted PPV/NPV
  external        external full-grid summary
  augablation     augmentation ablation summary
  detector        alternative R-peak detectors
  rpeak           R-peak detection against expert annotations
  afdb            naturally noisy Holter recordings
All tables are written to outputs/tables/rev_<section>*.csv, and a readable digest to docs/revision_results.md.
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
from ecgrobust.metrics import auroc, holm  # noqa: E402
from ecgrobust.utils import Paths, load_config, load_pickle, setup_logging  # noqa: E402

LOG = logging.getLogger("revision_stats")
B, BOOT_SEED = 1000, 99
NAMES = {"rule": "Rule", "logreg": "Logistic regression", "lgbm": "Gradient boosting", "cnn": "1-D CNN"}
DIGEST: list[str] = []


# =========================================================================== helpers
class Data:
    def __init__(self, cfg: dict):
        self.cfg, self.paths = cfg, Paths.from_cfg(cfg)
        self.rev = self.paths.root / "revision"
        self.tables = self.paths.tables
        self._ext_meta = None

    @property
    def ext_meta(self):
        if self._ext_meta is None:
            self._ext_meta = load_dataset(self.cfg, "external")["meta"]
        return self._ext_meta

    def main(self, variant: str, model: str, conds: list[str] | None = None) -> pd.DataFrame:
        cols = ["record_id", "y", "seed", "condition", "prob", "threshold"]
        f = self.paths.predictions / f"binary__{variant}__{model}.parquet"
        p = pd.read_parquet(f, columns=cols, filters=[("condition", "in", conds)] if conds else None)
        ext = [c for c in (conds or []) if c.startswith("external__") and c != "external__clean"]
        if ext or conds is None:
            g = self.paths.predictions / f"external_noise__{variant}__{model}.parquet"
            if g.exists():
                q = pd.read_parquet(g, columns=cols, filters=[("condition", "in", ext)] if conds else None)
                p = pd.concat([p, q], ignore_index=True)
        p["realization"] = 0
        return p

    def rev_preds(self, name: str, variant: str, model: str) -> pd.DataFrame | None:
        f = self.rev / "predictions" / f"{name}__{variant}__{model}.parquet"
        return pd.read_parquet(f) if f.exists() else None


def call_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per record: y and the share of (seed, realization) predictions that call AF."""
    d = df.assign(pos=(df.prob >= df.threshold).astype(float))
    return d.groupby("record_id").agg(y=("y", "first"), p=("pos", "mean"))


def cluster_sums(cr: pd.DataFrame, groups: pd.Series | None) -> np.ndarray:
    """(n_units, 4) sums of p*y, y, p*(1-y), (1-y) per resampling unit."""
    a = np.c_[cr.p * cr.y, cr.y, cr.p * (1 - cr.y), 1 - cr.y]
    if groups is None:
        return a
    g = groups.reindex(cr.index).to_numpy()
    return pd.DataFrame(a).groupby(g).sum().to_numpy()


def boot_idx(n: int, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (B, n))


def ss_from(s: np.ndarray) -> dict:
    """Sensitivity, specificity and PPV from summed columns (works on (4,) or (B, 4))."""
    s = np.asarray(s, float)
    tp, pos, fp, neg = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sensitivity": tp / pos, "specificity": 1 - fp / neg, "ppv": tp / (tp + fp)}


def summary_ci(cr: pd.DataFrame, groups=None) -> dict:
    s = cluster_sums(cr, groups)
    point = ss_from(s.sum(0))
    reps = ss_from(s[boot_idx(len(s))].sum(1))
    out = {}
    for k in point:
        out[k], out[f"{k}_lo"], out[f"{k}_hi"] = float(point[k]), *np.nanpercentile(reps[k], [2.5, 97.5])
    return out


def paired_ci(cr_a: pd.DataFrame, cr_b: pd.DataFrame, metric: str, groups=None) -> dict:
    """metric(A) - metric(B) on the same records, paired bootstrap over units."""
    ids = cr_a.index.intersection(cr_b.index)
    sa, sb = cluster_sums(cr_a.loc[ids], groups), cluster_sums(cr_b.loc[ids], groups)
    d0 = ss_from(sa.sum(0))[metric] - ss_from(sb.sum(0))[metric]
    ix = boot_idx(len(sa))
    d = ss_from(sa[ix].sum(1))[metric] - ss_from(sb[ix].sum(1))[metric]
    d = d[np.isfinite(d)]
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return {"diff": float(d0), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p": float(max(p, 1 / len(d)))}


def mean_auroc(df: pd.DataFrame) -> float:
    return float(np.mean([auroc(g.y.to_numpy(), g.prob.to_numpy()) for _, g in df.groupby(["seed", "realization"])]))


def fmt(v, lo=None, hi=None, nd=3):
    if lo is None:
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{lo:.{nd}f}–{hi:.{nd}f}]"


def write(df: pd.DataFrame, D: Data, name: str, title: str, cols: list[str] | None = None) -> None:
    df.to_csv(D.tables / f"rev_{name}.csv", index=False)
    show = df if cols is None else df[cols]
    DIGEST.extend([f"## {title}", "", f"`outputs/tables/rev_{name}.csv`", "", show.to_markdown(index=False, floatfmt=".3f"), ""])
    LOG.info("wrote rev_%s.csv", name)


def realization_frame(D: Data, variant: str, model: str) -> pd.DataFrame | None:
    """Internal predictions with realizations 0-4 where available."""
    r = D.rev_preds("realizations", variant, model)
    if r is None:
        return None
    conds0 = sorted({f"internal__{n}__{s:+g}dB__r0" for n, s in r[["noise", "snr_db"]].drop_duplicates().itertuples(index=False)})
    m = D.main(variant, model, conds0)
    keys = r[["noise", "snr_db"]].drop_duplicates()
    m["noise"] = m.condition.str.split("__").str[1]
    m["snr_db"] = m.condition.str.split("__").str[2].str.replace("dB", "").astype(float)
    cols = ["record_id", "y", "seed", "prob", "threshold", "noise", "snr_db", "realization"]
    out = pd.concat([m[cols], r[cols]], ignore_index=True)
    return out.merge(keys, on=["noise", "snr_db"])


# =========================================================================== sections
def variance(D: Data) -> None:
    rows = []
    for variant in ["standard", "augmented"]:
        for model in NAMES:
            df = realization_frame(D, variant, model)
            if df is None:
                return
            for (n, s), g in df.groupby(["noise", "snr_db"]):
                nreal = g.realization.nunique()
                per = []
                for (seed, r), h in g.groupby(["seed", "realization"]):
                    cr = call_rates(h)
                    v = ss_from(cluster_sums(cr, None).sum(0))
                    per.append({"seed": seed, "realization": r, "auroc": auroc(h.y.to_numpy(), h.prob.to_numpy()), **v})
                per = pd.DataFrame(per)
                ci = summary_ci(call_rates(g))
                row = {"variant": variant, "model": model, "noise": n, "snr_db": s, "n_realizations": nreal}
                for k in ["sensitivity", "specificity", "auroc"]:
                    row[k] = per[k].mean()
                    row[f"{k}_sd_realization"] = per.groupby("seed")[k].std(ddof=1).mean()   # noise draw, same model
                    row[f"{k}_sd_seed"] = per.groupby("realization")[k].std(ddof=1).mean()   # model seed, same noise
                    if k != "auroc":
                        row[f"{k}_lo"], row[f"{k}_hi"] = ci[f"{k}_lo"], ci[f"{k}_hi"]
                        row[f"{k}_se_sample"] = (ci[f"{k}_hi"] - ci[f"{k}_lo"]) / 3.92
                rows.append(row)
    t = pd.DataFrame(rows)
    write(t, D, "variance", "Variability: noise realization vs training seed vs test sample (pooled over realizations)",
          ["variant", "model", "noise", "snr_db", "n_realizations", "sensitivity", "sensitivity_sd_realization",
           "sensitivity_sd_seed", "sensitivity_se_sample", "specificity", "specificity_sd_realization",
           "specificity_sd_seed", "specificity_se_sample"])


PRIMARY = [
    # (label, dataset, condition, metric, (variant, model) A, (variant, model) B)
    ("H1 internal EM -6 dB: sensitivity, gradient boosting - CNN", "internal", ("em", -6.0), "sensitivity",
     ("standard", "lgbm"), ("standard", "cnn")),
    ("H2 internal EM -6 dB: specificity, CNN - gradient boosting", "internal", ("em", -6.0), "specificity",
     ("standard", "cnn"), ("standard", "lgbm")),
    ("H3 external EM -6 dB: sensitivity, gradient boosting - CNN", "external", ("em", -6.0), "sensitivity",
     ("standard", "lgbm"), ("standard", "cnn")),
    ("H4 external EM -6 dB: specificity, CNN - gradient boosting", "external", ("em", -6.0), "specificity",
     ("standard", "cnn"), ("standard", "lgbm")),
    ("H5 internal MA -6 dB: CNN sensitivity, augmented - standard", "internal", ("ma", -6.0), "sensitivity",
     ("augmented", "cnn"), ("standard", "cnn")),
    ("H6 internal EM -6 dB: gradient-boosting specificity, augmented - standard", "internal", ("em", -6.0),
     "specificity", ("augmented", "lgbm"), ("standard", "lgbm")),
]


def primary(D: Data) -> None:
    rows = []
    for label, ds, (n, s), metric, A, Bm in PRIMARY:
        if ds == "internal":
            fa, fb = realization_frame(D, *A), realization_frame(D, *Bm)
            if fa is None or fb is None:
                return
            # realization 0 is the draw on which the hypotheses were formed; test on the four fresh draws only
            fa = fa[(fa.noise == n) & (fa.snr_db == s) & (fa.realization >= 1)]
            fb = fb[(fb.noise == n) & (fb.snr_db == s) & (fb.realization >= 1)]
            groups, unit = None, "recordings (fresh noise realizations 1-4)"
        else:
            key = f"external__{n}__{s:+g}dB__r0"
            fa, fb = D.main(*A, [key]), D.main(*Bm, [key])
            groups, unit = D.ext_meta.patient_id, "patients (realization 0)"
        ca, cb = call_rates(fa), call_rates(fb)
        r = paired_ci(ca, cb, metric, groups)
        rows.append({"hypothesis": label, "metric": metric, "A": ss_from(cluster_sums(ca, None).sum(0))[metric],
                     "B": ss_from(cluster_sums(cb, None).sum(0))[metric], **r, "resampling_unit": unit})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.tolist())
    write(t, D, "primary", "Primary comparisons (Holm-adjusted across the six)")


def oracle(D: Data) -> None:
    rows = []
    for variant in ["standard", "augmented"]:
        for model in ["rule", "logreg", "lgbm"]:
            o = D.rev_preds("oracle", variant, model)
            if o is None:
                return
            clean = call_rates(D.main(variant, model, ["internal__clean"]))
            for (n, s), g in o.groupby(["noise", "snr_db"]):
                key = f"internal__{n}__{s:+g}dB__r0"
                norm = D.main(variant, model, [key])
                cn, co = call_rates(norm), call_rates(g)
                row = {"variant": variant, "model": model, "noise": n, "snr_db": s}
                for lab, cr in [("clean", clean), ("detected_peaks", cn), ("clean_peaks", co)]:
                    v = ss_from(cluster_sums(cr, None).sum(0))
                    row[f"spec_{lab}"], row[f"sens_{lab}"] = v["specificity"], v["sensitivity"]
                row["auroc_detected_peaks"], row["auroc_clean_peaks"] = mean_auroc(norm), mean_auroc(g.assign(realization=0))
                for m in ["specificity", "sensitivity"]:
                    d = paired_ci(co, cn, m)
                    row[f"{m[:4]}_gain"], row[f"{m[:4]}_gain_lo"], row[f"{m[:4]}_gain_hi"] = d["diff"], d["lo"], d["hi"]
                    lost = row[f"{m[:4]}_clean"] - row[f"{m[:4]}_detected_peaks"]
                    row[f"{m[:4]}_share_restored"] = d["diff"] / lost if abs(lost) > 0.005 else np.nan
                rows.append(row)
    write(pd.DataFrame(rows), D, "oracle", "Feature models with clean-signal R peaks (oracle) vs detected R peaks")


def detection(D: Data) -> None:
    feats = {}
    for f in sorted(D.paths.features.glob("internal__em__*__r0.parquet")):
        feats[f.stem] = pd.read_parquet(f)
    clean = pd.read_parquet(D.paths.features / "internal__clean.parquet")
    test = D.main("standard", "lgbm", ["internal__clean"]).drop_duplicates("record_id").set_index("record_id")
    ids, y = test.index, test.y
    dur = load_dataset(D.cfg, "internal")["meta"].loc[ids, "duration_s"]
    # feature drift
    rows = []
    keys = ["internal__clean"] + sorted(feats, key=lambda k: -float(k.split("__")[2].replace("dB", "")))
    for key in keys:
        F = (clean if key == "internal__clean" else feats[key]).loc[ids]
        for cls, m in [("non-AF", y == 0), ("AF", y == 1)]:
            G = F[m.to_numpy()]
            rows.append({"condition": key, "class": cls, "beats_per_min": float((G.rr_n_beats / dur[m] * 60).median()),
                         **{c: float(G[c].median()) for c in ["rr_rmssd_norm", "rr_pnn50", "rr_cosen", "rr_sampen",
                                                             "p_present_frac", "p_region_corr", "qrs_corr_min_frac"]}})
    write(pd.DataFrame(rows), D, "feature_drift", "Median feature values by class under electrode motion (realization 0)")
    # false calls vs detection errors (pooled over EM SNRs and realizations 0-4 where available)
    out = []
    for model in ["lgbm", "logreg", "rule", "cnn"]:
        df = realization_frame(D, "standard", model)
        if df is None:
            df = D.main("standard", model, [k for k in feats])
            df["noise"], df["snr_db"] = "em", df.condition.str.split("__").str[2].str.replace("dB", "").astype(float)
        df = df[df.noise == "em"]
        r = D.rev_preds("realizations", "standard", "rule")
        det = []
        for key, F in feats.items():
            s = float(key.split("__")[2].replace("dB", ""))
            det.append(pd.DataFrame({"record_id": F.index, "snr_db": s, "realization": 0, "beat_fp": F.beat_fp.to_numpy(),
                                     "beat_fn": F.beat_fn.to_numpy()}))
        if r is not None:
            rr = r[r.noise == "em"].drop_duplicates(["record_id", "snr_db", "realization"])
            det.append(rr[["record_id", "snr_db", "realization", "beat_fp", "beat_fn"]])
        det = pd.concat(det, ignore_index=True)
        d = df.assign(pos=(df.prob >= df.threshold).astype(float)).groupby(["record_id", "snr_db", "realization"]).agg(
            y=("y", "first"), pos=("pos", "mean")).reset_index().merge(det, on=["record_id", "snr_db", "realization"])
        d = d.merge(dur.rename("dur").reset_index(), on="record_id")
        d["spurious_per_min"], d["missed_per_min"] = d.beat_fp / d.dur * 60, d.beat_fn / d.dur * 60
        for cls, col, bins, labels in [(0, "spurious_per_min", [-0.1, 0.5, 2, 5, 10, np.inf], ["0", "0.5-2", "2-5", "5-10", ">10"]),
                                       (1, "missed_per_min", [-0.1, 0.5, 2, 5, 10, np.inf], ["0", "0.5-2", "2-5", "5-10", ">10"])]:
            g = d[d.y == cls].copy()
            g["bin"] = pd.cut(g[col], bins, labels=labels)
            for b, h in g.groupby("bin", observed=True):
                out.append({"model": model, "class": "non-AF" if cls == 0 else "AF", "detection_error": col, "bin": b,
                            "n_recording_conditions": len(h),
                            "af_call_rate": float(h.pos.mean())})
    write(pd.DataFrame(out), D, "detection_errors",
          "Share of recordings called AF by beat-detection error (electrode motion, all SNRs; non-AF: spurious beats, AF: missed beats)")


def cnn(D: Data) -> None:
    rows = []
    thr = D.rev / "cnn_noise_thresholds.csv"
    T = pd.read_csv(thr) if thr.exists() else None
    for variant in ["standard", "augmented"]:
        base = D.main(variant, "cnn", None)
        base = base[base.condition.str.startswith("internal")]
        clean = base[base.condition == "internal__clean"]
        for cond, g in base.groupby("condition"):
            if cond != "internal__clean" and not any(t in cond for t in ["__em__", "__ma__"]):
                continue
            row = {"variant": variant, "condition": cond, "mean_prob_af": g[g.y == 1].prob.mean(),
                   "mean_prob_non_af": g[g.y == 0].prob.mean(), "auroc": mean_auroc(g)}
            v = ss_from(cluster_sums(call_rates(g), None).sum(0))
            row.update(sens_fixed=v["sensitivity"], spec_fixed=v["specificity"])
            # diagnostic only (uses test labels): sensitivity at the threshold giving the clean specificity
            se = []
            for seed, h in g.groupby("seed"):
                c = clean[clean.seed == seed]
                spec_c = ((c.prob < c.threshold) & (c.y == 0)).sum() / (c.y == 0).sum()
                t_ = np.quantile(h[h.y == 0].prob, spec_c)
                se.append(((h.prob > t_) & (h.y == 1)).sum() / (h.y == 1).sum())
            row["sens_at_clean_specificity_diagnostic"] = float(np.mean(se))
            if T is not None and cond != "internal__clean":
                n, s = cond.split("__")[1], float(cond.split("__")[2].replace("dB", ""))
                tt = T[(T.variant == variant) & (T.noise == n) & (T.snr_db == s)].set_index("seed")
                if len(tt):
                    h = g.assign(threshold=g.seed.map(tt.threshold_noise))
                    v2 = ss_from(cluster_sums(call_rates(h), None).sum(0))
                    row.update(sens_noise_threshold=v2["sensitivity"], spec_noise_threshold=v2["specificity"],
                               threshold_default=float(tt.threshold_default.mean()), threshold_noise=float(tt.threshold_noise.mean()))
            rows.append(row)
    write(pd.DataFrame(rows), D, "cnn", "CNN under noise: probability shift, threshold diagnostics, noise-specific threshold")


def regional(D: Data) -> None:
    rows = []
    for variant in ["standard", "augmented"]:
        for model in NAMES:
            g = D.rev_preds("regional", variant, model)
            if g is None:
                return
            for (s, strat), h in g.groupby(["snr_db", "strategy"]):
                full = D.main(variant, model, [f"internal__em__{s:+g}dB__r0"])
                for lab, df in [("full", full), (strat, h)]:
                    v = summary_ci(call_rates(df))
                    rows.append({"variant": variant, "model": model, "snr_db": s, "region": lab,
                                 "auroc": mean_auroc(df.assign(realization=0)), **v})
    t = pd.DataFrame(rows).drop_duplicates(["variant", "model", "snr_db", "region"])
    write(t, D, "regional", "Electrode motion confined to QRS complexes or to the rest of the waveform",
          ["variant", "model", "snr_db", "region", "auroc", "sensitivity", "specificity"])


def altnoise(D: Data) -> None:
    rows = []
    for variant in ["standard", "augmented"]:
        for model in NAMES:
            g = D.rev_preds("altnoise", variant, model)
            if g is None:
                return
            for (n, s), h in g.groupby(["noise", "snr_db"]):
                base = D.main(variant, model, [f"internal__{n}__{s:+g}dB__r0"])
                for strat, df in [("power-based (main)", base)] + [(k, x) for k, x in h.groupby("strategy")]:
                    v = summary_ci(call_rates(df))
                    rows.append({"variant": variant, "model": model, "noise": n, "snr_db": s, "strategy": strat,
                                 "auroc": mean_auroc(df.assign(realization=0)), **v})
    write(pd.DataFrame(rows), D, "altnoise", "Alternative noise generators",
          ["variant", "model", "noise", "snr_db", "strategy", "auroc", "sensitivity", "specificity"])


def operating(D: Data) -> None:
    conds = ["internal__clean", "internal__em__-6dB__r0", "internal__ma__-6dB__r0",
             "external__clean", "external__em__-6dB__r0", "external__ma__-6dB__r0"]
    rows, prev_rows, st_rows = [], [], []
    for variant in ["standard", "augmented"]:
        for model in NAMES:
            df = D.main(variant, model, conds)
            bundles = {b["seed"]: b for b in (load_pickle(D.paths.models / "binary" / variant / f"{model}_seed{s}.pkl")
                                               for s in D.cfg["seeds"])}
            # sensitivity-target threshold: highest threshold with >= 90% sensitivity on the calibration rows
            t90 = {}
            for s, b in bundles.items():
                o = b["oof"]
                o = o[~o.get("is_aug", pd.Series(False, index=o.index)).astype(bool)]
                pa = np.sort(o.prob[o.y == 1].to_numpy())
                t90[s] = float(pa[int(np.floor(0.10 * len(pa)))])
            for cond, g in df.groupby("condition"):
                ext = cond.startswith("external")
                groups = D.ext_meta.patient_id if ext else None
                cr = call_rates(g)
                v = summary_ci(cr, groups)
                d = g.assign(pos=g.prob >= g.threshold)
                cm = d.groupby("seed").apply(lambda h: pd.Series({
                    "TP": ((h.pos) & (h.y == 1)).sum(), "FN": ((~h.pos) & (h.y == 1)).sum(),
                    "FP": ((h.pos) & (h.y == 0)).sum(), "TN": ((~h.pos) & (h.y == 0)).sum()})).mean()
                npv = cm.TN / (cm.TN + cm.FN)
                rows.append({"variant": variant, "model": model, "condition": cond, **cm.round(1).to_dict(),
                             "npv": npv, **v})
                h = g.assign(threshold=g.seed.map(t90))
                v90 = ss_from(cluster_sums(call_rates(h), None).sum(0))
                st_rows.append({"variant": variant, "model": model, "condition": cond,
                                "threshold_sens90": float(np.mean(list(t90.values()))),
                                "sensitivity": v90["sensitivity"], "specificity": v90["specificity"], "ppv": v90["ppv"]})
                for prev in [0.005, 0.01, 0.02, 0.05, 0.10]:
                    se, sp = v["sensitivity"], v["specificity"]
                    ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
                    npv_ = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
                    prev_rows.append({"variant": variant, "model": model, "condition": cond, "prevalence": prev,
                                      "sensitivity": se, "specificity": sp, "ppv_hypothetical": ppv,
                                      "npv_hypothetical": npv_,
                                      "false_alarms_per_true_af": (1 - sp) * (1 - prev) / (se * prev) if se > 0 else np.nan})
    write(pd.DataFrame(rows), D, "operating", "Confusion matrices (mean counts over seeds), NPV and 95% CIs",
          ["variant", "model", "condition", "TP", "FN", "FP", "TN", "sensitivity", "specificity", "ppv", "npv"])
    write(pd.DataFrame(st_rows), D, "sens90", "Sensitivity-target threshold (>= 0.90 on training calibration data)")
    write(pd.DataFrame(prev_rows), D, "prevalence", "Hypothetical PPV and NPV at screening prevalences (from observed sensitivity and specificity)")


def external(D: Data) -> None:
    f = D.tables / "external_noise.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    write(t[["variant", "model", "condition", "auroc", "auroc_lo", "auroc_hi", "sensitivity", "specificity", "ppv", "npv", "f1"]],
          D, "external_grid", "External (PTB-XL) full noise grid")


def augablation(D: Data) -> None:
    files = sorted((D.rev / "predictions").glob("augablation__*.parquet"))
    if not files:
        return
    rows = []
    for f in files:
        p = pd.read_parquet(f)
        for (cfgname, model, cond), g in p.groupby(["config", "model", "condition"]):
            groups = D.ext_meta.patient_id if cond.startswith("external") else None
            v = summary_ci(call_rates(g.assign(realization=0)), groups)
            rows.append({"config": cfgname, "model": model, "condition": cond, "n_seeds": g.seed.nunique(),
                         "auroc": mean_auroc(g.assign(realization=0)), **v})
    # the main study's standard models as the no-augmentation reference
    for model in ["logreg", "lgbm", "cnn"]:
        p = D.main("standard", model, None)
        for cond, g in p.groupby("condition"):
            if not (cond.endswith("r0") or cond.endswith("clean")):
                continue
            groups = D.ext_meta.patient_id if cond.startswith("external") else None
            v = summary_ci(call_rates(g), groups)
            rows.append({"config": "none (standard)", "model": model, "condition": cond, "n_seeds": g.seed.nunique(),
                         "auroc": mean_auroc(g), **v})
    t = pd.DataFrame(rows)
    write(t, D, "augablation", "Augmentation ablation",
          ["config", "model", "condition", "n_seeds", "auroc", "sensitivity", "specificity"])


def detector(D: Data) -> None:
    f = D.tables / "detector_ablation.csv"
    if f.exists():
        write(pd.read_csv(f), D, "detector", "Alternative R-peak detectors (standard training, five seeds)")


def rpeak(D: Data) -> None:
    for name, title in [("rpeak_validation.csv", "R-peak detection vs expert annotations (MIT-BIH Arrhythmia Database)"),
                        ("rpeak_pseudo_reference.csv", "Pseudo-reference (clean-signal detections) vs expert reference")]:
        f = D.tables / name
        if f.exists():
            t = pd.read_csv(f)
            if "rhythm" in t:
                t = t[t.rhythm == "all"]
            write(t, D, name.replace(".csv", ""), title)


def afdb(D: Data) -> None:
    f = D.tables / "afdb_by_quality.csv"
    if f.exists():
        write(pd.read_csv(f), D, "afdb", "Naturally noisy Holter recordings (MIT-BIH AF Database)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    setup_logging()
    D = Data(load_config(a.config))
    sections = {"variance": variance, "primary": primary, "oracle": oracle, "detection": detection, "cnn": cnn,
                "regional": regional, "altnoise": altnoise, "operating": operating, "external": external,
                "augablation": augablation, "detector": detector, "rpeak": rpeak, "afdb": afdb}
    for name, fn in sections.items():
        if a.only and name not in a.only:
            continue
        try:
            fn(D)
        except FileNotFoundError as e:
            LOG.warning("%s skipped: %s", name, e)
    out = ROOT / "docs" / "revision_results.md"
    out.write_text("# Revision analyses: results digest\n\nGenerated by `scripts/revision_stats.py`.\n\n" + "\n".join(DIGEST),
                   encoding="utf-8")
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()

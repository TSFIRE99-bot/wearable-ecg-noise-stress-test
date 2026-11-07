"""Tables and figures for the paper, computed from the saved predictions.

Outputs (under ``<output_dir>/tables`` and ``/figures``):

    table1_cohorts.csv                       cohort sizes, AF prevalence, PTB-XL age and sex
T1  main_internal.csv / main_external.csv    clean performance, mean (SD over seeds) [95% bootstrap CI]
    failure_modes.csv                        sensitivity / specificity per condition (direction of errors)
    subgroups_external.csv                   PTB-XL performance by sex and age band
T2  fourclass_challenge.csv                  CinC 2017 challenge score (comparability with prior work)
T3  noise_degradation.csv                    every metric at every noise type x SNR, and paired drop vs clean
T4  paired_tests.csv                         model-vs-model paired bootstrap tests (Holm-adjusted)
T5  augmentation_effect.csv                  augmented minus standard training, paired
T6  beat_detection.csv                       NeuroKit2 R-peak sensitivity / PPV / F1 under noise
T7  performance_by_beat_quality.csv          AUROC per model, stratified by per-record beat-detection F1
T8  abstention_curve.csv / abstention_fixed_threshold.csv
T9  calibration.csv                          ECE, calibration slope and intercept per condition
T10 shap_importance.csv / shap_rank_stability.csv
metrics_all_conditions.csv                   long table with every number above
summary.md                                   short human-readable digest
"""
from __future__ import annotations

import logging
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from .features import BEAT, FEATURES  # noqa: E402
from .metrics import (BINARY_METRICS, binary_metrics, bootstrap_indices, calibration_summary,  # noqa: E402
                      challenge_score, holm, paired_difference, summarize_condition)
from .pipeline import _cond_from_key, _model_path, binary_ids, feature_path  # noqa: E402
from .data import load_dataset  # noqa: E402
from .splits import load_split  # noqa: E402
from .utils import Paths, load_pickle, save_json  # noqa: E402

LOG = logging.getLogger(__name__)

MODELS = ["rule", "logreg", "lgbm", "cnn"]
LABEL = {"rule": "Rule (RMSSD/RR)", "logreg": "Logistic regression", "lgbm": "Gradient boosting", "cnn": "1-D CNN"}
# reference categorical palette, slots 1-4 (validated for adjacent series), plus distinct markers
COLOR = {"rule": "#2a78d6", "logreg": "#eb6834", "lgbm": "#1baf7a", "cnn": "#eda100"}
MARKER = {"rule": "o", "logreg": "s", "lgbm": "^", "cnn": "D"}
NOISE_LABEL = {"bw": "Baseline wander", "ma": "Muscle artifact", "em": "Electrode motion"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KEY_METRICS = ["auroc", "auprc", "f1", "sensitivity", "specificity"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _fmt(s: dict, k: str, nd: int = 3) -> str:
    return f"{s[k]:.{nd}f} (±{s[k + '_sd']:.{nd}f}) [{s[k + '_lo']:.{nd}f}–{s[k + '_hi']:.{nd}f}]"


class Reporter:
    def __init__(self, cfg: dict):
        self.cfg, self.paths = cfg, Paths.from_cfg(cfg)
        self.split = load_split(cfg)
        self.ext_meta = load_dataset(cfg, "external")["meta"]
        files = sorted(self.paths.predictions.glob("binary__*.parquet"))
        if not files:
            raise FileNotFoundError("No predictions found; run `evaluate` first.")
        self.pred = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        self.variants = [v for v in ["standard", "augmented"] if v in set(self.pred.variant)]
        self.models = [m for m in MODELS if m in set(self.pred.model)]
        self.conds = list(dict.fromkeys(self.pred.condition))
        self.B = cfg["evaluation"]["bootstrap"]
        self._boots: dict = {}
        self.summ: dict = {}

    # ------------------------------------------------------------------ helpers
    def _matrix(self, variant, model, cond):
        d = self.pred[(self.pred.variant == variant) & (self.pred.model == model) & (self.pred.condition == cond)]
        P = d.pivot(index="seed", columns="record_id", values="prob")
        first = d[d.seed == P.index[0]].set_index("record_id").loc[P.columns]
        thr = d.groupby("seed").threshold.first().loc[P.index].to_numpy()
        return first, P.to_numpy(), thr

    def boots(self, dataset: str, first: pd.DataFrame):
        if dataset not in self._boots:
            groups = first.patient_id.to_numpy() if dataset == "external" else None
            self._boots[dataset] = bootstrap_indices(len(first), self.B, self.cfg["evaluation"]["bootstrap_seed"], groups)
        return self._boots[dataset]

    def summary(self, variant, model, cond):
        key = (variant, model, cond)
        if key not in self.summ:
            first, P, thr = self._matrix(variant, model, cond)
            self.summ[key] = summarize_condition(first.y.to_numpy(), P, thr, self.boots(first.dataset.iloc[0], first))
        return self.summ[key]

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        rows = []
        for v in self.variants:
            for m in self.models:
                for c in self.conds:
                    s = self.summary(v, m, c)
                    cond = _cond_from_key(c)
                    rows.append({"variant": v, "model": m, "condition": c, "dataset": cond.dataset, "noise": cond.noise,
                                 "snr_db": cond.snr_db, **{k: x for k, x in s.items() if not k.startswith("_")}})
                LOG.info("summarized %s/%s", v, m)
        self.long = pd.DataFrame(rows)
        self.long.to_csv(self.paths.tables / "metrics_all_conditions.csv", index=False)
        self.cohort()
        self.main_tables()
        self.failure_modes()
        self.subgroups()
        self.fourclass()
        self.noise_tables()
        self.paired_tests()
        self.augmentation()
        self.beat_detection()
        self.abstention()
        self.calibration()
        self.shap()
        self.figures()
        self.write_summary()

    # ------------------------------------------------------------------ Table 1 (cohorts)
    def cohort(self):
        im = load_dataset(self.cfg, "internal")["meta"]
        em = self.ext_meta

        def iqr(s):
            q = s.quantile([0.25, 0.5, 0.75])
            return f"{q[0.5]:.1f} ({q[0.25]:.1f}–{q[0.75]:.1f})"

        rows = []
        for name, ids in [("CinC 2017 development", self.split["train"]), ("CinC 2017 internal test", self.split["test"])]:
            m = im.loc[ids]
            b = m[m.label4 != "~"]
            rows.append({"set": name, "recordings": len(m), "patients": "not available",
                         "class_counts": ", ".join(f"{k} {v}" for k, v in m.label4.value_counts().items()),
                         "binary_task_n": len(b), "AF_n": int(b.af.sum()), "AF_pct": round(100 * b.af.mean(), 1),
                         "duration_s_median_IQR": iqr(m.duration_s), "age_median_IQR": "not available",
                         "female_pct": "not available"})
        row = {"set": "PTB-XL lead I (external)", "recordings": len(em), "patients": em.patient_id.nunique(),
               "class_counts": "", "binary_task_n": len(em), "AF_n": int(em.af.sum()),
               "AF_pct": round(100 * em.af.mean(), 1), "duration_s_median_IQR": iqr(em.duration_s)}
        if "age" in em:
            age = em.age.where(em.age < 300, 90)  # PTB-XL encodes ages > 89 as 300
            row["age_median_IQR"] = iqr(age.dropna())
            row["female_pct"] = round(100 * (em.sex == 1).mean(), 1)  # PTB-XL: 0 = male, 1 = female
        rows.append(row)
        pd.DataFrame(rows).to_csv(self.paths.tables / "table1_cohorts.csv", index=False)

    # ------------------------------------------------------------------ T1
    def main_tables(self):
        for ds in ["internal", "external"]:
            out = []
            for v in self.variants:
                for m in self.models:
                    s = self.summary(v, m, f"{ds}__clean")
                    out.append({"training": v, "model": LABEL[m], **{k: _fmt(s, k) for k in KEY_METRICS + ["brier"]}})
            pd.DataFrame(out).to_csv(self.paths.tables / f"main_{ds}.csv", index=False)

    # ------------------------------------------------------------------ failure modes
    def failure_modes(self):
        """Sensitivity / specificity at the training-set threshold under each internal condition:
        do models fail by false alarms (specificity falls) or by missed AF (sensitivity falls)?"""
        L = self.long[self.long.dataset == "internal"].copy()
        L["condition"] = [("clean" if r.noise == "none" else f"{r.noise} {r.snr_db:+g} dB") for r in L.itertuples()]
        rows = []
        for (v, m, c), g in L.groupby(["variant", "model", "condition"], sort=False):
            r = g.iloc[0]
            rows.append({"variant": v, "model": LABEL[m], "condition": c,
                         "sensitivity": _fmt(r, "sensitivity"), "specificity": _fmt(r, "specificity"),
                         "ppv": _fmt(r, "ppv"), "f1": _fmt(r, "f1")})
        pd.DataFrame(rows).to_csv(self.paths.tables / "failure_modes.csv", index=False)

    # ------------------------------------------------------------------ subgroups (fairness, TRIPOD+AI)
    def subgroups(self):
        """External (PTB-XL, clean) performance by sex and age band, patient-level bootstrap."""
        em = self.ext_meta
        if "age" not in em or "external__clean" not in self.conds:
            return
        age = em.age.where(em.age < 300, 90)
        groups = {"sex=male": em.sex == 0, "sex=female": em.sex == 1, "age<65": age < 65, "age>=65": age >= 65}
        rows = []
        for v in self.variants:
            for m in self.models:
                first, P, thr = self._matrix(v, m, "external__clean")
                for g, mask in groups.items():
                    keep = mask.reindex(first.index).fillna(False).to_numpy()
                    sub = first[keep]
                    if sub.y.nunique() < 2:
                        continue
                    boots = bootstrap_indices(len(sub), self.B, self.cfg["evaluation"]["bootstrap_seed"],
                                              sub.patient_id.to_numpy())
                    s = summarize_condition(sub.y.to_numpy(), P[:, keep], thr, boots)
                    rows.append({"variant": v, "model": LABEL[m], "subgroup": g, "n": len(sub), "AF_n": int(sub.y.sum()),
                                 **{k: _fmt(s, k) for k in ["auroc", "auprc", "sensitivity", "specificity"]}})
        pd.DataFrame(rows).to_csv(self.paths.tables / "subgroups_external.csv", index=False)

    # ------------------------------------------------------------------ T2
    def fourclass(self):
        files = sorted(self.paths.predictions.glob("fourclass__*.parquet"))
        out = []
        for f in files:
            d = pd.read_parquet(f)
            seeds = sorted(d.seed.unique())
            per = {s: d[d.seed == s].sort_values("record_id") for s in seeds}
            y = per[seeds[0]].y.to_numpy()
            preds = np.stack([per[s].pred.to_numpy() for s in seeds])
            point = [challenge_score(y, p) for p in preds]
            boots = bootstrap_indices(len(y), self.B, self.cfg["evaluation"]["bootstrap_seed"])
            reps = [np.mean([challenge_score(y[ix], p[ix])["score"] for p in preds]) for ix in boots]
            sc = np.array([p["score"] for p in point])
            row = {"model": LABEL[d.model.iloc[0]], "n_seeds": len(seeds),
                   "challenge_F1_NAO": f"{sc.mean():.3f} (±{sc.std(ddof=1) if len(sc) > 1 else 0:.3f}) "
                                       f"[{np.percentile(reps, 2.5):.3f}–{np.percentile(reps, 97.5):.3f}]"}
            for k in ["f1_N", "f1_A", "f1_O", "f1_~", "score4"]:
                row[k] = round(float(np.nanmean([p[k] for p in point])), 3)
            out.append(row)
        if out:
            pd.DataFrame(out).to_csv(self.paths.tables / "fourclass_challenge.csv", index=False)

    # ------------------------------------------------------------------ T3
    def noise_tables(self):
        out = []
        for v in self.variants:
            for m in self.models:
                for c in self.conds:
                    cond = _cond_from_key(c)
                    if cond.is_clean:
                        continue
                    s, s0 = self.summary(v, m, c), self.summary(v, m, f"{cond.dataset}__clean")
                    row = {"variant": v, "model": m, "dataset": cond.dataset, "noise": cond.noise, "snr_db": cond.snr_db}
                    for k in KEY_METRICS:
                        j = BINARY_METRICS.index(k)
                        d = paired_difference(s["_reps"][:, j], s0["_reps"][:, j], s[k], s0[k])
                        row.update({k: s[k], f"{k}_drop": -d["diff"], f"{k}_drop_lo": -d["hi"], f"{k}_drop_hi": -d["lo"]})
                    out.append(row)
        self.noise = pd.DataFrame(out)
        self.noise.to_csv(self.paths.tables / "noise_degradation.csv", index=False)

    # ------------------------------------------------------------------ T4
    def paired_tests(self):
        pairs = [("logreg", "rule"), ("lgbm", "rule"), ("cnn", "rule"), ("lgbm", "logreg"), ("cnn", "lgbm")]
        pairs = [p for p in pairs if p[0] in self.models and p[1] in self.models]
        out = []
        for v in self.variants:
            for c in self.conds:
                for a, b in pairs:
                    sa, sb = self.summary(v, a, c), self.summary(v, b, c)
                    for k in ["auroc", "auprc", "f1"]:
                        j = BINARY_METRICS.index(k)
                        d = paired_difference(sa["_reps"][:, j], sb["_reps"][:, j], sa[k], sb[k])
                        out.append({"variant": v, "condition": c, "model_a": a, "model_b": b, "metric": k, **d})
        df = pd.DataFrame(out)
        # Holm within each (variant, condition, metric) family
        df["p_holm"] = np.nan
        for _, idx in df.groupby(["variant", "condition", "metric"]).groups.items():
            ok = df.loc[idx, "p"].notna()
            if ok.any():
                df.loc[idx[ok.to_numpy()], "p_holm"] = holm(df.loc[idx[ok.to_numpy()], "p"].tolist())
        df.to_csv(self.paths.tables / "paired_tests.csv", index=False)

    # ------------------------------------------------------------------ T5
    def augmentation(self):
        if "augmented" not in self.variants:
            return
        out = []
        for m in self.models:
            for c in self.conds:
                sa, ss = self.summary("augmented", m, c), self.summary("standard", m, c)
                for k in ["auroc", "auprc", "f1", "sensitivity", "specificity"]:
                    j = BINARY_METRICS.index(k)
                    out.append({"model": m, "condition": c, "metric": k, "standard": ss[k], "augmented": sa[k],
                                **paired_difference(sa["_reps"][:, j], ss["_reps"][:, j], sa[k], ss[k])})
        pd.DataFrame(out).to_csv(self.paths.tables / "augmentation_effect.csv", index=False)

    # ------------------------------------------------------------------ T6/T7
    def beat_detection(self):
        out, per_rec = [], []
        for c in self.conds:
            cond = _cond_from_key(c)
            if cond.is_clean:
                continue
            f = pd.read_parquet(feature_path(self.paths, c))[BEAT + ["rr_n_beats"]]
            tp, fp, fn = f.beat_tp.sum(), f.beat_fp.sum(), f.beat_fn.sum()
            rec_f1 = 2 * f.beat_tp / (2 * f.beat_tp + f.beat_fp + f.beat_fn).replace(0, np.nan)
            out.append({"condition": c, "dataset": cond.dataset, "noise": cond.noise, "snr_db": cond.snr_db,
                        "sensitivity": tp / (tp + fn), "ppv": tp / (tp + fp), "f1": 2 * tp / (2 * tp + fp + fn),
                        "median_record_f1": float(rec_f1.median()),
                        "frac_records_f1_below_0.9": float((rec_f1 < 0.9).mean()),
                        "frac_detection_failed": float((f.rr_n_beats < self.cfg["features"]["min_beats"]).mean())})
            per_rec.append(pd.DataFrame({"condition": c, "record_id": f.index, "beat_f1": rec_f1.to_numpy()}))
        self.beats = pd.DataFrame(out)
        self.beats.to_csv(self.paths.tables / "beat_detection.csv", index=False)
        if not per_rec:
            return
        pr = pd.concat(per_rec)
        bins = [-0.01, 0.5, 0.8, 0.95, 1.0]
        pr["beat_f1_bin"] = pd.cut(pr.beat_f1.fillna(0), bins, labels=["<0.50", "0.50–0.80", "0.80–0.95", "≥0.95"])
        rows = []
        for v in self.variants:
            for m in self.models:
                d = self.pred[(self.pred.variant == v) & (self.pred.model == m) & (self.pred.dataset == "internal")
                              & (self.pred.noise != "none")].merge(pr, on=["condition", "record_id"])
                for b, g in d.groupby("beat_f1_bin", observed=True):
                    per_seed = [binary_metrics(gs.y, gs.prob, gs.threshold.iloc[0]) for _, gs in g.groupby("seed")]
                    rows.append({"variant": v, "model": m, "beat_f1_bin": b, "n_record_conditions": len(g) // g.seed.nunique(),
                                 **{k: float(np.nanmean([p[k] for p in per_seed])) for k in ["auroc", "f1", "sensitivity", "specificity"]}})
        pd.DataFrame(rows).to_csv(self.paths.tables / "performance_by_beat_quality.csv", index=False)

    # ------------------------------------------------------------------ T8
    def abstention(self):
        qcol = self.cfg["evaluation"]["quality_score"]
        covs = self.cfg["evaluation"]["abstention_coverages"]
        pcts = self.cfg["evaluation"]["fixed_quality_percentiles"]
        curve, fixed = [], []
        for v in self.variants:
            for m in self.models:
                b0 = load_pickle(_model_path(self.paths, "binary", v, m, int(self.pred.seed.min())))
                tq = b0["train_quality"]
                for c in self.conds:
                    first, P, thr = self._matrix(v, m, c)
                    y = first.y.to_numpy()
                    q = first[qcol].to_numpy(float)
                    qs = np.where(np.isfinite(q), q, -np.inf)
                    order = np.argsort(-qs, kind="mergesort")
                    for cov in covs:
                        keep = order[: max(1, int(math.ceil(cov * len(y))))]
                        per = [binary_metrics(y[keep], P[s, keep], thr[s]) for s in range(len(P))]
                        curve.append({"variant": v, "model": m, "condition": c, "coverage": cov,
                                      **{k: float(np.nanmean([p[k] for p in per])) for k in KEY_METRICS}})
                    for p in pcts:
                        t = np.nanpercentile(tq, p)
                        keep = np.where(q >= t)[0]
                        per = [binary_metrics(y[keep], P[s, keep], thr[s]) for s in range(len(P))] if len(keep) else []
                        fixed.append({"variant": v, "model": m, "condition": c, "train_percentile": p, "q_threshold": t,
                                      "abstained_frac": 1 - len(keep) / len(y), "af_abstained_frac":
                                      1 - (y[keep] == 1).sum() / max((y == 1).sum(), 1),
                                      **{k: float(np.nanmean([d[k] for d in per])) if per else np.nan for k in KEY_METRICS}})
        self.abst = pd.DataFrame(curve)
        self.abst.to_csv(self.paths.tables / "abstention_curve.csv", index=False)
        pd.DataFrame(fixed).to_csv(self.paths.tables / "abstention_fixed_threshold.csv", index=False)

    # ------------------------------------------------------------------ T9
    def calibration(self):
        rows, self.cal_curves = [], {}
        nb = self.cfg["evaluation"]["calibration_bins"]
        for v in self.variants:
            for m in self.models:
                for c in self.conds:
                    first, P, _ = self._matrix(v, m, c)
                    y = first.y.to_numpy()
                    per = [calibration_summary(y, P[s], nb) for s in range(len(P))]
                    pooled = calibration_summary(y, P.mean(0), nb)
                    self.cal_curves[(v, m, c)] = pooled
                    rows.append({"variant": v, "model": m, "condition": c,
                                 **{k: float(np.nanmean([p[k] for p in per])) for k in ["ece", "cal_slope", "cal_intercept"]},
                                 "brier": self.summary(v, m, c)["brier"]})
        pd.DataFrame(rows).to_csv(self.paths.tables / "calibration.csv", index=False)

    # ------------------------------------------------------------------ T10
    def shap(self):
        if "lgbm" not in self.models:
            return
        import shap

        internal_meta = load_dataset(self.cfg, "internal")["meta"]
        ids = binary_ids(internal_meta, self.split["test"])
        rng = np.random.default_rng(0)
        if len(ids) > self.cfg["shap"]["max_samples"]:
            ids = sorted(rng.choice(ids, self.cfg["shap"]["max_samples"], replace=False))
        rows = []
        for v in self.variants:
            seeds = sorted(self.pred[(self.pred.model == "lgbm") & (self.pred.variant == v)].seed.unique())
            ests = [load_pickle(_model_path(self.paths, "binary", v, "lgbm", int(s)))["estimator"] for s in seeds]
            for c in [c for c in self.conds if c.startswith("internal")]:
                X = pd.read_parquet(feature_path(self.paths, c)).loc[ids, FEATURES]
                imp = []
                for est in ests:
                    sv = shap.TreeExplainer(est).shap_values(X)
                    sv = sv[1] if isinstance(sv, list) else sv
                    sv = sv[..., 1] if sv.ndim == 3 else sv
                    a = np.abs(sv).mean(0)
                    imp.append(a / a.sum())
                imp = np.mean(imp, 0)
                for f, x in zip(FEATURES, imp):
                    rows.append({"variant": v, "condition": c, "feature": f, "mean_abs_shap_share": float(x)})
        self.shap_df = pd.DataFrame(rows)
        self.shap_df.to_csv(self.paths.tables / "shap_importance.csv", index=False)
        stab = []
        for (v, c), g in self.shap_df.groupby(["variant", "condition"]):
            ref = self.shap_df[(self.shap_df.variant == v) & (self.shap_df.condition == "internal__clean")].set_index("feature")
            g = g.set_index("feature").loc[ref.index]
            tau = kendalltau(ref.mean_abs_shap_share, g.mean_abs_shap_share).statistic
            top = lambda s: set(s.sort_values(ascending=False).index[:5])  # noqa: E731
            stab.append({"variant": v, "condition": c, "kendall_tau_vs_clean": tau,
                         "top5_overlap_vs_clean": len(top(ref.mean_abs_shap_share) & top(g.mean_abs_shap_share)),
                         "top5": ", ".join(g.mean_abs_shap_share.sort_values(ascending=False).index[:5])})
        pd.DataFrame(stab).to_csv(self.paths.tables / "shap_rank_stability.csv", index=False)

    # ------------------------------------------------------------------ figures
    def _snr_axis(self, ax, snrs):
        ax.set_xticks(range(len(snrs) + 1))
        ax.set_xticklabels(["clean"] + [f"{s:g}" for s in snrs])
        ax.set_xlabel("SNR (dB)  →  noisier")

    def _snrs(self, dataset="internal"):
        return sorted({_cond_from_key(c).snr_db for c in self.conds if c.startswith(dataset) and "clean" not in c},
                      reverse=True)

    def figures(self):
        snrs = self._snrs()
        types = [t for t in ["bw", "ma", "em"] if any(f"__{t}__" in c for c in self.conds)]
        if not types:
            return
        L = self.long[self.long.dataset == "internal"]

        def series(v, m, metric, noise):
            d = L[(L.variant == v) & (L.model == m)]
            clean = d[d.noise == "none"].iloc[0]
            rows = [clean] + [d[(d.noise == noise) & (d.snr_db == s)].iloc[0] for s in snrs]
            return (np.array([r[metric] for r in rows]), np.array([r[f"{metric}_lo"] for r in rows]),
                    np.array([r[f"{metric}_hi"] for r in rows]))

        # Fig 1: robustness of each model (standard training)
        for metric, ylab in [("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1")]:
            fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 2.9), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, t in zip(axes, types):
                for m in self.models:
                    y, lo, hi = series("standard", m, metric, t)
                    x = np.arange(len(y))
                    ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.12, linewidth=0)
                    ax.plot(x, y, color=COLOR[m], marker=MARKER[m], label=LABEL[m])
                ax.set_title(NOISE_LABEL[t], color=INK, fontsize=9)
                self._snr_axis(ax, snrs)
            axes[0].set_ylabel(ylab)
            axes[-1].legend(loc="lower left", fontsize=7)
            fig.tight_layout()
            fig.savefig(self.paths.figures / f"fig1_noise_{metric}.png")
            plt.close(fig)

        # Fig 2: beat detection vs SNR, and AUROC tracking beat detection
        if len(self.beats):
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
            B = self.beats[self.beats.dataset == "internal"]
            ncol = {"bw": "#6250d6", "ma": "#e34948", "em": "#008300"}
            nmark = {"bw": "o", "ma": "s", "em": "^"}
            for t in types:
                d = B[B.noise == t].set_index("snr_db").loc[snrs]
                # clean = detections compared with themselves (F1 = 1 by definition)
                a1.plot(range(len(snrs) + 1), np.r_[1.0, d.f1.to_numpy()], color=ncol[t], marker=nmark[t],
                        label=NOISE_LABEL[t])
            self._snr_axis(a1, snrs)
            a1.set_ylabel("R-peak detection F1")
            a1.legend(fontsize=7, loc="lower left")
            a1.set_title("a  NeuroKit2 beat detection", loc="left", fontsize=9)
            N = self.noise[(self.noise.variant == "standard") & (self.noise.dataset == "internal")]
            Bk = B.set_index(["noise", "snr_db"])
            for m in self.models:
                d = N[N.model == m]
                xs = [Bk.loc[(r.noise, r.snr_db), "f1"] for r in d.itertuples()]
                a2.scatter(xs, d.auroc, color=COLOR[m], marker=MARKER[m], s=22, label=LABEL[m], edgecolor="white", linewidth=0.8)
            a2.set_xlabel("R-peak detection F1 in that condition")
            a2.set_ylabel("AUROC")
            a2.legend(fontsize=7, loc="lower right")
            a2.set_title("b  Classifier AUROC vs beat detection", loc="left", fontsize=9)
            fig.tight_layout()
            fig.savefig(self.paths.figures / "fig2_beat_detection.png")
            plt.close(fig)

        # Fig 3: augmentation (solid = standard, dashed = augmented)
        if "augmented" in self.variants:
            fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 2.9), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, t in zip(axes, types):
                for m in self.models:
                    for v, ls in [("standard", "-"), ("augmented", "--")]:
                        y, _, _ = series(v, m, "auroc", t)
                        ax.plot(range(len(y)), y, ls, color=COLOR[m], marker=MARKER[m],
                                label=f"{LABEL[m]}{' (aug.)' if v == 'augmented' else ''}")
                ax.set_title(NOISE_LABEL[t], color=INK, fontsize=9)
                self._snr_axis(ax, snrs)
            axes[0].set_ylabel("AUROC")
            axes[-1].legend(fontsize=6, loc="lower left", ncol=1)
            fig.tight_layout()
            fig.savefig(self.paths.figures / "fig3_augmentation_auroc.png")
            plt.close(fig)

        # Fig 4: abstention trade-off
        pick = [c for c in self.conds if c == "internal__clean"]
        em = [c for c in self.conds if c.startswith("internal__em__")]
        if em:
            mid = sorted(em, key=lambda c: abs(_cond_from_key(c).snr_db - 6))[0]
            pick.append(mid)
        fig, axes = plt.subplots(1, len(pick), figsize=(3.4 * len(pick), 2.9), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, pick):
            for m in self.models:
                d = self.abst[(self.abst.variant == "standard") & (self.abst.model == m) & (self.abst.condition == c)]
                ax.plot(1 - d.coverage, d.f1, color=COLOR[m], marker=MARKER[m], label=LABEL[m])
            cond = _cond_from_key(c)
            ax.set_title("Clean" if cond.is_clean else f"{NOISE_LABEL[cond.noise]}, {cond.snr_db:g} dB", fontsize=9)
            ax.set_xlabel("Fraction abstained (lowest quality first)")
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[0].set_ylabel("F1 on retained recordings")
        axes[-1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(self.paths.figures / "fig4_abstention.png")
        plt.close(fig)

        # Fig 5: calibration under noise (electrode motion)
        em_snrs = [s for s in snrs if s in (12.0, 0.0)] or snrs[-1:]
        cs = ["internal__clean"] + [c for c in self.conds if c.startswith("internal__em__")
                                    and _cond_from_key(c).snr_db in em_snrs]
        show = [m for m in ["lgbm", "cnn"] if m in self.models]
        fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3.0), sharey=True)
        axes = np.atleast_1d(axes)
        shades = ["#0b0b0b", "#52514e", "#a3a29c"]
        for ax, m in zip(axes, show):
            ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
            for c, col in zip(cs, shades):
                cur = self.cal_curves[("standard", m, c)]
                cond = _cond_from_key(c)
                ax.plot(cur["curve_pred"], cur["curve_obs"], color=col, marker="o",
                        label="clean" if cond.is_clean else f"EM {cond.snr_db:g} dB")
            ax.set_title(LABEL[m], fontsize=9)
            ax.set_xlabel("Predicted AF probability")
            ax.set_aspect("equal")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Observed AF fraction")
        axes[-1].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(self.paths.figures / "fig5_calibration.png")
        plt.close(fig)

        # Fig 6: SHAP importance, clean and across electrode-motion SNR
        if hasattr(self, "shap_df"):
            S = self.shap_df[self.shap_df.variant == "standard"]
            clean = S[S.condition == "internal__clean"].sort_values("mean_abs_shap_share", ascending=False).head(15)
            em_c = ["internal__clean"] + [c for c in self.conds if c.startswith("internal__em__")]
            H = S[S.condition.isin(em_c)].pivot(index="feature", columns="condition", values="mean_abs_shap_share")
            H = H.loc[clean.feature, em_c]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
            a1.barh(clean.feature[::-1], clean.mean_abs_shap_share[::-1], color=COLOR["lgbm"], height=0.7)
            a1.set_xlabel("Share of mean |SHAP|")
            a1.set_title("a  Gradient boosting, clean", loc="left", fontsize=9)
            a1.grid(axis="y", visible=False)
            im = a2.imshow(H.to_numpy(), aspect="auto", cmap="Greens")
            a2.set_yticks(range(len(H)))
            a2.set_yticklabels(H.index, fontsize=7)
            a2.set_xticks(range(len(em_c)))
            a2.set_xticklabels(["clean"] + [f"{_cond_from_key(c).snr_db:g}" for c in em_c[1:]])
            a2.set_xlabel("Electrode motion SNR (dB)")
            a2.grid(False)
            a2.set_title("b  Importance as noise increases", loc="left", fontsize=9)
            fig.colorbar(im, ax=a2, fraction=0.04, label="share of mean |SHAP|")
            fig.tight_layout()
            fig.savefig(self.paths.figures / "fig6_shap.png")
            plt.close(fig)

    # ------------------------------------------------------------------ digest
    def write_summary(self):
        lines = ["# Results digest", "", "Values: mean over seeds (±SD) [95% bootstrap CI]. Generated automatically.", ""]
        for ds in ["internal", "external"]:
            lines += [f"## {ds.capitalize()} test, clean", "", pd.read_csv(self.paths.tables / f"main_{ds}.csv").to_markdown(index=False), ""]
        f4 = self.paths.tables / "fourclass_challenge.csv"
        if f4.exists():
            lines += ["## Four-class challenge score", "", pd.read_csv(f4).to_markdown(index=False), ""]
        if len(self.noise):
            worst = self.noise[(self.noise.variant == "standard") & (self.noise.dataset == "internal")]
            worst = worst.loc[worst.groupby(["model", "noise"]).snr_db.idxmin()]
            lines += ["## AUROC at the noisiest level (standard training)", "",
                      worst[["model", "noise", "snr_db", "auroc", "auroc_drop", "auroc_drop_lo", "auroc_drop_hi"]]
                      .round(3).to_markdown(index=False), ""]
        if len(self.beats):
            lines += ["## Beat detection under noise", "", self.beats.round(3).to_markdown(index=False), ""]
        (self.paths.tables / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        save_json({"n_bootstrap": self.B, "conditions": self.conds, "models": self.models, "variants": self.variants},
                  self.paths.tables / "report_manifest.json")


def run_report(cfg: dict) -> None:
    import warnings

    with warnings.catch_warnings():
        # empty strata (e.g. no AF retained at extreme abstention) legitimately yield NaN metrics
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")
        Reporter(cfg).run()

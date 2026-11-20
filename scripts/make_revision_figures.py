"""Figures for the revised manuscript (journal format, same style and palette as make_journal_figures.py).

    python scripts/make_revision_figures.py --config configs/local.yaml [--only workflow examples ...]

workflow     study workflow diagram
examples     clean and noise-corrupted example ECGs with clean-signal and corrupted-signal R peaks (supplement)
errors       sensitivity and specificity versus electrode-motion SNR, internal and external
mechanism    (a) R-peak F1 against expert beats by detector, (b) specificity with detected vs clean-signal R peaks,
             (c) share of non-AF recordings called AF by spurious beats per minute
selective    coverage-risk curves for several quality measures, and the share of clean PTB-XL withheld
variability  noise-realization vs training-seed variability (supplement)
afdb         naturally noisy Holter windows by quality fifth (supplement)

Each figure is written to paper/submission/<name>.pdf and .png (15 cm wide, Arial, 600 dpi).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgrobust.utils import Paths, load_config  # noqa: E402

MODELS = ["rule", "logreg", "lgbm", "cnn"]
LABEL = {"rule": "Rule (RMSSD/RR)", "logreg": "Logistic regression", "lgbm": "Gradient boosting", "cnn": "1-D CNN"}
COLOR = {"rule": "#2a78d6", "logreg": "#eb6834", "lgbm": "#1baf7a", "cnn": "#eda100"}
MARKER = {"rule": "o", "logreg": "s", "lgbm": "^", "cnn": "D"}
NOISE_LABEL = {"bw": "Baseline wander", "ma": "Muscle artifact", "em": "Electrode motion"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W = 15 / 2.54
DPI, PAD = 600, 0.02

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.25, "lines.markersize": 4, "grid.linewidth": 0.5, "axes.unicode_minus": True,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/local.yaml")
ap.add_argument("--only", nargs="*")
A = ap.parse_args()
cfg = load_config(A.config)
P = Paths.from_cfg(cfg)
T = P.tables
OUT = ROOT / "paper" / "submission"


def save(fig, name: str) -> None:
    for ext, kw in [("pdf", {}), ("png", {"dpi": DPI})]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=PAD, facecolor="white", **kw)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf and .png")


def snr_txt(s: float) -> str:
    return "clean" if not np.isfinite(s) else f"{s:g}".replace("-", "−")


# ----------------------------------------------------------------------------- workflow
def workflow():
    fig, ax = plt.subplots(figsize=(W, 3.05))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.set_xlim(-0.6, 101.2)
    ax.set_ylim(0, 47.5)

    def box(x, y, w, h, title, body, fc="#f4f3f0"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=INK2, lw=0.6))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=6.6, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=5.7, color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2,
                                                                      shrinkA=0, shrinkB=0, mutation_scale=7))

    w4, gap = 23.5, 2.0
    xs = [0.5 + i * (w4 + gap) for i in range(4)]
    top = [("CinC 2017", "8,528 handheld\nsingle-lead ECGs, 300 Hz\n80% development,\n20% locked test"),
           ("MIT-BIH NSTDB", "baseline wander, muscle\nartifact, electrode motion\nfirst half: training\nsecond half: test"),
           ("PTB-XL lead I", "21,799 clinical ECGs\nfrom 18,869 patients\nexternal test only"),
           ("Annotated databases", "MIT-BIH Arrhythmia:\nexpert beat locations\nMIT-BIH AF: rhythm labels,\nnatural Holter noise")]
    for x, (a, b) in zip(xs, top):
        box(x, 33, w4, 14, a, b)
    w3 = 32.0
    x3 = [0.5, 0.5 + w3 + 1.75, 0.5 + 2 * (w3 + 1.75)]
    mid = [("Model development", "training data only; 5 seeds\nrule, logistic regression,\ngradient boosting (29 features)\n1-D CNN on the raw signal\nstandard and noise-augmented", "#eaf3ee"),
           ("Controlled corruption", "3 noise types x 6 SNRs\n(24 to -6 dB), identical for\nevery model; 5 noise draws;\nalternative generators; features\nre-extracted after corruption", "#fdf2e3"),
           ("Frozen models", "calibration and threshold from\ntraining data only; applied\nunchanged to CinC test set,\nPTB-XL and Holter windows", "#eaf3ee")]
    for x, (a, b, c) in zip(x3, mid):
        box(x, 15.5, w3, 14.5, a, b.replace("x 6", "× 6").replace("-6 dB", "−6 dB"), c)
    bottom = [("Error direction", "sensitivity, specificity,\nPPV, NPV at the deployed\nthreshold; confusion\nmatrices"),
              ("Mechanism", "R-peak errors vs expert\nbeats; clean-signal R peaks;\nother detectors;\nregional noise"),
              ("Remedies", "augmentation ablation;\n7 quality measures;\ncoverage-risk; transfer\nbetween devices"),
              ("Statistics", "paired bootstrap over\nrecordings or patients;\n6 Holm-adjusted primary\ncomparisons")]
    for x, (a, b) in zip(xs, bottom):
        box(x, 0.5, w4, 12.3, a, b)
    arrow(xs[0] + w4 / 2, 33, x3[0] + w3 / 2, 30.3)
    arrow(xs[1] + w4 / 2, 33, x3[1] + w3 / 2, 30.3)
    arrow(xs[2] + w4 / 2, 33, x3[2] + w3 / 2 - 3, 30.3)
    arrow(xs[3] + w4 / 2, 33, x3[2] + w3 / 2 + 6, 30.3)
    arrow(x3[0] + w3 + 0.2, 22.75, x3[1] - 0.2, 22.75)
    arrow(x3[1] + w3 + 0.2, 22.75, x3[2] - 0.2, 22.75)
    for x in xs:
        arrow(x3[2] + w3 / 2 if x > 50 else x3[1] + w3 / 2, 15.2, x + w4 / 2, 13.1)
    save(fig, "figure_workflow")


# ----------------------------------------------------------------------------- examples
def examples():
    import warnings

    import neurokit2 as nk

    from ecgrobust.data import load_dataset
    from ecgrobust.noise import Condition, NoiseBank, corrupt_for_test
    from ecgrobust.splits import load_split
    from ecgrobust.utils import load_pickle

    warnings.filterwarnings("ignore")
    internal = load_dataset(cfg, "internal")
    meta, split = internal["meta"], load_split(cfg)
    bank = NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])
    ref = load_pickle(P.features / "internal__refpeaks.pkl")
    feats = pd.read_parquet(P.features / "internal__em__-6dB__r0.parquet")
    test = [r for r in split["test"] if meta.at[r, "label4"] in ("N", "A") and meta.at[r, "duration_s"] >= 30]
    # a typical recording of each class: median spurious-beat count under EM -6 dB
    pick = {}
    for lab in ("N", "A"):
        ids = [r for r in test if meta.at[r, "label4"] == lab]
        f = feats.loc[ids].sort_values("beat_fp")
        pick[lab] = f.index[len(f) // 2]
    fs, seconds = cfg["fs"], 8
    conds = [("Clean", None), ("Baseline wander, −6 dB", Condition("internal", "bw", -6.0, 0)),
             ("Muscle artifact, −6 dB", Condition("internal", "ma", -6.0, 0)),
             ("Electrode motion, −6 dB", Condition("internal", "em", -6.0, 0))]
    fig, axes = plt.subplots(len(conds), 2, figsize=(W, 4.6), sharex=True)
    for j, (lab, title) in enumerate([("N", "Normal rhythm"), ("A", "Atrial fibrillation")]):
        rid = pick[lab]
        x = internal["signals"][rid]
        lo = 5 * fs
        seg = slice(lo, lo + seconds * fs)
        t = np.arange(seconds * fs) / fs
        clean_pk = np.asarray(ref[rid])
        for i, (name, cond) in enumerate(conds):
            ax = axes[i, j]
            y = x if cond is None else corrupt_for_test(x, rid, cond, bank, cfg["noise"]["seed"])
            ax.plot(t, y[seg], color=INK, lw=0.6)
            inside = clean_pk[(clean_pk >= lo) & (clean_pk < lo + seconds * fs)] - lo
            ax.plot(inside / fs, np.full(len(inside), 1.0), "v", color=COLOR["lgbm"], ms=3.2,
                    transform=ax.get_xaxis_transform(), clip_on=False)
            if cond is not None:
                try:
                    yi, _ = nk.ecg_invert(y, sampling_rate=fs)
                except Exception:
                    yi = y
                pk = nk.ecg_peaks(nk.ecg_clean(yi, sampling_rate=fs), sampling_rate=fs, correct_artifacts=False)[1]["ECG_R_Peaks"]
                pk = np.asarray(pk)
                pk = pk[(pk >= lo) & (pk < lo + seconds * fs)] - lo
                ax.plot(pk / fs, np.full(len(pk), 0.0), "^", color=COLOR["logreg"], ms=3.2,
                        transform=ax.get_xaxis_transform(), clip_on=False)
            ax.set_ylim(-1.3, 1.6)
            ax.set_yticks([-1, 0, 1])
            ax.grid(False)
            if j == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7.5)
            if i == 0:
                ax.set_title(f"({'ab'[j]}) {title}")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.legend(handles=[Line2D([], [], color=COLOR["lgbm"], marker="v", ls="", label="R peaks detected on the clean signal"),
                        Line2D([], [], color=COLOR["logreg"], marker="^", ls="", label="R peaks detected on the corrupted signal")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.04))
    fig.text(0.0, 0.5, "mV", rotation=90, va="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0.02, 0.04, 1, 1))
    save(fig, "figureS_examples")


# ----------------------------------------------------------------------------- errors
def errors():
    snrs = [24.0, 18.0, 12.0, 6.0, 0.0, -6.0]
    I = pd.read_csv(T / "metrics_all_conditions.csv")
    E = pd.read_csv(T / "external_noise.csv")
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.2), sharex=True, sharey=True)
    for c, (data, name) in enumerate([("internal", "Internal test set (CinC 2017)"), ("external", "External test set (PTB-XL)")]):
        for r, metric in enumerate(["sensitivity", "specificity"]):
            ax = axes[r, c]
            for m in MODELS:
                if data == "internal":
                    d = I[(I.dataset == "internal") & (I.variant == "standard") & (I.model == m)]
                    rows = [d[d.noise == "none"].iloc[0]] + [d[(d.noise == "em") & (d.snr_db == s)].iloc[0] for s in snrs]
                else:
                    d = E[(E.variant == "standard") & (E.model == m)]
                    keys = ["external__clean"] + [f"external__em__{s:+g}dB__r0" for s in snrs]
                    rows = [d[d.condition == k].iloc[0] for k in keys if (d.condition == k).any()]
                y = np.array([x[metric] for x in rows])
                lo = np.array([x[f"{metric}_lo"] for x in rows])
                hi = np.array([x[f"{metric}_hi"] for x in rows])
                xs = np.arange(len(y))
                ax.fill_between(xs, lo, hi, color=COLOR[m], alpha=0.12, lw=0)
                ax.plot(xs, y, color=COLOR[m], marker=MARKER[m], label=LABEL[m])
            ax.set_xticks(range(len(snrs) + 1))
            ax.set_xticklabels(["clean"] + [snr_txt(s) for s in snrs])
            ax.set_ylim(0, 1.02)
            if c == 0:
                ax.set_ylabel(metric.capitalize())
            if r == 0:
                ax.set_title(f"({'ab'[c]}) {name}")
            else:
                ax.set_title(f"({'cd'[c]})")
                ax.set_xlabel("Electrode-motion SNR (dB)  →  noisier")
    fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKER[m], label=LABEL[m]) for m in MODELS],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "figure_errors")


# ----------------------------------------------------------------------------- mechanism
def mechanism():
    rv = pd.read_csv(T / "rpeak_validation.csv")
    orc = pd.read_csv(T / "rev_oracle.csv")
    det = pd.read_csv(T / "rev_detection_errors.csv")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(W, 2.5))
    dets = {"neurokit": ("NeuroKit2 (study)", INK, "o"), "hamilton2002": ("Hamilton", "#2a78d6", "s"),
            "elgendi2010": ("Elgendi", "#eb6834", "^"), "pantompkins1985": ("Pan–Tompkins", "#1baf7a", "D"),
            "engzeemod2012": ("Engelse–Zeelenberg", "#9b59b6", "v"), "kalidas2017": ("Kalidas", "#eda100", "P"),
            "rodrigues2021": ("Rodrigues", "#7f8c8d", "X")}
    snrs = [12.0, 6.0, 0.0, -6.0]
    d = rv[(rv.rhythm == "all")]
    for k, (lab, col, mk) in dets.items():
        y = [d[(d.noise == "none") & (d.detector == k)].f1.iloc[0]] + \
            [d[(d.noise == "em") & (d.snr_db == s) & (d.detector == k)].f1.iloc[0] for s in snrs]
        a1.plot(range(len(y)), y, color=col, marker=mk, lw=1.4 if k == "neurokit" else 0.9,
                ms=3.5 if k == "neurokit" else 2.8, label=lab, zorder=3 if k == "neurokit" else 2)
    a1.set_xticks(range(len(snrs) + 1))
    a1.set_xticklabels(["clean"] + [snr_txt(s) for s in snrs])
    a1.set_xlabel("Electrode-motion SNR (dB)")
    a1.set_ylabel("R-peak F1 vs expert beats")
    a1.set_title("(a) Beat detection (MIT-BIH)")
    a1.legend(fontsize=6, loc="lower left", handlelength=1.4, labelspacing=0.25)
    o = orc[(orc.variant == "standard") & (orc.noise == "em")].sort_values("snr_db", ascending=False)
    for m in ["rule", "logreg", "lgbm"]:
        g = o[o.model == m]
        xs = np.arange(len(g))
        a2.plot(xs, g.spec_detected_peaks, color=COLOR[m], marker=MARKER[m], label=LABEL[m])
        a2.plot(xs, g.spec_clean_peaks, color=COLOR[m], marker=MARKER[m], ls="--", mfc="white")
    a2.set_xticks(range(len(o.snr_db.unique())))
    a2.set_xticklabels([snr_txt(s) for s in sorted(o.snr_db.unique(), reverse=True)])
    a2.set_xlabel("Electrode-motion SNR (dB)")
    a2.set_ylabel("Specificity (internal)")
    a2.set_title("(b) R peaks: detected vs clean")
    a2.legend(handles=[Line2D([], [], color=INK, label="peaks from noisy signal"),
                       Line2D([], [], color=INK, ls="--", mfc="white", marker="o", label="peaks from clean signal")],
              fontsize=6, loc="lower left")
    g = det[(det["class"] == "non-AF")]
    order = ["0", "0.5-2", "2-5", "5-10", ">10"]
    for m in MODELS:
        h = g[g.model == m].set_index("bin").reindex(order)
        a3.plot(range(len(order)), h.af_call_rate, color=COLOR[m], marker=MARKER[m], label=LABEL[m])
    a3.set_xticks(range(len(order)))
    a3.set_xticklabels([s.replace("-", "–") for s in order])
    a3.set_xlabel("Spurious beats/min")
    a3.set_ylabel("Non-AF recordings called AF")
    a3.set_title("(c) False AF calls")
    fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKER[m], label=LABEL[m]) for m in MODELS],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "figure_mechanism")


# ----------------------------------------------------------------------------- selective
def selective():
    cr = pd.read_csv(T / "selective_coverage_risk.csv")
    th = pd.read_csv(T / "selective_thresholds.csv")
    meas = {"avgqrs": ("Average QRS (NeuroKit2)", INK), "zhao": ("Zhao fuzzy", "#9b59b6"), "ksqi": ("kSQI", "#2a78d6"),
            "psqi": ("pSQI", "#eb6834"), "bassqi": ("basSQI", "#7f8c8d"), "bsqi": ("Detector agreement", "#1baf7a"),
            "learned": ("Learned (expert labels)", "#eda100")}
    fig = plt.figure(figsize=(W, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8])
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for k, (lab, col) in meas.items():
        for ax, metric in [(a1, "fpr"), (a2, "fnr")]:
            d = cr[(cr.model == "lgbm" if metric == "fpr" else cr.model == "cnn") & (cr.condition == "internal__em__-6dB__r0") &
                   (cr.measure == k)].sort_values("coverage", ascending=False)
            ax.plot(1 - d.coverage, d[metric], color=col, marker="o", ms=2.5, lw=1.0, label=lab)
    a1.set_title("(a) Gradient boosting, EM −6 dB")
    a1.set_ylabel("False-positive rate (retained)")
    a2.set_title("(b) CNN, EM −6 dB")
    a2.set_ylabel("False-negative rate (retained)")
    for ax in (a1, a2):
        ax.set_xlabel("Share withheld")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    s = th[(th.condition == "external__clean") & (th.model == "lgbm")]
    src = s[s.policy.str.startswith("source")].set_index("measure").reindex(list(meas))
    tgt = s[s.policy.str.startswith("target")].set_index("measure").reindex(list(meas))
    yy = np.arange(len(meas))
    a3.barh(yy + 0.2, src.withheld_all, height=0.38, color="#c0392b", label="CinC threshold")
    a3.barh(yy - 0.2, tgt.withheld_all, height=0.38, color="#95a5a6", label="PTB-XL threshold")
    a3.set_yticks(yy)
    a3.set_yticklabels([meas[k][0] for k in meas], fontsize=6.5)
    a3.invert_yaxis()
    a3.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a3.set_xlabel("Share of clean PTB-XL recordings withheld")
    a3.set_title("(c) Clean PTB-XL recordings withheld by a 5th-percentile quality threshold")
    a3.set_xlim(0, 0.6)
    a3.legend(fontsize=6.5, loc="lower right")
    a3.grid(axis="y", visible=False)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.505), fontsize=6.5, frameon=False)
    fig.tight_layout(h_pad=4.2)
    save(fig, "figure_selective")


# ----------------------------------------------------------------------------- variability
def variability():
    v = pd.read_csv(T / "rev_variance.csv")
    v = v[(v.variant == "standard") & (v.noise == "em")]
    order = sorted(v.snr_db.unique(), reverse=True)   # fixed positions: the CNN has extra draws at 0 and -6 dB only
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for ax, metric in zip(axes, ["sensitivity", "specificity"]):
        for m in MODELS:
            g = v[v.model == m].sort_values("snr_db", ascending=False)
            xs = np.array([order.index(s) for s in g.snr_db])
            ax.plot(xs, g[f"{metric}_sd_realization"], color=COLOR[m], marker=MARKER[m], label=LABEL[m])
            ax.plot(xs, g[f"{metric}_sd_seed"], color=COLOR[m], marker=MARKER[m], ls=":", mfc="white")
            ax.plot(xs, g[f"{metric}_se_sample"], color=COLOR[m], ls="--", lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([snr_txt(s) for s in order])
        ax.set_xlabel("Electrode-motion SNR (dB)")
        ax.set_title(f"({'ab'[metric == 'specificity']}) {metric.capitalize()}")
    axes[0].set_ylabel("Standard deviation")
    fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKER[m], label=LABEL[m]) for m in MODELS] +
               [Line2D([], [], color=INK, label="noise draw"), Line2D([], [], color=INK, ls=":", marker="o", mfc="white", label="training seed"),
                Line2D([], [], color=INK, ls="--", label="test sample (bootstrap)")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "figureS_variability")


# ----------------------------------------------------------------------------- afdb
def afdb():
    t = pd.read_csv(T / "afdb_by_quality.csv")
    # the cleanest fifth's natural row appears twice (by-quality and by-condition tables); keep the first
    nat = t[(t.condition == "natural") & (t.variant == "standard")].drop_duplicates(["model", "quality_fifth"])
    order = ["Q1 (noisiest)", "Q2", "Q3", "Q4", "Q5 (cleanest)"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for ax, metric in zip(axes, ["sensitivity", "specificity"]):
        for i, m in enumerate(MODELS):
            g = nat[nat.model == m].set_index("quality_fifth").reindex(order)
            xs = np.arange(len(order)) + (i - 1.5) * 0.08
            ax.errorbar(xs, g[metric], yerr=[g[metric] - g[f"{metric}_lo"], g[f"{metric}_hi"] - g[metric]],
                        color=COLOR[m], marker=MARKER[m], lw=1.0, capsize=1.5, label=LABEL[m])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["Q1\nnoisiest", "Q2", "Q3", "Q4", "Q5\ncleanest"])
        ax.set_xlabel("Quality fifth (learned quality score)")
        ax.set_title(f"({'ab'[metric == 'specificity']}) {metric.capitalize()}")
    axes[0].set_ylabel("Value (95% CI)")
    fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKER[m], label=LABEL[m]) for m in MODELS],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "figureS_afdb")


FIGS = {"workflow": workflow, "examples": examples, "errors": errors, "mechanism": mechanism, "selective": selective,
        "variability": variability, "afdb": afdb}
for name, fn in FIGS.items():
    if A.only and name not in A.only:
        continue
    try:
        fn()
    except FileNotFoundError as e:
        print(f"skipped {name}: {e}")

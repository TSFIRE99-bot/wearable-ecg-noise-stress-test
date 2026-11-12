"""Journal-format figures 1-6 (IOP Physiological Measurement style) from the saved results.

    python scripts/make_journal_figures.py --config configs/local.yaml

Writes paper/submission/figure1.pdf/.png ... figure6.pdf/.png. Data, colours, markers and
the content of each figure match the report figures (src/ecgrobust/report.py); only size,
fonts, panel labels and legend placement change, plus Figure 4's right panel, which now
shows electrode motion at -6 dB instead of +6 dB.

Format: multi-panel width 15 cm (all six figures have two or three panels), Arial,
8 pt ticks and legends, 9 pt axis labels and panel titles, fonts embedded (TrueType),
vector PDF plus 600 dpi PNG, saved with bbox_inches="tight" on a white background.
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
from matplotlib.ticker import PercentFormatter  # noqa: E402

# Only plotting libraries are needed, so this script avoids importing the model code (PyTorch,
# LightGBM) and runs in any Python with pandas, matplotlib, scikit-learn, pyarrow and PyYAML.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgrobust.metrics import calibration_summary  # noqa: E402
from ecgrobust.noise import Condition  # noqa: E402
from ecgrobust.utils import Paths, load_config  # noqa: E402

# same palette, markers and labels as src/ecgrobust/report.py
MODELS = ["rule", "logreg", "lgbm", "cnn"]
LABEL = {"rule": "Rule (RMSSD/RR)", "logreg": "Logistic regression", "lgbm": "Gradient boosting", "cnn": "1-D CNN"}
COLOR = {"rule": "#2a78d6", "logreg": "#eb6834", "lgbm": "#1baf7a", "cnn": "#eda100"}
MARKER = {"rule": "o", "logreg": "s", "lgbm": "^", "cnn": "D"}
NOISE_LABEL = {"bw": "Baseline wander", "ma": "Muscle artifact", "em": "Electrode motion"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _cond_from_key(key: str) -> Condition:
    parts = key.split("__")
    if parts[1] == "clean":
        return Condition(parts[0])
    return Condition(parts[0], parts[1], float(parts[2].replace("dB", "")), int(parts[3][1:]))

MULTI_W = 15 / 2.54     # 15 cm
SINGLE_W = 8.5 / 2.54   # 8.5 cm (not used: every figure here has more than one panel)
PAD = 0.02              # inches around the tight bounding box
DPI = 600

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.25, "lines.markersize": 4, "grid.linewidth": 0.5,
    "axes.unicode_minus": True, "figure.dpi": 100,
    # recessive grid and axes, as in the report figures
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/local.yaml")
cfg = load_config(ap.parse_args().config)
OUT = ROOT / "paper" / "submission"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- data (unchanged)
class Saved:
    """The parts of the report's state the figures need, read from the saved files."""

    def __init__(self, cfg):
        self.paths = Paths.from_cfg(cfg)
        self.pred = pd.concat([pd.read_parquet(self.paths.predictions / f"binary__standard__{m}.parquet")
                               for m in ("lgbm", "cnn")], ignore_index=True)
        self.conds = list(dict.fromkeys(self.pred.condition))
        self.models = MODELS

    def _matrix(self, variant, model, cond):  # same as Reporter._matrix
        d = self.pred[(self.pred.variant == variant) & (self.pred.model == model) & (self.pred.condition == cond)]
        P = d.pivot(index="seed", columns="record_id", values="prob")
        first = d[d.seed == P.index[0]].set_index("record_id").loc[P.columns]
        return first, P.to_numpy(), d.groupby("seed").threshold.first().loc[P.index].to_numpy()


r = Saved(cfg)
T = r.paths.tables
long = pd.read_csv(T / "metrics_all_conditions.csv")
L = long[long.dataset == "internal"]
noise_tab = pd.read_csv(T / "noise_degradation.csv")
beats = pd.read_csv(T / "beat_detection.csv")
abst = pd.read_csv(T / "abstention_curve.csv")
shap_df = pd.read_csv(T / "shap_importance.csv")
models = r.models
types = [t for t in ["bw", "ma", "em"] if any(f"__{t}__" in c for c in r.conds)]
snrs = sorted({_cond_from_key(c).snr_db for c in r.conds if c.startswith("internal") and "clean" not in c}, reverse=True)


def snr_label(s: float) -> str:
    return f"{s:g}".replace("-", "−")


def snr_axis(ax):
    ax.set_xticks(range(len(snrs) + 1))
    ax.set_xticklabels(["clean"] + [snr_label(s) for s in snrs])
    ax.set_xlabel("SNR (dB)  →  noisier")


def series(variant, model, metric, noise):
    d = L[(L.variant == variant) & (L.model == model)]
    rows = [d[d.noise == "none"].iloc[0]] + [d[(d.noise == noise) & (d.snr_db == s)].iloc[0] for s in snrs]
    return (np.array([x[metric] for x in rows]), np.array([x[f"{metric}_lo"] for x in rows]),
            np.array([x[f"{metric}_hi"] for x in rows]))


def model_handles():
    return [Line2D([], [], color=COLOR[m], marker=MARKER[m], label=LABEL[m]) for m in models]


# ----------------------------------------------------------------------------- figures
def fig1(w, h):
    """AUROC vs SNR per noise type; one shared legend below, in one row."""
    fig, axes = plt.subplots(1, len(types), figsize=(w, h), sharey=True, layout="constrained")
    for k, (ax, t) in enumerate(zip(axes, types)):
        for m in models:
            y, lo, hi = series("standard", m, "auroc", t)
            x = np.arange(len(y))
            ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.12, linewidth=0)
            ax.plot(x, y, color=COLOR[m], marker=MARKER[m], label=LABEL[m])
        ax.set_title(f"({'abc'[k]}) {NOISE_LABEL[t]}")
        snr_axis(ax)
    axes[0].set_ylabel("AUROC")
    fig.legend(handles=model_handles(), loc="outside lower center", ncol=len(models), frameon=False)
    return fig


def fig2(w, h):
    """(a) R-peak detection F1 vs SNR; (b) classifier AUROC vs beat-detection F1."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w, h), layout="constrained")
    B = beats[beats.dataset == "internal"]
    ncol = {"bw": "#6250d6", "ma": "#e34948", "em": "#008300"}
    nmark = {"bw": "o", "ma": "s", "em": "^"}
    for t in types:
        d = B[B.noise == t].set_index("snr_db").loc[snrs]
        a1.plot(range(len(snrs) + 1), np.r_[1.0, d.f1.to_numpy()], color=ncol[t], marker=nmark[t], label=NOISE_LABEL[t])
    snr_axis(a1)
    a1.set_ylabel("R-peak detection F1")
    a1.legend(loc="lower left", frameon=False)
    a1.set_title("(a)")
    N = noise_tab[(noise_tab.variant == "standard") & (noise_tab.dataset == "internal")]
    Bk = B.set_index(["noise", "snr_db"])
    for m in models:
        d = N[N.model == m]
        xs = [Bk.loc[(row.noise, row.snr_db), "f1"] for row in d.itertuples()]
        a2.scatter(xs, d.auroc, color=COLOR[m], marker=MARKER[m], s=16, label=LABEL[m], edgecolor="white", linewidth=0.6)
    a2.set_xlabel("R-peak detection F1 in that condition")
    a2.set_ylabel("AUROC")
    a2.legend(loc="lower right", frameon=False)
    a2.set_title("(b)")
    return fig


def fig3(w, h):
    """AUROC with standard (solid) vs noise-augmented (dashed) training; legend below."""
    fig, axes = plt.subplots(1, len(types), figsize=(w, h), sharey=True, layout="constrained")
    for k, (ax, t) in enumerate(zip(axes, types)):
        for m in models:
            for v, ls in [("standard", "-"), ("augmented", "--")]:
                y, _, _ = series(v, m, "auroc", t)
                ax.plot(range(len(y)), y, ls, color=COLOR[m], marker=MARKER[m])
        ax.set_title(f"({'abc'[k]}) {NOISE_LABEL[t]}")
        snr_axis(ax)
    axes[0].set_ylabel("AUROC")
    handles = model_handles() + [Line2D([], [], color=INK, ls="-", label="Standard training"),
                                 Line2D([], [], color=INK, ls="--", label="Noise-augmented training")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    return fig


def fig4(w, h):
    """F1 on retained recordings vs fraction abstained: clean and electrode motion at -6 dB."""
    panels = [("internal__clean", "Clean"), ("internal__em__-6dB__r0", "Electrode motion, −6 dB")]
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True, layout="constrained")
    for k, (ax, (c, title)) in enumerate(zip(axes, panels)):
        for m in models:
            d = abst[(abst.variant == "standard") & (abst.model == m) & (abst.condition == c)]
            ax.plot(1 - d.coverage, d.f1, color=COLOR[m], marker=MARKER[m], label=LABEL[m])
        ax.set_title(f"({'ab'[k]}) {title}")
        ax.set_xlabel("Fraction abstained (lowest quality first)")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("F1 on retained recordings")
    fig.legend(handles=model_handles(), loc="outside lower center", ncol=len(models), frameon=False)
    return fig


def fig5(w, h):
    """Reliability curves for gradient boosting and the CNN: clean vs electrode motion 12 and 0 dB."""
    em_snrs = [s for s in snrs if s in (12.0, 0.0)] or snrs[-1:]
    cs = ["internal__clean"] + [c for c in r.conds if c.startswith("internal__em__") and _cond_from_key(c).snr_db in em_snrs]
    show = [m for m in ["lgbm", "cnn"] if m in models]
    # square panels placed explicitly (inches), so the figure width is predictable
    left, gap, right, bottom, top = 0.48, 0.28, 0.04, 0.40, 0.24
    side = (w - left - gap - right) / len(show)
    fig = plt.figure(figsize=(w, side + bottom + top))
    axes = []
    for k in range(len(show)):
        x0 = left + k * (side + gap)
        axes.append(fig.add_axes([x0 / w, bottom / (side + bottom + top), side / w, side / (side + bottom + top)],
                                 sharey=axes[0] if axes else None))
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    shades = ["#0b0b0b", "#52514e", "#a3a29c"]
    for k, (ax, m) in enumerate(zip(axes, show)):
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
        for c, col in zip(cs, shades):
            cur = cal_curves[(m, c)]
            cond = _cond_from_key(c)
            ax.plot(cur["curve_pred"], cur["curve_obs"], color=col, marker="o",
                    label="clean" if cond.is_clean else f"EM {snr_label(cond.snr_db)} dB")
        ax.set_title(f"({'ab'[k]}) {LABEL[m]}")
        ax.set_xlabel("Predicted AF probability")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed AF fraction")
    axes[-1].legend(loc="upper left", frameon=False)
    return fig


def fig6(w, h):
    """(a) Gradient-boosting SHAP importance on clean data; (b) importance across electrode-motion SNR."""
    S = shap_df[shap_df.variant == "standard"]
    clean = S[S.condition == "internal__clean"].sort_values("mean_abs_shap_share", ascending=False).head(15)
    em_c = ["internal__clean"] + [c for c in r.conds if c.startswith("internal__em__")]
    H = S[S.condition.isin(em_c)].pivot(index="feature", columns="condition", values="mean_abs_shap_share")
    H = H.loc[clean.feature, em_c]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w, h), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]})
    a1.barh(clean.feature[::-1], clean.mean_abs_shap_share[::-1], color=COLOR["lgbm"], height=0.7)
    a1.set_xlabel("Share of mean |SHAP|")
    a1.set_title("(a)")
    a1.grid(axis="y", visible=False)
    im = a2.imshow(H.to_numpy(), aspect="auto", cmap="Greens")
    a2.set_yticks(range(len(H)))
    a2.set_yticklabels(H.index)
    a2.set_xticks(range(len(em_c)))
    a2.set_xticklabels(["clean"] + [snr_label(_cond_from_key(c).snr_db) for c in em_c[1:]])
    a2.set_xlabel("Electrode motion SNR (dB)")
    a2.grid(False)
    a2.set_title("(b)")
    fig.colorbar(im, ax=a2, fraction=0.05, label="Share of mean |SHAP|")
    return fig


# calibration curves come from the saved predictions (same computation as the report)
cal_curves = {}
for m in ("lgbm", "cnn"):
    for c in r.conds:
        if c == "internal__clean" or c.startswith("internal__em__"):
            first, probs, _ = r._matrix("standard", m, c)
            cal_curves[(m, c)] = calibration_summary(first.y.to_numpy(), probs.mean(0), cfg["evaluation"]["calibration_bins"])


# ----------------------------------------------------------------------------- export
def tight_width(fig) -> float:
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD


def export(name: str, draw, target_w: float, height: float, fit: str = "width"):
    """Resize the canvas until the tightly cropped figure is exactly target_w wide.

    fit="width" rescales the canvas width; fit="height" rescales the height instead, for
    figures whose panels have a fixed aspect ratio (their width follows the height)."""
    w, h = target_w, height
    for _ in range(12):
        fig = draw(w, h)
        actual = tight_width(fig)
        if abs(actual - target_w) < 0.003:
            break
        plt.close(fig)
        if fit == "width":
            w *= target_w / actual
        else:
            h *= target_w / actual
    for ext, kw in [("pdf", {}), ("png", {"dpi": DPI})]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=PAD, facecolor="white", **kw)
    plt.close(fig)


for name, draw, h, fit in [("figure1", fig1, 2.45, "width"), ("figure2", fig2, 2.5, "width"),
                           ("figure3", fig3, 2.6, "width"), ("figure4", fig4, 2.5, "width"),
                           ("figure5", fig5, 3.0, "width"), ("figure6", fig6, 3.6, "width")]:
    export(name, draw, MULTI_W, h, fit)
    print(f"wrote {OUT / name}.pdf and .png")

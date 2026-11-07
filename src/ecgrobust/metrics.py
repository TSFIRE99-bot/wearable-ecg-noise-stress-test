"""Metrics, bootstrap confidence intervals and paired bootstrap tests.

AUROC and AUPRC are implemented with numpy (tie-aware, identical to
scikit-learn's ``roc_auc_score`` / ``average_precision_score``; see tests) so
that thousands of bootstrap replicates stay fast.

Seeds and uncertainty
---------------------
For a condition we hold an array ``P`` of shape (n_seeds, n_records). The point
estimate is the mean of the per-seed metrics, the spread is their SD, and the
95% CI comes from resampling *records* (or patients, for PTB-XL) and averaging
the per-seed metrics inside each replicate. The same replicate indices are used
for every model and condition, so differences are paired.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

BINARY_METRICS = ["sensitivity", "specificity", "ppv", "npv", "f1", "accuracy", "auroc", "auprc", "brier"]
LABELS4 = ["N", "A", "O", "~"]


# --------------------------------------------------------------------------- core metrics
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(float)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-np.asarray(s, float), kind="mergesort")
    s_sorted, y_sorted = np.asarray(s, float)[order], y[order]
    thr_idx = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[thr_idx]
    fps = 1 + thr_idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def binary_metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, float)
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    sens, ppv = div(tp, tp + fn), div(tp, tp + fp)
    return {
        "sensitivity": sens,
        "specificity": div(tn, tn + fp),
        "ppv": ppv,
        "npv": div(tn, tn + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "accuracy": div(tp + tn, len(y)),
        "auroc": auroc(y, prob),
        "auprc": average_precision(y, prob),
        "brier": float(np.mean((prob - y) ** 2)),
    }


def challenge_score(y4: np.ndarray, pred4: np.ndarray) -> dict:
    """PhysioNet/CinC 2017 scoring. ``score`` is the official final metric,
    mean(F1_N, F1_A, F1_O); ``score4`` also averages the noisy class."""
    y4, pred4 = np.asarray(y4), np.asarray(pred4)
    out = {}
    for k, lab in enumerate(LABELS4):
        t, p = y4 == k, pred4 == k
        denom = t.sum() + p.sum()
        out[f"f1_{lab}"] = 2 * (t & p).sum() / denom if denom else np.nan
    out["score"] = float(np.nanmean([out["f1_N"], out["f1_A"], out["f1_O"]]))
    out["score4"] = float(np.nanmean([out[f"f1_{l}"] for l in LABELS4]))
    return out


def best_f1_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """Threshold maximizing F1 (chosen on out-of-fold *training* predictions)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-prob, kind="mergesort")
    p_sorted, y_sorted = prob[order], y[order]
    tps = np.cumsum(y_sorted)
    fps = np.arange(1, len(y) + 1) - tps
    f1 = 2 * tps / (tps + fps + y.sum())
    # only thresholds at distinct values
    last = np.r_[np.diff(p_sorted) != 0, True]
    i = int(np.argmax(np.where(last, f1, -1)))
    return float(p_sorted[i])


# --------------------------------------------------------------------------- calibration
def calibration_summary(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict:
    """ECE (equal-mass bins), calibration intercept (CITL) and slope."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    rows = []
    for k in np.unique(b):
        m = b == k
        rows.append((p[m].mean(), y[m].mean(), m.sum()))
    rows = np.array(rows)
    ece = float(np.sum(rows[:, 2] * np.abs(rows[:, 0] - rows[:, 1])) / len(p))
    lo = np.log(p / (1 - p))
    slope = intercept = np.nan
    if 0 < y.sum() < len(y):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(lo[:, None], y)
        slope = float(lr.coef_[0, 0])
        # calibration-in-the-large: intercept with slope fixed at 1 (one Newton solve)
        a = 0.0
        for _ in range(50):
            q = expit(lo + a)
            g, h = np.sum(y - q), np.sum(q * (1 - q))
            a += g / max(h, 1e-12)
        intercept = float(a)
    return {"ece": ece, "cal_slope": slope, "cal_intercept": intercept,
            "curve_pred": rows[:, 0].tolist(), "curve_obs": rows[:, 1].tolist(), "curve_n": rows[:, 2].tolist()}


# --------------------------------------------------------------------------- bootstrap
def bootstrap_indices(n: int, B: int, seed: int, groups: np.ndarray | None = None) -> list[np.ndarray]:
    """Record-level, or cluster (patient-level) bootstrap when ``groups`` is given."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return [rng.integers(0, n, n) for _ in range(B)]
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([members[g] for g in pick]))
    return out


def seed_metrics(y: np.ndarray, P: np.ndarray, thr: np.ndarray, idx: np.ndarray | None = None) -> dict:
    """Mean of per-seed metrics. P: (n_seeds, n). thr: (n_seeds,)."""
    if idx is not None:
        y, P = y[idx], P[:, idx]
    per = [binary_metrics(y, P[s], thr[s]) for s in range(P.shape[0])]
    return {k: float(np.nanmean([d[k] for d in per])) for k in BINARY_METRICS}


def summarize_condition(y, P, thr, boots, alpha=0.05) -> dict:
    """Point estimate (mean over seeds), SD over seeds, bootstrap CI of the seed mean."""
    y, P, thr = np.asarray(y), np.asarray(P), np.asarray(thr)
    per = [binary_metrics(y, P[s], thr[s]) for s in range(P.shape[0])]
    reps = np.array([[seed_metrics(y, P, thr, ix)[k] for k in BINARY_METRICS] for ix in boots])
    out = {}
    for j, k in enumerate(BINARY_METRICS):
        vals = np.array([d[k] for d in per], float)
        out[k] = float(np.nanmean(vals))
        out[f"{k}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f"{k}_lo"], out[f"{k}_hi"] = np.nanpercentile(reps[:, j], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out["_reps"] = reps  # kept for paired comparisons
    return out


def paired_difference(reps_a: np.ndarray, reps_b: np.ndarray, est_a: float, est_b: float) -> dict:
    """Paired bootstrap for metric(A) - metric(B) using shared replicate indices.
    Two-sided p-value from the percentile of 0 in the replicate distribution."""
    d = reps_a - reps_b
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": est_a - est_b, "lo": float(lo), "hi": float(hi), "p": float(min(max(p, 1 / len(d)), 1.0))}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()

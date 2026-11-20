"""Additional noise experiments on the internal test set (models are not retrained).

    python scripts/internal_revision.py --config configs/local.yaml realizations
    python scripts/internal_revision.py --config configs/local.yaml altnoise
    python scripts/internal_revision.py --config configs/local.yaml oracle
    python scripts/internal_revision.py --config configs/local.yaml regional
    python scripts/internal_revision.py --config configs/local.yaml cnn-threshold

realizations  Four more independent noise draws (realizations 1-4) for muscle artifact and electrode motion at
              every SNR, and two for baseline wander at 0 and -6 dB (CNN: MA and EM at 0 and -6 dB only). Realization 0
              is the main analysis.
altnoise      Two alternative noise generators for EM and MA at 6, 0 and -6 dB:
              * "qrsamp": the SNR is set relative to the QRS amplitude instead of the signal power. The signal
                level is the median peak-to-peak amplitude A of the beats detected on the clean signal, taken as
                the power of a sinusoid with that peak-to-peak amplitude (A^2/8).
              * "burst": noise is present only in bursts of 2-5 s (raised-cosine edges) that cover about 40% of
                the recording, at the stated SNR within the bursts.
oracle        Features of EM- and MA-corrupted recordings computed with the R peaks detected on the clean signal
              instead of the peaks detected on the corrupted signal. Everything else in the features (P waves,
              QRS morphology, quality) is still computed from the corrupted signal. This isolates the part of the
              feature models' failure that is caused by R-peak detection errors.
regional      Electrode-motion noise (same gain as the full corruption at 0 and -6 dB) confined to the QRS
              complexes (+-50 ms around each clean-signal R peak) or to everything outside them (P waves,
              T waves, baseline). An exploratory probe of which parts of the waveform matter; confining noise
              creates signals that never occur in practice, so it is interpreted with care.
cnn-threshold For the CNN: a noise-specific decision threshold chosen on the CNN's own calibration recordings
              (training records not used to fit its weights), corrupted with *training* noise at the same
              condition, then applied to the test set. This asks whether the missed AF is a threshold shift
              that a noise-aware device could correct. It needs the noise condition to be known.

All outputs go to outputs/revision/: features/<key>.parquet and predictions/<experiment>__<variant>__<model>.parquet
with one row per record x seed x condition (columns: record_id, y, seed, condition, noise, snr_db, realization,
strategy, prob, threshold, rr_n_beats, beat_fp, beat_fn, beat_offset_ms).
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrobust import models as M  # noqa: E402
from ecgrobust.data import load_dataset  # noqa: E402
from ecgrobust.features import extract_many  # noqa: E402
from ecgrobust.metrics import best_f1_threshold  # noqa: E402
from ecgrobust.noise import Condition, NoiseBank, add_noise_at_snr, corrupt_for_test  # noqa: E402
from ecgrobust.pipeline import binary_ids  # noqa: E402
from ecgrobust.splits import load_split  # noqa: E402
from ecgrobust.utils import Paths, load_config, load_pickle, setup_logging, stable_seed  # noqa: E402

import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) < (1, 8):   # models pickled with scikit-learn 1.9
    LogisticRegression.multi_class = "deprecated"

LOG = logging.getLogger("internal_revision")
MODELS, VARIANTS = ["rule", "logreg", "lgbm", "cnn"], ["standard", "augmented"]
FEATURE_MODELS = ["rule", "logreg", "lgbm"]


class Ctx:
    def __init__(self, cfg: dict):
        self.cfg, self.paths = cfg, Paths.from_cfg(cfg)
        self.out = self.paths.root / "revision"
        (self.out / "features").mkdir(parents=True, exist_ok=True)
        (self.out / "predictions").mkdir(parents=True, exist_ok=True)
        self.internal = load_dataset(cfg, "internal")
        self.meta = self.internal["meta"]
        self.ids = binary_ids(self.meta, load_split(cfg)["test"])
        self.y = self.meta.loc[self.ids, "af"].to_numpy(int)
        self.bank = NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])
        self.ref = load_pickle(self.paths.features / "internal__refpeaks.pkl")
        self.fs, self.seed = cfg["fs"], cfg["noise"]["seed"]
        self._bundles: dict = {}

    def bundles(self, variant: str, model: str) -> list:
        k = (variant, model)
        if k not in self._bundles:
            bs = [load_pickle(self.paths.models / "binary" / variant / f"{model}_seed{s}.pkl") for s in self.cfg["seeds"]]
            cms = [M.load_cnn(b, self.cfg) for b in bs] if model == "cnn" else [None] * len(bs)
            self._bundles[k] = list(zip(bs, cms))
        return self._bundles[k]

    def signals(self, cond: Condition) -> dict:
        return {r: corrupt_for_test(self.internal["signals"][r], r, cond, self.bank, self.seed) for r in self.ids}

    def features(self, key: str, sigs: dict, peaks: dict | None = None) -> pd.DataFrame:
        f = self.out / "features" / f"{key}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        df = extract_many(sigs, self.fs, self.cfg["features"], ref_peaks={r: self.ref[r] for r in sigs},
                          peaks=peaks, desc=key)
        df.to_parquet(f)
        return df

    def score(self, variant: str, model: str, feats: pd.DataFrame | None, sigs: dict | None) -> list[tuple]:
        """[(seed, prob, threshold)] for one condition."""
        res = []
        sig_list = [sigs[r] for r in self.ids] if model == "cnn" else None
        X = feats.loc[self.ids] if feats is not None else None
        for b, cm in self.bundles(variant, model):
            raw = M.raw_scores(b, X=X, signals=sig_list, cfg=self.cfg, cnn_model=cm)
            res.append((b["seed"], b["calibrator"].transform(raw), b["threshold"]))
        return res


def frame(ctx: Ctx, variant: str, model: str, scored: list[tuple], info: dict, feats: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for seed, prob, thr in scored:
        df = pd.DataFrame({"record_id": ctx.ids, "y": ctx.y, "seed": seed, "prob": prob, "threshold": thr})
        for k, v in info.items():
            df[k] = v
        df["variant"], df["model"] = variant, model
        if feats is not None:
            for c in ["rr_n_beats", "beat_fp", "beat_fn", "beat_offset_ms", "q_avgqrs"]:
                if c in feats:
                    df[c] = feats.loc[ctx.ids, c].to_numpy(float)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def run_experiment(ctx: Ctx, name: str, conditions: list[dict], models: list[str]) -> None:
    """conditions: dicts with key, info (columns to store), make (-> signals dict), peaks (optional)."""
    for variant in VARIANTS:
        for model in models:
            out = ctx.out / "predictions" / f"{name}__{variant}__{model}.parquet"
            old = pd.read_parquet(out) if out.exists() else None
            done = set(old.condition) if old is not None else set()
            parts = [] if old is None else [old]
            for c in conditions:
                if c["key"] in done:
                    continue
                sigs = c["make"]()
                feats = ctx.features(c["key"], sigs, c.get("peaks"))
                parts.append(frame(ctx, variant, model, ctx.score(variant, model, feats, sigs),
                                   {"condition": c["key"], **c["info"]}, feats))
                LOG.info("%s: scored %s/%s on %s", name, variant, model, c["key"])
                pd.concat(parts, ignore_index=True).to_parquet(out)   # save as we go (resumable)


# --------------------------------------------------------------------------- realizations
def realizations(ctx: Ctx) -> None:
    grid = [(n, s, r) for n in ("ma", "em") for s in (24.0, 18.0, 12.0, 6.0, 0.0, -6.0) for r in (1, 2, 3, 4)]
    grid += [("bw", s, r) for s in (0.0, -6.0) for r in (1, 2)]
    conds = []
    for n, s, r in grid:
        cond = Condition("internal", n, s, r)
        conds.append({"key": cond.key, "info": {"noise": n, "snr_db": s, "realization": r, "strategy": "additive"},
                      "make": (lambda cond=cond: ctx.signals(cond))})
    run_experiment(ctx, "realizations", conds, [m for m in MODELS if m != "cnn"])
    # CNN scoring is the slow part on a CPU: its extra draws are limited to muscle artifact and electrode motion at
    # 0 and -6 dB, where it degrades and where the primary comparisons and the variance analysis need them
    run_experiment(ctx, "realizations", [c for c in conds if c["info"]["noise"] in ("ma", "em")
                                          and c["info"]["snr_db"] in (0.0, -6.0)], ["cnn"])


# --------------------------------------------------------------------------- alternative noise generators
def qrs_amplitude(sig: np.ndarray, peaks: np.ndarray, fs: int) -> float:
    w = int(0.05 * fs)
    amps = [np.ptp(sig[max(p - w, 0):p + w]) for p in peaks if p - w >= 0 and p + w < len(sig)]
    return float(np.median(amps)) if amps else float(np.ptp(sig))


def corrupt_qrsamp(sig, rid, noise, snr, ctx: Ctx):
    rng = np.random.default_rng(stable_seed(ctx.seed, "test", rid, noise, 0))     # same segment as realization 0
    seg = ctx.bank.segment(noise, len(sig), rng, "test")
    seg = seg - seg.mean()
    a = qrs_amplitude(np.asarray(sig, float) - np.mean(sig), ctx.ref[rid], ctx.fs)
    ps, pn = a ** 2 / 8, np.mean(seg ** 2)
    if ps <= 0 or pn <= 0:
        return sig
    return (np.asarray(sig, float) + math.sqrt(ps / (pn * 10 ** (snr / 10))) * seg).astype(np.float32)


def burst_mask(n: int, fs: int, rng: np.random.Generator, cover: float = 0.4) -> np.ndarray:
    m = np.zeros(n)
    ramp = int(0.2 * fs)
    target = cover * n
    tries = 0
    while m.sum() < target and tries < 200:
        tries += 1
        L = int(rng.uniform(2, 5) * fs)
        if L >= n:
            L = n // 2
        start = int(rng.integers(0, n - L))
        win = np.ones(L)
        r = min(ramp, L // 2)
        if r > 0:
            edge = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
            win[:r], win[-r:] = edge, edge[::-1]
        m[start:start + L] = np.maximum(m[start:start + L], win)
    return m


def corrupt_burst(sig, rid, noise, snr, ctx: Ctx):
    rng = np.random.default_rng(stable_seed(ctx.seed, "burst", rid, noise))
    seg = ctx.bank.segment(noise, len(sig), rng, "test")
    seg = seg - seg.mean()
    mask = burst_mask(len(sig), ctx.fs, rng)
    s = np.asarray(sig, float)
    ps = np.mean((s - s.mean()) ** 2)
    on = mask > 0.5
    pn = np.mean(seg[on] ** 2) if on.any() else 0
    if ps <= 0 or pn <= 0:
        return sig
    return (s + math.sqrt(ps / (pn * 10 ** (snr / 10))) * seg * mask).astype(np.float32)


def altnoise(ctx: Ctx) -> None:
    conds = []
    for strat, fn in [("qrsamp", corrupt_qrsamp), ("burst", corrupt_burst)]:
        for n in ("ma", "em"):
            for s in (6.0, 0.0, -6.0):
                key = f"internal__{n}__{s:+g}dB__{strat}"
                conds.append({"key": key, "info": {"noise": n, "snr_db": s, "realization": 0, "strategy": strat},
                              "make": (lambda fn=fn, n=n, s=s: {r: fn(ctx.internal["signals"][r], r, n, s, ctx) for r in ctx.ids})})
    run_experiment(ctx, "altnoise", conds, MODELS)


# --------------------------------------------------------------------------- oracle R peaks
def oracle(ctx: Ctx) -> None:
    conds = []
    for n in ("ma", "em"):
        for s in (12.0, 6.0, 0.0, -6.0):
            cond = Condition("internal", n, s, 0)
            conds.append({"key": cond.key + "__oraclepeaks",
                          "info": {"noise": n, "snr_db": s, "realization": 0, "strategy": "oracle_peaks"},
                          "make": (lambda cond=cond: ctx.signals(cond)),
                          "peaks": {r: ctx.ref[r] for r in ctx.ids}})
    run_experiment(ctx, "oracle", conds, FEATURE_MODELS)


# --------------------------------------------------------------------------- regional corruption
def region_mask(n: int, peaks: np.ndarray, fs: int, half_ms: float = 50, ramp_ms: float = 20) -> np.ndarray:
    m = np.zeros(n)
    h, r = int(half_ms * fs / 1000), max(int(ramp_ms * fs / 1000), 1)
    edge = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
    for p in peaks:
        lo, hi = max(p - h, 0), min(p + h, n)
        m[lo:hi] = 1
        a0 = max(lo - r, 0)
        m[a0:lo] = np.maximum(m[a0:lo], edge[r - (lo - a0):])
        b1 = min(hi + r, n)
        m[hi:b1] = np.maximum(m[hi:b1], edge[::-1][:b1 - hi])
    return m


def regional(ctx: Ctx) -> None:
    conds = []
    for s in (0.0, -6.0):
        cond = Condition("internal", "em", s, 0)
        for region in ("qrs", "nonqrs"):
            def make(cond=cond, region=region):
                out = {}
                for r in ctx.ids:
                    x = ctx.internal["signals"][r]
                    full = corrupt_for_test(x, r, cond, ctx.bank, ctx.seed)
                    m = region_mask(len(x), ctx.ref[r], ctx.fs)
                    m = m if region == "qrs" else 1 - m
                    out[r] = (x + m * (full - x)).astype(np.float32)
                return out
            conds.append({"key": f"{cond.key}__{region}only", "info": {"noise": "em", "snr_db": s, "realization": 0,
                                                                     "strategy": f"{region}_only"}, "make": make})
    run_experiment(ctx, "regional", conds, MODELS)


# --------------------------------------------------------------------------- noise-specific CNN threshold
def cnn_threshold(ctx: Ctx) -> None:
    rows = []
    train_sig = ctx.internal["signals"]
    for variant in VARIANTS:
        for b, cm in ctx.bundles(variant, "cnn"):
            cal = [i for i in map(str, b["oof"].index) if "#aug" not in i]
            ycal = ctx.meta.loc[cal, "af"].to_numpy(int)
            for n in ("ma", "em"):
                for s in (6.0, 0.0, -6.0):
                    sig = []
                    for r in cal:
                        rng = np.random.default_rng(stable_seed(ctx.seed, "calib-noise", r, n, s))
                        seg = ctx.bank.segment(n, len(train_sig[r]), rng, "train")      # training noise only
                        sig.append(add_noise_at_snr(train_sig[r], seg, s))
                    raw = M.raw_scores(b, signals=sig, cfg=ctx.cfg, cnn_model=cm)
                    prob = b["calibrator"].transform(raw)
                    rows.append({"variant": variant, "seed": b["seed"], "noise": n, "snr_db": s,
                                 "threshold_default": b["threshold"], "threshold_noise": best_f1_threshold(ycal, prob),
                                 "n_calibration": len(cal), "n_calibration_af": int(ycal.sum())})
                    LOG.info("cnn-threshold %s seed %d %s %+g dB: %.3f -> %.3f", variant, b["seed"], n, s,
                             b["threshold"], rows[-1]["threshold_noise"])
    pd.DataFrame(rows).to_csv(ctx.out / "cnn_noise_thresholds.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("what", choices=["realizations", "altnoise", "oracle", "regional", "cnn-threshold", "all"])
    a = ap.parse_args()
    setup_logging()
    ctx = Ctx(load_config(a.config))
    steps = {"realizations": realizations, "altnoise": altnoise, "oracle": oracle, "regional": regional,
             "cnn-threshold": cnn_threshold}
    for name, fn in steps.items():
        if a.what in (name, "all"):
            LOG.info("=== %s", name)
            fn(ctx)


if __name__ == "__main__":
    main()

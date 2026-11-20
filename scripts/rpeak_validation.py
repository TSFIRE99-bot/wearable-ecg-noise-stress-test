"""R-peak detection under real noise, scored against expert beat annotations.

    python scripts/rpeak_validation.py --config configs/local.yaml [--n-jobs 7]

The main study measures beat detection on CinC 2017 against the detections made on the clean signal
(a pseudo-reference), because CinC has no beat annotations. This script checks that assumption on the
MIT-BIH Arrhythmia Database (mitdb), whose beats were annotated by cardiologists:

* lead MLII of the 44 non-paced records (102, 104, 107 and 217 are excluded, as in ANSI/AAMI EC57),
  resampled from 360 to 300 Hz and cut into non-overlapping 30 s windows;
* each window is corrupted with the study's NSTDB *test* noise (same NoiseBank, same seeding rule, same
  power-based SNR) for baseline wander, muscle artifact and electrode motion at 12, 6, 0 and -6 dB;
* seven NeuroKit2 R-peak detectors are run on every clean and corrupted window;
* detections are matched to the expert beats within +-150 ms (EC57), excluding 0.3 s at each window edge.

For the NeuroKit2 detector used by the AF pipeline it also reports agreement with its own clean-signal
detections, so the pseudo-reference used in the paper can be compared with the expert reference.

Each window's rhythm (majority of samples) is taken from the expert rhythm annotations: normal sinus
rhythm, AF, or other.

Outputs
  outputs/tables/rpeak_validation_windows.parquet   one row per window x condition x detector
  outputs/tables/rpeak_validation.csv               totals by condition x detector x rhythm, with 95% CIs
                                                    from a record-level bootstrap
  outputs/tables/rpeak_pseudo_reference.csv         pseudo-reference versus expert reference (NeuroKit2)
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrobust.data import load_dataset  # noqa: E402
from ecgrobust.noise import Condition, NoiseBank, corrupt_for_test  # noqa: E402
from ecgrobust.utils import Paths, load_config, setup_logging  # noqa: E402

LOG = logging.getLogger("rpeak_validation")
PACED = {"102", "104", "107", "217"}
BEATS = set("NLRBAaJSVrFejnE/fQ")          # beat annotation symbols (non-beat marks are ignored)
DETECTORS = ["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010", "engzeemod2012", "kalidas2017",
             "rodrigues2021"]
CONDITIONS = [("none", np.inf)] + [(n, s) for n in ("bw", "ma", "em") for s in (12.0, 6.0, 0.0, -6.0)]
WIN_S, EDGE_S, TOL_MS = 30, 0.3, 150


def match(ref: np.ndarray, det: np.ndarray, tol: int) -> tuple[int, int, int, float]:
    """Greedy one-to-one matching within +-tol samples: TP, FP, FN and the summed absolute offset of matches."""
    ref, det = np.sort(ref), np.sort(det)
    i = j = tp = 0
    off = 0.0
    while i < len(ref) and j < len(det):
        d = det[j] - ref[i]
        if abs(d) <= tol:
            tp += 1
            off += abs(d)
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return tp, len(det) - tp, len(ref) - tp, off


def detect(sig: np.ndarray, fs: int, method: str) -> np.ndarray:
    import neurokit2 as nk

    warnings.filterwarnings("ignore")
    try:
        try:
            sig, _ = nk.ecg_invert(sig, sampling_rate=fs)
        except Exception:
            pass
        try:
            clean = nk.ecg_clean(sig, sampling_rate=fs, method=method)
        except Exception:
            clean = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method=method, correct_artifacts=False)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def load_windows(mitdb_dir: Path, fs: int) -> list[dict]:
    import wfdb

    out = []
    recs = sorted(p.stem for p in mitdb_dir.glob("*.hea"))
    for rec in recs:
        if rec in PACED:
            continue
        r = wfdb.rdrecord(str(mitdb_dir / rec))
        a = wfdb.rdann(str(mitdb_dir / rec), "atr")
        x = resample_poly(r.p_signal[:, r.sig_name.index("MLII")].astype(np.float64), 5, 6).astype(np.float32)
        scale = fs / r.fs
        beats = np.array([round(s * scale) for s, sym in zip(a.sample, a.symbol) if sym in BEATS], dtype=int)
        # rhythm per sample from the '+' annotations
        rh = np.full(len(x), "", dtype=object)
        cur, start = "(N", 0
        for s, sym, aux in zip(a.sample, a.symbol, a.aux_note):
            if sym == "+" and aux.strip():
                rh[start:round(s * scale)] = cur
                cur, start = aux.strip().rstrip("\x00"), round(s * scale)
        rh[start:] = cur
        n = WIN_S * fs
        for k in range(len(x) // n):
            lo, hi = k * n, (k + 1) * n
            vals, cnt = np.unique(rh[lo:hi], return_counts=True)
            major = vals[np.argmax(cnt)]
            rhythm = "N" if major == "(N" else "AF" if major == "(AFIB" else "other"
            b = beats[(beats >= lo) & (beats < hi)] - lo
            out.append({"record": rec, "window": k, "id": f"mitdb_{rec}_{k:03d}", "rhythm": rhythm,
                        "signal": x[lo:hi], "beats": b})
    return out


def score_window(w: dict, bank: NoiseBank, base_seed: int, fs: int) -> list[dict]:
    tol, edge, n = round(TOL_MS * fs / 1000), round(EDGE_S * fs), WIN_S * fs
    inside = lambda p: p[(p >= edge) & (p < n - edge)]  # noqa: E731
    ref = inside(w["beats"])
    rows, clean_nk = [], None
    for noise, snr in CONDITIONS:
        cond = Condition("mitdb", noise, snr, 0)
        sig = w["signal"] if noise == "none" else corrupt_for_test(w["signal"], w["id"], cond, bank, base_seed)
        for det in DETECTORS:
            pk = inside(detect(sig, fs, det))
            tp, fp, fn, off = match(ref, pk, tol)
            row = {"record": w["record"], "window": w["window"], "rhythm": w["rhythm"], "noise": noise,
                   "snr_db": snr, "detector": det, "n_ref": len(ref), "tp": tp, "fp": fp, "fn": fn,
                   "abs_offset_ms": off / fs * 1000}
            if det == "neurokit":
                if noise == "none":
                    clean_nk = pk
                ptp, pfp, pfn, _ = match(clean_nk, pk, tol)
                row.update(pseudo_tp=ptp, pseudo_fp=pfp, pseudo_fn=pfn, pseudo_n_ref=len(clean_nk))
            rows.append(row)
    return rows


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan


def summarize(df: pd.DataFrame, B: int = 1000, seed: int = 99) -> pd.DataFrame:
    """Totals by condition x detector x rhythm; record-level bootstrap CI for the pooled F1."""
    rng = np.random.default_rng(seed)
    recs = np.array(sorted(df.record.unique()))
    picks = [rng.choice(recs, len(recs)) for _ in range(B)]
    rows = []
    for rhythm in ["all", "N", "AF", "other"]:
        d = df if rhythm == "all" else df[df.rhythm == rhythm]
        g = d.groupby(["noise", "snr_db", "detector", "record"])[["tp", "fp", "fn", "abs_offset_ms"]].sum()
        for (noise, snr, det), sub in g.groupby(level=[0, 1, 2]):
            sub = sub.droplevel([0, 1, 2])
            tot = sub.sum()
            boot = []
            for pk in picks:
                s = sub.reindex(pk).dropna().sum()
                boot.append(f1(s.tp, s.fp, s.fn))
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            n_win = int(((d.noise == noise) & (d.snr_db == snr) & (d.detector == det)).sum())
            minutes = n_win * WIN_S / 60
            rows.append({"rhythm": rhythm, "noise": noise, "snr_db": snr, "detector": det, "n_windows": n_win,
                         "n_beats": int(tot.tp + tot.fn), "sensitivity": tot.tp / (tot.tp + tot.fn),
                         "ppv": tot.tp / (tot.tp + tot.fp) if tot.tp + tot.fp else np.nan,
                         "f1": f1(tot.tp, tot.fp, tot.fn), "f1_lo": lo, "f1_hi": hi,
                         "missed_per_min": tot.fn / minutes, "spurious_per_min": tot.fp / minutes,
                         "mean_abs_offset_ms": tot.abs_offset_ms / tot.tp if tot.tp else np.nan})
    return pd.DataFrame(rows)


def pseudo_vs_expert(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.detector == "neurokit"].copy()
    d["f1_expert"] = [f1(a, b, c) for a, b, c in zip(d.tp, d.fp, d.fn)]
    d["f1_pseudo"] = [f1(a, b, c) for a, b, c in zip(d.pseudo_tp, d.pseudo_fp, d.pseudo_fn)]
    rows = []
    for (noise, snr), g in d.groupby(["noise", "snr_db"]):
        e = (g.tp.sum(), g.fp.sum(), g.fn.sum())
        p = (g.pseudo_tp.sum(), g.pseudo_fp.sum(), g.pseudo_fn.sum())
        rows.append({"noise": noise, "snr_db": snr, "f1_expert": f1(*e), "f1_pseudo": f1(*p),
                     "f1_pseudo_minus_expert": f1(*p) - f1(*e),
                     "window_f1_corr": float(np.corrcoef(g.f1_expert, g.f1_pseudo)[0, 1]) if g.f1_expert.std() > 0 else np.nan,
                     "frac_windows_f1_below_0.9_expert": float((g.f1_expert < 0.9).mean()),
                     "frac_windows_f1_below_0.9_pseudo": float((g.f1_pseudo < 0.9).mean())})
    return pd.DataFrame(rows)


def main():
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mitdb", default=None, help="folder with the mitdb records (default: <raw_dir>/mitdb)")
    ap.add_argument("--n-jobs", type=int, default=7)
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    paths = Paths.from_cfg(cfg)
    fs = cfg["fs"]
    mitdb = Path(a.mitdb or Path(cfg["paths"]["raw_dir"]) / "mitdb")
    bank = NoiseBank(load_dataset(cfg, "nstdb"), cfg["noise"]["train_fraction"])
    wins = load_windows(mitdb, fs)
    LOG.info("mitdb: %d windows from %d records; rhythms %s", len(wins), len({w["record"] for w in wins}),
             pd.Series([w["rhythm"] for w in wins]).value_counts().to_dict())
    out_rows = paths.tables / "rpeak_validation_windows.parquet"
    if out_rows.exists():
        df = pd.read_parquet(out_rows)
    else:
        res = Parallel(n_jobs=a.n_jobs, batch_size=4, verbose=5)(
            delayed(score_window)(w, bank, cfg["noise"]["seed"], fs) for w in wins)
        df = pd.DataFrame([r for rows in res for r in rows])
        df.to_parquet(out_rows)
    summarize(df).to_csv(paths.tables / "rpeak_validation.csv", index=False)
    pseudo_vs_expert(df).to_csv(paths.tables / "rpeak_pseudo_reference.csv", index=False)
    LOG.info("wrote rpeak_validation.csv and rpeak_pseudo_reference.csv")


if __name__ == "__main__":
    main()

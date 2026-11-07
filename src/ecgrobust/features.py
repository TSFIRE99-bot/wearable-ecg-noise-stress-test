"""Hand-crafted AF features, signal-quality indices and beat-detection accuracy.

Feature groups
--------------
* ``rr_*``  beat-timing irregularity (time-domain + nonlinear). No frequency-domain
  HRV: LF/HF are not interpretable on 9-60 s recordings.
* ``p_*``   atrial activity: P-wave presence/consistency from NeuroKit2 DWT
  delineation, plus a delineation-free P-region template correlation.
* ``qrs_*`` ventricular morphology consistency (mainly helps the 4-class task).
* ``q_*``   signal-quality indices. They are not model inputs; they are used for
  abstention and to describe the noise conditions.
* ``beat_*`` R-peak detection agreement with the clean-signal detections
  (only for corrupted conditions).

R-peak artifact correction is turned off because it would replace the irregular
RR intervals that define AF.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

LOG = logging.getLogger(__name__)

RR_FEATURES = [
    "rr_n_beats", "rr_valid_frac", "rr_mean", "rr_median", "rr_hr", "rr_sdnn", "rr_rmssd", "rr_pnn50",
    "rr_pnn20", "rr_cv", "rr_rmssd_norm", "rr_sd1", "rr_sd2", "rr_sd1_sd2", "rr_mad_norm", "rr_iqr_norm",
    "rr_tpr", "rr_sampen", "rr_cosen",
]
P_FEATURES = ["p_present_frac", "p_amp_ratio_mean", "p_amp_cv", "p_pr_sd", "p_pr_cv", "p_region_corr"]
QRS_FEATURES = ["qrs_dur_mean", "qrs_corr_mean", "qrs_corr_min_frac", "qrs_ramp_cv"]
FEATURES = RR_FEATURES + P_FEATURES + QRS_FEATURES
QUALITY = ["q_avgqrs", "q_zhao", "q_ksqi", "q_psqi", "q_bassqi"]
BEAT = ["beat_ref", "beat_tp", "beat_fp", "beat_fn"]


# --------------------------------------------------------------------------- helpers
def sample_entropy(x: np.ndarray, m: int, r: float) -> tuple[float, int]:
    """SampEn and the number of (m+1)-length matches A. NaN if undefined."""
    x = np.asarray(x, float)
    n = len(x)
    if n <= m + 1 or r <= 0:
        return np.nan, 0

    def count(mm: int) -> int:
        # Richman & Moorman: the same N-m template start points for lengths m and m+1
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        return int((np.sum(d <= r) - len(emb)) // 2)

    b, a = count(m), count(m + 1)
    if a == 0 or b == 0:
        return np.nan, a
    return float(-np.log(a / b)), a


def cosen(rr_s: np.ndarray) -> float:
    """Coefficient of sample entropy (Lake & Moorman 2011), designed for short AF series."""
    for r in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        se, a = sample_entropy(rr_s, 1, r)
        if a >= 5 and np.isfinite(se):
            return float(se + np.log(2 * r) - np.log(np.mean(rr_s)))
    return np.nan


def match_beats(ref: np.ndarray, test: np.ndarray, tol: int) -> tuple[int, int, int]:
    """Greedy one-to-one matching within +-tol samples. Returns TP, FP, FN."""
    ref, test = np.sort(ref), np.sort(test)
    i = j = tp = 0
    while i < len(ref) and j < len(test):
        d = test[j] - ref[i]
        if abs(d) <= tol:
            tp += 1
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return tp, len(test) - tp, len(ref) - tp


def _template_corr(sig: np.ndarray, centers: np.ndarray, pre: int, post: int) -> np.ndarray:
    idx = [c for c in centers if c - pre >= 0 and c + post < len(sig)]
    if len(idx) < 3:
        return np.array([])
    segs = np.stack([sig[c - pre:c + post] for c in idx])
    tmpl = np.median(segs, axis=0)
    segs_c = segs - segs.mean(1, keepdims=True)
    t_c = tmpl - tmpl.mean()
    denom = np.linalg.norm(segs_c, axis=1) * np.linalg.norm(t_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (segs_c @ t_c) / denom


def _band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(pxx[m], f[m])) if m.sum() > 1 else 0.0


# --------------------------------------------------------------------------- main
def extract_one(sig: np.ndarray, fs: int, fcfg: dict, ref_peaks: np.ndarray | None = None,
                return_peaks: bool = False) -> dict:
    import neurokit2 as nk

    out: dict = {k: np.nan for k in FEATURES + QUALITY}
    sig = np.nan_to_num(np.asarray(sig, float))
    warnings.filterwarnings("ignore")

    # --- detection-free quality indices (on the raw signal) ---
    try:
        f, pxx = welch(sig - sig.mean(), fs=fs, nperseg=min(len(sig), 4 * fs))
        p5_15, p5_40 = _band_power(f, pxx, 5, 15), _band_power(f, pxx, 5, 40)
        p0_1, p0_40 = _band_power(f, pxx, 0, 1), _band_power(f, pxx, 0, 40)
        out["q_psqi"] = p5_15 / p5_40 if p5_40 > 0 else np.nan
        out["q_bassqi"] = 1 - p0_1 / p0_40 if p0_40 > 0 else np.nan
    except Exception:
        pass

    peaks = np.array([], dtype=int)
    try:
        try:
            sig_inv, _ = nk.ecg_invert(sig, sampling_rate=fs)
        except Exception:
            sig_inv = sig
        clean = nk.ecg_clean(sig_inv, sampling_rate=fs, method="neurokit")
        out["q_ksqi"] = float(kurtosis(clean, fisher=False))
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit", correct_artifacts=False)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception as e:  # catastrophic failure -> all NaN
        LOG.debug("peak detection failed: %s", e)
        clean = sig

    if ref_peaks is not None:
        tp, fp, fn = match_beats(np.asarray(ref_peaks), peaks, int(round(fcfg["beat_match_tolerance_ms"] * fs / 1000)))
        out.update(beat_ref=len(ref_peaks), beat_tp=tp, beat_fp=fp, beat_fn=fn)

    out["rr_n_beats"] = len(peaks)
    if len(peaks) >= fcfg["min_beats"]:
        out.update(_rr_features(peaks, fs, fcfg))
        out.update(_morph_features(clean, peaks, fs))
        try:
            q = nk.ecg_quality(clean, rpeaks=peaks, sampling_rate=fs, method="averageQRS")
            out["q_avgqrs"] = float(np.nanmean(q))
        except Exception:
            pass
        try:
            z = nk.ecg_quality(clean, rpeaks=peaks, sampling_rate=fs, method="zhao2018", approach="fuzzy")
            out["q_zhao"] = {"Unacceptable": 0.0, "Barely acceptable": 1.0, "Excellent": 2.0}.get(z, np.nan)
        except Exception:
            pass
    if return_peaks:
        out["_peaks"] = peaks
    return out


def _rr_features(peaks: np.ndarray, fs: int, fcfg: dict) -> dict:
    rr = np.diff(peaks) / fs * 1000.0
    ok = (rr >= fcfg["rr_min_ms"]) & (rr <= fcfg["rr_max_ms"])
    o = {"rr_valid_frac": float(ok.mean()) if len(rr) else np.nan}
    v = rr[ok]
    if len(v) < 4:
        return o
    # successive differences only between adjacent *valid* intervals
    pair_ok = ok[1:] & ok[:-1]
    d = np.diff(rr)[pair_ok]
    mean = v.mean()
    sdnn = v.std(ddof=1)
    o.update(rr_mean=mean, rr_median=float(np.median(v)), rr_hr=60000.0 / mean, rr_sdnn=sdnn, rr_cv=sdnn / mean)
    if len(d) >= 2:
        rmssd = float(np.sqrt(np.mean(d**2)))
        sd_d = d.std(ddof=1)
        sd1 = np.sqrt(0.5) * sd_d
        sd2 = np.sqrt(max(2 * sdnn**2 - 0.5 * sd_d**2, 0.0))
        o.update(rr_rmssd=rmssd, rr_rmssd_norm=rmssd / mean, rr_pnn50=float(np.mean(np.abs(d) > 50)),
                 rr_pnn20=float(np.mean(np.abs(d) > 20)), rr_sd1=sd1, rr_sd2=sd2,
                 rr_sd1_sd2=sd1 / sd2 if sd2 > 0 else np.nan)
    med = np.median(v)
    o["rr_mad_norm"] = float(np.median(np.abs(v - med)) / med)
    q75, q25 = np.percentile(v, [75, 25])
    o["rr_iqr_norm"] = float((q75 - q25) / med)
    if len(v) >= 3:
        tp = np.sum((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]) | (v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))
        o["rr_tpr"] = float(tp / (len(v) - 2))
    o["rr_sampen"], _ = sample_entropy(v, 2, 0.2 * sdnn)
    o["rr_cosen"] = cosen(v / 1000.0)
    return o


def _morph_features(clean: np.ndarray, peaks: np.ndarray, fs: int) -> dict:
    import neurokit2 as nk

    o: dict = {}
    ms = lambda t: int(round(t * fs / 1000))  # noqa: E731
    # delineation-free P-region consistency: window 250..60 ms before each R peak
    c = _template_corr(clean, peaks, ms(250), -ms(60))
    if len(c):
        o["p_region_corr"] = float(np.nanmean(c))
    # QRS consistency (ectopy / aberrancy)
    c = _template_corr(clean, peaks, ms(100), ms(100))
    if len(c):
        o["qrs_corr_mean"] = float(np.nanmean(c))
        o["qrs_corr_min_frac"] = float(np.mean(c < 0.8))
    r_amp = clean[peaks]
    if np.mean(np.abs(r_amp)) > 0:
        o["qrs_ramp_cv"] = float(np.std(r_amp) / np.mean(np.abs(r_amp)))
    try:
        _, waves = nk.ecg_delineate(clean, peaks, sampling_rate=fs, method="dwt")
        get = lambda k: np.asarray(waves.get(k, [np.nan] * len(peaks)), dtype=float)  # noqa: E731
        p_pk, p_on = get("ECG_P_Peaks"), get("ECG_P_Onsets")
        r_on, r_off = get("ECG_R_Onsets"), get("ECG_R_Offsets")
        n = min(len(peaks), len(p_pk))
        pk, p_pk, p_on = peaks[:n], p_pk[:n], p_on[:n]
        o["p_present_frac"] = float(np.mean(np.isfinite(p_pk)))
        has = np.isfinite(p_pk)
        if has.sum() >= 3:
            ratio = clean[p_pk[has].astype(int)] / np.where(clean[pk[has]] == 0, np.nan, clean[pk[has]])
            o["p_amp_ratio_mean"] = float(np.nanmean(ratio))
            o["p_amp_cv"] = float(np.nanstd(ratio) / (abs(np.nanmean(ratio)) + 1e-9))
        has_on = np.isfinite(p_on)
        if has_on.sum() >= 3:
            pr = (pk[has_on] - p_on[has_on]) / fs * 1000
            o["p_pr_sd"] = float(np.std(pr, ddof=1))
            o["p_pr_cv"] = float(np.std(pr, ddof=1) / np.mean(pr)) if np.mean(pr) > 0 else np.nan
        m = np.isfinite(r_on[:n]) & np.isfinite(r_off[:n])
        if m.sum() >= 3:
            o["qrs_dur_mean"] = float(np.mean((r_off[:n][m] - r_on[:n][m]) / fs * 1000))
    except Exception as e:
        LOG.debug("delineation failed: %s", e)
    return o


def extract_many(signals: dict, fs: int, fcfg: dict, ref_peaks: dict | None = None,
                 return_peaks: bool = False, desc: str = "") -> pd.DataFrame:
    from joblib import Parallel, delayed

    ids = list(signals)
    rows = Parallel(n_jobs=fcfg["n_jobs"], batch_size=16)(
        delayed(extract_one)(signals[r], fs, fcfg, None if ref_peaks is None else ref_peaks[r], return_peaks)
        for r in ids
    )
    df = pd.DataFrame(rows, index=pd.Index(ids, name="record_id"))
    LOG.info("features %-40s n=%d  peak-detection failures=%d", desc, len(df), int((df.rr_n_beats < fcfg["min_beats"]).sum()))
    return df

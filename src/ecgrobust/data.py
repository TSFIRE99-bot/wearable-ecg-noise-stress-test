"""Loaders for the three public datasets, all mapped to one in-memory format.

A processed dataset is a dict::

    {"meta": DataFrame indexed by record_id, "signals": {record_id: float32 array}, "fs": 300}

``meta`` always has columns ``label4`` (N/A/O/~ or 'ext' for PTB-XL), ``af`` (0/1),
``patient_id`` and ``duration_s``. CinC 2017 has no patient identifiers, so its
``patient_id`` is the record id (one record per assumed patient). The paper lists
this as a limitation, and PTB-XL, which has real patient ids, is the external test.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .utils import Paths, load_pickle, save_pickle, stable_seed

LOG = logging.getLogger(__name__)

LABELS4 = ["N", "A", "O", "~"]


# --------------------------------------------------------------------------- CinC 2017
def _find_reference(cinc_dir: Path, name: str) -> Path:
    for cand in [cinc_dir / name, cinc_dir / "training2017" / name,
                 cinc_dir / "training2017" / "REFERENCE.csv", cinc_dir / "REFERENCE.csv"]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No CinC 2017 reference file under {cinc_dir}")


def load_cinc2017(cfg: dict) -> dict:
    import wfdb

    root = Path(cfg["data"]["cinc2017"]["dir"])
    ref = _find_reference(root, cfg["data"]["cinc2017"]["reference"])
    LOG.info("CinC 2017 labels from %s", ref)
    meta = pd.read_csv(ref, header=None, names=["record_id", "label4"])
    rec_dir = root / "training2017" if (root / "training2017").exists() else root
    signals = {}
    for rid in meta.record_id:
        rec = wfdb.rdrecord(str(rec_dir / rid))
        assert rec.fs == 300, f"{rid}: unexpected fs {rec.fs}"
        signals[rid] = rec.p_signal[:, 0].astype(np.float32)  # mV
    meta["af"] = (meta.label4 == "A").astype(int)
    meta["patient_id"] = meta.record_id  # not available in the public release
    meta["duration_s"] = [len(signals[r]) / 300 for r in meta.record_id]
    meta = meta.set_index("record_id")
    LOG.info("CinC 2017 class counts: %s", meta.label4.value_counts().to_dict())
    return {"meta": meta, "signals": signals, "fs": 300}


# --------------------------------------------------------------------------- PTB-XL
def _read_ptbxl_lead(path: str, lead: str, fs_out: int) -> np.ndarray:
    import wfdb

    rec = wfdb.rdrecord(path)
    idx = rec.sig_name.index(lead)
    x = rec.p_signal[:, idx].astype(np.float64)
    x = np.nan_to_num(x)
    g = np.gcd(int(rec.fs), fs_out)
    return resample_poly(x, fs_out // g, int(rec.fs) // g).astype(np.float32)


def load_ptbxl(cfg: dict) -> dict:
    from joblib import Parallel, delayed

    pcfg = cfg["data"]["ptbxl"]
    root = Path(pcfg["dir"])
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db.scp_codes.apply(ast.literal_eval)
    af_codes, excl = set(pcfg["af_codes"]), set(pcfg.get("exclude_codes") or [])
    if excl:
        keep = ~db.scp_codes.apply(lambda d: bool(excl & set(d)))
        LOG.info("PTB-XL: excluding %d records with codes %s", (~keep).sum(), sorted(excl))
        db = db[keep]
    fs = cfg["fs"]
    paths = [str(root / f) for f in db.filename_hr]
    sigs = Parallel(n_jobs=cfg["features"]["n_jobs"], batch_size=64)(
        delayed(_read_ptbxl_lead)(p, pcfg["lead"], fs) for p in paths
    )
    ids = [f"ptbxl_{i:05d}" for i in db.index]
    meta = pd.DataFrame({
        "record_id": ids,
        "label4": "ext",
        "af": db.scp_codes.apply(lambda d: int(bool(af_codes & set(d)))).values,
        "patient_id": db.patient_id.astype(int).astype(str).values,
        "age": db.age.values,
        "sex": db.sex.values,
        "strat_fold": db.strat_fold.values,
        "duration_s": [len(s) / fs for s in sigs],
    }).set_index("record_id")
    LOG.info("PTB-XL: %d records, %d patients, %d AF", len(meta), meta.patient_id.nunique(), meta.af.sum())
    return {"meta": meta, "signals": dict(zip(ids, sigs)), "fs": fs}


# --------------------------------------------------------------------------- NSTDB
def load_nstdb(cfg: dict) -> dict:
    """Return {'bw'|'ma'|'em': array (2, n)} resampled 360 -> fs Hz."""
    import wfdb

    root = Path(cfg["data"]["nstdb"]["dir"])
    fs = cfg["fs"]
    out = {}
    for name in cfg["data"]["nstdb"]["records"]:
        rec = wfdb.rdrecord(str(root / name))
        g = np.gcd(int(rec.fs), fs)
        out[name] = np.stack([
            resample_poly(rec.p_signal[:, c], fs // g, int(rec.fs) // g) for c in range(rec.p_signal.shape[1])
        ]).astype(np.float32)
        LOG.info("NSTDB %s: %s at %d Hz", name, out[name].shape, fs)
    return out


# --------------------------------------------------------------------------- synthetic
def _synthetic_record(rng: np.random.Generator, label: str, fs: int, dur: float) -> np.ndarray:
    """Crude stand-in signals for the smoke test only (not physiological)."""
    import neurokit2 as nk

    hr = rng.uniform(55, 95)
    if label == "A":  # irregular RR, no P waves
        n = int(dur * hr / 60 * 1.3) + 5
        rr = np.clip(rng.normal(60 / hr, 0.18 * 60 / hr, n), 0.3, 1.6)
    elif label == "O":  # occasional premature beats
        n = int(dur * hr / 60 * 1.3) + 5
        rr = np.full(n, 60 / hr) * rng.normal(1, 0.02, n)
        prem = rng.random(n) < 0.12
        rr[prem] *= 0.65
    else:
        n = int(dur * hr / 60 * 1.3) + 5
        rr = np.full(n, 60 / hr) * rng.normal(1, 0.02, n)
    beat = nk.ecg_simulate(duration=1.0, sampling_rate=fs, heart_rate=60, method="simple", random_state=int(rng.integers(1e6)))
    beat = beat - np.median(beat)
    if label == "A":  # suppress P wave (first ~35% of the simulated cycle)
        beat[: int(0.30 * fs)] *= 0.1
    out = np.zeros(int(dur * fs) + fs)
    t = 0
    for r in rr:
        i = int(t * fs)
        if i >= len(out):
            break
        seg = beat[: min(len(beat), len(out) - i)]
        out[i:i + len(seg)] += seg
        t += r
    out = out[: int(dur * fs)]
    out += 0.02 * rng.standard_normal(len(out))
    if label == "~":
        out += np.cumsum(rng.standard_normal(len(out))) * 0.05
    if label == "A":
        out += 0.04 * np.sin(2 * np.pi * rng.uniform(5, 8) * np.arange(len(out)) / fs)
    return out.astype(np.float32)


def make_synthetic(cfg: dict) -> tuple[dict, dict, dict]:
    fs = cfg["fs"]
    rng = np.random.default_rng(stable_seed("synthetic", 0))

    def build(n, prefix, labels, probs, dur_range, n_patients=None):
        lab = rng.choice(labels, size=n, p=probs)
        ids = [f"{prefix}{i:05d}" for i in range(n)]
        sigs = {rid: _synthetic_record(rng, l, fs, float(rng.uniform(*dur_range))) for rid, l in zip(ids, lab)}
        pid = ids if n_patients is None else [f"p{int(i)}" for i in rng.integers(0, n_patients, n)]
        meta = pd.DataFrame({"record_id": ids, "label4": lab, "af": (lab == "A").astype(int), "patient_id": pid,
                             "duration_s": [len(sigs[r]) / fs for r in ids]}).set_index("record_id")
        return {"meta": meta, "signals": sigs, "fs": fs}

    internal = build(cfg["data"]["synthetic_n_internal"], "A", LABELS4, [0.55, 0.15, 0.25, 0.05], (9, 40))
    external = build(cfg["data"]["synthetic_n_external"], "ptbxl_", ["N", "A"], [0.85, 0.15], (10, 10),
                     n_patients=cfg["data"]["synthetic_n_external"] // 2)
    external["meta"]["label4"] = "ext"
    t = np.arange(fs * 600) / fs
    nst = {
        "bw": np.stack([0.4 * np.sin(2 * np.pi * 0.25 * t + p) + 0.1 * np.cumsum(rng.standard_normal(len(t))) / 300 for p in (0, 1)]),
        "ma": np.stack([0.1 * rng.standard_normal(len(t)) for _ in range(2)]),
        "em": np.stack([np.repeat(rng.standard_normal(len(t) // 30 + 1), 30)[: len(t)] * 0.3 for _ in range(2)]),
    }
    return internal, external, {k: v.astype(np.float32) for k, v in nst.items()}


# --------------------------------------------------------------------------- prepare / access
def prepare(cfg: dict) -> None:
    paths = Paths.from_cfg(cfg)
    if cfg["data"].get("synthetic"):
        internal, external, nst = make_synthetic(cfg)
    else:
        internal, external, nst = load_cinc2017(cfg), load_ptbxl(cfg), load_nstdb(cfg)
    save_pickle(internal, paths.dataset("internal"))
    save_pickle(external, paths.dataset("external"))
    save_pickle(nst, paths.dataset("nstdb"))
    LOG.info("Processed datasets written to %s", paths.processed)


def load_dataset(cfg: dict, name: str) -> dict:
    return load_pickle(Paths.from_cfg(cfg).dataset(name))

"""Real-noise corruption using the MIT-BIH Noise Stress Test Database.

Design choices (all stated in the Methods):

* SNR is power-based: ``SNR_dB = 10 log10(P_signal / P_noise)`` with both
  powers computed on the mean-removed recording and noise segment. The original
  ``nst`` tool scales by QRS amplitude instead; power-based SNR is defined for
  every recording, including ones where QRS detection fails.
* Each NSTDB record is split in time. The first ``train_fraction`` is used only
  for training augmentation and the remainder only for test corruption, so no
  noise sample is shared between training and evaluation.
* For a given (record, noise type, realization) the same noise segment is used
  at every SNR and only its gain changes, so differences across SNR levels come
  from the noise level alone. Every model sees identical corrupted signals,
  which makes the paired comparisons valid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .utils import stable_seed


@dataclass(frozen=True)
class Condition:
    dataset: str          # 'internal' | 'external'
    noise: str = "none"   # 'none' | 'bw' | 'ma' | 'em'
    snr_db: float = math.inf
    realization: int = 0

    @property
    def key(self) -> str:
        if self.noise == "none":
            return f"{self.dataset}__clean"
        return f"{self.dataset}__{self.noise}__{self.snr_db:+g}dB__r{self.realization}"

    @property
    def is_clean(self) -> bool:
        return self.noise == "none"


def evaluation_conditions(cfg: dict) -> list[Condition]:
    out = [Condition("internal")]
    ncfg = cfg["noise"]
    grid = [(n, float(s), r) for n in ncfg["types"] for s in ncfg["snr_db"] for r in range(ncfg["test_realizations"])]
    out += [Condition("internal", n, s, r) for n, s, r in grid]
    out.append(Condition("external"))
    if cfg["evaluation"].get("external_noise"):
        out += [Condition("external", n, s, r) for n, s, r in grid]
    return out


class NoiseBank:
    def __init__(self, nstdb: dict[str, np.ndarray], train_fraction: float = 0.5):
        self.parts = {"train": {}, "test": {}}
        for name, arr in nstdb.items():
            cut = int(arr.shape[1] * train_fraction)
            self.parts["train"][name] = arr[:, :cut]
            self.parts["test"][name] = arr[:, cut:]

    def segment(self, noise: str, length: int, rng: np.random.Generator, partition: str) -> np.ndarray:
        arr = self.parts[partition][noise]
        ch = int(rng.integers(arr.shape[0]))
        x = arr[ch]
        if length > len(x):  # mirror-extend (avoids a step discontinuity)
            reps = int(np.ceil(length / len(x)))
            x = np.concatenate([x if i % 2 == 0 else x[::-1] for i in range(reps)])
        start = int(rng.integers(0, len(x) - length + 1))
        return x[start:start + length].astype(np.float64)


def add_noise_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    s = signal.astype(np.float64)
    n = noise - noise.mean()
    ps = np.mean((s - s.mean()) ** 2)
    pn = np.mean(n**2)
    if ps <= 0 or pn <= 0:
        return signal.astype(np.float32)
    gain = math.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (s + gain * n).astype(np.float32)


def corrupt_for_test(signal: np.ndarray, record_id: str, cond: Condition, bank: NoiseBank, base_seed: int) -> np.ndarray:
    if cond.is_clean:
        return signal
    rng = np.random.default_rng(stable_seed(base_seed, "test", record_id, cond.noise, cond.realization))
    seg = bank.segment(cond.noise, len(signal), rng, "test")
    return add_noise_at_snr(signal, seg, cond.snr_db)


def augmentation_spec(record_ids, cfg: dict) -> list[dict]:
    """Deterministic table of noisy training copies used by the feature models."""
    a, types = cfg["augmentation"], cfg["noise"]["types"]
    spec = []
    for rid in record_ids:
        for k in range(a["copies"]):
            rng = np.random.default_rng(stable_seed(a["seed"], "spec", rid, k))
            spec.append({
                "record_id": rid, "copy": k,
                "noise": types[int(rng.integers(len(types)))],
                "snr_db": float(rng.uniform(*a["snr_range"])),
                "seed": int(rng.integers(2**62)),
            })
    return spec


def corrupt_for_train(signal: np.ndarray, spec: dict, bank: NoiseBank) -> np.ndarray:
    rng = np.random.default_rng(spec["seed"])
    seg = bank.segment(spec["noise"], len(signal), rng, "train")
    return add_noise_at_snr(signal, seg, spec["snr_db"])


def random_train_corruption(signal: np.ndarray, rng: np.random.Generator, bank: NoiseBank, cfg: dict) -> np.ndarray:
    """On-the-fly augmentation for the CNN (training noise partition only)."""
    types = cfg["noise"]["types"]
    noise = types[int(rng.integers(len(types)))]
    snr = float(rng.uniform(*cfg["augmentation"]["snr_range"]))
    return add_noise_at_snr(signal, bank.segment(noise, len(signal), rng, "train"), snr)

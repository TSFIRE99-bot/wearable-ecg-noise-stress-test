import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ecgrobust.features import match_beats, sample_entropy
from ecgrobust.metrics import (auroc, average_precision, best_f1_threshold, binary_metrics, bootstrap_indices,
                               challenge_score, holm)
from ecgrobust.noise import Condition, NoiseBank, add_noise_at_snr, corrupt_for_test


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.random(500) < 0.1
    s = np.round(rng.random(500) + y * 0.5, 2)  # rounding creates ties
    assert math.isclose(auroc(y, s), roc_auc_score(y, s), abs_tol=1e-12)
    assert math.isclose(average_precision(y, s), average_precision_score(y, s), abs_tol=1e-12)


def test_snr_is_exact():
    rng = np.random.default_rng(0)
    sig = np.sin(np.linspace(0, 60, 9000)).astype(np.float32)
    noise = rng.standard_normal(9000)
    for snr in [24, 6, 0, -6]:
        noisy = add_noise_at_snr(sig, noise, snr)
        n = noisy.astype(float) - sig
        got = 10 * np.log10(np.mean((sig - sig.mean()) ** 2) / np.mean(n**2))
        assert abs(got - snr) < 1e-3


def test_same_noise_segment_across_snr_and_partitions_disjoint():
    arr = np.arange(2 * 1000, dtype=np.float32).reshape(2, 1000)
    bank = NoiseBank({"em": arr}, 0.5)
    assert set(bank.parts["train"]["em"][0]).isdisjoint(bank.parts["test"]["em"][0])
    sig = np.sin(np.linspace(0, 20, 300)).astype(np.float32)
    a = corrupt_for_test(sig, "r1", Condition("internal", "em", 12.0), bank, 1) - sig
    b = corrupt_for_test(sig, "r1", Condition("internal", "em", 0.0), bank, 1) - sig
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.999  # same segment, different gain


def test_beat_matching():
    assert match_beats(np.array([100, 400, 700]), np.array([105, 690, 900]), 45) == (2, 1, 1)


def test_sample_entropy_regular_vs_random():
    rng = np.random.default_rng(1)
    regular = np.tile([800.0, 810.0], 30)
    rnd = rng.normal(800, 100, 60)
    se_reg, _ = sample_entropy(regular, 2, 0.2 * regular.std())
    se_rnd, _ = sample_entropy(rnd, 2, 0.2 * rnd.std())
    assert se_reg < se_rnd


def test_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    t = best_f1_threshold(y, p)
    m = binary_metrics(y, p, t)
    assert m["f1"] == 1.0 and t == 0.7


def test_challenge_score_perfect():
    y = np.array([0, 1, 2, 3, 0])
    s = challenge_score(y, y)
    assert s["score"] == 1.0 and s["score4"] == 1.0


def test_cluster_bootstrap_keeps_patients_together():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    for ix in bootstrap_indices(6, 20, 0, groups):
        g = groups[ix]
        for p in set(g):
            assert (g == p).sum() % (groups == p).sum() == 0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

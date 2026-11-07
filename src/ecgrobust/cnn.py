"""Small 1-D residual CNN on the raw single-lead signal.

Training uses random 9 s crops (one per record per epoch). Inference averages
the predicted probabilities of overlapping 9 s windows (stride 3 s) covering
the whole recording. Each window is z-scored on its own, so the network never
depends on recording-level amplitude or on any beat detector.
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from torch import nn

from .metrics import average_precision, challenge_score
from .noise import NoiseBank, random_train_corruption
from .utils import set_global_seed

LOG = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, k: int = 9, p_drop: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout),
        )
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ECGNet(nn.Module):
    def __init__(self, n_out: int = 1, c: int = 32, dropout: float = 0.3, stem_pool: int = 2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, c, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(c), nn.ReLU(),
                                  nn.MaxPool1d(stem_pool))
        chans = [c, c, 2 * c, 2 * c, 4 * c, 4 * c]
        blocks, cin = [], c
        for i, co in enumerate(chans):
            blocks.append(ResBlock(cin, co, stride=2 if i % 2 == 0 and i > 0 else 1, p_drop=0.1))
            cin = co
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_out))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# --------------------------------------------------------------------------- windows
def _zscore(w: np.ndarray) -> np.ndarray:
    w = w - np.median(w, axis=-1, keepdims=True)
    return w / (w.std(axis=-1, keepdims=True) + 1e-6)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, (0, n - len(x)), mode="reflect" if len(x) > 1 else "constant") if len(x) < n else x


def window_starts(length: int, win: int, stride: int) -> list[int]:
    if length <= win:
        return [0]
    starts = list(range(0, length - win + 1, stride))
    if starts[-1] != length - win:
        starts.append(length - win)
    return starts


@torch.no_grad()
def predict(model: nn.Module, signals: list[np.ndarray], fs: int, ccfg: dict, n_out: int,
            batch: int = 256) -> np.ndarray:
    """Window-averaged probabilities. Shape (n,) for binary, (n, n_out) otherwise."""
    model.eval()
    win, stride = int(ccfg["window_s"] * fs), int(ccfg["stride_s"] * fs)
    wins, owner = [], []
    for i, s in enumerate(signals):
        s = _fit_length(np.asarray(s, np.float32), win)
        for st in window_starts(len(s), win, stride):
            wins.append(s[st:st + win])
            owner.append(i)
    wins = _zscore(np.stack(wins).astype(np.float32))
    owner = np.asarray(owner)
    probs = []
    for b in range(0, len(wins), batch):
        logits = model(torch.from_numpy(wins[b:b + batch, None, :]))
        probs.append(torch.sigmoid(logits[:, 0]).numpy() if n_out == 1 else torch.softmax(logits, 1).numpy())
    probs = np.concatenate(probs)
    if n_out == 1:
        return np.bincount(owner, weights=probs, minlength=len(signals)) / np.bincount(owner, minlength=len(signals))
    out = np.zeros((len(signals), n_out))
    np.add.at(out, owner, probs)
    return out / np.bincount(owner, minlength=len(signals))[:, None]


# --------------------------------------------------------------------------- training
def train(signals: list[np.ndarray], y: np.ndarray, fs: int, cfg: dict, hp: dict, seed: int,
          n_classes: int = 2, bank: NoiseBank | None = None) -> tuple[nn.Module, dict]:
    """Fit on ``signals``; an inner stratified split provides early stopping."""
    from sklearn.model_selection import train_test_split

    ccfg = cfg["models"]["cnn"]
    if ccfg.get("num_threads"):
        torch.set_num_threads(ccfg["num_threads"])
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(signals))
    tr, va = train_test_split(idx, test_size=ccfg["inner_val_fraction"], stratify=y, random_state=seed)
    n_out = 1 if n_classes == 2 else n_classes
    model = build(n_out, cfg, hp)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps_per_epoch = int(np.ceil(len(tr) / ccfg["batch_size"]))
    # one-cycle needs enough steps for its warm-up phase (only tiny smoke runs lack them)
    if ccfg.get("schedule", "cosine") == "onecycle" and steps_per_epoch * ccfg["max_epochs"] >= 20:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], epochs=ccfg["max_epochs"],
                                                    steps_per_epoch=steps_per_epoch, pct_start=0.25)
        per_batch = True
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ccfg["max_epochs"])
        per_batch = False
    if n_out == 1:
        pos = max(y[tr].sum(), 1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr) - pos) / pos, dtype=torch.float32))
    else:
        cnt = np.bincount(y[tr], minlength=n_classes).astype(float)
        w = len(tr) / (n_classes * np.maximum(cnt, 1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    win = int(ccfg["window_s"] * fs)
    best, best_score, best_epoch, bad, hist = None, -np.inf, 0, 0, []
    for epoch in range(ccfg["max_epochs"]):
        model.train()
        order = rng.permutation(tr)
        losses = []
        for b in range(0, len(order), ccfg["batch_size"]):
            bi = order[b:b + ccfg["batch_size"]]
            xb = []
            for i in bi:
                s = _fit_length(np.asarray(signals[i], np.float32), win)
                st = int(rng.integers(0, len(s) - win + 1))
                w = s[st:st + win]
                if bank is not None and rng.random() < cfg["augmentation"]["p_noise_cnn"]:
                    w = random_train_corruption(w, rng, bank, cfg)
                if ccfg.get("sign_flip") and rng.random() < 0.5:
                    w = -w
                xb.append(w)
            xb = torch.from_numpy(_zscore(np.stack(xb).astype(np.float32))[:, None, :])
            out = model(xb)
            yb = torch.from_numpy(y[bi])
            loss = loss_fn(out[:, 0], yb.float()) if n_out == 1 else loss_fn(out, yb.long())
            opt.zero_grad()
            loss.backward()
            if ccfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), ccfg["grad_clip"])
            opt.step()
            if per_batch:
                sched.step()
            losses.append(loss.item())
        if not per_batch:
            sched.step()
        pv = predict(model, [signals[i] for i in va], fs, ccfg, n_out)
        score = average_precision(y[va], pv) if n_out == 1 else challenge_score(y[va], pv.argmax(1))["score"]
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": float(score)})
        LOG.info("  cnn epoch %2d loss %.4f val %.4f", epoch, hist[-1]["loss"], score)
        if score > best_score:
            best, best_score, best_epoch, bad, best_pv = copy.deepcopy(model.state_dict()), score, epoch, 0, pv
        else:
            bad += 1
            # no early stopping while the learning rate is still high (one-cycle warm-up / plateau):
            # validation scores are noisy there and an early peak is not a converged model
            if bad >= ccfg["patience"] and epoch + 1 >= ccfg.get("min_epochs", 0):
                break
    model.load_state_dict(best)
    LOG.debug("cnn seed=%d hp=%s best_epoch=%d val=%.4f", seed, hp, best_epoch, best_score)
    return model, {"best_epoch": best_epoch, "inner_val": best_score, "history": hist, "va_idx": va, "va_pred": best_pv}


def build(n_out: int, cfg: dict, hp: dict) -> nn.Module:
    c = cfg["models"]["cnn"]
    return ECGNet(n_out, c["base_channels"], hp["dropout"], c.get("stem_pool", 2))

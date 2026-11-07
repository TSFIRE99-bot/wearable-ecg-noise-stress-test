"""Gate before unlocking the test set: every trained CNN must have converged.

A run counts as converged when either
  * early stopping fired (training ended before max_epochs), or
  * it completed the schedule and its validation curve had plateaued: the best score of
    the last 10 epochs improves on the best of the 10 epochs before by less than 0.01.
Exit code 0 = all converged, 1 = at least one did not.

    python scripts/check_cnn_converged.py --config configs/local.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ecgrobust.utils import load_config, load_pickle

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
cfg = load_config(ap.parse_args().config)
max_ep = cfg["models"]["cnn"]["max_epochs"]
files = sorted(Path(cfg["output_dir"], "models").rglob("cnn_seed*.pkl"))
if not files:
    sys.exit("no CNN models found")
ok = True
for f in files:
    b = load_pickle(f)
    val = [h["val"] for h in b["history"]]
    early = len(val) < max_ep
    gain = max(val[-10:]) - max(val[-20:-10]) if len(val) >= 20 else float("inf")
    conv = early or gain < 0.01
    ok &= conv
    print(f"{'OK ' if conv else 'NOT'} {f.parent.parent.name}/{f.parent.name}/{f.name}: best epoch "
          f"{b['final_fit']['best_epoch']}, trained {len(val)}, best inner-val {max(val):.3f}, "
          f"last-10 gain {gain:+.3f}")
sys.exit(0 if ok else 1)

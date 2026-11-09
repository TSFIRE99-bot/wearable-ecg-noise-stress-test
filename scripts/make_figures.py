"""Redraw every paper figure from the saved results, without re-running the 2.6-hour report.

    python scripts/make_figures.py --config configs/local.yaml

Uses the same drawing code as the full report (Reporter.figures in src/ecgrobust/report.py)
and writes fig0 to fig6 to <output_dir>/figures. Needs the saved tables in
<output_dir>/tables and predictions in <output_dir>/predictions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ecgrobust.metrics import calibration_summary
from ecgrobust.report import Reporter
from ecgrobust.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/local.yaml")
cfg = load_config(ap.parse_args().config)

r = Reporter(cfg)
tables = r.paths.tables
r.long = pd.read_csv(tables / "metrics_all_conditions.csv")
r.noise = pd.read_csv(tables / "noise_degradation.csv")
r.beats = pd.read_csv(tables / "beat_detection.csv")
r.abst = pd.read_csv(tables / "abstention_curve.csv")
r.shap_df = pd.read_csv(tables / "shap_importance.csv")

# Fig 5 plots reliability curves, which are computed from the saved predictions
r.cal_curves = {}
for model in ("lgbm", "cnn"):
    for cond in r.conds:
        if cond == "internal__clean" or cond.startswith("internal__em__"):
            first, probs, _ = r._matrix("standard", model, cond)
            r.cal_curves[("standard", model, cond)] = calibration_summary(
                first.y.to_numpy(), probs.mean(0), cfg["evaluation"]["calibration_bins"])

r.figures()
flow = Path(__file__).parent / "make_flow_diagram.py"
subprocess.run([sys.executable, str(flow), "--output-dir", cfg["output_dir"]], check=True)
print(f"figures written to {r.paths.figures}")

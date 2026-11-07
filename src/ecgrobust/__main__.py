"""Command line: ``python -m ecgrobust <stage> --config configs/default.yaml``.

Stages, in order:
  prepare      read the raw datasets into a common 300 Hz format
  split        make the stratified 80/20 internal hold-out (once; never overwritten)
  features     NeuroKit2 features for clean, augmented and corrupted signals
  train        tune (5-fold CV) and fit all models x seeds x training variants
  unlock-test  freeze model development and record it (run once)
  evaluate     predict on internal test (clean + noise grid) and PTB-XL
  report       tables, figures, SHAP, calibration, abstention
  all          everything up to train; add --unlock to continue through report
"""
from __future__ import annotations

import argparse
import logging
import time

from .utils import load_config, setup_logging

STAGES = ["prepare", "split", "features", "train", "unlock-test", "evaluate", "report", "all"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ecgrobust", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="*", help="train only these models (e.g. --models cnn)")
    ap.add_argument("--unlock", action="store_true", help="with `all`: unlock the test set and evaluate + report")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    setup_logging(logging.DEBUG if a.verbose else logging.INFO)
    cfg = load_config(a.config)

    from . import data, pipeline, report, splits

    run = {
        "prepare": lambda: data.prepare(cfg),
        "split": lambda: splits.make_split(cfg),
        "features": lambda: pipeline.run_features(cfg),
        "train": lambda: pipeline.run_train(cfg, a.models),
        "unlock-test": lambda: splits.unlock_test(cfg),
        "evaluate": lambda: pipeline.run_evaluate(cfg),
        "report": lambda: report.run_report(cfg),
    }
    order = ["prepare", "split", "features", "train"] + (["unlock-test", "evaluate", "report"] if a.unlock else [])
    for st in order if a.stage == "all" else [a.stage]:
        t = time.time()
        logging.getLogger("ecgrobust").info(">>> %s", st)
        run[st]()
        logging.getLogger("ecgrobust").info("<<< %s done in %.1f min", st, (time.time() - t) / 60)


if __name__ == "__main__":
    main()

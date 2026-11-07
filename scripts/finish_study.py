"""Wait for `train` to finish, check CNN convergence, then unlock-test -> evaluate -> report.

    python scripts/finish_study.py --config configs/local.yaml --train-log run_cnn.log

While it runs it asks Windows not to idle-sleep (SetThreadExecutionState; released
automatically when this process exits; no system settings are changed).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--train-log", required=True)
a = ap.parse_args()

if sys.platform == "win32":
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


log = Path(a.train_log)
say(f"waiting for training to finish ({log})")
while True:
    txt = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    if "<<< train done" in txt:
        break
    if "Traceback" in txt:
        sys.exit("training failed; see the training log")
    time.sleep(60)

say("training finished; checking CNN convergence")
here = Path(__file__).parent
if subprocess.run([sys.executable, str(here / "check_cnn_converged.py"), "--config", a.config]).returncode:
    sys.exit("at least one CNN did not converge; test set left LOCKED for review")

for stage in ["unlock-test", "evaluate", "report"]:
    say(f"running {stage}")
    if subprocess.run([sys.executable, "-m", "ecgrobust", stage, "--config", a.config]).returncode:
        sys.exit(f"{stage} failed")
say("STUDY COMPLETE")

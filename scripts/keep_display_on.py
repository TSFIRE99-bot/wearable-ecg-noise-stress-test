"""Keep the display (and so the machine) awake until a long run finishes.

On laptops with Modern Standby, Windows suspends everything when the screen turns
off, even on AC power. This asks Windows to keep the display on
(SetThreadExecutionState) until the finisher log reports completion or failure,
or `--max-hours` elapses. No settings are changed; the request ends when this
process exits. Lock the screen with Win+L when you step away; the run continues.

    python scripts/keep_display_on.py --watch run_finish.log --max-hours 16
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
DONE = ("STUDY COMPLETE", "failed", "LOCKED", "did not converge")

ap = argparse.ArgumentParser()
ap.add_argument("--watch", required=True)
ap.add_argument("--max-hours", type=float, default=16)
a = ap.parse_args()
if sys.platform != "win32":
    sys.exit("Windows only")

ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
print(time.strftime("%H:%M:%S"), "holding display on", flush=True)
deadline = time.time() + a.max_hours * 3600
log = Path(a.watch)
try:
    while time.time() < deadline:
        txt = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
        if any(k in txt for k in DONE):
            break
        time.sleep(60)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print(time.strftime("%H:%M:%S"), "released", flush=True)

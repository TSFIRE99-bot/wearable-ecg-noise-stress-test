"""Draw the participant flow diagram (TRIPOD+AI) from the saved split and cohort table.

    python scripts/make_flow_diagram.py --output-dir outputs

Writes <output-dir>/figures/fig0_flow.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#8f8e88"

ap = argparse.ArgumentParser()
ap.add_argument("--output-dir", default="outputs")
out = Path(ap.parse_args().output_dir)
split = json.loads((out / "splits.json").read_text())
cohort = pd.read_csv(out / "tables" / "table1_cohorts.csv").set_index("set")
tr, te = split["class_counts_train"], split["class_counts_test"]
ptb = cohort.loc["PTB-XL lead I (external)"]


def counts(c: dict) -> str:
    return f"N {c['N']:,} · AF {c['A']:,} · Other {c['O']:,} · Noisy {c['~']:,}"


total = {k: tr[k] + te[k] for k in tr}
n_tr, n_te = sum(tr.values()), sum(te.values())

fig, ax = plt.subplots(figsize=(7.2, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, body, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.0, edgecolor=EDGE, facecolor="white"))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=8.5, color=INK,
            fontweight="bold" if bold else "normal")
    ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top", fontsize=7.3, color=INK2, linespacing=1.45)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))


# development data (left column)
box(4, 82, 56, 15, "PhysioNet/CinC 2017 training set", f"{n_tr + n_te:,} single-lead recordings (AliveCor, 300 Hz)\n{counts(total)}", True)
arrow(32, 82, 32, 76.5)
ax.text(33.5, 79.2, "stratified 80/20 split (seed 2017), made once", fontsize=6.8, color=INK2, va="center")
box(4, 58, 26, 17, "Development set", f"{n_tr:,} recordings\nN {tr['N']:,} · AF {tr['A']:,}\nOther {tr['O']:,} · Noisy {tr['~']:,}")
box(34, 58, 26, 17, "Internal test set (locked)", f"{n_te:,} recordings\nN {te['N']:,} · AF {te['A']:,}\nOther {te['O']:,} · Noisy {te['~']:,}")
arrow(17, 58, 17, 50.5)
arrow(47, 58, 47, 50.5)
ax.text(32, 54.2, "noisy class excluded from the binary task", fontsize=6.8, color=INK2, ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))
box(4, 33, 26, 17, "Binary training data", f"{n_tr - tr['~']:,} recordings ({tr['A']:,} AF)\n+ 1 noisy copy each\n(augmented training)")
box(34, 33, 26, 17, "Binary internal test", f"{n_te - te['~']:,} recordings ({te['A']:,} AF)\nclean + 3 noise types\n× 6 SNR levels")
ax.text(17, 29, "4-class models use all\ndevelopment recordings", fontsize=6.8, color=INK2, ha="center", va="top")

# external data (right column)
box(66, 82, 30, 15, "PTB-XL v1.0.3", f"{int(ptb['recordings']):,} 10-s 12-lead ECGs\n{int(ptb['patients']):,} patients (500 Hz)", True)
arrow(81, 82, 81, 50.5)
ax.text(82.2, 66, "lead I only,\nresampled to 300 Hz;\nno exclusions", fontsize=6.8, color=INK2, va="center")
box(66, 33, 30, 17, "External test set", f"{int(ptb['binary_task_n']):,} recordings\n{int(ptb['AF_n']):,} AF ({ptb['AF_pct']}%)\npatient-level bootstrap")

# noise source
box(4, 4, 92, 13, "MIT-BIH Noise Stress Test Database (bw, ma, em; resampled to 300 Hz)",
    "first half of each record → training augmentation only;  second half → test corruption only")

fig.tight_layout()
dst = out / "figures" / "fig0_flow.png"
fig.savefig(dst, dpi=300, facecolor="white")
print(f"wrote {dst}")

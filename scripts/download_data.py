"""Download the three public datasets from PhysioNet (open access, no login).

    python scripts/download_data.py            # all three
    python scripts/download_data.py --only nstdb

Approximate sizes: CinC 2017 ~100 MB, NSTDB ~20 MB, PTB-XL ~1.7 GB (zip; ~3 GB unpacked).
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
CINC = "https://physionet.org/files/challenge-2017/1.0.0/"
PTBXL_ZIP = ("https://physionet.org/static/published-projects/ptb-xl/"
             "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"exists: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)
    return dest


def cinc2017() -> None:
    d = RAW / "challenge-2017"
    z = fetch(CINC + "training2017.zip", d / "training2017.zip")
    if not (d / "training2017").exists():
        zipfile.ZipFile(z).extractall(d)
    fetch(CINC + "REFERENCE-v3.csv", d / "REFERENCE-v3.csv")


def nstdb() -> None:
    import wfdb

    wfdb.dl_database("nstdb", str(RAW / "nstdb"), records=["bw", "ma", "em"])


def ptbxl() -> None:
    d = RAW / "ptb-xl"
    if (d / "ptbxl_database.csv").exists():
        print("exists: PTB-XL")
        return
    z = fetch(PTBXL_ZIP, RAW / "ptb-xl.zip")
    # strip the long top-level folder while extracting (keeps paths under Windows' 260-char limit)
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
            if not rel or info.is_dir():
                continue
            if not (rel.startswith("records500/") or rel.endswith(".csv")):
                continue  # only 500 Hz records and metadata are needed
            out = d / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cinc2017", "nstdb", "ptbxl"], nargs="*")
    ap.add_argument("--root", default="data/raw", help="where raw data go (use a short path on Windows)")
    a = ap.parse_args()
    RAW = Path(a.root)
    for name in a.only or ["cinc2017", "nstdb", "ptbxl"]:
        globals()[name]()
    print("done")

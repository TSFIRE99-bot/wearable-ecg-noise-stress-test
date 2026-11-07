"""Build the supplementary-material Word file (medRxiv wants supplements as a separate file).

    python scripts/build_supplement_docx.py

Writes paper/submission/supplementary.docx with Supplementary Fig. S1 (participant flow),
Table S1 (four-class challenge scores), Table S2 (external subgroups) and Table S3
(TRIPOD+AI mapping), built from outputs/ and docs/.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
OUT = ROOT / "paper" / "submission" / "supplementary.docx"
TITLE = "Stress-testing wearable ECG atrial fibrillation classifiers with real recorded noise"
FONT = "Calibri"
ci = lambda s: re.sub(r" \(±[0-9.]+\)", "", str(s))  # noqa: E731  (drop the seed SD; keep mean [95% CI])

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1))
st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
for name, size in [("Title", 15), ("Heading 1", 12.5)]:
    s = doc.styles[name]
    s.font.name, s.font.size, s.font.bold = FONT, Pt(size), True
    s.font.color.rgb = RGBColor(0, 0, 0)
    s.paragraph_format.keep_with_next = True


def table(df: pd.DataFrame, size: float = 8):
    t = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    t.style, t.alignment = "Table Grid", WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate([list(df.columns)] + df.astype(str).values.tolist()):
        for c, val in enumerate(row):
            p = t.cell(r, c).paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val.replace("`", ""))
            run.font.size, run.bold = Pt(size), r == 0
    doc.add_paragraph()


def caption(label: str, text: str):
    p = doc.add_paragraph()
    p.add_run(label + " ").bold = True
    p.add_run(text)


doc.add_paragraph(f"Supplementary material for: {TITLE}", style="Title")
doc.add_paragraph("Tejas Menneni, Columbus State Community College, Columbus, Ohio, USA (tejas.menneni@gmail.com)")
doc.add_paragraph("Code and all result tables: https://github.com/TSFIRE99-bot/wearable-ecg-noise-stress-test "
                  "(archived at https://doi.org/10.5281/zenodo.24042753).")

# Figure S1
doc.add_heading("Supplementary Figure S1", level=1)
pic = doc.add_paragraph(); pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
pic.add_run().add_picture(str(FIG / "fig0_flow.png"), width=Inches(6.0))
caption("Supplementary Figure S1.", "Participant flow: CinC 2017 split into development and locked internal test sets, "
        "exclusion of the noisy class for the binary task, the PTB-XL external test set, and the separate training and "
        "test partitions of the MIT-BIH Noise Stress Test Database.")

# Table S1: four-class challenge score
doc.add_page_break()
doc.add_heading("Supplementary Table S1", level=1)
f4 = pd.read_csv(TAB / "fourclass_challenge.csv")
f4 = f4.rename(columns={"model": "Model", "n_seeds": "Seeds", "challenge_F1_NAO": "Challenge score, mean F1 (N, A, O) [95% CI]",
                        "f1_N": "F1 Normal", "f1_A": "F1 AF", "f1_O": "F1 Other", "f1_~": "F1 Noisy", "score4": "Mean F1, 4 classes"})
f4["Challenge score, mean F1 (N, A, O) [95% CI]"] = f4["Challenge score, mean F1 (N, A, O) [95% CI]"].map(ci)
for c in ["F1 Normal", "F1 AF", "F1 Other", "F1 Noisy", "Mean F1, 4 classes"]:
    f4[c] = f4[c].map(lambda x: f"{x:.3f}")
caption("Supplementary Table S1.", "Four-class performance on the internal test set (1,706 recordings), PhysioNet/CinC 2017 "
        "challenge score. The CNN was trained with one seed for computational reasons; the other models with five.")
table(f4)

# Table S2: external subgroups
doc.add_heading("Supplementary Table S2", level=1)
sg = pd.read_csv(TAB / "subgroups_external.csv")
sg = sg[sg.variant == "standard"].drop(columns="variant")
for c in ["auroc", "auprc", "sensitivity", "specificity"]:
    sg[c] = sg[c].map(ci)
sg["subgroup"] = sg.subgroup.str.replace("sex=", "", regex=False).str.replace(">=", "≥", regex=False)
sg = sg.rename(columns={"model": "Model", "subgroup": "Subgroup", "n": "n", "AF_n": "AF n", "auroc": "AUROC",
                        "auprc": "AUPRC", "sensitivity": "Sensitivity", "specificity": "Specificity"})
caption("Supplementary Table S2.", "External (PTB-XL, clean) performance by sex and age, standard training. Mean over "
        "five seeds [95% patient-level bootstrap CI].")
table(sg, size=7.5)

# Table S3: TRIPOD+AI mapping
doc.add_page_break()
doc.add_heading("Supplementary Table S3", level=1)
md = (ROOT / "docs" / "TRIPOD_AI_checklist.md").read_text(encoding="utf-8")
rows = [l.strip().strip("|").split("|") for l in md.splitlines() if l.startswith("|")]
rows = [[c.strip() for c in r] for r in rows if not all(set(c.strip()) <= set("-:") for c in r)]
tr = pd.DataFrame(rows[1:], columns=rows[0])
caption("Supplementary Table S3.", "Where the manuscript addresses each TRIPOD+AI reporting topic (Collins et al., BMJ "
        "2024). Topic names are paraphrased; file names refer to the public code repository.")
table(tr, size=7.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")

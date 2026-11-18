"""Build the supplementary-material Word file (Physiological Measurement layout).

    python scripts/build_supplement_docx.py                 # named version
    python scripts/build_supplement_docx.py --anonymous     # double-anonymous review version
    python scripts/build_supplement_docx.py --no-figures    # placeholder instead of figure S1

Writes paper/submission/physiol_meas/supplementary[_anonymous][_no_figures].docx with supplementary
figure S1 (flow of recordings), table S1 (four-class challenge scores), table S2 (external
subgroups) and table S3 (TRIPOD+AI mapping), built from outputs/ and docs/. IOP asks for a
title (30 characters or fewer) and a description (30 words or fewer) inside the file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
NO_FIGURES = "--no-figures" in sys.argv
ANONYMOUS = "--anonymous" in sys.argv
OUT = ROOT / "paper" / "submission" / "physiol_meas" / ("supplementary" + ("_anonymous" if ANONYMOUS else "")
                                       + ("_no_figures" if NO_FIGURES else "") + ".docx")
TITLE = "Stress-testing wearable ECG atrial fibrillation classifiers with real recorded noise"
SUPP_TITLE = "Study flow and extra tables"    # 27 characters
SUPP_DESCRIPTION = ("Flow of recordings (figure S1), four-class challenge scores (table S1), external performance "
                    "by sex and age (table S2) and the TRIPOD+AI reporting checklist mapping (table S3).")
assert len(SUPP_TITLE) <= 30 and len(SUPP_DESCRIPTION.split()) <= 30
FONT = "Times New Roman"
ci = lambda s: re.sub(r" \(±[0-9.]+\)", "", str(s))  # noqa: E731  (drop the seed SD; keep mean [95% CI])
MODEL = {"Rule (RMSSD/RR)": "Rule (RMSSD/RR)", "rule": "Rule (RMSSD/RR)", "logreg": "Logistic regression",
         "lgbm": "Gradient boosting", "cnn": "1-D CNN"}

doc = Document()
doc.core_properties.author = "" if ANONYMOUS else "Tejas Menneni"
doc.core_properties.last_modified_by = "" if ANONYMOUS else "Tejas Menneni"
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1))
st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.line_spacing = 1.5
for name, size in [("Title", 15), ("Heading 1", 13)]:
    s = doc.styles[name]
    s.font.name, s.font.size, s.font.bold = FONT, Pt(size), True
    s.font.color.rgb = RGBColor(0, 0, 0)
    fonts = s.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), FONT)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(attr), None)
    s.paragraph_format.keep_with_next = True

title_ppr = doc.styles["Title"].element.pPr   # drop the coloured rule Word's Title style draws
if title_ppr is not None and title_ppr.find(qn("w:pBdr")) is not None:
    title_ppr.remove(title_ppr.find(qn("w:pBdr")))


def table(df: pd.DataFrame, size: float = 8):
    t = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    t.style, t.alignment = "Table Grid", WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate([list(df.columns)] + df.astype(str).values.tolist()):
        for c, val in enumerate(row):
            p = t.cell(r, c).paragraphs[0]
            p.paragraph_format.space_after, p.paragraph_format.line_spacing = Pt(0), 1.0
            p.paragraph_format.keep_with_next = r < len(df)   # keep each table on one page
            run = p.add_run(val.replace("`", ""))
            run.font.size, run.bold = Pt(size), r == 0


def caption(label: str, text: str, above: bool = True):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = above
    p.add_run(label + " ").bold = True
    p.add_run(text)


def note(text: str):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing, p.paragraph_format.space_before = 1.0, Pt(3)
    p.add_run(text).font.size = Pt(10)


doc.add_paragraph(SUPP_TITLE, style="Title")
p = doc.add_paragraph(); p.add_run("Description: ").bold = True; p.add_run(SUPP_DESCRIPTION)
p = doc.add_paragraph(); p.add_run("Supplementary material for: ").bold = True; p.add_run(TITLE)
if not ANONYMOUS:
    doc.add_paragraph("Tejas Menneni, Columbus State Community College, Columbus, OH, USA (tejas.menneni@gmail.com)")
    doc.add_paragraph("Code and all result tables: https://github.com/TSFIRE99-bot/wearable-ecg-noise-stress-test "
                      "(release v1.0.3, archived at https://doi.org/10.5281/zenodo.24352330).")

# Figure S1
doc.add_heading("Supplementary figure S1", level=1).paragraph_format.page_break_before = True
pic = doc.add_paragraph(); pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
pic.paragraph_format.keep_with_next, pic.paragraph_format.line_spacing = True, 1.0
journal_s1 = ROOT / "paper" / "submission" / "figureS1.png"   # journal-format version from make_journal_figures.py
if NO_FIGURES:
    pic.add_run("[Insert supplementary figure S1 here: figureS1.png, 15 cm wide]").bold = True
elif journal_s1.exists():
    pic.add_run().add_picture(str(journal_s1), width=Inches(15 / 2.54))
else:
    pic.add_run().add_picture(str(FIG / "fig0_flow.png"), width=Inches(6.0))
caption("Figure S1.", "Flow of recordings through the study. The PhysioNet/Computing in Cardiology (CinC) Challenge "
        "2017 data were split once into a development set and a locked internal test set; recordings labeled noisy "
        "were excluded from the binary atrial fibrillation (AF) task. PTB-XL lead I served as the external test set. "
        "Each record of the MIT-BIH Noise Stress Test Database was divided in time, with the first half used only for "
        "training augmentation and the second half only for test corruption (bw, baseline wander; ma, muscle "
        "artifact; em, electrode motion; SNR, signal-to-noise ratio).", above=False)

# Table S1: four-class challenge score
doc.add_heading("Supplementary table S1", level=1).paragraph_format.page_break_before = True
f4 = pd.read_csv(TAB / "fourclass_challenge.csv")
f4["model"] = f4["model"].map(lambda m: MODEL.get(m, m))
f4 = f4.rename(columns={"model": "Model", "n_seeds": "Seeds", "challenge_F1_NAO": "Challenge score, mean F1 (N, A, O) [95% CI]",
                        "f1_N": "F1 N", "f1_A": "F1 A", "f1_O": "F1 O", "f1_~": "F1 ~", "score4": "Mean F1, 4 classes"})
f4["Challenge score, mean F1 (N, A, O) [95% CI]"] = f4["Challenge score, mean F1 (N, A, O) [95% CI]"].map(ci)
for c in ["F1 N", "F1 A", "F1 O", "F1 ~", "Mean F1, 4 classes"]:
    f4[c] = f4[c].map(lambda x: f"{x:.3f}")
caption("Table S1.", "Four-class performance on the internal test set (1,706 recordings), scored with the "
        "PhysioNet/Computing in Cardiology Challenge 2017 metric (mean F1 score over the normal, atrial fibrillation "
        "and other-rhythm classes).")
table(f4)
note("N, normal rhythm; A, atrial fibrillation; O, other rhythm; ~, noisy; CI, confidence interval; 1-D CNN, "
     "one-dimensional convolutional neural network; RMSSD/RR, root mean square of successive RR-interval differences "
     "divided by the mean RR interval. The CNN was trained with one seed for computational reasons and the other "
     "models with five.")

# Table S2: external subgroups
doc.add_heading("Supplementary table S2", level=1).paragraph_format.page_break_before = True
sg = pd.read_csv(TAB / "subgroups_external.csv")
sg = sg[sg.variant == "standard"].drop(columns="variant")
sg["model"] = sg["model"].map(lambda m: MODEL.get(m, m))
for c in ["auroc", "auprc", "sensitivity", "specificity"]:
    sg[c] = sg[c].map(ci)
sg["subgroup"] = sg.subgroup.str.replace("sex=", "", regex=False).str.replace(">=", "≥", regex=False)
sg = sg.rename(columns={"model": "Model", "subgroup": "Subgroup", "n": "n", "AF_n": "AF n", "auroc": "AUROC",
                        "auprc": "AUPRC", "sensitivity": "Sensitivity", "specificity": "Specificity"})
caption("Table S2.", "Performance on the external test set (PTB-XL lead I, clean recordings) by sex and by age, "
        "with standard training. Values are means over five seeds with 95% patient-level bootstrap confidence "
        "intervals.")
table(sg, size=7.5)
note("Sex is as recorded in the PTB-XL metadata. AF, atrial fibrillation; AUROC, area under the receiver operating "
     "characteristic curve; AUPRC, area under the precision-recall curve.")

# Table S3: TRIPOD+AI mapping
doc.add_heading("Supplementary table S3", level=1).paragraph_format.page_break_before = True
md = (ROOT / "docs" / "TRIPOD_AI_checklist.md").read_text(encoding="utf-8")
rows = [l.strip().strip("|").split("|") for l in md.splitlines() if l.startswith("|")]
rows = [[c.strip() for c in r] for r in rows if not all(set(c.strip()) <= set("-:") for c in r)]
tr = pd.DataFrame(rows[1:], columns=rows[0])
caption("Table S3.", "Where the manuscript addresses each topic of the TRIPOD+AI reporting guideline (Collins et al "
        "2024). Topic names are paraphrased; file names refer to the public code repository.")
table(tr, size=7.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")

"""Build the supplementary-material Word file (Physiological Measurement layout).

    python scripts/build_supplement_docx.py                 # named version
    python scripts/build_supplement_docx.py --anonymous     # double-anonymous review version
    python scripts/build_supplement_docx.py --no-figures    # placeholders instead of the figures

Writes paper/submission/physiol_meas/supplementary[_anonymous][_no_figures].docx with the supplementary
methods (paper/supplement_methods.md), figure S1 (flow of recordings), tables S1-S3 built from outputs/ and
docs/ (four-class scores, external subgroups, TRIPOD+AI mapping), and the tables and figures moved out of the
main text (paper/supplement_items.md: tables S4-S18, figures S2-S6). IOP asks for a title (30 characters or
fewer) and a description (30 words or fewer) inside the file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
NO_FIGURES = "--no-figures" in sys.argv
ANONYMOUS = "--anonymous" in sys.argv
OUT = ROOT / "paper" / "submission" / "physiol_meas" / ("supplementary" + ("_anonymous" if ANONYMOUS else "")
                                       + ("_no_figures" if NO_FIGURES else "") + ".docx")
TITLE = "Stress-testing wearable ECG atrial fibrillation classifiers with real recorded noise"
SUPP_TITLE = "Extended methods and results"    # 28 characters
SUPP_DESCRIPTION = ("Extended methods, data audit, literature comparison, full external noise grid, beat-detection "
                    "validation, mechanism, natural-noise, augmentation-ablation, quality-measure and operating-point "
                    "results.")
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
for name, size in [("Title", 15), ("Heading 1", 13), ("Heading 2", 12)]:
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


TEXT_WIDTH = 6.5   # inches between the 1 in margins of a Letter page
CELL_PAD = 0.16    # inches of left + right cell margin in Word's Table Grid style


def _text_inches(text: str, size: float, bold: bool = False) -> float:
    """Printed width of text in Times New Roman, measured from the installed font (fallback: estimate)."""
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("timesbd.ttf" if bold else "times.ttf", 100)
        return font.getlength(text) * size / 100 / 72
    except Exception:
        return len(text) * size * 0.45 / 72


def column_widths(rows: list[list[str]], size: float) -> list[float] | None:
    """Same rule as the manuscript builder: each column gets the width of its longest cell (headers may wrap); if the
    table is too wide, number columns keep their width and only text columns wrap. None if even that does not fit."""
    ncol = len(rows[0])
    need, floor, numeric = [], [], []
    for j in range(ncol):
        body = [r[j].strip() for r in rows[1:]]
        head = rows[0][j].strip()
        w_body = max((_text_inches(x, size) for x in body), default=0)
        w_head = max((_text_inches(w, size, True) for w in head.split()), default=0)
        need.append(max(w_body, w_head) + CELL_PAD)
        short = max((len(x) for x in body), default=0) <= 12
        floor.append(need[-1] if short else
                     max([_text_inches(w, size) for x in body for w in x.split()] + [w_head]) + CELL_PAD)
        numeric.append(sum(ch.isdigit() for x in body for ch in x) > 0.4 * max(1, sum(len(x) for x in body)))
    if sum(need) <= TEXT_WIDTH:
        return [w * TEXT_WIDTH / sum(need) for w in need]
    fixed = sum(n for n, num in zip(need, numeric) if num)
    rest = TEXT_WIDTH - fixed
    text_need = sum(n for n, num in zip(need, numeric) if not num)
    widths = [n if num else max(f, n * rest / text_need) if text_need else n
              for n, f, num in zip(need, floor, numeric)]
    return widths if sum(widths) <= TEXT_WIDTH + 1e-6 else None


def table(df: pd.DataFrame, size: float = 8):
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    rows = [[c.replace("`", "") for c in r] for r in rows]
    ncol = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style, t.alignment, t.autofit = "Table Grid", WD_TABLE_ALIGNMENT.CENTER, False
    widths = None
    while widths is None and size >= 6.5:
        widths = column_widths(rows, size)
        if widths is None:
            size -= 0.5
    if widths is None:
        size, widths = 6.5, [TEXT_WIDTH / ncol] * ncol
    long = len(rows) > 28   # a table longer than a page may break; its header row repeats
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.width = Inches(widths[c])
            p = cell.paragraphs[0]
            p.paragraph_format.space_after, p.paragraph_format.line_spacing = Pt(0), 1.0
            p.paragraph_format.keep_with_next = (r < len(rows) - 1) if not long else (r < 2)
            run = p.add_run(val)
            run.font.size, run.bold = Pt(size), r == 0
    for r, tr in enumerate(t.rows):
        trpr = tr._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if r == 0 and long:
            hdr = OxmlElement("w:tblHeader"); hdr.set(qn("w:val"), "true"); trpr.append(hdr)


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


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|<sub>[^<]+</sub>)")


def runs(par, text: str, size: float | None = None):
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**"):
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("<sub>"):
            r = par.add_run(tok[5:-6]); r.font.subscript = True
        elif tok.startswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); r.italic = True
        else:
            r = par.add_run(tok)
        if size:
            r.font.size = Pt(size)


def render_md(path: Path):
    """Render a small markdown file: ## (new page) and ### headings, paragraphs, lists, tables, figures.

    A paragraph that starts with **Table** is a caption above its table, one that starts with **Figure** a caption
    below its figure, and the paragraph right after a table is that table's note."""
    lines, i, buf, after_table = path.read_text(encoding="utf-8").splitlines(), 0, [], False

    def flush():
        nonlocal buf, after_table
        if not buf:
            return
        text = " ".join(s.strip() for s in buf)
        m = re.match(r"\*\*((?:Table|Figure) S\d+[a-z]?(?: \(continued\))?\.)\*\* (.*)", text)
        if m:
            caption(m.group(1), m.group(2), above=m.group(1).startswith("Table"))
        elif after_table:
            note(text)
        else:
            runs(doc.add_paragraph(), text)
        buf, after_table = [], False

    while i < len(lines):
        ln, s = lines[i], lines[i].strip()
        if not s:
            flush(); i += 1; continue
        if ln.startswith("## "):
            flush(); doc.add_heading(ln[3:].strip(), level=1).paragraph_format.page_break_before = True; i += 1; continue
        if ln.startswith("### "):
            flush(); doc.add_heading(ln[4:].strip(), level=2); i += 1; continue
        fm = re.match(r"!\[[^\]]*\]\(([^)]+)\)", s)
        if fm:
            flush()
            pic = doc.add_paragraph(); pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pic.paragraph_format.keep_with_next, pic.paragraph_format.line_spacing = True, 1.0
            if NO_FIGURES:
                pic.add_run(f"[Insert {fm.group(1)} here, 15 cm wide]").bold = True
            else:
                pic.add_run().add_picture(str(ROOT / "paper" / "submission" / fm.group(1)), width=Inches(15 / 2.54))
            i += 1; continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            table(pd.DataFrame(rows[1:], columns=rows[0]), size=8.5 if len(rows[0]) <= 7 else 8)
            after_table = True
            continue
        if s.startswith("- "):
            flush()
            text = s[2:]; i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                text += " " + lines[i].strip(); i += 1
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3); p.paragraph_format.line_spacing = 1.15
            runs(p, text, size=11)
            continue
        buf.append(ln); i += 1
    flush()


doc.add_paragraph(SUPP_TITLE, style="Title")
p = doc.add_paragraph(); p.add_run("Description: ").bold = True; p.add_run(SUPP_DESCRIPTION)
p = doc.add_paragraph(); p.add_run("Supplementary material for: ").bold = True; p.add_run(TITLE)
if not ANONYMOUS:
    doc.add_paragraph("Tejas Menneni, Columbus State Community College, Columbus, OH, USA (tejas.menneni@gmail.com)")
    doc.add_paragraph("Code and all result tables: https://github.com/TSFIRE99-bot/wearable-ecg-noise-stress-test "
                      "(release v1.0.3, archived at https://doi.org/10.5281/zenodo.24352330).")

render_md(ROOT / "paper" / "supplement_methods.md")

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

render_md(ROOT / "paper" / "supplement_items.md")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")

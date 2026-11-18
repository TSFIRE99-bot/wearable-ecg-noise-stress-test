"""Build the submission Word file from paper/manuscript.md (Physiological Measurement layout).

    python scripts/build_manuscript_docx.py                 # named version, figures embedded
    python scripts/build_manuscript_docx.py --anonymous     # double-anonymous review version
    python scripts/build_manuscript_docx.py --no-figures    # placeholders instead of pictures

Writes paper/submission/physiol_meas/manuscript[_anonymous][_no_figures].docx. Layout follows the IOP
author guidelines: Times New Roman 12 pt, 1.5 line spacing, figures and tables placed in the
text next to their captions. Figures are read from paper/submission/figureN.png (made by
scripts/make_journal_figures.py) and inserted 15 cm wide. Markdown lines of the form
![Figure N](figureN.png) mark where each figure goes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "manuscript.md"
SUBMISSION = ROOT / "paper" / "submission"          # journal figures live here
OUT_DIR = SUBMISSION / "physiol_meas"               # Physiological Measurement files
FIG_DIRS = (SUBMISSION, ROOT / "outputs" / "figures")
NO_FIGURES = "--no-figures" in sys.argv
ANONYMOUS = "--anonymous" in sys.argv
OUT = OUT_DIR / ("manuscript" + ("_anonymous" if ANONYMOUS else "") + ("_no_figures" if NO_FIGURES else "") + ".docx")
FONT = "Times New Roman"
AUTHOR = "Tejas Menneni"

# Identifying details removed from the anonymous version.
ANON_SUBS = [
    (r"https://github\.com/TSFIRE99-bot/wearable-ecg-noise-stress-test", "[repository link withheld for anonymous review]"),
    (r"https://doi\.org/10\.5281/zenodo\.\d+", "[archive DOI withheld for anonymous review]"),
    (AUTHOR, "[author name withheld for anonymous review]"),
]

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|<sub>[^<]+</sub>)")


def anonymise(text: str) -> str:
    if ANONYMOUS:
        for pat, rep in ANON_SUBS:
            text = re.sub(pat, rep, text)
    return text


def add_runs(par, text: str, size: float | None = None, bold_all: bool = False):
    for tok in INLINE.split(anonymise(text)):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("`") and tok.endswith("`"):
            r = par.add_run(tok[1:-1]); r.font.name = "Courier New"
        elif tok.startswith("<sub>"):
            r = par.add_run(tok[5:-6]); r.font.subscript = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); r.italic = True
        else:
            r = par.add_run(tok)
        if bold_all:
            r.bold = True
        if size:
            r.font.size = Pt(size)


# --------------------------------------------------------------------------- document setup
doc = Document()
doc.core_properties.author = "" if ANONYMOUS else AUTHOR
doc.core_properties.last_modified_by = "" if ANONYMOUS else AUTHOR
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1))
# continuous line numbers in the margin, so reviewers can refer to lines
ln_num = OxmlElement("w:lnNumType")
ln_num.set(qn("w:countBy"), "1"); ln_num.set(qn("w:restart"), "continuous"); ln_num.set(qn("w:distance"), "283")
cols = sec._sectPr.find(qn("w:cols"))
if cols is not None:
    cols.addprevious(ln_num)
else:
    sec._sectPr.append(ln_num)

st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.5
for name, size, before in [("Title", 16, 0), ("Heading 1", 13, 14), ("Heading 2", 12, 10)]:
    s = doc.styles[name]
    s.font.name, s.font.size, s.font.bold = FONT, Pt(size), True
    s.font.italic = False
    s.font.color.rgb = RGBColor(0, 0, 0)
    rpr = s.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts"); rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), FONT)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(attr), None)
    s.paragraph_format.space_before, s.paragraph_format.space_after = Pt(before), Pt(6)
    s.paragraph_format.keep_with_next = True
    s.paragraph_format.line_spacing = 1.15
title_ppr = doc.styles["Title"].element.pPr   # drop the coloured rule Word's Title style draws
if title_ppr is not None and title_ppr.find(qn("w:pBdr")) is not None:
    title_ppr.remove(title_ppr.find(qn("w:pBdr")))
for name in ("List Number", "List Bullet"):
    doc.styles[name].font.name = FONT

# page numbers in the footer
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run()
for tag, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
    if tag:
        el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
    else:
        el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
    run._r.append(el)


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
    """Give each column the width of its longest cell (headers may wrap between words). If the table is too wide,
    columns of numbers keep their full width and only the text columns wrap. None if even that does not fit."""
    plain = lambda s: re.sub(r"\*\*|`|<sub>|</sub>", "", s.strip())  # noqa: E731
    ncol = len(rows[0])
    need, floor, numeric = [], [], []
    for j in range(ncol):
        body = [plain(r[j]) for r in rows[1:]]
        head = plain(rows[0][j])
        w_body = max((_text_inches(s, size) for s in body), default=0)
        w_head = max((_text_inches(w, size, True) for w in head.split()), default=0)
        need.append(max(w_body, w_head) + CELL_PAD)
        short = max((len(s) for s in body), default=0) <= 12   # short labels such as "EM −6 dB" never wrap
        floor.append(need[-1] if short else
                     max([_text_inches(w, size) for s in body for w in s.split()] + [w_head]) + CELL_PAD)
        numeric.append(sum(ch.isdigit() for s in body for ch in s) > 0.4 * max(1, sum(len(s) for s in body)))
    if sum(need) <= TEXT_WIDTH:
        return [w * TEXT_WIDTH / sum(need) for w in need]
    fixed = sum(n for n, num in zip(need, numeric) if num)
    rest = TEXT_WIDTH - fixed
    text_need = sum(n for n, num in zip(need, numeric) if not num)
    widths = [n if num else max(f, n * rest / text_need) if text_need else n
              for n, f, num in zip(need, floor, numeric)]
    return widths if sum(widths) <= TEXT_WIDTH + 1e-6 else None


def add_table(rows: list[list[str]]):
    """Black-and-white grid table; cell text is single spaced and smaller than body text. Columns are sized to their
    contents so that numbers with confidence intervals stay on one line where possible."""
    ncol = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    size, widths = 9.0, None
    while widths is None and size >= 7:
        widths = column_widths(rows, size)
        if widths is None:
            size -= 0.5
    if widths is None:   # nothing fits: equal columns at the smallest size, cells wrap
        size, widths = 7.0, [TEXT_WIDTH / ncol] * ncol
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = i < len(rows) - 1   # keep the whole table on one page
            add_runs(p, cell.strip(), size=size, bold_all=(i == 0))
            # size the paragraph mark too, so an empty cell is not as tall as a line of body text
            mark = OxmlElement("w:rPr")   # rPr is the last child of pPr here, so appending keeps schema order
            sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2))); mark.append(sz)
            p._p.get_or_add_pPr().append(mark)
    for r in t.rows:   # keep each table row on one page
        trpr = r._tr.get_or_add_trPr()
        el = OxmlElement("w:cantSplit"); trpr.append(el)


def small_paragraph(text: str, size: float = 10):
    """Table notes: single spaced, slightly smaller."""
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(3)
    add_runs(p, text, size=size)


def add_figure(fname: str, label: str):
    pic = doc.add_paragraph(); pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.keep_with_next = True
    pic.paragraph_format.line_spacing = 1.0
    if NO_FIGURES:
        pic.add_run(f"[Insert {label} here: {fname}, 15 cm wide]").bold = True
        return
    src = next((d / fname for d in FIG_DIRS if (d / fname).exists()), None)
    if src is None:
        raise SystemExit(f"missing figure file {fname}; run scripts/make_journal_figures.py first")
    pic.add_run().add_picture(str(src), width=Inches(15 / 2.54))


# --------------------------------------------------------------------------- parse markdown
lines = SRC.read_text(encoding="utf-8").splitlines()
i, buf = 0, []
section = ""
after_table = False   # the paragraph right after a table is its note


def flush():
    global buf, after_table
    if buf:
        text = " ".join(s.strip() for s in buf)
        if after_table:
            small_paragraph(text)
        else:
            p = doc.add_paragraph()
            if text.startswith("**Table") or text.startswith("**Figure"):
                p.paragraph_format.line_spacing = 1.15
                if text.startswith("**Table"):
                    p.paragraph_format.keep_with_next = True   # caption stays above its table
            add_runs(p, text)
        buf, after_table = [], False


while i < len(lines):
    ln = lines[i]
    s = ln.strip()
    if s.startswith(">") or s == "---":
        flush(); i += 1; continue
    if not s:
        flush(); i += 1; continue
    if ln.startswith("# "):
        flush(); doc.add_paragraph(ln[2:].strip(), style="Title"); i += 1
        if ANONYMOUS:   # drop the author, affiliation and e-mail lines that follow the title
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
        continue
    if ln.startswith("## "):
        flush(); section = ln[3:].strip()
        doc.add_heading(section, level=1); i += 1; continue
    if ln.startswith("### "):
        flush(); doc.add_heading(ln[4:].strip(), level=2); i += 1; continue
    fm = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
    if fm:
        flush(); add_figure(fm.group(2), fm.group(1)); i += 1; continue
    if s.startswith("|"):
        flush()
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = lines[i].strip().strip("|").split("|")
            if not all(re.fullmatch(r"\s*:?-{3,}:?\s*", c) for c in cells):
                rows.append(cells)
            i += 1
        add_table(rows)
        after_table = True
        continue
    m = re.match(r"^(\d+)\.\s+(.*)", s) or re.match(r"^-\s+(.*)", s)
    if m:
        flush()
        text = m.group(m.lastindex)
        i += 1
        while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
            text += " " + lines[i].strip(); i += 1
        refs = section == "References"
        # literal numbers (Word's auto-numbering would run on from one list to the next)
        lead = "" if refs else (s.split(".")[0] + ".	" if s[0].isdigit() else "•	")
        p = doc.add_paragraph()
        p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.3), Inches(-0.3)
        if not refs:
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
        else:   # the reference list is single spaced
            p.paragraph_format.line_spacing, p.paragraph_format.space_after = 1.0, Pt(4)
        add_runs(p, lead + text)
        continue
    buf.append(ln); i += 1
flush()

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")

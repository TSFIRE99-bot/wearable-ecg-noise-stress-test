"""Build the submission Word file from paper/manuscript.md.

    python scripts/build_manuscript_docx.py

Writes paper/submission/manuscript.docx. Internal author notes are removed: the
draft-status box, the "(Table N <- file.csv)" source pointers, and the reference-list
reminder. Figures are embedded above their legends.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (kept for easy landscape tweaks)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "manuscript.md"
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "paper" / "submission" / "manuscript.docx"
FONT = "Calibri"


# --------------------------------------------------------------------------- text clean-up
FILE = r"`[^`]+\.(?:csv|png)`"


def strip_pointers(text: str) -> str:
    """Turn author notes like '*(Table 4 <- `failure_modes.csv`)*' into '(Table 4)'.

    Arrows and output file names are removed; the reader-facing labels (Table N, Fig. N)
    and any real remark stay. Segments left as a single lowercase word are dropped."""
    text = re.sub(rf"\s*←\s*{FILE}(?:\s*,\s*{FILE})*", "", text)

    def tidy(m):
        parts = [p.strip() for p in m.group(1).split(";")]
        parts = [p for p in parts if p and not re.fullmatch(FILE, p) and not re.fullmatch(r"[a-z]+", p)]
        return f"({'; '.join(parts)})" if parts else ""

    text = re.sub(r"\*?\(([^()]*)\)\*?", lambda m: tidy(m) if (m.group(0).startswith("*(") or "`" in m.group(1)
                                                               or "Table" in m.group(1) or "Fig" in m.group(1))
                  else m.group(0), text)
    text = re.sub(rf"\s*\({FILE}\)", "", text)
    return re.sub(r"\s{2,}", " ", text).replace("( ", "(")


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|<sub>[^<]+</sub>)")


def add_runs(par, text: str, size: float | None = None, bold_all: bool = False):
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("`") and tok.endswith("`"):
            r = par.add_run(tok[1:-1]); r.font.name = "Consolas"
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
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1))

st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for name, size, before in [("Title", 16, 0), ("Heading 1", 13, 14), ("Heading 2", 11.5, 10)]:
    s = doc.styles[name]
    s.font.name, s.font.size, s.font.bold = FONT, Pt(size), True
    s.font.color.rgb = RGBColor(0, 0, 0)
    s.element.rPr.rFonts.set(qn("w:asciiTheme"), "") if s.element.rPr is not None else None
    s.paragraph_format.space_before, s.paragraph_format.space_after = Pt(before), Pt(6)
    s.paragraph_format.keep_with_next = True

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


def add_table(rows: list[list[str]]):
    ncol = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    size = 8 if ncol <= 5 else 7
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            add_runs(p, cell.strip(), size=size, bold_all=(i == 0))
    doc.add_paragraph()


# --------------------------------------------------------------------------- parse markdown
lines = SRC.read_text(encoding="utf-8").splitlines()
i, buf, in_refs, in_legends = 0, [], False, False


def flush():
    global buf
    if buf:
        text = strip_pointers(" ".join(s.strip() for s in buf))
        if text and not text.startswith("*(Verify every entry"):
            add_runs(doc.add_paragraph(), text)
    buf = []


while i < len(lines):
    ln = lines[i]
    s = ln.strip()
    if s.startswith(">") or s == "---":                      # draft-status box, rules
        flush(); i += 1; continue
    if not s:
        flush(); i += 1; continue
    if ln.startswith("# "):
        flush(); doc.add_paragraph(ln[2:].strip(), style="Title"); i += 1; continue
    if ln.startswith("## "):
        flush()
        title = ln[3:].strip()
        in_refs, in_legends = title == "References", title == "Figure legends"
        if title == "Tables":
            doc.add_page_break()
        doc.add_heading(title, level=1); i += 1; continue
    if ln.startswith("### "):
        flush(); doc.add_heading(ln[4:].strip(), level=2); i += 1; continue
    if s.startswith("|"):                                     # table block
        flush()
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r"\s*:?-{3,}:?\s*", c) for c in cells):
                rows.append(cells)
            i += 1
        add_table(rows); continue
    m = re.match(r"^(\d+)\.\s+(.*)", s) or re.match(r"^-\s+(.*)", s)
    if m and not in_legends:                                   # list item (+ indented continuation)
        flush()
        text = m.group(m.lastindex)
        i += 1
        while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
            text += " " + lines[i].strip(); i += 1
        style = "List Number" if s[0].isdigit() else ("Normal" if in_refs else "List Bullet")
        p = doc.add_paragraph(style=style)
        if in_refs:
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.3), Inches(-0.3)
        add_runs(p, strip_pointers(text))
        continue
    if in_legends and s.startswith("**") and "Figure" in s:   # legend: embed the figure first
        flush()
        fm = re.search(r"`(fig[^`]+\.png)`", s)
        if fm and (FIG / fm.group(1)).exists():
            doc.add_page_break()
            pic = doc.add_paragraph(); pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
            width = 6.0 if "flow" in fm.group(1) else 6.5
            pic.add_run().add_picture(str(FIG / fm.group(1)), width=Inches(width))
        add_runs(doc.add_paragraph(), strip_pointers(s))
        i += 1; continue
    buf.append(ln); i += 1
flush()

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")

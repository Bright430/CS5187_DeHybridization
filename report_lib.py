"""Build the IEEE-format project report from the experiment outputs.

Base: the official IEEE Transactions Word template (styles are reused).
Run:  python build_report.py
"""

import os
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(
    HERE,
    "IEEE-Transactions-Word-templates-and-instructions",
    "Transactions-template-and-instructions-on-how-to-create-your-article-formatted (4).docx",
)
OUT = os.path.join(HERE, "CS5187_Report_DeHybridization.docx")
FONT = "Times New Roman"


# --------------------------------------------------------------------------
# low level helpers
# --------------------------------------------------------------------------
def clean(doc):
    """Drop every template paragraph, keep the final sectPr."""
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def set_cols(section, num, space_twips=288):
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    if num == 1:
        cols.set(qn("w:equalWidth"), "1")


def fmt(par, size=10, align=None, before=0, after=0, indent=None, spacing=1.0):
    pf = par.paragraph_format
    if align is not None:
        pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    if indent is not None:
        pf.first_line_indent = Pt(indent)
    for r in par.runs:
        r.font.name = FONT
        r.font.size = Pt(size)
    return par


def run(par, text, size=None, bold=False, italic=False, caps=False, sub=False, sup=False):
    r = par.add_run(text)
    r.font.name = FONT
    if size:
        r.font.size = Pt(size)
    r.font.bold = bold or None
    r.font.italic = italic or None
    r.font.small_caps = caps or None
    if sub:
        r.font.subscript = True
    if sup:
        r.font.superscript = True
    return r


def math_runs(par, text, size=10, italic_var=True):
    """Render _{sub} and ^{sup} markers as real sub/superscripts."""
    for tok in re.split(r"([_^]\{[^}]*\})", text):
        if not tok:
            continue
        if tok.startswith("_{"):
            run(par, tok[2:-1], size=size, sub=True)
        elif tok.startswith("^{"):
            run(par, tok[2:-1], size=size, sup=True)
        else:
            run(par, tok, size=size)


def body(doc, text, indent=10.1, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=0):
    p = doc.add_paragraph(style="Text")
    math_runs(p, text, size=size)
    fmt(p, size=size, align=align, indent=indent, after=after, spacing=1.0)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph(style="Heading 1" if level == 1 else "Heading 2")
    if level == 1:
        run(p, text, size=10, caps=True)
        fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, before=11, after=5)
    else:
        run(p, text, size=10, italic=True)
        fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=3)
    return p


def figure(doc, path, width, caption, cap_width=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(os.path.join(HERE, path), width=Inches(width))
    c = doc.add_paragraph(style="Figure Caption")
    math_runs(c, caption, size=8)
    fmt(c, size=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=8, spacing=1.0)
    return c


def _border(el, edge, sz):
    tag = "w:" + edge
    e = OxmlElement(tag)
    e.set(qn("w:val"), "single" if sz else "nil")
    if sz:
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
    el.append(e)


def table(doc, caption, header, rows, widths, first_left=True, size=8):
    cap = doc.add_paragraph(style="Table Title")
    run(cap, "TABLE " + caption.split("|")[0].strip(), size=8, caps=True)
    fmt(cap, size=8, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=0)
    cap2 = doc.add_paragraph(style="Table Title")
    run(cap2, caption.split("|")[1].strip(), size=8, caps=True)
    fmt(cap2, size=8, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=3)

    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    _border(borders, "top", 12)
    _border(borders, "bottom", 12)
    _border(borders, "left", 0)
    _border(borders, "right", 0)
    _border(borders, "insideH", 0)
    _border(borders, "insideV", 0)
    tblPr.append(borders)

    def fill(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER):
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = align
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        math_runs(p, text, size=size)

    for j, h in enumerate(header):
        fill(t.rows[0].cells[j], h)
        tcPr = t.rows[0].cells[j]._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        _border(b, "bottom", 6)
        tcPr.append(b)

    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            al = WD_ALIGN_PARAGRAPH.LEFT if (first_left and j == 0) else WD_ALIGN_PARAGRAPH.CENTER
            fill(t.rows[i].cells[j], v, align=al)

    for j, w in enumerate(widths):
        for r in t.rows:
            r.cells[j].width = Inches(w)

    for r in t.rows:                      # keep rows from splitting over columns
        trPr = r._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))

    sp = doc.add_paragraph()
    fmt(sp, size=4, before=0, after=4)
    return t


def caption_note(doc, text):
    p = doc.add_paragraph()
    math_runs(p, text, size=7.5)
    fmt(p, size=7.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=8)
    return p

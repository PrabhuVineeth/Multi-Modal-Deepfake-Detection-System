"""
Minimal Markdown -> Word (.docx) converter for the project docs.

Handles the constructs used in PROJECT_DOCUMENTATION.md: #/##/### headings,
pipe tables (with a header row), - bullet and 1. numbered lists, > blockquotes,
--- rules, and inline **bold**, *italic*, `code`.

Usage:
    python md_to_docx.py PROJECT_DOCUMENTATION.md PROJECT_DOCUMENTATION.docx
"""

import re
import sys

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")


def add_runs(paragraph, text):
    """Add text to a paragraph, honoring inline **bold**, *italic*, `code`."""
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = paragraph.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("*") and tok.endswith("*"):
            r = paragraph.add_run(tok[1:-1]); r.italic = True
        elif tok.startswith("`") and tok.endswith("`"):
            r = paragraph.add_run(tok[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(10)
        else:
            paragraph.add_run(tok)


def is_table_row(line):
    return line.strip().startswith("|") and line.strip().endswith("|")


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md_path, docx_path):
    lines = open(md_path, encoding="utf-8").read().splitlines()
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── table block ──
        if is_table_row(line) and i + 1 < len(lines) and set(cells(lines[i + 1])[0]) <= set("-: "):
            header = cells(line)
            body = []
            j = i + 2
            while j < len(lines) and is_table_row(lines[j]):
                body.append(cells(lines[j]))
                j += 1
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Light Grid Accent 1"
            for k, h in enumerate(header):
                cell = table.rows[0].cells[k]
                cell.paragraphs[0].text = ""
                run = cell.paragraphs[0].add_run(re.sub(r"\*\*|\*|`", "", h))
                run.bold = True
            for row in body:
                r = table.add_row().cells
                for k, val in enumerate(row):
                    if k < len(r):
                        r[k].paragraphs[0].text = ""
                        add_runs(r[k].paragraphs[0], val)
            doc.add_paragraph()
            i = j
            continue

        # ── headings ──
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1:
                h = doc.add_heading(level=0 if text and i < 3 else 1)
                add_runs(h, text)
            else:
                h = doc.add_heading(level=min(level, 4))
                add_runs(h, text)
            i += 1
            continue

        # ── horizontal rule ──
        if stripped in ("---", "***", "___"):
            i += 1
            continue

        # ── blockquote ──
        if stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(18)
            run_text = stripped.lstrip("> ").strip()
            r = p.add_run(re.sub(r"\*(.+?)\*", r"\1", run_text)); r.italic = True
            i += 1
            continue

        # ── bullet / numbered lists ──
        m_bullet = re.match(r"^(\s*)[-*]\s+(.*)", line)
        m_num = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if m_bullet:
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, m_bullet.group(2))
            i += 1
            continue
        if m_num:
            p = doc.add_paragraph(style="List Number")
            add_runs(p, m_num.group(2))
            i += 1
            continue

        # ── blank line ──
        if not stripped:
            i += 1
            continue

        # ── normal paragraph ──
        p = doc.add_paragraph()
        add_runs(p, stripped)
        i += 1

    doc.save(docx_path)
    print(f"Wrote {docx_path}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "PROJECT_DOCUMENTATION.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".docx"
    convert(src, dst)

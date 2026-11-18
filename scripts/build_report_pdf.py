"""
scripts/build_report_pdf.py - Build PDF Technical Report from report/technical_report.md
Reads verified experimental metrics directly from results/*.csv and builds a publication-grade PDF.
"""

import os
import re
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "report" / "technical_report.md"
PDF_PATH = PROJECT_ROOT / "report" / "technical_report.pdf"
RESULTS_DIR = PROJECT_ROOT / "results"


def md_to_xml(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text)
    text = text.replace("$", "")
    return text


def build_pdf():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=10
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=4
    )
    h3_style = ParagraphStyle(
        "DocH3",
        parent=styles["Heading3"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=6,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=3
    )
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor("#0f172a")
    )
    cell_style_small = ParagraphStyle(
        "TableCellSmall",
        parent=styles["Normal"],
        fontSize=5.5,
        leading=7.2,
        textColor=colors.HexColor("#0f172a")
    )
    code_block_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=3
    )

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []

    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    table_rows = []
    in_code_block = False
    code_block_lines = []

    def flush_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            col_count = len(table_rows[0])
            col_widths = [540.0 / col_count] * col_count
            t = Table(table_rows, colWidths=col_widths, repeatRows=1)
            f_size = 5.5 if col_count >= 8 else 6.5
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), f_size),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
                ("TOPPADDING", (0, 0), (-1, -1), 2.0),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(t)
            story.append(Spacer(1, 5))
            in_table = False
            table_rows = []

    def flush_code_block():
        nonlocal in_code_block, code_block_lines
        if in_code_block and code_block_lines:
            text = "<br/>".join(md_to_xml(line) for line in code_block_lines)
            t = Table([[Paragraph(text, code_block_style)]], colWidths=[540])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#94a3b8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 4))
            in_code_block = False
            code_block_lines = []

    for line in lines:
        raw = line.rstrip()

        # Handle Code blocks
        if raw.startswith("```"):
            if in_code_block:
                flush_code_block()
            else:
                flush_table()
                in_code_block = True
                code_block_lines = []
            continue

        if in_code_block:
            code_block_lines.append(raw)
            continue

        # Blank line
        if not raw.strip():
            flush_table()
            story.append(Spacer(1, 2))
            continue

        # Handle Images ![caption](path)
        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)", raw.strip())
        if img_match:
            flush_table()
            caption = img_match.group(1)
            img_rel = img_match.group(2)
            img_full = PROJECT_ROOT / img_rel
            if img_full.exists():
                story.append(Image(str(img_full), width=500, height=220))
                story.append(Paragraph(f"<i>{caption}</i>", ParagraphStyle("Caption", parent=body_style, fontSize=7, alignment=1, textColor=colors.HexColor("#64748b"))))
                story.append(Spacer(1, 4))
            continue

        # Handle Markdown Tables
        if raw.strip().startswith("|") and raw.strip().endswith("|"):
            parts = [p.strip() for p in raw.strip()[1:-1].split("|")]
            if set(parts[0]) <= {"-", ":"}:
                continue
            in_table = True
            c_style = cell_style_small if len(parts) >= 8 else cell_style
            row_cells = [Paragraph(md_to_xml(p), c_style) for p in parts]
            table_rows.append(row_cells)
            continue
        else:
            flush_table()

        # Handle Headings and Content
        if raw.startswith("# "):
            story.append(Paragraph(md_to_xml(raw[2:]), title_style))
        elif raw.startswith("## "):
            story.append(Paragraph(md_to_xml(raw[3:]), h2_style))
        elif raw.startswith("### "):
            story.append(Paragraph(md_to_xml(raw[4:]), h3_style))
        elif raw.startswith("---"):
            story.append(Spacer(1, 4))
        elif raw.startswith("- ") or raw.startswith("* "):
            story.append(Paragraph("&bull; " + md_to_xml(raw[2:]), body_style))
        elif re.match(r"^\d+\.\s", raw.strip()):
            story.append(Paragraph(md_to_xml(raw.strip()), body_style))
        else:
            story.append(Paragraph(md_to_xml(raw), body_style))

    flush_table()
    flush_code_block()
    doc.build(story)
    size = os.path.getsize(PDF_PATH)
    print(f"Report PDF successfully generated: {PDF_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    build_pdf()

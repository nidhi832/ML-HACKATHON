"""
scripts/build_report_pdf.py - Build PDF Technical Report from report/technical_report.md
Uses reportlab to generate a publication-ready PDF deliverable.
"""

import os
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "report" / "technical_report.md"
PDF_PATH = PROJECT_ROOT / "report" / "technical_report.pdf"


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
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=5
    )
    h3_style = ParagraphStyle(
        "DocH3",
        parent=styles["Heading3"],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4
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

    def flush_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            col_count = len(table_rows[0])
            col_widths = [540.0 / col_count] * col_count
            t = Table(table_rows, colWidths=col_widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))
            in_table = False
            table_rows = []

    for line in lines:
        raw = line.strip()
        if not raw:
            flush_table()
            story.append(Spacer(1, 3))
            continue

        if raw.startswith("|") and raw.endswith("|"):
            parts = [p.strip() for p in raw[1:-1].split("|")]
            if set(parts[0]) <= {"-", ":"}:
                continue
            in_table = True
            row_cells = [Paragraph(md_to_xml(p), body_style) for p in parts]
            table_rows.append(row_cells)
            continue
        else:
            flush_table()

        if raw.startswith("# "):
            story.append(Paragraph(md_to_xml(raw[2:]), title_style))
        elif raw.startswith("## "):
            story.append(Paragraph(md_to_xml(raw[3:]), h2_style))
        elif raw.startswith("### "):
            story.append(Paragraph(md_to_xml(raw[4:]), h3_style))
        elif raw.startswith("---"):
            story.append(Spacer(1, 6))
        elif raw.startswith("- ") or raw.startswith("* "):
            story.append(Paragraph("&bull; " + md_to_xml(raw[2:]), body_style))
        elif re.match(r"^\d+\.\s", raw):
            story.append(Paragraph(md_to_xml(raw), body_style))
        else:
            story.append(Paragraph(md_to_xml(raw), body_style))

    flush_table()
    doc.build(story)
    print(f"Report PDF successfully generated: {PDF_PATH} ({os.path.getsize(PDF_PATH):,} bytes)")


if __name__ == "__main__":
    build_pdf()

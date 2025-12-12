import os
import json
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm


# =====================================================
#                 DATA LOADING
# =====================================================

def load_hti_and_charts():
    """
    Load HTI Excel from ./data/hti/employees_hti.xlsx
    and chart images from ./data/results/charts/.
    Does NOT modify the Excel file.
    """
    hti_path = "./data/hti/employees_hti.xlsx"
    charts_dir = "./data/charts/"

    if not os.path.exists(hti_path):
        raise FileNotFoundError(f"HTI file not found at: {hti_path}")

    hti_df = pd.read_excel(hti_path)

    chart_files: List[str] = []
    if os.path.exists(charts_dir):
        chart_files = sorted(
            [
                os.path.join(charts_dir, f)
                for f in os.listdir(charts_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    return hti_df, chart_files


def find_hti_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic to find the HTI score column."""
    candidates = [c for c in df.columns if "hti" in c.lower() or "threat" in c.lower()]
    if candidates:
        return candidates[0]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        return numeric_cols[-1]

    return None


# =====================================================
#            FALLBACK SUMMARY (NO LLM)
# =====================================================

def default_summary(df: pd.DataFrame, hti_col: Optional[str]) -> Dict[str, Any]:
    """Fallback summary if Ollama is not available or fails."""
    total_employees = len(df)
    if hti_col and hti_col in df.columns:
        col = df[hti_col].astype(float)
        exec_summary = (
            f"This report presents the Human Threat Index (HTI) analysis "
            f"for {total_employees} employees. The HTI score column is '{hti_col}', "
            f"with values ranging from {col.min():.3f} to {col.max():.3f} "
            f"and an average of {col.mean():.3f}."
        )
        key_findings = [
            "HTI scores show a spread of risk levels across the workforce, indicating heterogeneous insider risk.",
            "Employees in the higher HTI band should be prioritized for contextual review and closer monitoring.",
            "Mid-range HTI scores may indicate users who could benefit from targeted awareness and policy reinforcement.",
            "Low HTI scores still require baseline controls, periodic review, and monitoring for changes over time.",
            "Correlating HTI with role, access level, and historical events will refine risk prioritization."
        ]
    else:
        exec_summary = (
            f"This report presents the Human Threat Index (HTI) dataset for "
            f"{total_employees} employees. No single numeric HTI column could be "
            f"automatically identified, but the complete dataset is included for analysis."
        )
        key_findings = [
            "The HTI dataset aggregates employee-related attributes that can be used to derive a risk score.",
            "A dedicated numeric risk model can be constructed from this dataset via feature engineering.",
            "Integration with SOC and UEBA pipelines can support continuous insider risk detection.",
            "Segmentation by department, role, or geography can highlight concentrated areas of risk.",
            "Maintaining regular updates to the HTI dataset enables longitudinal tracking of risk posture."
        ]

    recommendations = (
        "Use this HTI dataset as a foundation for risk-based monitoring, access governance, and awareness programs. "
        "Higher-risk cohorts should be reviewed with security operations, HR, and compliance to ensure appropriate "
        "controls, while maintaining privacy, proportionality, and fairness in all decisions."
    )

    return {
        "executive_summary": exec_summary,
        "key_findings": key_findings,
        "recommendations": recommendations,
    }


# =====================================================
#                 OLLAMA UTILITIES
# =====================================================

def call_ollama(prompt: str, model: str = "llama3") -> str:
    """
    Call a local Ollama model via CLI.
    Assumes 'ollama' is installed and model is available.
    """
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr.strip()}")

    return result.stdout.strip()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from arbitrary text.
    If it fails, raise ValueError.
    """
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object inside the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("No valid JSON object found in LLM output")


def generate_llm_summary(df: pd.DataFrame, hti_col: Optional[str]) -> Dict[str, Any]:
    """
    Use local Ollama LLM to generate a structured JSON summary of the HTI dataset.
    Does NOT modify any records; only reads them.
    If anything fails, falls back to default_summary().
    """
    model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3.1")

    try:
        total_employees = len(df)

        if hti_col and hti_col in df.columns:
            col = df[hti_col].astype(float)
            stats = {
                "hti_column": hti_col,
                "min_hti": float(col.min()),
                "max_hti": float(col.max()),
                "avg_hti": float(col.mean()),
            }
        else:
            stats = {
                "hti_column": None,
                "min_hti": None,
                "max_hti": None,
                "avg_hti": None,
            }

        # Limit sample for prompt size
        sample_rows = df.head(20)
        sample_csv = sample_rows.to_csv(index=False)

        # Build prompt template without f-string complexity
        template = """
You are a cybersecurity analyst specializing in insider threat and human risk modeling.

You are given:
- A Human Threat Index (HTI) dataset of {{TOTAL_EMPLOYEES}} employees.
- Statistics about the HTI score column (if available).
- A small sample of the dataset in CSV format.

Your tasks:
1. Write a concise executive summary (4–7 sentences) describing the overall risk posture.
2. Provide exactly 5 key findings as short bullet-style strings.
3. Provide a recommendations paragraph (3–5 sentences) focused on next steps for security, monitoring, and governance.

Rules:
- Do NOT generate any employee names, IDs, departments, or fabricated records.
- Speak only in aggregate terms (e.g., "some employees", "a subset of users").
- Base reasoning strictly on HTI statistics and sample data.
- Output must be valid JSON ONLY.

HTI statistics (JSON):
<<HTI_STATS>>
{{HTI_STATS_JSON}}
<<END_STATS>>

Sample of the dataset (CSV):
<<CSV_SAMPLE>>
{{CSV_SAMPLE}}
<<END_CSV>>

Return ONLY valid JSON in this structure:

{
  "executive_summary": "string",
  "key_findings": [
    "string",
    "string",
    "string",
    "string",
    "string"
  ],
  "recommendations": "string"
}
"""

        prompt = (
            template
            .replace("{{TOTAL_EMPLOYEES}}", str(total_employees))
            .replace("{{HTI_STATS_JSON}}", json.dumps(stats, indent=2))
            .replace("{{CSV_SAMPLE}}", sample_csv)
        )

        raw_output = call_ollama(prompt, model=model_name)
        parsed = extract_json_from_text(raw_output)

        # Basic shape validation
        if (
            isinstance(parsed, dict)
            and "executive_summary" in parsed
            and "key_findings" in parsed
            and "recommendations" in parsed
        ):
            return parsed

        print("⚠️ Ollama JSON shape unexpected – using default summary.")
        return default_summary(df, hti_col)

    except Exception as e:
        print(f"⚠️ Ollama LLM call failed: {e}")
        return default_summary(df, hti_col)


# =====================================================
#            PDF BUILDING HELPERS
# =====================================================

def build_exec_section(summary: Dict[str, Any], styles):
    elements: List[Any] = []

    elements.append(Paragraph("<b>1. Executive Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    exec_text = summary.get("executive_summary", "")
    for block in exec_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        elements.append(Paragraph(block.replace("\n", "<br/>"), styles["BodyText"]))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>2. Key Findings</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    key_findings = summary.get("key_findings", [])
    for item in key_findings:
        text = f"• {item}"
        elements.append(Paragraph(text, styles["BodyText"]))
        elements.append(Spacer(1, 3))

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>3. Recommendations</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    rec_text = summary.get("recommendations", "")
    for block in rec_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        elements.append(Paragraph(block.replace("\n", "<br/>"), styles["BodyText"]))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 18))
    return elements


def build_full_hti_table(df: pd.DataFrame, styles):
    """
    Build a table with the ENTIRE HTI Excel content (all rows, all columns).
    ReportLab will auto-split it across pages.
    """
    elements: List[Any] = []

    elements.append(PageBreak())
    elements.append(
        Paragraph("<b>4. Full Human Threat Index (HTI) Dataset</b>", styles["Heading2"])
    )
    elements.append(Spacer(1, 8))

    columns = df.columns.tolist()
    table_data = [columns]

    for _, row in df.iterrows():
        table_data.append([str(row[col]) for col in columns])

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 12))

    note = Paragraph(
        "Note: This table contains the full HTI dataset exactly as stored in "
        "<i>./data/hti/employees_hti.xlsx</i>. No records were modified during report generation.",
        styles["Italic"],
    )
    elements.append(note)

    return elements


def build_charts_section(chart_files: List[str], styles):
    """Create a section with all charts, if any, each on its own block."""
    if not chart_files:
        return []

    elements: List[Any] = []

    elements.append(PageBreak())
    elements.append(Paragraph("<b>5. Visualization Charts</b>", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    for idx, img_path in enumerate(chart_files, start=1):
        title = os.path.basename(img_path)
        elements.append(
            Paragraph(f"<b>Chart {idx}:</b> {title}", styles["Heading3"])
        )
        elements.append(Spacer(1, 6))

        img = Image(img_path)
        max_width = 16 * cm
        max_height = 10 * cm
        img._restrictSize(max_width, max_height)

        elements.append(img)
        elements.append(Spacer(1, 18))

    return elements


# =====================================================
#                 MAIN REPORT LOGIC
# =====================================================

def create_pdf_report(hti_df: pd.DataFrame, chart_files: List[str]):
    os.makedirs("./reports", exist_ok=True)
    pdf_path = "./reports/final_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    styles["Title"].fontSize = 20
    styles["Heading2"].fontSize = 14
    styles["Heading3"].fontSize = 12
    styles["Italic"].fontSize = 9

    story: List[Any] = []

    # ---------- COVER / HEADER ----------
    story.append(Paragraph("HERA Threat Intelligence Report", styles["Title"]))
    story.append(Spacer(1, 12))

    subtitle = "Human Engineered Research Analysis – Internal Threat Assessment"
    story.append(Paragraph(subtitle, styles["Heading2"]))
    story.append(Spacer(1, 6))

    date_str = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"Generated on: {date_str}", styles["BodyText"]))
    story.append(Spacer(1, 18))

    total_employees = len(hti_df)
    story.append(
        Paragraph(
            f"This report is generated from the HTI dataset containing "
            f"<b>{total_employees}</b> employees stored in "
            f"<i>./data/hti/employees_hti.xlsx</i>.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 18))

    # ---------- LLM SUMMARY (OLLAMA) ----------
    hti_col = find_hti_column(hti_df)
    summary = generate_llm_summary(hti_df, hti_col)
    story.extend(build_exec_section(summary, styles))

    # ---------- FULL HTI TABLE ----------
    story.extend(build_full_hti_table(hti_df, styles))

    # ---------- CHARTS ----------
    story.extend(build_charts_section(chart_files, styles))

    # ---------- FOOTER ----------
    story.append(Spacer(1, 24))
    story.append(
        Paragraph(
            "<i>Generated automatically by HERA — Human Engineered Research Analysis.</i>",
            styles["Italic"],
        )
    )

    doc.build(story)
    print(f"✅ PDF report generated at: {pdf_path}")


def main():
    print("🔍 Loading HTI Excel and charts...")
    hti_df, chart_files = load_hti_and_charts()

    print("🧠 Generating Ollama-based summary (read-only on Excel data)...")
    create_pdf_report(hti_df, chart_files)

    print("✅ Phase 5 complete.")


if __name__ == "__main__":
    main()

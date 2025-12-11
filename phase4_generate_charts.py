#!/usr/bin/env python3
"""
Phase 4: Generate charts from data/HTI/employees_HTI.xlsx

- Input:  ./data/HTI/employees_HTI.xlsx
- Output: ./data/charts/*.png
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
DEFAULT_DATA_DIR = Path("data")
DEFAULT_INPUT_FILE = DEFAULT_DATA_DIR / "HTI" / "employees_HTI.xlsx"
DEFAULT_CHARTS_DIR = DEFAULT_DATA_DIR / "charts"


# -----------------------------
# Helpers
# -----------------------------
def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to read Excel file: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print(f"[ERROR] Input file is empty: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(df)} rows and {len(df.columns)} columns from {input_path}")
    return df


def ensure_output_dir(charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Charts will be saved to: {charts_dir}")


def detect_key_columns(df: pd.DataFrame) -> dict:
    cols_lower = {c.lower(): c for c in df.columns}

    # HTI
    hti_col = None
    for col in df.columns:
        c = col.lower()
        if "hti" in c and pd.api.types.is_numeric_dtype(df[col]):
            hti_col = col
            break

    # Department
    dept_col = None
    for key in ["department", "dept", "team"]:
        if key in cols_lower:
            dept_col = cols_lower[key]
            break

    # Risk category
    risk_col = None
    for col in df.columns:
        if "risk" in col.lower():
            risk_col = col
            break

    print("[INFO] Auto-detected columns:")
    print(f"       HTI column       : {hti_col}")
    print(f"       Department column: {dept_col}")
    print(f"       Risk column      : {risk_col}")

    return {
        "hti": hti_col,
        "department": dept_col,
        "risk": risk_col
    }


def save_plot(output_path: Path):
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved chart: {output_path}")


# -----------------------------
# Charts
# -----------------------------
def plot_hti_distribution(df, hti_col, charts_dir):
    if hti_col is None:
        print("[WARN] No HTI column. Skipping HTI distribution chart.")
        return

    plt.figure(figsize=(8, 5))
    df[hti_col].hist(bins=20)
    plt.title("Distribution of Human Threat Index")
    plt.xlabel(hti_col)
    plt.ylabel("Number of Employees")
    plt.grid(axis="y", alpha=0.3)

    save_plot(charts_dir / "hti_distribution.png")


def plot_hti_by_department(df, hti_col, dept_col, charts_dir):
    if hti_col is None or dept_col is None:
        print("[WARN] Missing HTI/Department column. Skipping boxplot.")
        return

    plt.figure(figsize=(10, 6))
    df.boxplot(column=hti_col, by=dept_col)
    plt.title("HTI by Department")
    plt.suptitle("")
    plt.xticks(rotation=45)
    plt.ylabel(hti_col)

    save_plot(charts_dir / "hti_by_department_boxplot.png")


def plot_risk_counts(df, risk_col, charts_dir):
    if risk_col is None:
        print("[WARN] No risk column. Skipping risk distribution.")
        return

    plt.figure(figsize=(8, 5))
    df[risk_col].value_counts().sort_index().plot(kind="bar")
    plt.title("Employees by Risk Level")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)

    save_plot(charts_dir / "risk_levels.png")


def plot_numeric_histograms(df, charts_dir, exclude=None):
    if exclude is None:
        exclude = []

    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude
    ]

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        df[col].hist(bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(axis="y", alpha=0.3)

        save_plot(charts_dir / f"hist_{col.replace(' ', '_')}.png")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Chart generation")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to employees_HTI.xlsx"
    )
    parser.add_argument(
        "--charts-dir",
        default=str(DEFAULT_CHARTS_DIR),
        help="Where charts should be saved"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    charts_dir = Path(args.charts_dir)

    df = load_data(input_path)
    ensure_output_dir(charts_dir)

    columns = detect_key_columns(df)
    hti_col = columns["hti"]
    dept_col = columns["department"]
    risk_col = columns["risk"]

    # Generate charts
    plot_hti_distribution(df, hti_col, charts_dir)
    plot_hti_by_department(df, hti_col, dept_col, charts_dir)
    plot_risk_counts(df, risk_col, charts_dir)

    # Extra numeric columns
    plot_numeric_histograms(df, charts_dir, exclude=[hti_col])

    print("\n[DONE] Charts generated successfully in ./data/charts/\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 1 – Synthetic Employee Data Generator

- Generates synthetic data for 100 banking employees
- Includes PII + behavioural features + activity context
- Saves to: data/raw/employees_synthetic.csv
- If the file already exists, asks user whether to reuse or generate fresh data
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

# ----------------- CONFIG -----------------

NUM_EMPLOYEES = 10

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
EMPLOYEE_FILE = DATA_RAW_DIR / "employees_synthetic.csv"

fake = Faker()

DEPARTMENTS = [
    "Retail Banking",
    "Corporate Banking",
    "Wealth Management",
    "IT Security",
    "Risk Management",
    "Compliance",
    "Operations",
    "Treasury",
    "HR",
    "Fraud Analytics",
]

ROLES_BY_DEPT = {
    "Retail Banking": ["Branch Manager", "Customer Advisor", "Teller"],
    "Corporate Banking": ["Relationship Manager", "Credit Analyst"],
    "Wealth Management": ["Portfolio Manager", "Investment Advisor"],
    "IT Security": ["Security Engineer", "SOC Analyst", "IAM Specialist"],
    "Risk Management": ["Risk Analyst", "Quantitative Analyst"],
    "Compliance": ["Compliance Officer", "AML Specialist"],
    "Operations": ["Operations Analyst", "Process Manager"],
    "Treasury": ["Treasury Analyst", "Liquidity Manager"],
    "HR": ["HR Generalist", "Recruiter"],
    "Fraud Analytics": ["Fraud Analyst", "Data Scientist"],
}

ACCESS_LEVELS = [1, 2, 3, 4, 5]  # 1 = low, 5 = highly privileged

ACTIVITY_TEMPLATES = [
    "Accessed {system} from office network and downloaded {size} report.",
    "Sent an email to {recipient} with subject '{subject}'.",
    "Attempted to access {system} outside business hours.",
    "Uploaded files to {system} from corporate laptop.",
    "Viewed {count} customer accounts in a short time window.",
    "Exported a transaction report for {region} region.",
    "Logged in from a new device and changed account settings.",
]

SYSTEMS = [
    "Core Banking",
    "SWIFT Gateway",
    "Customer 360",
    "Fraud Analytics Portal",
    "Data Warehouse",
    "Email Gateway",
]

SUBJECTS = [
    "Monthly Performance Report",
    "Client Onboarding Documents",
    "Urgent Access Request",
    "Updated AML Procedures",
    "Q4 Financial Summary",
]

REGIONS = ["North America", "Europe", "APAC", "LATAM", "Middle East"]


# ----------------- HELPERS -----------------


def random_hire_date(years_back: int = 10) -> datetime:
    today = datetime.today()
    start = today - timedelta(days=365 * years_back)
    return Faker().date_time_between(start_date=start, end_date=today)


def random_last_login(days_back: int = 30) -> datetime:
    today = datetime.today()
    start = today - timedelta(days=days_back)
    return Faker().date_time_between(start_date=start, end_date=today)


def random_activity_summary() -> str:
    tmpl = random.choice(ACTIVITY_TEMPLATES)
    return tmpl.format(
        system=random.choice(SYSTEMS),
        size=f"{random.randint(50, 800)}MB",
        recipient=fake.company_email(),
        subject=random.choice(SUBJECTS),
        count=random.randint(5, 200),
        region=random.choice(REGIONS),
    )


def generate_employee_record(emp_id: int) -> dict:
    """Generate a single synthetic employee record with PII + behaviour + activities."""
    name = fake.name()
    email = fake.company_email()
    phone = fake.phone_number()
    address = fake.address().replace("\n", ", ")

    dept = random.choice(DEPARTMENTS)
    role = random.choice(ROLES_BY_DEPT.get(dept, ["Analyst"]))
    manager_name = fake.name()

    access_level = random.choice(ACCESS_LEVELS)
    is_contractor = random.random() < 0.2  # 20% are contractors
    privileged_account = access_level >= 4

    hire_date = random_hire_date()
    last_login = random_last_login()

    # Behavioural metrics (useful for insider threat)
    logins_30 = random.randint(5, 80)
    failed_logins = max(0, int(random.gauss(mu=1, sigma=2)))
    late_night_logins = max(0, int(random.gauss(mu=2, sigma=3)))
    large_file_transfers = max(0, int(random.gauss(mu=1, sigma=2)))

    avg_daily_emails = max(5, int(random.gauss(mu=60, sigma=25)))
    external_email_ratio = min(1.0, max(0.0, random.gauss(mu=0.25, sigma=0.15)))
    suspicious_attachments = max(0, int(random.gauss(mu=1, sigma=1.5)))

    training_done = random.random() < 0.8
    last_training = (
        fake.date_between(start_date="-2y", end_date="today")
        if training_done
        else None
    )

    performance_rating = random.choice([1, 2, 3, 4, 5])

    # Recent email + activity fields (good for masking in Phase 2)
    recent_email_to = fake.company_email()
    recent_email_subject = random.choice(SUBJECTS)
    recent_email_external = random.random() < external_email_ratio

    activity_summary = random_activity_summary()

    return {
        # Core identity
        "employee_id": f"EMP{emp_id:04d}",
        "employee_name": name,
        "email": email,
        "phone": phone,
        "address": address,
        "department": dept,
        "role": role,
        "manager_name": manager_name,
        "access_level": access_level,
        "is_contractor": is_contractor,
        "privileged_account": privileged_account,
        # Employment & login
        "hire_date": hire_date.date().isoformat(),
        "last_login": last_login.isoformat(timespec="seconds"),
        # Behavioural metrics
        "logins_past_30d": logins_30,
        "failed_logins_past_30d": failed_logins,
        "late_night_logins_past_30d": late_night_logins,
        "large_file_transfers_past_30d": large_file_transfers,
        "avg_daily_email_count": avg_daily_emails,
        "external_email_ratio": round(external_email_ratio, 2),
        "suspicious_attachments_past_30d": suspicious_attachments,
        # Training / HR
        "security_training_completed": training_done,
        "last_security_training_date": last_training.isoformat()
        if last_training
        else None,
        "performance_rating": performance_rating,
        # Email + activity context (nice for Presidio in Phase 2)
        "recent_email_to": recent_email_to,
        "recent_email_subject": recent_email_subject,
        "recent_email_is_external": recent_email_external,
        "activity_summary": activity_summary,
    }


def generate_dataset(n: int = NUM_EMPLOYEES) -> pd.DataFrame:
    return pd.DataFrame([generate_employee_record(i + 1) for i in range(n)])


# ----------------- MAIN ENTRYPOINT -----------------


# ----------------- MAIN ENTRYPOINT -----------------

def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    excel_file = DATA_RAW_DIR / "employees_synthetic.xlsx"

    if excel_file.exists():
        print(f"[Phase 1] Existing dataset found at: {excel_file}")
        choice = input("Use existing data? (y = use old, n = generate fresh) [y/n]: ").strip().lower()
        if choice in ("y", "yes", ""):
            print("[Phase 1] Using existing dataset. No new data generated.")
            return
        print("[Phase 1] Generating fresh dataset and overwriting old file...")

    else:
        print("[Phase 1] No dataset found. Generating new synthetic data...")

    df = generate_dataset(NUM_EMPLOYEES)

    # SAVE AS EXCEL – NOT CSV
    df.to_excel(excel_file, index=False, sheet_name="employees")

    print(f"[Phase 1] Generated {len(df)} employees.")
    print(f"[Phase 1] Saved synthetic dataset to Excel: {excel_file}")


if __name__ == "__main__":
    main()

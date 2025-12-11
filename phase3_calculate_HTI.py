import os
import re
import json
import glob
import requests
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError

# =========================================================
# CONFIG SECTION – EDIT THESE VALUES IF NEEDED
# =========================================================

# 1) Input Excel file (masked data from Phase 2)
#    Option A: Hard-code the exact file:
INPUT_FILE = "./data/masked/employees_masked.xlsx"

#    Option B (fallback): If INPUT_FILE doesn't exist, we auto-pick
#    the first .xlsx file from ./data/masked/
AUTO_DETECT_IF_MISSING = True

# 2) Output Excel file (HTI scores)
OUTPUT_FILE = "./data/hti/employees_HTI.xlsx"

# 3) Ollama endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 4) Model name – change this to any local model you have in `ollama list`
#    e.g.: "llama3.1", "deepseek-r1:latest", "mistral:latest"
OLLAMA_MODEL = "llama3"

# 5) HTTP timeout for Ollama (seconds)
OLLAMA_TIMEOUT = 300

# 6) (Optional) Restrict which columns are sent to the LLM.
#    If this list is empty, ALL columns from the Excel row will be sent.
SELECTED_COLUMNS = [
    # "employee_id",
    # "department",
    # "role",
    # "failed_logins_7d",
    # "files_downloaded_7d",
    # "emails_sent_7d",
    # "after_hours_activity_score",
    # "usb_activity_score",
    # "privilege_change_30d",
]
# Leave commented / empty to send full row.


# =========================================================
# PROMPT BUILDER
# =========================================================

def build_prompt(record: dict) -> str:
    """
    Build the HTI scoring prompt for a single employee record.
    Uses explicit formulas (BRS, CRS, IIS, HTI) and strict JSON output.
    """
    prompt = f"""
You are an AI model operating inside an air-gapped insider-threat analysis pipeline.
You MUST NOT hallucinate, invent data, or use any information outside the employee record provided.
If any numeric field required for a formula is missing or null, treat that metric as "INSUFFICIENT_DATA"
and DO NOT guess values.

You must compute a HUMAN THREAT INDEX (HTI) using the following definitions and formulas.

------------------------------------------------
INPUT EMPLOYEE RECORD (JSON)
------------------------------------------------
You will receive an employee record like this (fields may vary):

{json.dumps(record, indent=2)}

Only use the fields that actually exist. If a field is missing or null, treat it as missing.

------------------------------------------------
SCORING MODEL (MANDATORY)
------------------------------------------------

1) Behavioral Risk Score (BRS)
--------------------------------
Use anomaly / activity indicators. Map from your available columns as follows
(only if they exist; otherwise mark BRS as INSUFFICIENT_DATA):

- login_failures_30d
- privilege_escalations_30d
- suspicious_file_access_30d
- email_anomalies_30d

Formula:

BRS =
  (login_failures_30d       * 0.15) +
  (privilege_escalations_30d * 0.25) +
  (suspicious_file_access_30d * 0.35) +
  (email_anomalies_30d       * 0.25)

If BRS > 100, cap it at 100.

If ANY of the above fields needed for the formula is missing → BRS = "INSUFFICIENT_DATA".

If your dataset uses different but clearly equivalent features
(e.g. failed_logins_7d, usb_activity_score, privilege_change_30d),
you may adapt them logically, but NEVER invent values.

2) Compliance Risk Score (CRS)
--------------------------------
Uses security training and historical incidents where available:

Required fields (if present):
- security_training_score (0–100; higher is better)
- previous_incidents (count of prior security incidents)

Formula:

CRS =
  (100 - security_training_score) * 0.6
  + (previous_incidents * 10) * 0.4

Cap CRS at 100.

If a required field is missing → CRS = "INSUFFICIENT_DATA".

3) Insider Intent Score (IIS)
--------------------------------
Uses psychometric and tenure data (if available):

Required fields:
- psychometric_risk_score (0–100; higher = riskier)
- tenure_years (years in organization, may be float)

Formula:

IIS =
  (psychometric_risk_score * 0.7)
  + (max(0, (5 - tenure_years)) * 6)

Cap IIS at 100.

If a required field is missing → IIS = "INSUFFICIENT_DATA".

4) HUMAN THREAT INDEX (HTI)
--------------------------------
If ANY of BRS, CRS, or IIS is "INSUFFICIENT_DATA" → HTI = "INSUFFICIENT_DATA".

Otherwise:

HTI = (BRS * 0.4) + (CRS * 0.3) + (IIS * 0.3)

Round HTI to 2 decimal places.

5) RISK LEVEL MAPPING
--------------------------------
Based on numeric HTI (0–100):

- HTI in [0, 45]    → "Low"
- HTI in [46, 80]   → "Medium"
- HTI in [81, 100]  → "High"

If HTI = "INSUFFICIENT_DATA" → risk_level = "Unknown".

------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
------------------------------------------------
Return JSON in EXACT format:

{{
  "hti_score": <number or "INSUFFICIENT_DATA">,
  "risk_level": "<Low|Medium|High|Unknown>",
  "explanation": "<short explanation referencing BRS, CRS, IIS and key drivers>"
}}

Rules:
- Only output the JSON object (no commentary, no markdown).
- Do NOT invent or guess missing numeric values.
- If you had to mark anything as INSUFFICIENT_DATA, mention that in the explanation.
- The explanation must be concise (3–5 short sentences).

------------------------------------------------
GENERATION & HALLUCINATION CONTROL
------------------------------------------------
You are running with:
- temperature = 0.0
- top_p = 0.1
- top_k = 20
- deterministic seed

This means you MUST behave deterministically and avoid any creative or speculative output.
Just apply the formulas above to the provided employee record.

Now, compute the HTI for the given Employee Data and output ONLY the JSON.
"""
    return prompt


# =========================================================
# LLM CALL (OLLAMA)
# =========================================================

def call_llm(prompt: str) -> dict:
    """
    Call a local Ollama model via HTTP and return parsed JSON.
    If there is a timeout or connection issue, return a safe fallback dict.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an advanced threat-modeling assistant inside an air-gapped environment. "
                    "You must follow the user's instructions exactly, avoid hallucinations, and output ONLY strict JSON."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.1,
            "top_k": 20,
            "seed": 42
        }
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
    except ReadTimeout:
        print("⏰ Ollama request timed out. Returning default 'Unknown' result for this employee.")
        return {
            "hti_score": 0,
            "risk_level": "Unknown",
            "explanation": "LLM timeout while computing HTI."
        }
    except ConnectionError as e:
        print(f"❌ Could not connect to Ollama at {OLLAMA_API_URL}. Is the Ollama server running?")
        raise e

    data = response.json()

    # Typical Ollama chat format: {"message": {"role": "...", "content": "..."}, ...}
    if "message" in data and "content" in data["message"]:
        text = data["message"]["content"]
    else:
        # Some variants may use a different key
        text = data.get("response", "")

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except Exception:
        # Fallback: search for a JSON block in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"LLM did not return valid JSON. Raw output:\n{text}")
        return json.loads(match.group(0))


# =========================================================
# HELPER: RESOLVE INPUT FILE
# =========================================================

def resolve_input_file() -> str:
    """
    Ensure we have a valid input Excel path.
    If the configured INPUT_FILE doesn't exist and AUTO_DETECT_IF_MISSING is True,
    pick the first .xlsx file under ./data/masked/.
    """
    if os.path.exists(INPUT_FILE):
        return INPUT_FILE

    if AUTO_DETECT_IF_MISSING:
        candidates = glob.glob("./data/masked/*.xlsx")
        if not candidates:
            raise FileNotFoundError(
                f"No Excel files found in ./data/masked/, and INPUT_FILE does not exist: {INPUT_FILE}"
            )
        print(f"⚠️ Configured INPUT_FILE not found. Using auto-detected file: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(f"Input file not found at {INPUT_FILE}")


# =========================================================
# MAIN WORKFLOW
# =========================================================

def main():
    # 1) Resolve input path
    input_path = resolve_input_file()

    print("\n🔍 Loading masked dataset from:", input_path)
    df = pd.read_excel(input_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    results = []

    for idx, row in df.iterrows():
        # 2) Prepare record to send to LLM
        if SELECTED_COLUMNS:
            record = {col: row.get(col) for col in SELECTED_COLUMNS if col in df.columns}
        else:
            record = row.to_dict()

        emp_id = record.get("employee_id", f"row_{idx + 1}")

        print(f"\n🧠 Calling LLM for Employee row {idx + 1} (employee_id={emp_id})...")

        try:
            prompt = build_prompt(record)
            llm_result = call_llm(prompt)
        except Exception as e:
            print(f"❌ Error while processing employee {emp_id}: {e}")
            llm_result = {
                "hti_score": 0,
                "risk_level": "Unknown",
                "explanation": f"Error while calling LLM: {e}"
            }

        hti_score = llm_result.get("hti_score", 0)
        risk_level = llm_result.get("risk_level", "Unknown")
        explanation = llm_result.get("explanation", "")

        results.append({
            "EmployeeID": emp_id,
            "HTI_Score": hti_score,
            "Risk_Level": risk_level,
            "Explanation": explanation
        })

        print(f"✅ Processed → HTI {hti_score} | Risk = {risk_level}")

    # 3) Save result Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(OUTPUT_FILE, index=False)

    print("\n🎯 Phase 3 Completed Successfully!")
    print(f"📁 HTI Output saved at: {OUTPUT_FILE}")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()

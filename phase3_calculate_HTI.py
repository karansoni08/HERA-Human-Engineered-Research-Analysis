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
    """
    prompt = f"""
You are an expert insider-threat analyst. 
Your goal is to calculate a HUMAN THREAT INDEX (HTI) for each employee.

HTI Scale:
0-25 = Low Risk
26-60 = Medium Risk
61-100 = High Risk

Using the employee’s masked behavioral data below, identify:
1. Behavioral anomalies
2. Risk factors
3. Weighted scoring logic
4. Final HTI score (0–100)
5. Explanation in 4–5 bullet points

Return JSON in EXACT format:
{{
  "hti_score": <number>,
  "risk_level": "<Low|Medium|High>",
  "explanation": "<short explanation>"
}}

Only output the JSON. Do not include any extra text.

Employee Data:
{json.dumps(record, indent=2)}
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
            {"role": "system", "content": "You are an advanced threat-modeling assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
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

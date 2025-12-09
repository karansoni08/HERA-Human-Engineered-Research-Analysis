#!/usr/bin/env python3
"""
Phase 2 – PII Detection (Microsoft Presidio + spaCy) and Masking with Faker

Flow:
1. Read Phase 1 Excel:
       data/raw/employees_synthetic.xlsx
2. For selected columns, use Presidio + spaCy (en_core_web_lg) to detect PII.
3. Replace detected PII with realistic fake values using Faker,
   with deterministic mapping: same original -> same fake.
4. Save masked data to:
       data/masked/employees_masked.xlsx
   and mapping to:
       data/masked/pii_mappings.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from faker import Faker
from tqdm import tqdm

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpEngine

# ----------------- PATHS & CONFIG -----------------

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "employees_synthetic.xlsx"
MASKED_DIR = PROJECT_ROOT / "data" / "masked"
MASKED_FILE = MASKED_DIR / "employees_masked.xlsx"
MAPPINGS_FILE = MASKED_DIR / "pii_mappings.json"

LANGUAGE = "en"
SPACY_MODEL = "en_core_web_lg"

fake = Faker()

# Columns that we treat as PII and will process
PII_COLUMNS: List[str] = [
    "employee_id",
    "employee_name",
    "manager_name",
    "email",
    "recent_email_to",
    "phone",
    "address",
    "recent_email_subject",
    "activity_summary",
]

# Column -> logical type (used as hint + for fallback when no entity detected)
COLUMN_HINT_TYPE: Dict[str, str] = {
    "employee_id": "EMP_ID",
    "employee_name": "PERSON",
    "manager_name": "PERSON",
    "email": "EMAIL_ADDRESS",
    "recent_email_to": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER",
    "address": "LOCATION",
    "recent_email_subject": "SUBJECT",
    "activity_summary": "ACTIVITY",
}

# Presidio entities to detect
PII_ENTITY_TYPES: List[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
]

SUBJECT_TEMPLATES = [
    "Quarterly performance update",
    "Client review scheduled",
    "Internal compliance reminder",
    "Access request confirmation",
    "Security policy update",
]

ACTIVITY_TEMPLATES = [
    "Generated a report in the {system}.",
    "Accessed customer records via the {system}.",
    "Uploaded documents to the {system}.",
    "Reviewed accounts in the {region} region.",
    "Submitted an incident update in the {system}.",
]

SYSTEMS = ["Core Banking Portal", "Fraud Analytics Dashboard", "Customer 360", "HR Suite"]
REGIONS = ["North America", "Europe", "APAC", "LATAM", "Middle East"]


# ----------------- DETERMINISTIC MAPPING CACHE -----------------

class MappingCache:
    """
    (entity_or_hint_type, original_value) -> fake_value

    Used so that the same original PII always becomes the same fake.
    """

    def __init__(self, path: Path):
        self.path = path
        self.mapping: Dict[str, str] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)

    @staticmethod
    def _key(kind: str, original: str) -> str:
        return f"{kind}|||{original}"

    def get_or_create(self, kind: str, original: str, generator) -> str:
        k = self._key(kind, original)
        if k in self.mapping:
            return self.mapping[k]
        fake_val = generator()
        self.mapping[k] = fake_val
        return fake_val

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, indent=2, ensure_ascii=False)


# ----------------- PII MASKER -----------------

class PiiMasker:
    def __init__(self):
        self.faker = Faker()
        self.mapping_cache = MappingCache(MAPPINGS_FILE)
        self.analyzer = self._init_analyzer()

    def _init_analyzer(self) -> AnalyzerEngine:
        """
        Initialize Presidio Analyzer with spaCy en_core_web_lg.
        """
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": LANGUAGE, "model_name": SPACY_MODEL}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine: NlpEngine = provider.create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[LANGUAGE])

    # ---------- Type-correct fake generators ----------

    def _fake_for(self, entity_type: str | None, original: str, column_name: str) -> str:
        """
        Generate fake value based on:
        - Presidio entity_type (PERSON, EMAIL_ADDRESS, etc.)
        - Column hint (EMP_ID, SUBJECT, ACTIVITY)
        """
        entity_type = (entity_type or "").upper()
        hint = COLUMN_HINT_TYPE.get(column_name, "").upper()

        # Employee IDs (always mask full value)
        if hint == "EMP_ID":
            return self.mapping_cache.get_or_create(
                "EMP_ID",
                original,
                lambda: self.faker.bothify(text="EMP_MASK_####"),
            )

        # Person names
        if entity_type == "PERSON" or hint == "PERSON":
            return self.mapping_cache.get_or_create(
                "PERSON",
                original,
                lambda: self.faker.name(),
            )

        # Emails
        if entity_type == "EMAIL_ADDRESS" or hint == "EMAIL_ADDRESS":
            return self.mapping_cache.get_or_create(
                "EMAIL_ADDRESS",
                original,
                lambda: self.faker.company_email(),
            )

        # Phone numbers
        if entity_type == "PHONE_NUMBER" or hint == "PHONE_NUMBER":
            return self.mapping_cache.get_or_create(
                "PHONE_NUMBER",
                original,
                lambda: self.faker.phone_number(),
            )

        # Locations / addresses
        if entity_type == "LOCATION" or hint == "LOCATION":
            return self.mapping_cache.get_or_create(
                "LOCATION",
                original,
                lambda: self.faker.address().replace("\n", ", "),
            )

        # Subject – treat full cell as PII
        if hint == "SUBJECT":
            return self.mapping_cache.get_or_create(
                "SUBJECT",
                original,
                lambda: self.faker.random_element(SUBJECT_TEMPLATES),
            )

        # Activity – treat full cell as PII
        if hint == "ACTIVITY":
            template = self.faker.random_element(ACTIVITY_TEMPLATES)
            fake_activity = template.format(
                system=self.faker.random_element(SYSTEMS),
                region=self.faker.random_element(REGIONS),
            )
            return self.mapping_cache.get_or_create(
                "ACTIVITY",
                original,
                lambda: fake_activity,
            )

        # Fallback (should almost never be used)
        return self.mapping_cache.get_or_create(
            entity_type or "GENERIC",
            original,
            lambda: self.faker.text(max_nb_chars=len(original) + 10),
        )

    # ---------- Text masking using Presidio ----------

    def mask_text(self, text: str, column_name: str) -> str:
        """
        - Run Presidio over the text.
        - Replace detected spans with Faker values.
        - If nothing detected but the column is marked PII,
          mask the whole cell with a Faker value based on column hint.
        """
        if not isinstance(text, str) or not text.strip():
            return text

        results: List[RecognizerResult] = self.analyzer.analyze(
            text=text,
            language=LANGUAGE,
            entities=PII_ENTITY_TYPES,
        )

        # If Presidio finds nothing but schema says "this is PII", mask whole cell
        if not results:
            if column_name in COLUMN_HINT_TYPE:
                return self._fake_for(None, text, column_name)
            return text

        # Sort by start index and rebuild string with fake replacements
        results = sorted(results, key=lambda r: r.start)

        masked_chunks: List[str] = []
        cursor = 0

        for res in results:
            if res.start > cursor:
                masked_chunks.append(text[cursor:res.start])

            original_val = text[res.start:res.end]
            fake_val = self._fake_for(res.entity_type, original_val, column_name)
            masked_chunks.append(fake_val)

            cursor = res.end

        masked_chunks.append(text[cursor:])
        return "".join(masked_chunks)

    # ---------- DataFrame-level masking ----------

    def mask_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        masked_df = df.copy()

        for col in PII_COLUMNS:
            if col not in masked_df.columns:
                print(f"[Phase 2] Column '{col}' not found. Skipping.")
                continue

            print(f"[Phase 2] Masking column: {col}")
            tqdm.pandas()
            masked_df[col] = masked_df[col].progress_apply(
                lambda v: self.mask_text(str(v), col) if pd.notna(v) else v
            )

        # Save mapping for audit / reproducibility
        self.mapping_cache.save()
        return masked_df


# ----------------- MAIN -----------------

def main() -> None:
    if not RAW_FILE.exists():
        print(f"[Phase 2] ERROR: input Excel not found: {RAW_FILE}")
        print("          Run phase1.py first.")
        return

    MASKED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 2] Reading Excel: {RAW_FILE}")
    df = pd.read_excel(RAW_FILE)  # <- INPUT is Excel

    masker = PiiMasker()
    masked_df = masker.mask_dataframe(df)

    print(f"[Phase 2] Writing masked Excel to: {MASKED_FILE}")
    masked_df.to_excel(MASKED_FILE, index=False, sheet_name="employees_masked")

    print(f"[Phase 2] Mapping saved to: {MAPPINGS_FILE}")
    print("[Phase 2] Done – Excel read -> PII detect -> mask -> Excel write.")


if __name__ == "__main__":
    main()

"""Unified prompts. Same extraction prompt used across all pure-model configs."""
from __future__ import annotations

from .schema import SCHEMA_JSON_EXAMPLE


UNIFIED_EXTRACTION_PROMPT = """You are an invoice auditor for a Danish construction-industry SaaS called DiFacto.

Your job: extract structured data from the attached invoice PDF and cross-check
it against the supplier's pre-negotiated price agreement (provided below).

PRICE AGREEMENT DATA (from {agreement_filename}):
---BEGIN AGREEMENT---
{agreement_text}
---END AGREEMENT---

Instructions:
1. Read the PDF. Danish text. Amounts may use "1.234,56" or "1234.56".
2. Classify document_type as either "invoice" or "credit_note".
   - A credit note (kreditnota) reverses a prior invoice. Amounts should be
     reported as negative when sign_convention is "negative".
   - If this is a credit note, populate credit_note_handling with is_credit_note,
     sign_convention, and references_invoice (the original invoice number, if any).
   - If this is a regular invoice, set credit_note_handling to null.
3. For each line item, find the matching product in the agreement by description.
   Set agreed_unit_price from the agreement. Set has_discrepancy=true if
   unit_price differs from agreed_unit_price by more than 0.01. Set
   discrepancy_amount = (unit_price - agreed_unit_price) * quantity.
4. Compute expected_rebate from any bonus/rebate rules in the agreement
   (e.g. volume-based rebate on subtotal). Set rebate_applied from the invoice
   if present, otherwise 0.

Output strictly-valid JSON matching this schema (no prose, no code fences):

{schema}

Dates: ISO 8601 (YYYY-MM-DD). Numbers: plain numeric (no thousand separators).
Currency: ISO 4217 code (e.g. "DKK"). Return null for fields you cannot
determine. Do not invent data."""


HYBRID_EXTRACTION_PROMPT = """You are extracting structured data from a Danish construction-industry invoice PDF.

Instructions:
1. Read the PDF carefully. Danish text. Amounts may use "1.234,56" or "1234.56".
2. Classify document_type as either "invoice" or "credit_note" (kreditnota).
3. If credit note, populate credit_note_handling. If not, set it to null.
4. Do NOT compare against any agreement - just extract what's on the invoice.
   Leave agreed_unit_price, has_discrepancy, discrepancy_amount, and
   expected_rebate as null. Set rebate_applied from the invoice if present.

Output strictly-valid JSON matching this schema (no prose, no code fences):

{schema}

Dates: ISO 8601 (YYYY-MM-DD). Numbers: plain numeric (no thousand separators).
Currency: ISO 4217. Return null for unknowns. Do not invent data."""


HYBRID_REASONING_PROMPT = """You are auditing an already-extracted Danish invoice against a supplier price agreement.

EXTRACTED INVOICE DATA (JSON):
---BEGIN INVOICE---
{invoice_json}
---END INVOICE---

PRICE AGREEMENT DATA (from {agreement_filename}):
---BEGIN AGREEMENT---
{agreement_text}
---END AGREEMENT---

Instructions:
1. Keep the extracted fields as-is. Do not change supplier_name, invoice_number,
   invoice_date, document_type, currency, subtotal, vat, total, rebate_applied,
   or credit_note_handling.
2. For each line item, find the matching product in the agreement by description.
   Set agreed_unit_price from the agreement. Set has_discrepancy=true if
   unit_price differs from agreed_unit_price by more than 0.01. Set
   discrepancy_amount = (unit_price - agreed_unit_price) * quantity.
3. Compute expected_rebate from any bonus/rebate rules in the agreement.

Output the full enriched invoice as strictly-valid JSON matching this schema
(no prose, no code fences):

{schema}"""


def build_unified_prompt(agreement_filename: str, agreement_text: str) -> str:
    return UNIFIED_EXTRACTION_PROMPT.format(
        agreement_filename=agreement_filename,
        agreement_text=agreement_text,
        schema=SCHEMA_JSON_EXAMPLE,
    )


def build_hybrid_extraction_prompt() -> str:
    return HYBRID_EXTRACTION_PROMPT.format(schema=SCHEMA_JSON_EXAMPLE)


def build_hybrid_reasoning_prompt(
    invoice_json: str, agreement_filename: str, agreement_text: str
) -> str:
    return HYBRID_REASONING_PROMPT.format(
        invoice_json=invoice_json,
        agreement_filename=agreement_filename,
        agreement_text=agreement_text,
        schema=SCHEMA_JSON_EXAMPLE,
    )

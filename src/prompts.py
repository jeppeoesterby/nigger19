"""Prompt assembly. Prompts are built dynamically based on what's available
for each job: invoice may be PDF (attached) or XLSX (text inline); agreements
may be PDFs (attached) or XLSX (text inline); any mix is allowed."""
from __future__ import annotations

from .schema import SCHEMA_JSON_EXAMPLE


INTRO = """You are an invoice auditor for DiFacto, a Danish B2B SaaS that audits
supplier invoices in the construction industry against pre-negotiated price agreements."""

INSTRUCTIONS_EXTRACTION_ONLY = """Instructions:
1. Extract all fields from the invoice. Danish text. Amounts may use "1.234,56" or "1234.56".
2. Classify document_type as "invoice" or "credit_note" (kreditnota).
3. For credit notes, populate credit_note_handling with is_credit_note, sign_convention,
   and references_invoice (the original invoice number if any). For normal invoices
   set credit_note_handling to null.
4. Do NOT compare against any agreement in this step. Leave agreed_unit_price,
   has_discrepancy, discrepancy_amount, and expected_rebate as null. Set
   rebate_applied from the invoice if present.

Output strictly-valid JSON matching this schema (no prose, no code fences):

{schema}

Dates: ISO 8601 (YYYY-MM-DD). Numbers: plain numeric. Currency: ISO 4217. Null unknowns. Do not invent."""


INSTRUCTIONS_WITH_AGREEMENT = """Instructions:
1. Read the invoice. Danish text. Amounts may use "1.234,56" or "1234.56".
2. Classify document_type as "invoice" or "credit_note" (kreditnota).
   - A credit note reverses a prior invoice. If this is one, populate
     credit_note_handling with is_credit_note, sign_convention, references_invoice.
   - Otherwise set credit_note_handling to null.
3. For each line item, find the matching product in the agreement by description.
   Set agreed_unit_price from the agreement. Set has_discrepancy=true if
   unit_price differs from agreed_unit_price by more than 0.01. Set
   discrepancy_amount = (unit_price - agreed_unit_price) * quantity.
4. Compute expected_rebate from any bonus/rebate rules in the agreement.
   Set rebate_applied from the invoice if present, else 0.
5. If multiple agreements are provided, pick the one that matches the invoice's
   supplier. If none match, leave agreed_unit_price and rebate fields null.

Output strictly-valid JSON matching this schema (no prose, no code fences):

{schema}

Dates: ISO 8601 (YYYY-MM-DD). Numbers: plain numeric. Currency: ISO 4217. Null unknowns. Do not invent."""


REASONING_INSTRUCTIONS = """Instructions:
1. Keep the extracted fields as-is. Do not change supplier_name, invoice_number,
   invoice_date, document_type, currency, subtotal, vat, total, rebate_applied,
   or credit_note_handling.
2. For each line item, find the matching product in the agreement by description.
   Set agreed_unit_price. Set has_discrepancy=true if unit_price differs from
   agreed_unit_price by more than 0.01. Set discrepancy_amount.
3. Compute expected_rebate from any bonus/rebate rules.
4. If multiple agreements provided, match by supplier.

Output the full enriched invoice as strictly-valid JSON matching this schema:

{schema}"""


def _section(name: str, body: str) -> str:
    return f"---BEGIN {name}---\n{body}\n---END {name}---"


def build_unified_prompt(
    invoice_text: str | None,
    agreement_text: str | None,
    *,
    has_invoice_pdf: bool,
    agreement_pdf_count: int,
    with_agreement: bool,
) -> str:
    """Full extraction + audit prompt for pure-model configs.

    Any combination of attached PDFs and inline text is supported. If
    `with_agreement` is False (no agreements available), we ask for extraction
    only and skip discrepancy/rebate reasoning.
    """
    parts = [INTRO]

    if has_invoice_pdf and invoice_text:
        parts.append(
            "The invoice is provided both as an attached PDF AND as extracted "
            "spreadsheet text below. Use the PDF as authoritative; use the text "
            "as a hint only."
        )
        parts.append(_section("INVOICE SPREADSHEET TEXT", invoice_text))
    elif has_invoice_pdf:
        parts.append("The invoice is provided as an attached PDF document.")
    elif invoice_text:
        parts.append(
            "The invoice is provided as extracted spreadsheet text (no PDF)."
        )
        parts.append(_section("INVOICE SPREADSHEET TEXT", invoice_text))

    if with_agreement:
        if agreement_pdf_count and agreement_text:
            parts.append(
                f"{agreement_pdf_count} price agreement PDF(s) attached, plus "
                "additional agreement data extracted from spreadsheets below."
            )
            parts.append(_section("AGREEMENT SPREADSHEET TEXT", agreement_text))
        elif agreement_pdf_count:
            parts.append(
                f"{agreement_pdf_count} price agreement PDF(s) attached."
            )
        elif agreement_text:
            parts.append(_section("AGREEMENT DATA", agreement_text))

        parts.append(INSTRUCTIONS_WITH_AGREEMENT.format(schema=SCHEMA_JSON_EXAMPLE))
    else:
        parts.append(
            "No price agreement is provided. Extract invoice fields only."
        )
        parts.append(INSTRUCTIONS_EXTRACTION_ONLY.format(schema=SCHEMA_JSON_EXAMPLE))

    return "\n\n".join(parts)


def build_hybrid_extraction_prompt(
    invoice_text: str | None, *, has_invoice_pdf: bool
) -> str:
    """Step 1 of hybrid: Gemini extracts invoice fields only, no agreement."""
    parts = [INTRO]
    if has_invoice_pdf and invoice_text:
        parts.append(
            "Invoice is provided as attached PDF plus extracted spreadsheet text below."
        )
        parts.append(_section("INVOICE SPREADSHEET TEXT", invoice_text))
    elif has_invoice_pdf:
        parts.append("The invoice is provided as an attached PDF document.")
    elif invoice_text:
        parts.append("The invoice is provided as extracted spreadsheet text.")
        parts.append(_section("INVOICE SPREADSHEET TEXT", invoice_text))
    parts.append(INSTRUCTIONS_EXTRACTION_ONLY.format(schema=SCHEMA_JSON_EXAMPLE))
    return "\n\n".join(parts)


def build_hybrid_reasoning_prompt(
    invoice_json: str,
    agreement_text: str | None,
    *,
    agreement_pdf_count: int,
) -> str:
    """Step 2 of hybrid: Claude does price-match + rebate reasoning."""
    parts = [INTRO]
    parts.append(_section("EXTRACTED INVOICE JSON", invoice_json))
    if agreement_pdf_count and agreement_text:
        parts.append(
            f"{agreement_pdf_count} agreement PDF(s) attached plus spreadsheet "
            "agreement text below."
        )
        parts.append(_section("AGREEMENT SPREADSHEET TEXT", agreement_text))
    elif agreement_pdf_count:
        parts.append(f"{agreement_pdf_count} agreement PDF(s) attached.")
    elif agreement_text:
        parts.append(_section("AGREEMENT DATA", agreement_text))
    else:
        parts.append(
            "No agreement data provided. Leave agreement-related fields null."
        )
    parts.append(REASONING_INSTRUCTIONS.format(schema=SCHEMA_JSON_EXAMPLE))
    return "\n\n".join(parts)

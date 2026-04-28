"""Prompt assembly with user-editable templates.

Prompts are built from four template blocks plus mechanical glue (which
invoice/agreement inputs are attached vs. inline). The template blocks live
at ``paths.prompts_file`` (JSON on disk) so Ivan can tune wording from the
web UI without touching code. Missing or empty blocks fall back to defaults.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .schema import SCHEMA_JSON_EXAMPLE


# ---- Default template blocks ------------------------------------------------

DEFAULT_INTRO = """You are an invoice auditor for DiFacto, a Danish B2B SaaS that audits
supplier invoices in the construction industry against pre-negotiated price agreements."""


DEFAULT_INSTRUCTIONS_EXTRACTION_ONLY = """Instructions:
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


DEFAULT_INSTRUCTIONS_WITH_AGREEMENT = """Instructions:
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


DEFAULT_REASONING_INSTRUCTIONS = """Instructions:
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


# ---- Template storage -------------------------------------------------------

TEMPLATE_FIELDS = (
    "intro",
    "instructions_extraction_only",
    "instructions_with_agreement",
    "reasoning_instructions",
)


@dataclass
class PromptTemplates:
    intro: str = DEFAULT_INTRO
    instructions_extraction_only: str = DEFAULT_INSTRUCTIONS_EXTRACTION_ONLY
    instructions_with_agreement: str = DEFAULT_INSTRUCTIONS_WITH_AGREEMENT
    reasoning_instructions: str = DEFAULT_REASONING_INSTRUCTIONS

    @classmethod
    def defaults(cls) -> "PromptTemplates":
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "PromptTemplates":
        """Load templates from disk. Missing file / missing fields -> defaults."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(data, dict):
            return cls()
        d = cls()
        for f in TEMPLATE_FIELDS:
            v = data.get(f)
            if isinstance(v, str) and v.strip():
                setattr(d, f, v)
        return d

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def is_default(self) -> bool:
        return (
            self.intro == DEFAULT_INTRO
            and self.instructions_extraction_only == DEFAULT_INSTRUCTIONS_EXTRACTION_ONLY
            and self.instructions_with_agreement == DEFAULT_INSTRUCTIONS_WITH_AGREEMENT
            and self.reasoning_instructions == DEFAULT_REASONING_INSTRUCTIONS
        )


# ---- Prompt assembly --------------------------------------------------------


def _replace_schema(text: str) -> str:
    """Substitute {schema} -> the JSON schema example. Use .replace (not .format)
    so user text with other curly braces doesn't break."""
    return text.replace("{schema}", SCHEMA_JSON_EXAMPLE)


def _section(name: str, body: str) -> str:
    return f"---BEGIN {name}---\n{body}\n---END {name}---"


def build_unified_prompt(
    invoice_text: str | None,
    agreement_text: str | None,
    *,
    has_invoice_pdf: bool,
    agreement_pdf_count: int,
    with_agreement: bool,
    templates: Optional[PromptTemplates] = None,
) -> str:
    t = templates or PromptTemplates()
    parts = [t.intro]

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

        parts.append(_replace_schema(t.instructions_with_agreement))
    else:
        parts.append(
            "No price agreement is provided. Extract invoice fields only."
        )
        parts.append(_replace_schema(t.instructions_extraction_only))

    return "\n\n".join(parts)


def build_hybrid_extraction_prompt(
    invoice_text: str | None,
    *,
    has_invoice_pdf: bool,
    templates: Optional[PromptTemplates] = None,
) -> str:
    t = templates or PromptTemplates()
    parts = [t.intro]
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
    parts.append(_replace_schema(t.instructions_extraction_only))
    return "\n\n".join(parts)


def build_hybrid_reasoning_prompt(
    invoice_json: str,
    agreement_text: str | None,
    *,
    agreement_pdf_count: int,
    templates: Optional[PromptTemplates] = None,
) -> str:
    t = templates or PromptTemplates()
    parts = [t.intro]
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
    parts.append(_replace_schema(t.reasoning_instructions))
    return "\n\n".join(parts)

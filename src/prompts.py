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

DEFAULT_INTRO = """You are a meticulous invoice auditor for DiFacto, a Danish B2B SaaS that audits
supplier invoices in the construction industry against pre-negotiated price agreements.

Your PRIMARY JOB is to find pricing errors, missing rebates, and credit-note mishandling.
Be skeptical and thorough. Report EVERY discrepancy, no matter how small (down to 0.01 DKK).
Do NOT round, approximate, or assume. If something looks unusual, flag it. The user is
counting on you to catch supplier overcharges that humans miss."""


DEFAULT_INSTRUCTIONS_EXTRACTION_ONLY = """Instructions:

A) READ THE INVOICE
1. Danish text. Decimals are written with comma: "1.234,56" means 1234.56. "Kr." or
   "DKK" indicates Danish Kroner.
2. Extract: supplier_name, invoice_number, invoice_date (YYYY-MM-DD),
   currency (ISO 4217), subtotal, vat, total.

B) DOCUMENT TYPE
3. document_type:
   - "credit_note" if the heading says "Kreditnota", document number starts with KN/CN,
     or all amounts are negative.
   - Otherwise "invoice".
4. credit_note_handling (only for credit notes; null for normal invoices):
   - is_credit_note: true
   - sign_convention: "negative" if amounts shown with minus signs, else "positive"
   - references_invoice: original invoice number if cited (often after "vedr.", "ref",
     "kreditering af"). null if not found.

C) LINE ITEMS
5. Extract EVERY line item — description, quantity, unit_price, line_total. Do not
   skip lines. If a line is unclear, extract what you can and leave unclear fields null.

D) NO AGREEMENT COMPARISON YET
6. This is extraction only. Leave agreed_unit_price, has_discrepancy, discrepancy_amount,
   and expected_rebate as null. Set rebate_applied if the invoice itself shows a
   rebate/discount/"rabat"/"bonus" line, else null.

Output strictly-valid JSON matching this schema (no prose, no code fences, no markdown):

{schema}

Numbers: plain numerics in output (decimal point, no thousands separator). Use null
for unknowns. NEVER invent values."""


DEFAULT_INSTRUCTIONS_WITH_AGREEMENT = """Instructions:

A) READ THE INVOICE
1. Danish text. Decimals use comma: "1.234,56" = 1234.56.
2. Extract: supplier_name, invoice_number, invoice_date (YYYY-MM-DD), currency,
   subtotal, vat, total.
3. document_type: "credit_note" if heading says "Kreditnota", number starts with KN/CN,
   or amounts are negative; else "invoice". For credit notes populate credit_note_handling
   (is_credit_note=true, sign_convention, references_invoice). Else credit_note_handling=null.
4. Extract every line item: description, quantity, unit_price, line_total.

B) MATCH AGREEMENT TO SUPPLIER
5. Pick the agreement that matches the invoice's supplier (by name; ignore case, A/S,
   ApS, CVR numbers). If NONE match, leave agreed_unit_price and rebate fields null
   and skip C-D.

C) FIND PRICING DISCREPANCIES — THE MOST IMPORTANT PART
6. For each line item, find the agreement product whose description best matches:
   - Exact match preferred.
   - Fall back to fuzzy match on Danish words; ignore plural endings, sizes, item codes.
   - If no plausible match, set agreed_unit_price=null, has_discrepancy=null,
     discrepancy_amount=null and move on.
7. For matched lines:
   - agreed_unit_price = price from agreement
   - has_discrepancy = TRUE if abs(unit_price - agreed_unit_price) > 0.01
   - discrepancy_amount = round((unit_price - agreed_unit_price) * quantity, 2)
   - Positive value = supplier overcharged; negative = undercharged.
8. Sanity-check the invoice math (do NOT correct; report what's printed):
   - For each line: unit_price * quantity should ≈ line_total (within 0.02).
   - Sum of line_totals should ≈ subtotal. subtotal + vat should ≈ total.
   - If math is off, the invoice has its own error — still extract the printed values.

D) REBATE
9. rebate_applied: from the invoice (rabat/bonus/afslag/discount line). 0 if not present.
10. expected_rebate: compute from the agreement's bonus/rebate clause applied to this
    invoice's subtotal. If the agreement has no rebate clause, set null.

Output strictly-valid JSON matching this schema (no prose, no code fences, no markdown):

{schema}

Numbers: plain numerics in output (decimal point, no thousands separator). Currency: ISO 4217.
Use null for unknowns. NEVER invent values. Report what you see, not what you think is right."""


DEFAULT_REASONING_INSTRUCTIONS = """Instructions:

You receive an already-extracted invoice (JSON) and the price agreement(s). Your job:
fill in the agreement-related fields and find pricing discrepancies. Be thorough —
catching overcharges is the whole point.

1. KEEP all already-extracted fields exactly as given: supplier_name, invoice_number,
   invoice_date, document_type, currency, subtotal, vat, total, rebate_applied,
   credit_note_handling. Do not "correct" or change them.

2. Match agreement to supplier:
   - Pick the agreement whose supplier name matches the invoice supplier (ignore case,
     A/S, ApS, CVR numbers).
   - If multiple agreements match, pick the most specific.
   - If NONE match, leave agreed_unit_price and rebate fields null on every line and
     output the rest unchanged.

3. For EACH line item, find the agreement product whose description best matches:
   - Exact match preferred. Fall back to fuzzy on Danish words; ignore plural endings,
     sizes, item codes.
   - If no plausible match: agreed_unit_price=null, has_discrepancy=null,
     discrepancy_amount=null on that line.
   - If matched:
     - agreed_unit_price = agreement price
     - has_discrepancy = TRUE if abs(unit_price - agreed_unit_price) > 0.01
     - discrepancy_amount = round((unit_price - agreed_unit_price) * quantity, 2).
       Positive = supplier overcharged.

4. expected_rebate: compute from the agreement's bonus/rebate clauses applied to this
   invoice's subtotal. If the agreement has no rebate clause, set null.

Output the FULL enriched invoice as strictly-valid JSON matching this schema (no prose,
no code fences, no markdown):

{schema}

Numbers: plain numerics in output. Use null for genuinely unknown agreement values; do
NOT invent."""


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

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
supplier invoices in the construction industry against pre-negotiated price agreements
(project-based agreements / projektaftaler, fixed-price agreements / faste prisaftaler,
or a mix). Suppliers vary widely — STARK, Bygma, Stark, Davidsen, Optimera, etc. —
and invoice/agreement layouts vary accordingly.

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
5. Extract EVERY line item: description, item_number (varenummer / SKU exactly as
   printed, including spaces; null if not shown), quantity, unit_price, line_total.
   Do not skip lines. If a line is unclear, extract what you can and leave unclear
   fields null.

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
4. Extract every line item: description, item_number (varenummer / SKU exactly as
   printed, including spaces; null if not shown), quantity, unit_price, line_total.

B) MATCH AGREEMENT TO SUPPLIER
5. Pick the agreement that matches the invoice's supplier. Match loosely: ignore case,
   ignore legal suffixes (A/S, ApS, IVS), ignore CVR numbers, ignore branch/department
   suffixes (e.g. "STARK Aarhus C" matches a "STARK" agreement). Agreements may be
   tabular, bulleted, or prose; do not assume a fixed structure.
   If NONE match, leave agreed_unit_price and rebate fields null and skip C-D.

C) FIND PRICING DISCREPANCIES — THE PRIMARY GOAL

For EVERY single line item — without exception, even if the price looks ordinary —
do these checks. Skipping lines that "look fine" is the most common mistake.

6. MATCH the line to the agreement (priority order):
   a) item_number (varenummer / SKU): exact match, normalized first.
      Treat these as the SAME number — strip whitespace and inconsistent separators:
      "1325 0471007", "13250471007", "1325-0471007", "1325.0471007" all denote the
      same item. This is the most reliable signal; use whenever both sides have one.
   b) Description prefix: first 3-5 significant Danish keywords match, ignoring
      sizes, dimensions, lengths, units, and packaging suffixes. Fuzzy is OK on
      adjectives and word forms ("REGLAR" / "REGLER", "PLADE" / "PLADER", etc.).
   c) If neither (a) nor (b) gives a plausible match: set agreed_unit_price=null,
      has_discrepancy=null, discrepancy_amount=null — move on to the next line.

7. COMPARE prices on EVERY matched line — never skip:
   - has_discrepancy = TRUE iff abs(unit_price - agreed_unit_price) > 0.01
   - discrepancy_amount = round((unit_price - agreed_unit_price) * quantity, 2)
   - Positive value = supplier overcharged; negative = undercharged.

   The threshold is 0,01 DKK in ABSOLUTE terms — NOT relative to the price.
   A 0,55 kr per-unit overcharge IS a discrepancy. Do not use judgment about
   whether a small difference is "meaningful"; the math decides, not your sense
   of what's worth flagging.

   Worked example: a line has unit_price 77,97 kr, quantity 5,954. The agreement
   gives this item at 77,42 kr/stk. Difference per unit = 0,55 kr. Discrepancy
   amount = round(0.55 * 5.954, 2) = 3,27 kr. Set has_discrepancy=TRUE,
   agreed_unit_price=77.42, discrepancy_amount=3.27. Yes — this is a real
   overcharge worth flagging, even though the per-unit difference is small.

8. Sanity-check the invoice math (report as-printed; do NOT correct):
   - Per line: unit_price * quantity ≈ line_total (within 0.02)
   - Sum of line_totals ≈ subtotal. subtotal + vat ≈ total.
   - If math is off, the invoice itself has an arithmetic error — extract and
     report the printed numbers anyway.

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

2. Match agreement to supplier (loose matching):
   - Ignore case, legal suffixes (A/S, ApS, IVS), CVR numbers, and branch/department
     suffixes (e.g. "STARK Aarhus C" matches a "STARK" agreement).
   - Agreement layouts vary — tabular, bulleted, or prose; project agreements
     (projektaftaler) and fixed-price agreements (faste prisaftaler) are both common.
   - If multiple agreements match, pick the most specific.
   - If NONE match, leave agreed_unit_price and rebate fields null on every line and
     output the rest unchanged.

3. For EVERY line item in the input JSON — without exception, do not skip any:

   a) MATCH the line to the agreement (priority order):
      - item_number (varenummer / SKU): exact match after NORMALIZING whitespace
        and separators. Treat these as the same number: "1325 0471007",
        "13250471007", "1325-0471007", "1325.0471007". Use whenever both sides
        have an item_number — most reliable signal.
      - Description prefix: first 3-5 significant Danish keywords match, ignoring
        sizes/dimensions/lengths/units/packaging. Fuzzy on adjectives + word forms.
      - If neither matches plausibly: agreed_unit_price=null, has_discrepancy=null,
        discrepancy_amount=null on that line and move on.

   b) For matched lines, COMPARE PRICES (do not skip even if they look close):
      - agreed_unit_price = agreement price
      - has_discrepancy = TRUE iff abs(unit_price - agreed_unit_price) > 0.01
      - discrepancy_amount = round((unit_price - agreed_unit_price) * quantity, 2)
      - Positive = supplier overcharged; negative = undercharged.

   The threshold is 0,01 DKK ABSOLUTE — not relative to the price. A 0,55 kr
   per-unit difference IS a discrepancy. Do not skip "small" diffs.

   Worked example: line has unit_price 77,97 kr, quantity 5,954. Agreement lists
   this item at 77,42 kr/stk. Difference per unit = 0,55 kr. discrepancy_amount
   = round(0.55 * 5.954, 2) = 3,27. Flag it: has_discrepancy=TRUE,
   agreed_unit_price=77.42, discrepancy_amount=3.27.

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

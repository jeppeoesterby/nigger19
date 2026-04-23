"""Pydantic models for the invoice extraction schema."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str
    quantity: float
    unit_price: float
    line_total: float
    agreed_unit_price: Optional[float] = None
    has_discrepancy: Optional[bool] = None
    discrepancy_amount: Optional[float] = None


class CreditNoteHandling(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_credit_note: bool
    sign_convention: Literal["negative", "positive"]
    references_invoice: Optional[str] = None


class ExtractedInvoice(BaseModel):
    """Matches the ground-truth schema. All fields optional so partial outputs still validate."""

    model_config = ConfigDict(extra="ignore")

    supplier_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    document_type: Optional[Literal["invoice", "credit_note"]] = None
    agreement_file: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    vat: Optional[float] = None
    total: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)
    rebate_applied: Optional[float] = None
    expected_rebate: Optional[float] = None
    credit_note_handling: Optional[CreditNoteHandling] = None


SCHEMA_JSON_EXAMPLE = """{
  "supplier_name": "string",
  "invoice_number": "string",
  "invoice_date": "YYYY-MM-DD",
  "document_type": "invoice | credit_note",
  "currency": "DKK",
  "subtotal": 0.0,
  "vat": 0.0,
  "total": 0.0,
  "line_items": [
    {
      "description": "string",
      "quantity": 0.0,
      "unit_price": 0.0,
      "line_total": 0.0,
      "agreed_unit_price": 0.0,
      "has_discrepancy": false,
      "discrepancy_amount": 0.0
    }
  ],
  "rebate_applied": 0.0,
  "expected_rebate": 0.0,
  "credit_note_handling": {
    "is_credit_note": false,
    "sign_convention": "negative | positive",
    "references_invoice": "string or null"
  }
}"""

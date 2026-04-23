# DiFacto LLM Evaluation Harness

A small Python CLI that runs real invoices through six Claude / Gemini / hybrid
configurations and emits a single multi-sheet Excel report. The goal is to
make **one decision**: which LLM configuration goes into production for
DiFacto's supplier-invoice audit pipeline.

This is a test harness, not production code. No database, no web UI, no Docker.

## Test matrix

| # | Configuration     | Extraction       | Reasoning          |
|---|-------------------|------------------|--------------------|
| 1 | Gemini 2.5 Pro    | Gemini 2.5 Pro   | Gemini 2.5 Pro     |
| 2 | Gemini 3.1 Pro    | Gemini 3.1 Pro   | Gemini 3.1 Pro     |
| 3 | Claude Sonnet 4.6 | Claude Sonnet 4.6| Claude Sonnet 4.6  |
| 4 | Claude Opus 4.7   | Claude Opus 4.7  | Claude Opus 4.7    |
| 5 | Hybrid Sonnet     | Gemini 3.1 Pro   | Claude Sonnet 4.6  |
| 6 | Hybrid Opus       | Gemini 3.1 Pro   | Claude Opus 4.7    |

Pure configs use a single unified prompt (PDF + agreement data + extract +
audit). Hybrid configs split into two calls: Gemini extracts fields from the
PDF, then Claude reasons over the extracted JSON plus the agreement.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and GOOGLE_API_KEY
```

Before the first real run, verify model IDs and per-token pricing in
`config.yaml` against the official provider docs. The defaults shipped here
are our best guess at spring-2026 values.

## Data layout

```
data/
├── invoices/
│   ├── invoice_001.pdf
│   └── ...
├── agreements/
│   ├── supplier_A_agreement.xlsx
│   └── ...
└── ground_truth.json
```

`data/` is git-ignored because it contains real customer data. Ivan/Marius
provide `ground_truth.json`; the harness will not synthesize one — it fails
loudly if the file is missing.

### ground_truth.json shape

```json
{
  "invoice_001.pdf": {
    "supplier_name": "Stark A/S",
    "invoice_number": "2026-10042",
    "invoice_date": "2026-03-15",
    "document_type": "invoice",
    "agreement_file": "supplier_A_agreement.xlsx",
    "currency": "DKK",
    "subtotal": 45200.00,
    "vat": 11300.00,
    "total": 56500.00,
    "line_items": [
      {
        "description": "Gipsplade 13mm 900x2500",
        "quantity": 100,
        "unit_price": 89.50,
        "line_total": 8950.00,
        "agreed_unit_price": 85.00,
        "has_discrepancy": true,
        "discrepancy_amount": 450.00
      }
    ],
    "rebate_applied": 0.00,
    "expected_rebate": 2260.00,
    "credit_note_handling": null
  }
}
```

## Running

```bash
# Full run: all 6 configs, all invoices
python run_eval.py

# Subset of configs
python run_eval.py --configs "Claude Sonnet 4.6,Hybrid Sonnet"

# Limit to first N invoices (good for smoke tests)
python run_eval.py --limit 5

# Validate inputs, print the plan, no API calls
python run_eval.py --dry-run
```

Each run writes a timestamped Excel file to `results/eval_YYYY-MM-DD_HHMM.xlsx`.
Runs never overwrite previous results.

## Adding invoices

1. Drop the PDF into `data/invoices/`.
2. If it's a new supplier, drop the agreement `.xlsx` into `data/agreements/`.
3. Add an entry to `data/ground_truth.json` keyed by the PDF filename.
4. Re-run `python run_eval.py --limit N` where N covers the new doc.

## Output — how to read the Excel

- **Summary** (one row per config). This is Ivan's tab. Columns in order:
  accuracy for each of the 4 dimensions, composite score, latency, token usage,
  cost, failures. Credit-note accuracy is its own prominent column (not buried
  in field extraction).
- **Per-Invoice** (one row per invoice × config). Spot which documents each
  model struggles with.
- **Per-Field** (one row per field × config). Shows where each model loses
  points — e.g. does Gemini miss `invoice_number` more than Claude?
- **Raw-Outputs** — full ground truth, model output, and a line-by-line diff
  for debugging. JSON is pretty-printed with UTF-8 preserved (Danish text
  reads normally).

## Scoring

Weights are configurable in `config.yaml`. Defaults:

- **Field extraction (0.35)** — per-field 1.0 / 0.5 / 0.0 with Danish-aware
  normalization: numeric tolerance 0.01, dates to ISO 8601, supplier names
  fuzzy-matched (threshold 90) after stripping `A/S`, `ApS`, etc.
- **Price match (0.35)** — F1 over the binary `has_discrepancy` flag, using
  fuzzy description match (threshold 85) to align line items.
- **Credit note (0.20)** — pass/fail. Requires correct `document_type`,
  correct sign convention, and correct `references_invoice` (when GT has one).
- **Rebate (0.10)** — within 1 DKK → 1.0; within 1% → 0.5; else 0.0.

`composite = 0.35·field + 0.35·price + 0.20·credit_note + 0.10·rebate`

## Project layout

```
.
├── run_eval.py              # entry point
├── config.yaml              # model IDs, pricing, scoring weights
├── requirements.txt
├── .env.example             # ANTHROPIC_API_KEY / GOOGLE_API_KEY
└── src/
    ├── clients.py           # Claude + Gemini wrappers (unified interface)
    ├── configs.py           # the 6 configurations
    ├── excel_writer.py      # 4-sheet output
    ├── pricing.py           # token → USD
    ├── prompts.py           # unified + hybrid prompts
    ├── runner.py            # main loop
    ├── schema.py            # Pydantic models
    └── scorer.py            # 4 scoring functions + composite
```

## Non-goals

- No production-grade error handling beyond retries + logged failures.
- No parallelization — invoices run sequentially.
- No UI, no dashboard, no auto-updating pricing.
- No synthetic ground truth.
- No per-model prompt tuning. Same prompt for every pure config by design.

# Onboarding — DiFacto LLM Evaluation Harness

Short spec for a developer joining the project. Target: you can run the tool
locally and know where to touch each feature within 15 minutes.

## 1. What this is

A test harness. It runs Danish construction-industry supplier invoices through
a matrix of LLM configurations (Claude / Gemini / hybrids) and writes one Excel
file per run. The output drives a single decision: **which LLM configuration
goes into production** for DiFacto's invoice-audit pipeline.

- Not production code. No DB, no UI framework, no background queue.
- One flat repo. Flask web UI + Python CLI + one `config.yaml`.
- Deployed to Render (free tier). Data lives on ephemeral disk and wipes on
  restart — design around that.

## 2. The matrix

Seven configurations, defined in `src/configs.py`:

| # | Name | Extraction | Reasoning |
|---|---|---|---|
| 1 | Gemini 3.1 Pro | `gemini-3.1-pro-preview` | same |
| 2 | Gemini 2.5 Pro | `gemini-2.5-pro` | same |
| 3 | Gemini 2.5 Flash | `gemini-2.5-flash` | same |
| 4 | Claude Sonnet 4.6 | `claude-sonnet-4-6` | same |
| 5 | Claude Opus 4.7 | `claude-opus-4-7` | same |
| 6 | Hybrid Sonnet | Gemini 3.1 Pro | Claude Sonnet 4.6 |
| 7 | Hybrid Opus | Gemini 3.1 Pro | Claude Opus 4.7 |

Hybrids do two API calls per invoice: extraction on cheap Gemini, reasoning
(discrepancy detection + rebate) on premium Claude.

Model IDs and per-token pricing live in `config.yaml`. Update there, not in
Python. The `/models` endpoint returns what each API key can actually see —
use it before adding a new config.

## 3. Repo layout

```
run_web.py                 # web UI entry point
run_eval.py                # CLI entry point
wsgi.py                    # gunicorn entry (Render production)
config.yaml                # model IDs, pricing, scoring weights, parallelism
Dockerfile / render.yaml   # deploy config
src/
  configs.py               # the 7 model configurations
  clients.py               # Claude + Gemini wrappers, semaphores, retries
  prompts.py               # prompt templates + assembly
  schema.py                # Pydantic schema for extracted invoice data
  scorer.py                # 4 scoring dimensions + composite
  pricing.py               # token × price → USD
  runner.py                # main loop: configs × invoices, circuit breaker, file cache
  excel_writer.py          # 4-sheet Excel output
webapp/
  app.py                   # Flask routes
  templates/               # Jinja2 templates (base, index, run, prompts, results)
```

## 4. Data flow

```
Home page
  → upload PDFs/xlsx to data/invoices/ + data/agreements/
  → optional: upload ground_truth.json for automatic scoring
POST /start
  → spawns background thread running runner.run()
  → in-memory RUNS dict keyed by run_id
runner.run()
  → for each (config, invoice) pair in a ThreadPoolExecutor:
      → build prompt from templates + invoice/agreement content
      → call Claude or Gemini (semaphore-gated per provider)
      → validate response against Pydantic schema
      → score against ground truth (if present)
  → write Excel to results/eval_YYYY-MM-DD_HHMM.xlsx
  → emit event back to RUNS dict for the polling UI
GET /runs/<id>/status
  → returns current log + progress + summary as JSON (polled 1.5s)
GET /results/<filename>
  → downloads Excel
```

## 5. How to run locally

```bash
git clone https://github.com/jeppeoesterby/nigger19.git
cd nigger19
git checkout claude/llm-evaluation-harness-e47hv
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in ANTHROPIC_API_KEY + GOOGLE_API_KEY
python run_web.py             # → http://127.0.0.1:5000
```

Or CLI:

```bash
python run_eval.py --configs "Claude Sonnet 4.6" --limit 3 --dry-run
```

## 6. Scoring (when ground_truth.json is present)

- **Field extraction (35 %)** — per-field 1.0 / 0.5 / 0.0 with Danish-aware
  normalization: numeric tolerance 0.01, dates to ISO 8601, supplier names
  fuzzy-matched after stripping `A/S`, `ApS` etc.
- **Price match (35 %)** — F1 over the binary `has_discrepancy` flag on
  fuzzy-matched line items.
- **Credit note (20 %)** — pass/fail: correct document_type, correct sign
  convention, correct referenced invoice.
- **Rebate (10 %)** — within 1 DKK → 1.0; within 1 % → 0.5; else 0.0.

`composite = 0.35·field + 0.35·price + 0.20·credit + 0.10·rebate`.
Weights are in `config.yaml` and editable without code changes.

Without ground truth, the harness still runs ("extraction-only" mode),
outputs model predictions to the Excel, and skips the accuracy columns.

## 7. Performance model

Two tiers of parallelism:

- `api.parallelism` (default 10) — size of the global `ThreadPoolExecutor`
  that dispatches (config, invoice) pairs.
- `api.claude_max_concurrent` (default 2) — semaphore inside `ClaudeClient`.
- `api.gemini_max_concurrent` (default 8) — semaphore inside `GeminiClient`.

So Gemini can run 8-wide while Claude is capped at 2 by Anthropic tier-1 rate
limits. Bump `claude_max_concurrent` after upgrading Anthropic tier.

**Prompt caching** — `ClaudeClient.call()` takes `cached_prefix_pdfs`. The
runner passes agreement PDFs there; they're sent first with
`cache_control: ephemeral` on the last one. Anthropic caches the prefix for
5 min; subsequent invoices in the same config get a ~90 % discount on those
input tokens + lower server-side compute (faster).

**Circuit breaker** — 5 consecutive *permanent* failures for a config disable
the rest of its invoices for that run. Transient failures (429, 5xx, timeouts)
reset the counter. Classification is in `clients._is_retriable()`.

**File cache** — `runner._FileCache` memoizes PDF bytes and flattened xlsx
text per run. Each unique file is read from disk exactly once even when
referenced by 648 (invoice, config) pairs.

## 8. Prompts — editable without deploying

`src/prompts.py` defines four template blocks (intro, instructions-with-
agreement, instructions-extraction-only, hybrid-reasoning). Users edit them
at `/prompts` in the browser. Overrides land in `data/prompts.json` which
`PromptTemplates.load()` reads at run start (snapshotted so mid-run edits
don't mix).

`{schema}` is substituted with the JSON schema example at send time. Use
`.replace()` not `.format()` so user text with stray braces doesn't break.

## 9. Common gotchas

- **Ephemeral disk on Render free tier.** Uploads and results are wiped on
  redeploy / spin-down. "Files on disk" card on the home page is the source
  of truth; Start button is disabled when `invoice_count_on_disk == 0`.
- **Model IDs rot.** Hit `/models` to see what each key can actually call.
  Preview models (like `gemini-3.1-pro-preview`) may not be on every tier.
- **Rate limits look like empty responses.** Claude returning near-zero
  output tokens is usually input throttling, not model failure. Raw response
  text is captured in the Excel `raw_response_text` column for debugging.
- **`{}` in model_output_json + notes empty** means JSON parsing succeeded
  on an empty object; check `raw_response_text` to see what Claude actually
  said. The `[EMPTY RESPONSE DEBUG]` marker surfaces `stop_reason`,
  content-block types, etc.
- **Don't commit `.env`.** It's in `.gitignore`; verify before pushing.

## 10. How to add a new model

1. Verify the model ID via `/models` or provider docs.
2. Add an entry to `config.yaml` under `models` and `pricing`.
3. Add a `ModelConfig` entry to `src/configs.py`.
4. No other code changes needed — the runner dispatches by `provider` string.

## 11. How to add a new scoring dimension

1. Add a `score_<dimension>()` function to `src/scorer.py` returning float in
   [0, 1].
2. Call it from `score_invoice()` and include in the composite.
3. Add weight key to `config.yaml` under `scoring_weights` (sum must = 1.0).
4. Add column to `excel_writer.py` Summary + Per-Invoice sheets.

## 12. Branch and CI

- Dev branch: `claude/llm-evaluation-harness-e47hv`.
- Main branch is empty placeholder — don't merge to main until we're
  shipping.
- Render auto-deploys on push via `render.yaml` blueprint config.

## 13. Keys

- `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` in Render dashboard (Environment
  panel), NOT in code.
- Rotate keys if they ever appear in chat/logs/screenshots.

---

That's everything. Point a new dev at this file, a fresh API key, and
`http://127.0.0.1:5000` — they should be able to kick off a run within 15
minutes and debug extraction output within an hour.

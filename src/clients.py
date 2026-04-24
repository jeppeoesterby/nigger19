"""Unified client wrappers for Claude (Anthropic) and Gemini (Google)."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


def _is_retriable(exc: BaseException) -> bool:
    """Retry only on transient failures: rate limits, timeouts, 5xx, network
    errors. Never retry on 4xx client errors (wrong model, bad key, bad
    request) — those are configuration bugs that retrying can't fix."""
    name = type(exc).__name__
    msg = str(exc).lower()
    # Non-retriable signals anywhere in the class name or message.
    non_retriable_markers = (
        "notfound",
        "not_found",
        "permission",
        "unauthor",
        "authentic",
        "badrequest",
        "bad_request",
        "invalidargument",
        "invalid_argument",
        "unprocessable",
    )
    if any(m in name.lower() for m in non_retriable_markers):
        return False
    if any(m in msg for m in ("404", "401", "403", "400", "422")):
        return False
    # Retriable signals.
    retriable_markers = (
        "ratelimit",
        "rate_limit",
        "timeout",
        "connection",
        "unavailable",
        "internalserver",
        "apiconnection",
        "apitimeout",
    )
    if any(m in name.lower() for m in retriable_markers):
        return True
    if any(m in msg for m in ("429", "500", "502", "503", "504", "timeout", "temporarily")):
        return True
    # Unknown -> retry once. `stop_after_attempt(3)` still caps us.
    return True


@dataclass
class ModelCall:
    """Result of a single model invocation."""

    raw_text: str
    latency_sec: float
    input_tokens: int
    output_tokens: int
    model_id: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_int(v) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _gemini_usage(resp) -> tuple[int, int]:
    """Extract (prompt, output) token counts from a Gemini response.

    The google-genai SDK sometimes returns a UsageMetadata with attributes
    equal to None (e.g. on safety-filtered or empty candidates). Some
    versions of the SDK use snake_case, others camelCase. Be defensive.
    """
    usage = getattr(resp, "usage_metadata", None)
    if usage is None:
        return 0, 0
    in_tok = (
        getattr(usage, "prompt_token_count", None)
        or getattr(usage, "promptTokenCount", None)
        or 0
    )
    out_tok = (
        getattr(usage, "candidates_token_count", None)
        or getattr(usage, "candidatesTokenCount", None)
        or getattr(usage, "output_token_count", None)
        or 0
    )
    return _safe_int(in_tok), _safe_int(out_tok)


def parse_model_json(raw_text: str) -> dict:
    cleaned = _strip_code_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


_B64_CACHE: dict[str, str] = {}
_B64_CACHE_LOCK = threading.Lock()


def _pdf_b64(pdf: bytes) -> str:
    """Base64-encode a PDF once and cache the string. Hashing 2 MB of bytes is
    faster than re-encoding them on every call (~3x speedup)."""
    key = hashlib.md5(pdf).hexdigest()
    with _B64_CACHE_LOCK:
        cached = _B64_CACHE.get(key)
    if cached is not None:
        return cached
    encoded = base64.standard_b64encode(pdf).decode("ascii")
    with _B64_CACHE_LOCK:
        _B64_CACHE[key] = encoded
    return encoded


class ClaudeClient:
    def __init__(
        self,
        api_key: str,
        max_tokens: int = 8000,
        timeout_sec: int = 300,
        max_concurrent: int = 2,
    ):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key, timeout=timeout_sec)
        self.max_tokens = max_tokens
        # Provider-local semaphore gates in-flight API calls so the global
        # thread pool can be larger than the strictest provider rate limit.
        self._sem = threading.Semaphore(max_concurrent)

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=3, min=3, max=90),
        retry=retry_if_exception(_is_retriable),
        reraise=True,
    )
    def _raw_call(
        self,
        model: str,
        prompt: str,
        pdf_documents: Sequence[bytes],
        cached_prefix_pdfs: Sequence[bytes] = (),
    ) -> ModelCall:
        content: list = []
        # Cacheable prefix first: PDFs identical across calls in this config
        # (typically the agreement). Mark the last one with cache_control so
        # Anthropic server-side caches the prefix for 5 min. Subsequent calls
        # with identical prefix get a ~90% discount on those input tokens AND
        # lower server-side compute (faster response).
        cached = list(cached_prefix_pdfs)
        for i, pdf in enumerate(cached):
            block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": _pdf_b64(pdf),
                },
            }
            if i == len(cached) - 1:
                block["cache_control"] = {"type": "ephemeral"}
            content.append(block)
        # Per-invoice documents (not cached)
        for pdf in pdf_documents:
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": _pdf_b64(pdf),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        t0 = time.perf_counter()
        msg = self._client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        latency = time.perf_counter() - t0

        blocks = list(getattr(msg, "content", []) or [])
        text = "".join(
            b.text for b in blocks if getattr(b, "type", None) == "text"
        )
        usage = getattr(msg, "usage", None)
        in_tok = _safe_int(getattr(usage, "input_tokens", 0)) if usage else 0
        out_tok = _safe_int(getattr(usage, "output_tokens", 0)) if usage else 0

        # If we got no text, attach diagnostic metadata so the user can see
        # WHY the response was empty (stop_reason, block types, etc.) in
        # the Excel raw_response column.
        if not text.strip():
            stop_reason = getattr(msg, "stop_reason", None)
            stop_sequence = getattr(msg, "stop_sequence", None)
            block_types = [getattr(b, "type", "?") for b in blocks]
            # Non-text blocks sometimes carry explanation (tool_use, thinking,
            # etc.). Dump their repr so we can see them.
            non_text_preview = []
            for b in blocks:
                if getattr(b, "type", None) != "text":
                    try:
                        non_text_preview.append(repr(b)[:500])
                    except Exception:
                        non_text_preview.append(f"<{type(b).__name__}>")
            text = (
                f"[EMPTY RESPONSE DEBUG] model={model}  "
                f"stop_reason={stop_reason!r}  stop_sequence={stop_sequence!r}  "
                f"input_tokens={in_tok}  output_tokens={out_tok}  "
                f"content_blocks={block_types}"
            )
            if non_text_preview:
                text += "\nNon-text blocks:\n" + "\n".join(non_text_preview)

        return ModelCall(
            raw_text=text,
            latency_sec=float(latency),
            input_tokens=in_tok,
            output_tokens=out_tok,
            model_id=model,
        )

    def call(
        self,
        model: str,
        prompt: str,
        pdf_documents: Optional[Sequence[bytes]] = None,
        cached_prefix_pdfs: Optional[Sequence[bytes]] = None,
    ) -> ModelCall:
        try:
            with self._sem:
                return self._raw_call(
                    model,
                    prompt,
                    pdf_documents or (),
                    cached_prefix_pdfs or (),
                )
        except Exception as e:
            log.warning("Claude call failed (%s): %s", model, e)
            return ModelCall("", 0.0, 0, 0, model, error=str(e))


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        max_tokens: int = 8000,
        timeout_sec: int = 300,
        max_concurrent: int = 8,
    ):
        from google import genai
        from google.genai import types as genai_types

        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self._sem = threading.Semaphore(max_concurrent)

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=3, min=3, max=90),
        retry=retry_if_exception(_is_retriable),
        reraise=True,
    )
    def _raw_call(
        self, model: str, prompt: str, pdf_documents: Sequence[bytes]
    ) -> ModelCall:
        parts: list = []
        for pdf in pdf_documents:
            parts.append(
                self._types.Part.from_bytes(data=pdf, mime_type="application/pdf")
            )
        parts.append(prompt)

        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=model,
            contents=parts,
            config=self._types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        latency = time.perf_counter() - t0
        text = resp.text or ""
        in_tok, out_tok = _gemini_usage(resp)

        if not text.strip():
            # Pull diagnostic metadata from the candidates list.
            finish_reasons = []
            safety_notes: list[str] = []
            block_reason = None
            try:
                candidates = getattr(resp, "candidates", None) or []
                for c in candidates:
                    fr = getattr(c, "finish_reason", None)
                    if fr is not None:
                        finish_reasons.append(str(fr))
                    sr = getattr(c, "safety_ratings", None) or []
                    for s in sr:
                        cat = getattr(s, "category", None)
                        prob = getattr(s, "probability", None)
                        if cat is not None:
                            safety_notes.append(f"{cat}={prob}")
                pf = getattr(resp, "prompt_feedback", None)
                if pf is not None:
                    block_reason = getattr(pf, "block_reason", None)
            except Exception as e:  # diagnostics must never raise
                finish_reasons.append(f"(diag error: {e})")
            text = (
                f"[EMPTY RESPONSE DEBUG] model={model}  "
                f"finish_reasons={finish_reasons}  "
                f"prompt_block_reason={block_reason!r}  "
                f"input_tokens={in_tok}  output_tokens={out_tok}"
            )
            if safety_notes:
                text += "\nSafety ratings: " + ", ".join(safety_notes)

        return ModelCall(
            raw_text=text,
            latency_sec=float(latency),
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
            model_id=model,
        )

    def call(
        self,
        model: str,
        prompt: str,
        pdf_documents: Optional[Sequence[bytes]] = None,
        cached_prefix_pdfs: Optional[Sequence[bytes]] = None,
    ) -> ModelCall:
        # Gemini doesn't expose an equivalent of Anthropic prompt caching in
        # the unified SDK yet; concatenate all attachments.
        all_pdfs = list(cached_prefix_pdfs or ()) + list(pdf_documents or ())
        try:
            with self._sem:
                return self._raw_call(model, prompt, all_pdfs)
        except Exception as e:
            log.warning("Gemini call failed (%s): %s", model, e)
            return ModelCall("", 0.0, 0, 0, model, error=str(e))


def build_clients(cfg: dict) -> dict:
    anth_key = os.environ.get("ANTHROPIC_API_KEY")
    goog_key = os.environ.get("GOOGLE_API_KEY")
    if not anth_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (see .env.example)")
    if not goog_key:
        raise RuntimeError("GOOGLE_API_KEY not set (see .env.example)")
    api = cfg.get("api", {})
    return {
        "claude": ClaudeClient(
            anth_key,
            max_tokens=api.get("max_tokens", 8000),
            timeout_sec=api.get("timeout_sec", 300),
            max_concurrent=api.get("claude_max_concurrent", 2),
        ),
        "gemini": GeminiClient(
            goog_key,
            max_tokens=api.get("max_tokens", 8000),
            timeout_sec=api.get("timeout_sec", 300),
            max_concurrent=api.get("gemini_max_concurrent", 8),
        ),
    }

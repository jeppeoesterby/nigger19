"""Unified client wrappers for Claude (Anthropic) and Gemini (Google)."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


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


class ClaudeClient:
    def __init__(self, api_key: str, max_tokens: int = 8000, timeout_sec: int = 300):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key, timeout=timeout_sec)
        self.max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _raw_call(
        self, model: str, prompt: str, pdf_documents: Sequence[bytes]
    ) -> ModelCall:
        content: list = []
        for pdf in pdf_documents:
            b64 = base64.standard_b64encode(pdf).decode("ascii")
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64,
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
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        )
        usage = getattr(msg, "usage", None)
        in_tok = _safe_int(getattr(usage, "input_tokens", 0)) if usage else 0
        out_tok = _safe_int(getattr(usage, "output_tokens", 0)) if usage else 0
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
    ) -> ModelCall:
        try:
            return self._raw_call(model, prompt, pdf_documents or ())
        except Exception as e:
            log.warning("Claude call failed (%s): %s", model, e)
            return ModelCall("", 0.0, 0, 0, model, error=str(e))


class GeminiClient:
    def __init__(self, api_key: str, max_tokens: int = 8000, timeout_sec: int = 300):
        from google import genai
        from google.genai import types as genai_types

        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
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
    ) -> ModelCall:
        try:
            return self._raw_call(model, prompt, pdf_documents or ())
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
        ),
        "gemini": GeminiClient(
            goog_key,
            max_tokens=api.get("max_tokens", 8000),
            timeout_sec=api.get("timeout_sec", 300),
        ),
    }

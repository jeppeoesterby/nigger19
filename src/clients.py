"""Unified client wrappers for Claude (Anthropic) and Gemini (Google)."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
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
    """Remove ```json ... ``` fences if the model returned them anyway."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_model_json(raw_text: str) -> dict:
    """Parse JSON from a model response, tolerating code fences and trailing prose."""
    cleaned = _strip_code_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort: grab the first {...} block.
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
    def _call_with_pdf(self, model: str, pdf_bytes: bytes, prompt: str) -> ModelCall:
        b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")
        t0 = time.perf_counter()
        msg = self._client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        latency = time.perf_counter() - t0
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )
        return ModelCall(
            raw_text=text,
            latency_sec=latency,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            model_id=model,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_text_only(self, model: str, prompt: str) -> ModelCall:
        t0 = time.perf_counter()
        msg = self._client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        latency = time.perf_counter() - t0
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )
        return ModelCall(
            raw_text=text,
            latency_sec=latency,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            model_id=model,
        )

    def extract_from_pdf(self, model: str, pdf_bytes: bytes, prompt: str) -> ModelCall:
        try:
            return self._call_with_pdf(model, pdf_bytes, prompt)
        except Exception as e:
            log.warning("Claude extract_from_pdf failed (%s): %s", model, e)
            return ModelCall("", 0.0, 0, 0, model, error=str(e))

    def reason_text(self, model: str, prompt: str) -> ModelCall:
        try:
            return self._call_text_only(model, prompt)
        except Exception as e:
            log.warning("Claude reason_text failed (%s): %s", model, e)
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
    def _call_with_pdf(self, model: str, pdf_bytes: bytes, prompt: str) -> ModelCall:
        pdf_part = self._types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=model,
            contents=[pdf_part, prompt],
            config=self._types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        latency = time.perf_counter() - t0
        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return ModelCall(
            raw_text=text,
            latency_sec=latency,
            input_tokens=in_tok or 0,
            output_tokens=out_tok or 0,
            model_id=model,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_text_only(self, model: str, prompt: str) -> ModelCall:
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=model,
            contents=[prompt],
            config=self._types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        latency = time.perf_counter() - t0
        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return ModelCall(
            raw_text=text,
            latency_sec=latency,
            input_tokens=in_tok or 0,
            output_tokens=out_tok or 0,
            model_id=model,
        )

    def extract_from_pdf(self, model: str, pdf_bytes: bytes, prompt: str) -> ModelCall:
        try:
            return self._call_with_pdf(model, pdf_bytes, prompt)
        except Exception as e:
            log.warning("Gemini extract_from_pdf failed (%s): %s", model, e)
            return ModelCall("", 0.0, 0, 0, model, error=str(e))

    def reason_text(self, model: str, prompt: str) -> ModelCall:
        try:
            return self._call_text_only(model, prompt)
        except Exception as e:
            log.warning("Gemini reason_text failed (%s): %s", model, e)
            return ModelCall("", 0.0, 0, 0, model, error=str(e))


def build_clients(cfg: dict) -> dict:
    """Instantiate clients keyed by provider name. Fail loudly if keys are missing."""
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

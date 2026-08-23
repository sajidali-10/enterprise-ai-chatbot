"""Phase 34B — OpenAI-compatible Vision provider.

Speaks the OpenAI Chat Completions Vision protocol. Works against:
    * OpenAI (``https://api.openai.com/v1``)
    * OpenRouter (``https://openrouter.ai/api/v1``)
    * Azure OpenAI (configure ``VISION_BASE_URL``)
    * Local vLLM / llama.cpp / Ollama OpenAI-compatible gateways

Configuration (independent from text LLM settings):
    VISION_PROVIDER=openai-compatible
    VISION_BASE_URL=https://api.openai.com/v1
    VISION_MODEL=gpt-4o-mini
    VISION_API_KEY=<env var, read at runtime, never stored>
    VISION_TIMEOUT_SECONDS=30
    VISION_MAX_RETRIES=1

The provider sends the image as a base64 data URL inside a
``chat/completions`` request. The model is asked to respond with
strict JSON so the result can be parsed deterministically into a
``VisionResult``. Parsing is lenient — if the model returns a
plain paragraph we still extract ``description`` so the pipeline
does not break on a stricter provider.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.services.vision.base import (
    VisionProvider,
    VisionProviderError,
    VisionProviderTimeoutError,
    VisionProviderUnavailableError,
    VisionResult,
    VISION_SCHEMA_VERSION,
    timing_wrapper,
)

logger = logging.getLogger(__name__)

# Avoid pulling httpx in at import time — providers should fail soft.
try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - httpx is a runtime dependency
    httpx = None  # type: ignore


_SYSTEM_PROMPT = (
    "You are analyzing an enterprise technical screenshot. Describe "
    "visually observable facts only. Do not infer undocumented product "
    "behavior or troubleshooting steps. Preserve visible error codes, "
    "labels, status indicators, relationships, graph trends, selected "
    "controls, and warnings exactly as they appear. Mark anything you "
    "cannot see clearly as 'uncertain'. Avoid unnecessary verbose "
    "descriptions. Respond with STRICT JSON of the form: "
    "{\"description\": str, \"image_type\": str, "
    "\"visual_findings\": [str], \"detected_entities\": [str], "
    "\"visual_states\": [str], \"tags\": [str], \"confidence\": int}."
)


class OpenAICompatibleVisionProvider:
    """OpenAI Chat Completions Vision provider.

    Failures are surfaced as ``VisionProviderError`` subclasses so
    the router can degrade gracefully to OCR evidence.
    """

    name = "openai-compatible"
    model: str = ""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        # API key is read from env at runtime, NEVER stored in
        # settings — same pattern as LANGSMITH_API_KEY.
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("VISION_API_KEY", "")
        )
        self.base_url = (
            (base_url or os.getenv("VISION_BASE_URL", "") or "").rstrip("/")
        )
        self.model = (
            model
            if model is not None and model
            else os.getenv("VISION_MODEL", "gpt-4o-mini")
        )
        # Bounded defaults that can be overridden via constructor
        # for tests.
        self.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else os.getenv("VISION_TIMEOUT_SECONDS", "30")
        )
        self.max_retries = int(
            max_retries
            if max_retries is not None
            else os.getenv("VISION_MAX_RETRIES", "1")
        )

        if not self.base_url:
            # Fail-soft at construction so callers can detect this
            # before sending an image. The router treats this as
            # provider-unavailable and falls back to OCR.
            raise VisionProviderUnavailableError(
                "VISION_BASE_URL is not configured. Set VISION_BASE_URL "
                "to the OpenAI-compatible Vision endpoint."
            )
        if not self.api_key:
            raise VisionProviderUnavailableError(
                "VISION_API_KEY is not set. The openai-compatible Vision "
                "provider cannot authenticate."
            )
        if httpx is None:
            raise VisionProviderUnavailableError(
                "httpx is not importable — openai-compatible Vision "
                "provider requires httpx."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timing_wrapper
    def analyze_image(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        ocr_text: str = "",
        context_hint: str = "",
        max_tokens: int = 600,
        timeout_s: Optional[float] = None,
    ) -> VisionResult:
        if not image_bytes:
            raise VisionProviderUnavailableError(
                "Refusing to call openai-compatible Vision with empty bytes."
            )

        effective_timeout = float(timeout_s) if timeout_s is not None else self.timeout_s
        data_url = _build_data_url(image_bytes, mime_type or "image/png")

        user_prompt = self._build_user_prompt(
            prompt=prompt, ocr_text=ocr_text, context_hint=context_hint
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "max_tokens": int(max(64, min(max_tokens, 4096))),
            "temperature": 0.0,
        }

        response_text = self._call_with_retries(payload, effective_timeout)
        parsed = _parse_json_lenient(response_text)
        return _build_vision_result(parsed, provider=self.name, model=self.model)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self, *, prompt: str, ocr_text: str, context_hint: str
    ) -> str:
        # Defensive truncation so a 200KB OCR blob does not blow the
        # context window of a small Vision model.
        ocr_snippet = (ocr_text or "").strip()
        if len(ocr_snippet) > 2000:
            ocr_snippet = ocr_snippet[:2000].rsplit(" ", 1)[0] + "…"
        hint = (context_hint or "").strip()
        blocks = []
        if hint:
            blocks.append(f"Context: {hint}")
        blocks.append(
            f"User question: {(prompt or '').strip() or 'Describe the image.'}"
        )
        if ocr_snippet:
            blocks.append(
                "OCR text already extracted from the image (use it as "
                "ground truth for any text/numbers/labels, do not invent):\n"
                f"{ocr_snippet}"
            )
        else:
            blocks.append(
                "No OCR text was extracted — describe only what is "
                "visually observable."
            )
        return "\n\n".join(blocks)

    def _call_with_retries(self, payload: Dict[str, Any], timeout_s: float) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        attempts = max(0, int(self.max_retries)) + 1
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                with httpx.Client(timeout=timeout_s) as client:  # type: ignore[name-defined]
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    # Retry on transient errors.
                    last_error = VisionProviderUnavailableError(
                        f"provider status {resp.status_code}: {resp.text[:200]}"
                    )
                    continue
                if resp.status_code >= 400:
                    raise VisionProviderUnavailableError(
                        f"provider status {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
                return _extract_message_text(data)
            except Exception as exc:
                # Timeouts and connection errors are transient — retry
                # up to max_retries times.
                last_error = exc
                if isinstance(exc, VisionProviderError):
                    # Non-retryable provider errors should bubble up.
                    raise
                if attempt < attempts:
                    continue
                if "timeout" in repr(exc).lower() or "timed out" in repr(exc).lower():
                    raise VisionProviderTimeoutError(
                        f"openai-compatible Vision timed out after "
                        f"{attempt} attempts"
                    ) from exc
                raise VisionProviderUnavailableError(
                    f"openai-compatible Vision failed: {exc}"
                ) from exc
        # All retries exhausted.
        raise VisionProviderUnavailableError(
            f"openai-compatible Vision failed after {attempts} attempts: {last_error}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_data_url(image_bytes: bytes, mime_type: str) -> str:
    safe_mime = (mime_type or "image/png").split(";")[0].strip().lower()
    if safe_mime not in ("image/png", "image/jpeg", "image/jpg",
                         "image/webp", "image/gif"):
        safe_mime = "image/png"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{safe_mime};base64,{b64}"


def _extract_message_text(data: Dict[str, Any]) -> str:
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        # Some providers return content as a list of segments.
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif "text" in item:
                        parts.append(item.get("text", ""))
            return "\n".join(p for p in parts if p)
        return str(content or "")
    except Exception:
        return ""


def _parse_json_lenient(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response.

    Many models wrap JSON in ```json fences or add a brief preamble.
    We try strict JSON first, then a fenced block, then the first
    balanced brace pair. If everything fails we return a dict with
    the raw text as ``description`` so the result is never empty.
    """
    if not text:
        return {}
    text = text.strip()

    # 1) Direct JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Fenced ```json ... ```.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) First balanced { ... } substring.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 4) Fallback — treat the whole response as a free description.
    return {"description": text}


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        # Split on newlines / semicolons so a single-string field
        # still yields multiple findings.
        parts = re.split(r"[\n;]+", value)
        return [p.strip(" -•\t") for p in parts if p.strip()]
    return [str(value)]


def _coerce_confidence(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except Exception:
        return None
    return max(0, min(100, n))


def _build_vision_result(
    parsed: Dict[str, Any], *, provider: str, model: str
) -> VisionResult:
    return VisionResult(
        description=str(parsed.get("description") or "").strip(),
        image_type=str(parsed.get("image_type") or "unknown").strip()
        or "unknown",
        visual_findings=_coerce_list(parsed.get("visual_findings")),
        detected_entities=_coerce_list(parsed.get("detected_entities")),
        visual_states=_coerce_list(parsed.get("visual_states")),
        tags=_coerce_list(parsed.get("tags")),
        confidence=_coerce_confidence(parsed.get("confidence")),
        provider=provider,
        model=model,
        processing_time_ms=0,  # timing_wrapper fills this
        schema_version=VISION_SCHEMA_VERSION,
        raw=None,  # Do NOT forward provider internals to LLM
    )

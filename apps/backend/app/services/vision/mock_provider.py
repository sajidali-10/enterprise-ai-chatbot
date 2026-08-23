"""Phase 34B — Mock Vision provider (safe default).

The mock provider:

* is fully deterministic,
* performs ZERO network calls,
* returns a structured ``VisionResult`` synthesised from the OCR
  text + filename + a small prompt-derived heuristic,
* is the safe default for unit tests, dev environments, and
  deployments without a Vision-capable model.

The mock is intentionally good enough to drive the OCR-first
evidence flow during local development without paying for a real
Vision API. It MUST NOT be used for production visual grounding —
operators flip ``VISION_PROVIDER=openai-compatible`` in deployment.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from app.services.vision.base import (
    VisionProvider,
    VisionResult,
    VISION_SCHEMA_VERSION,
    timing_wrapper,
)


class MockVisionProvider:
    """Deterministic offline Vision provider.

    Builds a ``VisionResult`` from the supplied OCR text plus the
    prompt. The output is intentionally conservative: it describes
    what OCR observed and tags the image type from textual cues,
    but it never invents entities, troubleshooting steps, or
    product-specific meanings.

    Behaviour matrix:
        * empty OCR + empty prompt -> "No visual evidence available"
          with empty arrays (success=True, description non-empty).
        * OCR-only                -> description summarises the
          visible text; tags derived from keyword heuristics.
        * OCR + visual prompt     -> description includes a
          "Visual indicators (heuristic):" line summarising any
          chart/graph/diagram cues found in the OCR text.

    The mock is registered under ``name="mock"`` and ``model="mock-v1"``
    so the cache layer can key on a stable identifier.
    """

    name = "mock"
    model = "mock-v1"

    # Conservative keyword -> tag mapping. These mirror the Phase 34B
    # spec examples and exist purely so the mock output looks
    # plausible in observability; they MUST NOT influence routing.
    _TAG_KEYWORDS = {
        "dashboard": ("dashboard", "overview"),
        "chart": ("chart", "graph", "trend", "spike", "drop"),
        "diagram": ("diagram", "architecture", "topology", "layout"),
        "table": ("table", "row", "column"),
        "ui": ("button", "icon", "menu", "tab", "control", "selected", "disabled"),
        "alert": ("error", "warning", "alert", "critical", "red indicator"),
        "log": ("log", "exception", "stacktrace"),
    }

    _DIAGRAM_TOKENS = ("chart", "graph", "dashboard", "diagram",
                       "architecture", "topology")

    def __init__(self, model: Optional[str] = None) -> None:
        if model:
            self.model = model

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
        ocr_text = (ocr_text or "").strip()
        prompt_lower = (prompt or "").lower()

        # 1) Image type inference — purely textual heuristic.
        image_type = self._infer_image_type(ocr_text, prompt_lower)

        # 2) Tag inference — only from text already in scope.
        tags = self._infer_tags(ocr_text, prompt_lower)

        # 3) Description — summarise what OCR observed, never invent.
        description = self._summarise(ocr_text, prompt_lower, image_type)

        # 4) Visual findings — short, grounded strings.
        findings = self._extract_findings(ocr_text, prompt_lower, image_type)

        # 5) Detected entities — copy short, visible identifiers only.
        entities = self._extract_entities(ocr_text)

        # 6) Visual states — only when the user prompt asks for them.
        states = self._extract_states(prompt_lower, ocr_text)

        # 7) Confidence — deterministic but byte-derived so two calls
        # with identical inputs produce identical confidence scores.
        confidence = self._synth_confidence(image_bytes)

        return VisionResult(
            description=description,
            image_type=image_type,
            visual_findings=findings,
            detected_entities=entities,
            visual_states=states,
            tags=tags,
            confidence=confidence,
            provider=self.name,
            model=self.model,
            processing_time_ms=0,  # timing_wrapper fills this
            schema_version=VISION_SCHEMA_VERSION,
            raw={
                "mock": True,
                "ocr_text_length": len(ocr_text),
                "mime_type": mime_type,
                "byte_length": len(image_bytes or b""),
            },
        )

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def _infer_image_type(self, ocr_text: str, prompt_lower: str) -> str:
        text = f"{ocr_text}\n{prompt_lower}".lower()
        for token in self._DIAGRAM_TOKENS:
            if token in text:
                return token
        if "log" in text and ("exception" in text or "stacktrace" in text):
            return "log"
        if "table" in text or "row" in text:
            return "table"
        if any(t in text for t in ("button", "icon", "menu", "control")):
            return "ui"
        if ocr_text:
            return "screenshot"
        return "unknown"

    def _infer_tags(self, ocr_text: str, prompt_lower: str) -> List[str]:
        text = f"{ocr_text}\n{prompt_lower}".lower()
        tags: List[str] = []
        for tag, keywords in self._TAG_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                tags.append(tag)
        return tags

    def _summarise(self, ocr_text: str, prompt_lower: str, image_type: str) -> str:
        if not ocr_text:
            return (
                "No textual content was extracted from the image (mock "
                "Vision provider — no visual content is described beyond "
                "OCR text)."
            )
        # Truncate to a reasonable length so the prompt does not blow
        # up when combined with conversation context.
        snippet = ocr_text.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "…"
        prefix = f"OCR-extracted text ({image_type}):"
        if image_type in self._DIAGRAM_TOKENS:
            return (
                f"{prefix} {snippet}\n"
                "Visual indicators (heuristic): graph/chart/dashboard "
                "structure inferred from textual cues only — no actual "
                "visual grounding was performed."
            )
        return f"{prefix} {snippet}"

    def _extract_findings(
        self, ocr_text: str, prompt_lower: str, image_type: str
    ) -> List[str]:
        findings: List[str] = []
        if image_type in self._DIAGRAM_TOKENS:
            findings.append(
                "Image type suggests a chart / dashboard layout based "
                "on textual cues."
            )
        if "spike" in prompt_lower or "spike" in ocr_text.lower():
            findings.append("Prompt mentions a spike — actual trend is not visually grounded.")
        if "drop" in prompt_lower or "drop" in ocr_text.lower():
            findings.append("Prompt mentions a drop — actual trend is not visually grounded.")
        if not findings:
            findings.append(
                "Mock provider did not perform real visual analysis — "
                "OCR text is the only observable signal."
            )
        return findings

    def _extract_entities(self, ocr_text: str) -> List[str]:
        # Cap at 8 short visible identifiers. We extract numeric codes,
        # mixed alphanumeric identifiers, and quoted labels — but never
        # guesses.
        if not ocr_text:
            return []
        text = ocr_text[:2000]  # bounded scan
        entities: List[str] = []
        seen: set = set()

        # Error codes like 902, ERR_42, ABC-123.
        for m in re.finditer(r"\b[A-Z]?[A-Z0-9]{2,3}[-_]?[A-Z0-9]{2,6}\b", text):
            token = m.group(0).strip()
            if token and token not in seen and len(token) <= 12:
                seen.add(token)
                entities.append(token)
                if len(entities) >= 8:
                    return entities

        # Pure numeric codes (error codes, status codes).
        if len(entities) < 8:
            for m in re.finditer(r"\b\d{3,5}\b", text):
                token = m.group(0).strip()
                if token and token not in seen:
                    seen.add(token)
                    entities.append(token)
                    if len(entities) >= 8:
                        break

        return entities

    def _extract_states(self, prompt_lower: str, ocr_text: str) -> List[str]:
        states: List[str] = []
        text = (ocr_text or "").lower()
        if "selected" in prompt_lower or "selected" in text:
            states.append("selected-control-referenced")
        if "disabled" in prompt_lower or "disabled" in text:
            states.append("disabled-control-referenced")
        if "red" in text or "red indicator" in prompt_lower:
            states.append("red-indicator-referenced")
        if "green" in text or "green indicator" in prompt_lower:
            states.append("green-indicator-referenced")
        return states

    def _synth_confidence(self, image_bytes: bytes) -> int:
        if not image_bytes:
            return 0
        digest = hashlib.sha256(image_bytes).digest()
        # Map first byte to 30..85 so mock output looks plausible but
        # stays bounded.
        return 30 + (digest[0] % 56)

"""Phase 34B — Image Evidence Builder.

Combines OCR + optional ``VisionResult`` into a concise,
structured ``ImageEvidence`` payload that the existing RAG
pipeline can render into prompts and citations.

Design rules:

    * OCR is ALWAYS the baseline. Vision is additive.
    * The evidence payload is intentionally bounded — we never
      concatenate uncontrolled paragraphs into the prompt. Each
      field has a length cap so a large Vision response cannot
      blow the LLM context window.
    * The evidence payload surfaces ``processing_mode`` and
      ``vision_used`` so the LLM prompt can phrase its answer
      with the right grounding language ("The screenshot shows
      X" vs "Based on the retrieved documents, ...").
    * The evidence is rendered into prompt context via a small
      helper so the prompt builder does not need to import the
      Vision subsystem directly.

Citation model
--------------
Image citations stay tied to the underlying document / image. The
evidence payload does NOT introduce a new citation type — the
existing OCR citation pipeline already cites the document the
image came from. When Vision adds visual observations, the LLM
still cites [N] back to the image-derived chunks, and the
"Source: <filename>" line is preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.vision.base import VisionResult
from app.vision.router import (
    ImageProcessingDecision,
    ProcessingMode,
    RoutingSignal,
)

logger = logging.getLogger(__name__)


# Hard caps on evidence field lengths — prevent a hostile provider
# response from inflating the prompt. Values chosen to comfortably
# fit a 4k-token Vision model output.
MAX_DESCRIPTION_CHARS = 800
MAX_FINDING_CHARS = 240
MAX_ENTITY_CHARS = 80
MAX_FINDINGS = 6
MAX_ENTITIES = 12
MAX_STATES = 6
MAX_TAGS = 8


@dataclass
class ImageEvidence:
    """Structured evidence for one image, ready to merge into the RAG prompt.

    Fields are bounded strings / lists. The renderer (``to_prompt_section``)
    formats them into a single prompt section. ``processing_mode`` and
    ``vision_used`` are surfaced verbatim so observability can map an
    answer back to the routing decision that produced it.
    """

    document_id: Optional[int] = None
    image_id: Optional[int] = None
    image_filename: Optional[str] = None
    source_type: Optional[str] = None
    ocr_text: str = ""
    ocr_confidence: Optional[int] = None
    vision_used: bool = False
    vision_description: str = ""
    visual_findings: List[str] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)
    visual_states: List[str] = field(default_factory=list)
    vision_tags: List[str] = field(default_factory=list)
    vision_image_type: str = "unknown"
    vision_confidence: Optional[int] = None
    vision_provider: str = ""
    vision_model: str = ""
    vision_cache_hit: bool = False
    vision_error: Optional[str] = None
    vision_processing_time_ms: int = 0
    processing_mode: ProcessingMode = ProcessingMode.OCR_ONLY
    routing_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "image_id": self.image_id,
            "image_filename": self.image_filename,
            "source_type": self.source_type,
            "ocr_text": self.ocr_text,
            "ocr_confidence": self.ocr_confidence,
            "vision_used": self.vision_used,
            "vision_description": self.vision_description,
            "visual_findings": list(self.visual_findings),
            "detected_entities": list(self.detected_entities),
            "visual_states": list(self.visual_states),
            "vision_tags": list(self.vision_tags),
            "vision_image_type": self.vision_image_type,
            "vision_confidence": self.vision_confidence,
            "vision_provider": self.vision_provider,
            "vision_model": self.vision_model,
            "vision_cache_hit": self.vision_cache_hit,
            "vision_error": self.vision_error,
            "vision_processing_time_ms": self.vision_processing_time_ms,
            "processing_mode": self.processing_mode.value,
            "routing_reasons": list(self.routing_reasons),
        }

    def to_prompt_section(self) -> str:
        """Render the evidence as a single prompt-context section.

        Returns an empty string when neither OCR text nor Vision
        evidence is present (the upstream caller decides whether to
        still render the section header).

        The renderer never embeds secrets / API keys / raw image
        bytes — only structured fields.
        """
        lines: List[str] = []
        label_bits: List[str] = []
        if self.image_filename:
            label_bits.append(f"file: {self.image_filename}")
        if self.source_type:
            label_bits.append(f"source_type: {self.source_type}")
        if self.ocr_confidence is not None:
            label_bits.append(f"ocr_confidence: {self.ocr_confidence}")
        label = ", ".join(label_bits) if label_bits else "image evidence"

        lines.append(f"IMAGE EVIDENCE ({label})")

        if self.ocr_text:
            ocr_block = self.ocr_text.strip()
            if len(ocr_block) > 800:
                ocr_block = ocr_block[:800].rsplit(" ", 1)[0] + "…"
            lines.append("OCR text:")
            lines.append(ocr_block)

        if self.vision_used:
            lines.append(
                f"Vision analysis "
                f"(provider={self.vision_provider}, model={self.vision_model}, "
                f"image_type={self.vision_image_type}, "
                f"confidence={self.vision_confidence}, "
                f"cache_hit={self.vision_cache_hit}):"
            )
            if self.vision_description:
                lines.append(_truncate(self.vision_description, MAX_DESCRIPTION_CHARS))
            if self.visual_findings:
                lines.append("Visual findings:")
                for finding in self.visual_findings[:MAX_FINDINGS]:
                    lines.append(f"- {_truncate(finding, MAX_FINDING_CHARS)}")
            if self.detected_entities:
                entities = ", ".join(
                    _truncate(e, MAX_ENTITY_CHARS)
                    for e in self.detected_entities[:MAX_ENTITIES]
                )
                lines.append(f"Detected entities: {entities}")
            if self.visual_states:
                states = ", ".join(self.visual_states[:MAX_STATES])
                lines.append(f"Visual states: {states}")
            if self.vision_tags:
                tags = ", ".join(self.vision_tags[:MAX_TAGS])
                lines.append(f"Tags: {tags}")
        elif self.vision_error:
            lines.append(
                f"Vision analysis unavailable ({self.vision_error}). "
                "Using OCR-only evidence."
            )

        if self.processing_mode == ProcessingMode.OCR_ONLY:
            lines.append("Routing: OCR-only — Vision was not invoked.")
        elif self.processing_mode == ProcessingMode.OCR_PLUS_VISION:
            lines.append("Routing: OCR + Vision — both sources contribute.")
        elif self.processing_mode == ProcessingMode.VISION_FALLBACK:
            lines.append("Routing: Vision fallback — OCR was insufficient.")

        return "\n".join(lines)


def build_image_evidence(
    *,
    decision: ImageProcessingDecision,
    ocr_text: str,
    ocr_confidence: Optional[int] = None,
    document_id: Optional[int] = None,
    image_id: Optional[int] = None,
    image_filename: Optional[str] = None,
    source_type: Optional[str] = None,
    vision_result: Optional[VisionResult] = None,
    vision_cache_hit: bool = False,
    vision_error: Optional[str] = None,
) -> ImageEvidence:
    """Combine OCR + Vision into a single ``ImageEvidence``.

    The function NEVER raises. Bad inputs collapse to OCR-only
    evidence so the chat pipeline is always answerable.
    """
    evidence = ImageEvidence(
        document_id=document_id,
        image_id=image_id,
        image_filename=image_filename,
        source_type=source_type,
        ocr_text=(ocr_text or "").strip(),
        ocr_confidence=_safe_int(ocr_confidence),
        processing_mode=decision.processing_mode,
        routing_reasons=[s.value for s in decision.trigger_reasons],
        vision_used=False,
    )

    if vision_result is not None and vision_result.is_successful():
        evidence.vision_used = True
        evidence.vision_description = _truncate(
            vision_result.description, MAX_DESCRIPTION_CHARS
        )
        evidence.visual_findings = [
            _truncate(f, MAX_FINDING_CHARS)
            for f in (vision_result.visual_findings or [])[:MAX_FINDINGS]
            if f
        ]
        evidence.detected_entities = [
            _truncate(e, MAX_ENTITY_CHARS)
            for e in (vision_result.detected_entities or [])[:MAX_ENTITIES]
            if e
        ]
        evidence.visual_states = list(
            (vision_result.visual_states or [])[:MAX_STATES]
        )
        evidence.vision_tags = list((vision_result.tags or [])[:MAX_TAGS])
        evidence.vision_image_type = vision_result.image_type or "unknown"
        evidence.vision_confidence = _safe_int(vision_result.confidence)
        evidence.vision_provider = vision_result.provider or ""
        evidence.vision_model = vision_result.model or ""
        evidence.vision_cache_hit = bool(vision_cache_hit)
        evidence.vision_processing_time_ms = int(
            vision_result.processing_time_ms or 0
        )
    elif vision_error:
        evidence.vision_error = str(vision_error)[:400]

    return evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(value: str, max_chars: int) -> str:
    s = (value or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(1, max_chars - 1)].rsplit(" ", 1)[0] + "…"


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except Exception:
        return None
    return max(0, min(100, n))

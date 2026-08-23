"""Phase 34B — Image Intelligence Router.

Decides — deterministically — whether an uploaded image should be
processed as OCR_ONLY, OCR_PLUS_VISION, or VISION_FALLBACK.

The router is INTENTIONALLY cheap:

    * no LLM call to decide whether to call Vision,
    * regex-based intent detection on the user question,
    * threshold comparisons against OCR confidence + OCR text length,
    * short image-type inference from OCR keywords.

The router never decides on its own that Vision should REPLACE OCR.
OCR is always the baseline; Vision is additive evidence.

Cost rule
----------
Vision MUST NOT be called for every image. The router only calls
Vision when at least one of these strong signals is present:

    * the user question is about visual state (layout, status
      indicators, colour-coded controls, charts/graphs/diagrams,
      "what's wrong with this screen"),
    * OCR confidence is below ``VISION_OCR_CONFIDENCE_THRESHOLD``,
    * OCR text length is below ``VISION_MIN_OCR_TEXT_LENGTH`` and
      the question is not a pure identifier-lookup.

The decision is returned as an ``ImageProcessingDecision`` so the
chat pipeline can:
    1. Build evidence with the right combination of OCR + Vision,
    2. Cache the Vision result on the DocumentImage row,
    3. Surface ``processing_mode`` + ``trigger_reason`` to
       observability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


class ProcessingMode(str, Enum):
    """Three explicit processing modes for image questions.

    Values are lowercase strings so they survive JSON serialisation
    into LangSmith traces and admin config endpoints without extra
    mapping.
    """

    OCR_ONLY = "ocr_only"
    OCR_PLUS_VISION = "ocr_plus_vision"
    VISION_FALLBACK = "vision_fallback"


class RoutingSignal(str, Enum):
    """Why the router chose the mode it did.

    Used in observability + tests so failures can be diagnosed
    without re-running the request.
    """

    OCR_SUFFICIENT = "ocr_sufficient"
    VISUAL_INTENT = "visual_intent"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    LOW_OCR_TEXT_LENGTH = "low_ocr_text_length"
    OCR_ERROR = "ocr_error"
    VISION_DISABLED = "vision_disabled"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


# ---------------------------------------------------------------------------
# Visual intent detection
# ---------------------------------------------------------------------------


# Conservative intent phrases. We deliberately include BOTH the
# spec examples AND common variations so the regex is robust
# without becoming a giant enumeration of every possible sentence.
#
# IMPORTANT: identifier-lookup questions ("What error code is
# shown?", "What is the receiver name?", "What number is
# displayed?") are intentionally NOT in this list. They are pure
# OCR questions; Vision should not be called for them.
_VISUAL_INTENT_PATTERNS = [
    # Direct "what is happening on screen" phrasings.
    r"\bwhat(?:\s+visually)?\s+(?:is|does)\s+(?:wrong|happening|going\s+on)\b",
    r"\bwhat(?:'s|\s+is)\s+wrong\s+with\s+(?:this|that|the)\s+(?:screen|page|view|window|dashboard|screenshot|image)\b",
    r"\bwhat\s+looks?\s+wrong\b",
    r"\bwhat\s+(?:do\s+you\s+see|are\s+you\s+seeing)\b",
    r"\bdescribe\s+(?:this|that|the)?\s*(?:image|screenshot|screen|page|view|window|dashboard|diagram|chart|graph|picture)\b",
    r"\bexplain\s+(?:this|that|the)?\s*(?:screenshot|screen|image|diagram|chart|graph|architecture|topology|dashboard|table|figure)\b",
    # Spatial / selection / control state.
    r"\bwhich\s+(?:button|icon|server|row|field|item|control|menu|tab|column|server|node|box|line)\b",
    r"\bwhat\s+is\s+(?:selected|highlighted|disabled|active|inactive|focused|clicked|enabled|red|green|yellow|amber)\b",
    r"\bwhere\s+is\s+(?:the\s+)?[a-z0-9 _-]{2,40}\b",
    # Indicators / status.
    r"\b(?:red|green|yellow|amber|orange|blue)\s+indicator\b",
    r"\bstatus\s+indicator\b",
    r"\bwarning\s+(?:indicator|state|message|icon)\b",
    # Visually-shaped content.
    r"\b(?:chart|graph|diagram|dashboard|architecture|topology|layout)\b",
    r"\b(?:trend|spike|drop|anomaly|outlier|peak|valley)\b",
    r"\b(?:visual|visually|graphically|on\s+screen|in\s+the\s+(?:image|screenshot|picture))\b",
]

# Phrases that EXPLICITLY ask for visual content but the spec
# treats as identifier-style — we exclude them from the broad
# pattern. The router must NOT call Vision for "What error code is
# shown?".
_IDENTIFIER_LOOKUP_PATTERNS = [
    r"\bwhat\s+(?:error\s+code|error\s+number|code|number|status\s+code|status\s+number)\s+is\s+shown\b",
    r"\bwhat\s+(?:text|message|words?)\s+(?:does|do|is)\s+(?:this|the)\s+(?:screenshot|image|page)\s+(?:contain|show|display|say)\b",
    r"\bwhat\s+is\s+the\s+(?:receiver|sender|recipient|customer|user|name|address|phone|email|amount|value)\b",
    r"\bwhat\s+number\s+is\s+displayed\b",
    r"\bread\s+(?:out|back)\s+the\s+(?:text|error|message)\b",
]

_COMPILED_VISUAL_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _VISUAL_INTENT_PATTERNS
]
_COMPILED_IDENTIFIER_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _IDENTIFIER_LOOKUP_PATTERNS
]


def detect_visual_intent(question: str) -> bool:
    """Return True if the question expresses visual intent.

    The detector is conservative:

        1. Identifier-lookup questions ("What error code is shown?")
           return False even if they incidentally match a visual
           phrase. Vision should not be called for identifier
           lookups; OCR handles them.
        2. Otherwise, the question must match at least one of the
           curated visual-intent patterns.
    """
    if not question or not question.strip():
        return False

    text = question.strip()

    # Identifier-lookups always win — OCR handles them.
    for pattern in _COMPILED_IDENTIFIER_PATTERNS:
        if pattern.search(text):
            return False

    for pattern in _COMPILED_VISUAL_PATTERNS:
        if pattern.search(text):
            return True

    return False


# ---------------------------------------------------------------------------
# Image type inference (cheap, deterministic)
# ---------------------------------------------------------------------------


_IMAGE_TYPE_KEYWORDS: dict = {
    "dashboard": ("dashboard", "overview panel", "summary panel"),
    "chart": ("chart", "line chart", "bar chart", "pie chart", "scatter"),
    "graph": ("graph", "line graph", "trend line"),
    "diagram": ("diagram", "architecture", "topology", "flow", "sequence", "uml"),
    "table": ("table", "grid", "spreadsheet"),
    "log": ("log", "stacktrace", "trace", "exception"),
    "ui": ("button", "icon", "menu", "tab", "dialog", "modal", "toolbar"),
}


def infer_image_type_hint(ocr_text: str) -> str:
    """Best-effort image-type inference from OCR text.

    Conservative: returns ``"unknown"`` when OCR text is empty or
    contains no obvious category keywords. The router never uses
    this as a hard signal — it only uses it to enrich the
    ``ImageProcessingDecision`` for observability.
    """
    if not ocr_text:
        return "unknown"
    text = ocr_text.lower()
    # Iterate in priority order — first hit wins.
    for category, keywords in _IMAGE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    return "unknown"


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass
class ImageProcessingDecision:
    """Result of routing an image question.

    The chat pipeline treats ``processing_mode`` as authoritative.
    ``trigger_reason`` lists which signals fired so observability +
    tests can assert on routing behaviour. ``vision_required`` is a
    convenience flag — it is True for any mode that would call the
    Vision provider.
    """

    processing_mode: ProcessingMode = ProcessingMode.OCR_ONLY
    vision_required: bool = False
    trigger_reasons: List[RoutingSignal] = field(default_factory=list)
    ocr_confidence: Optional[int] = None
    ocr_text_length: int = 0
    visual_intent_detected: bool = False
    image_type_hint: str = "unknown"

    def has_signal(self, signal: RoutingSignal) -> bool:
        return signal in self.trigger_reasons

    def to_dict(self) -> dict:
        return {
            "processing_mode": self.processing_mode.value,
            "vision_required": self.vision_required,
            "trigger_reasons": [s.value for s in self.trigger_reasons],
            "ocr_confidence": self.ocr_confidence,
            "ocr_text_length": self.ocr_text_length,
            "visual_intent_detected": self.visual_intent_detected,
            "image_type_hint": self.image_type_hint,
        }


def decide_processing_mode(
    question: str,
    *,
    ocr_text: str = "",
    ocr_confidence: Optional[int] = None,
    ocr_status: str = "success",
    ocr_error: Optional[str] = None,
    vision_enabled: bool = True,
    vision_router_enabled: bool = True,
    vision_ocr_confidence_threshold: int = 55,
    vision_min_ocr_text_length: int = 25,
) -> ImageProcessingDecision:
    """Decide how to process the image attached to ``question``.

    Args:
        question: The user's chat question (already cleaned).
        ocr_text: OCR-extracted text for the image (may be empty).
        ocr_confidence: 0..100 OCR confidence, or None.
        ocr_status: One of ``success | failed | disabled | empty |
            low_confidence | pending``.
        ocr_error: Optional error string when OCR failed.
        vision_enabled: ``settings.VISION_ENABLED`` value.
        vision_router_enabled: ``settings.VISION_ROUTER_ENABLED``.
        vision_ocr_confidence_threshold: ``settings.VISION_OCR_CONFIDENCE_THRESHOLD``.
        vision_min_ocr_text_length: ``settings.VISION_MIN_OCR_TEXT_LENGTH``.

    Returns:
        ``ImageProcessingDecision``. The router never raises — bad
        inputs collapse to ``OCR_ONLY`` so the chat pipeline is
        always answerable.
    """
    text_len = len((ocr_text or "").strip())
    visual_intent = detect_visual_intent(question or "")
    image_type_hint = infer_image_type_hint(ocr_text or "")

    decision = ImageProcessingDecision(
        ocr_confidence=_safe_int(ocr_confidence),
        ocr_text_length=text_len,
        visual_intent_detected=visual_intent,
        image_type_hint=image_type_hint,
    )

    # --- Tier 0: Vision fully disabled -------------------------------
    # The spec requires VISION_ENABLED=false to restore the exact
    # existing OCR-only architecture. We must not record Vision as
    # "required" even if every other signal fires.
    if not vision_enabled or not vision_router_enabled:
        decision.processing_mode = ProcessingMode.OCR_ONLY
        decision.vision_required = False
        decision.trigger_reasons.append(
            RoutingSignal.VISION_DISABLED
            if not vision_enabled
            else RoutingSignal.OCR_SUFFICIENT
        )
        return decision

    # --- Tier 1: OCR did not run or errored ---------------------------
    # Without OCR we cannot answer text questions; Vision is the
    # only source of evidence.
    if ocr_status in ("failed", "disabled", "pending"):
        decision.processing_mode = ProcessingMode.VISION_FALLBACK
        decision.vision_required = True
        decision.trigger_reasons.append(
            RoutingSignal.OCR_ERROR if ocr_status == "failed"
            else RoutingSignal.LOW_OCR_TEXT_LENGTH
        )
        return decision

    # --- Tier 2: OCR produced almost nothing useful ------------------
    # Diagrams and visually-meaningful screenshots often OCR to
    # near-empty strings. Vision is a fallback.
    if text_len < max(1, int(vision_min_ocr_text_length)):
        # Identifier-only lookups ("What error code is shown?") do
        # not benefit from Vision — if the answer is NOT in OCR it
        # is not visually observable either. We keep OCR_ONLY so the
        # user sees a grounded insufficient-information response.
        if not visual_intent:
            decision.processing_mode = ProcessingMode.OCR_ONLY
            decision.vision_required = False
            decision.trigger_reasons.append(RoutingSignal.OCR_SUFFICIENT)
            return decision
        decision.processing_mode = ProcessingMode.VISION_FALLBACK
        decision.vision_required = True
        decision.trigger_reasons.append(RoutingSignal.LOW_OCR_TEXT_LENGTH)
        return decision

    # --- Tier 3: Visual intent detected ------------------------------
    # OCR alone cannot describe visual state, indicators, charts,
    # graphs, or diagram relationships. Vision is required.
    if visual_intent:
        decision.processing_mode = ProcessingMode.OCR_PLUS_VISION
        decision.vision_required = True
        decision.trigger_reasons.append(RoutingSignal.VISUAL_INTENT)
        return decision

    # --- Tier 4: Low OCR confidence ----------------------------------
    # OCR produced text but it may be noisy. Vision can complement
    # it for visual grounding questions. Identifier lookups remain
    # OCR_ONLY so we do not pay for Vision unnecessarily.
    if (
        decision.ocr_confidence is not None
        and decision.ocr_confidence < max(0, int(vision_ocr_confidence_threshold))
    ):
        decision.processing_mode = ProcessingMode.OCR_PLUS_VISION
        decision.vision_required = True
        decision.trigger_reasons.append(RoutingSignal.LOW_OCR_CONFIDENCE)
        return decision

    # --- Default: OCR is sufficient ----------------------------------
    decision.processing_mode = ProcessingMode.OCR_ONLY
    decision.vision_required = False
    decision.trigger_reasons.append(RoutingSignal.OCR_SUFFICIENT)
    return decision


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except Exception:
        return None
    return max(0, min(100, n))

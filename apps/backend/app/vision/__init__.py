"""Phase 34B — Vision subsystem package.

Exposes the Image Intelligence Router, Evidence Builder, persistence
helpers, and prompt construction used by the chat pipeline.

Architectural rules:

    * OCR remains the default evidence source. Vision is ADDITIONAL
      visual understanding capability on top of OCR.
    * The router is deterministic — it MUST NOT call an LLM to
      decide whether to call Vision. That would defeat the cost
      and latency goals of Phase 34B.
    * Vision is supplementary. If Vision fails, OCR evidence is
      always retained.
"""

from app.vision.router import (
    ImageProcessingDecision,
    ProcessingMode,
    RoutingSignal,
    decide_processing_mode,
    detect_visual_intent,
)

__all__ = [
    "ImageProcessingDecision",
    "ProcessingMode",
    "RoutingSignal",
    "decide_processing_mode",
    "detect_visual_intent",
]

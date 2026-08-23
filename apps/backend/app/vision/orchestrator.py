"""Phase 34B — Vision Orchestrator.

The single entry point for "process an image". It chains:

    1. Image resolution (find the right DocumentImage row),
    2. Router decision (OCR_ONLY / OCR_PLUS_VISION / VISION_FALLBACK),
    3. Cache lookup (reuse existing successful Vision rows),
    4. Provider call (only when justified + cache miss + provider available),
    5. Persistence (write successful Vision results back),
    6. Evidence Builder (combine OCR + optional Vision).

The orchestrator NEVER raises. Failures degrade gracefully to
OCR-only evidence so the chat pipeline remains answerable.

RBAC: the orchestrator trusts the caller. The chat endpoint
already gates image_context.document_id / image_id against the
authenticated user. The orchestrator does NOT re-check RBAC —
doing so would duplicate logic and risk drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.vision.base import (
    VISION_SCHEMA_VERSION,
    VisionProvider,
    VisionProviderError,
    VisionResult,
)
from app.services.vision.factory import get_vision_provider
from app.vision.evidence import ImageEvidence, build_image_evidence
from app.vision.persistence import (
    PersistenceResult,
    lookup_cached_vision,
    persist_vision_failure,
    persist_vision_result,
)
from app.vision.router import (
    ImageProcessingDecision,
    ProcessingMode,
    RoutingSignal,
    decide_processing_mode,
)

logger = logging.getLogger(__name__)


# Cap image bytes at the orchestrator level — guards against a
# hostile DocumentImage row claiming a huge storage_key.
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass
class OrchestratorOutcome:
    """Result of running the orchestrator for one image + question.

    The chat pipeline reads ``evidence`` to render prompt context
    and ``metadata`` to surface processing_mode / vision_called /
    vision_cache_hit / vision_latency_ms into the response /
    LangSmith trace.
    """

    evidence: ImageEvidence
    decision: ImageProcessingDecision
    vision_called: bool = False
    vision_cache_hit: bool = False
    vision_provider: str = ""
    vision_model: str = ""
    vision_latency_ms: int = 0
    vision_error: Optional[str] = None
    image_bytes_source: str = "missing"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.metadata or {})
        out.update({
            "image_processing_mode": self.decision.processing_mode.value,
            "vision_required": self.decision.vision_required,
            "vision_called": self.vision_called,
            "vision_cache_hit": self.vision_cache_hit,
            "vision_provider": self.vision_provider,
            "vision_model": self.vision_model,
            "vision_latency_ms": self.vision_latency_ms,
            "vision_error": self.vision_error,
            "image_bytes_source": self.image_bytes_source,
            "ocr_confidence": self.decision.ocr_confidence,
            "ocr_text_length": self.decision.ocr_text_length,
            "routing_reasons": [s.value for s in self.decision.trigger_reasons],
            "image_type_hint": self.decision.image_type_hint,
        })
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_image_for_question(
    db: Optional[Session],
    *,
    question: str,
    document_id: Optional[int] = None,
    image_id: Optional[int] = None,
    image_filename: Optional[str] = None,
    source_type: Optional[str] = None,
    ocr_text: str = "",
    ocr_confidence: Optional[int] = None,
    ocr_status: str = "success",
    ocr_error: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    context_hint: str = "",
    max_image_bytes: Optional[int] = None,
) -> OrchestratorOutcome:
    """Run the full Vision pipeline for one image + question.

    Args:
        db: SQLAlchemy session (required when ``image_id`` is set
            so the orchestrator can look up the DocumentImage row
            and persist Vision results).
        question: User question (used for intent detection + as the
            provider prompt).
        document_id / image_id / image_filename / source_type:
            Provenance fields. ``image_id`` is preferred — the
            orchestrator can load OCR + cache + provenance from
            the row. With only ``document_id`` the orchestrator
            will still cache per-image if the row is found.
        ocr_text / ocr_confidence / ocr_status / ocr_error:
            Pre-computed OCR signals (the chat endpoint already has
            these). When ``image_id`` is provided the orchestrator
            will load them from the row if not passed in.
        image_bytes: Raw image bytes for the provider. Optional —
            only required when Vision will actually be called.
        image_mime_type: MIME for the bytes (defaults to png).
        context_hint: Free-text context (e.g. "dashboard screenshot")
            forwarded to the provider as part of the prompt.
        max_image_bytes: Hard cap on bytes forwarded to the
            provider.

    Returns:
        ``OrchestratorOutcome`` (never raises).
    """
    max_bytes = int(
        max_image_bytes
        if max_image_bytes is not None
        else getattr(settings, "VISION_MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES)
    )

    # --- Step 1: resolve + load row data (best effort) --------------
    row_image_bytes: Optional[bytes] = None
    row_mime: Optional[str] = None
    actual_document_id = document_id
    actual_image_id = image_id

    if db is not None and image_id is not None:
        try:
            from app.models.document import DocumentImage

            row = (
                db.query(DocumentImage)
                .filter(DocumentImage.id == int(image_id))
                .first()
            )
            if row is not None:
                actual_document_id = actual_document_id or row.document_id
                actual_image_id = actual_image_id or row.id
                image_filename = image_filename or row.original_filename
                source_type = source_type or row.source_type
                if not ocr_text and row.ocr_status == "success":
                    # OCR text is in the index / chunks; we don't
                    # store raw OCR on the row, but the chat
                    # pipeline feeds it in via ocr_text. If not
                    # provided we leave it empty.
                    pass
                if ocr_confidence is None:
                    ocr_confidence = row.ocr_confidence
                if not ocr_status:
                    ocr_status = row.ocr_status or "success"
                if not ocr_error:
                    ocr_error = row.ocr_error
        except Exception as exc:
            logger.debug("orchestrator: image row load failed: %s", exc)

    # --- Step 2: router decision ------------------------------------
    decision = decide_processing_mode(
        question or "",
        ocr_text=ocr_text or "",
        ocr_confidence=ocr_confidence,
        ocr_status=ocr_status or "success",
        ocr_error=ocr_error,
        vision_enabled=getattr(settings, "VISION_ENABLED", False),
        vision_router_enabled=getattr(settings, "VISION_ROUTER_ENABLED", True),
        vision_ocr_confidence_threshold=getattr(
            settings, "VISION_OCR_CONFIDENCE_THRESHOLD", 55
        ),
        vision_min_ocr_text_length=getattr(
            settings, "VISION_MIN_OCR_TEXT_LENGTH", 25
        ),
    )

    outcome = OrchestratorOutcome(
        evidence=None,  # set below
        decision=decision,
    )

    # --- Step 3: cache lookup (only if we will call Vision) ---------
    cached: Optional[PersistenceResult] = None
    if (
        decision.vision_required
        and getattr(settings, "VISION_ENABLED", False)
        and db is not None
        and actual_image_id is not None
    ):
        try:
            provider = get_vision_provider()
            provider_name = (
                getattr(provider, "name", "") if provider is not None else "mock"
            )
            model_name = (
                getattr(provider, "model", "mock-v1")
                if provider is not None
                else "mock-v1"
            )
            cached = lookup_cached_vision(
                db,
                document_image_id=int(actual_image_id),
                provider=provider_name,
                model=model_name,
                schema_version=getattr(
                    settings, "VISION_CACHE_SCHEMA_VERSION",
                    VISION_SCHEMA_VERSION
                ),
            )
        except Exception as exc:
            cached = PersistenceResult(
                cache_hit=False, error=f"cache_lookup_exception: {str(exc)[:200]}"
            )

    # --- Step 4: provider call -------------------------------------
    vision_result: Optional[VisionResult] = None
    vision_error: Optional[str] = None
    vision_latency = 0

    if cached is not None and cached.cache_hit and cached.vision_result is not None:
        vision_result = cached.vision_result
        outcome.vision_cache_hit = True
        outcome.vision_called = False
        outcome.vision_provider = vision_result.provider
        outcome.vision_model = vision_result.model
        outcome.vision_latency_ms = 0
    elif decision.vision_required and getattr(settings, "VISION_ENABLED", False):
        vision_result, vision_error, vision_latency = _call_provider(
            image_bytes=image_bytes,
            row_image_bytes=row_image_bytes,
            row_mime=row_mime,
            image_mime_type=image_mime_type,
            question=question,
            ocr_text=ocr_text,
            context_hint=context_hint,
            max_bytes=max_bytes,
        )
        outcome.vision_called = vision_result is not None
        outcome.vision_latency_ms = vision_latency
        if vision_result is not None:
            outcome.vision_provider = vision_result.provider
            outcome.vision_model = vision_result.model
            # Persist successful result.
            if db is not None and actual_image_id is not None:
                try:
                    persist_vision_result(
                        db,
                        document_image_id=int(actual_image_id),
                        vision_result=vision_result,
                    )
                except Exception as exc:
                    logger.debug(
                        "orchestrator: persist_vision_result failed: %s", exc
                    )
        elif vision_error is not None:
            outcome.vision_error = vision_error[:300]
            # Record the failure for observability.
            if db is not None and actual_image_id is not None:
                try:
                    persist_vision_failure(
                        db,
                        document_image_id=int(actual_image_id),
                        provider=outcome.vision_provider or "",
                        model=outcome.vision_model or "",
                        error=vision_error[:400],
                    )
                except Exception as exc:
                    logger.debug(
                        "orchestrator: persist_vision_failure failed: %s", exc
                    )

    # --- Step 5: evidence ------------------------------------------
    outcome.evidence = build_image_evidence(
        decision=decision,
        ocr_text=ocr_text or "",
        ocr_confidence=ocr_confidence,
        document_id=actual_document_id,
        image_id=actual_image_id,
        image_filename=image_filename,
        source_type=source_type,
        vision_result=vision_result,
        vision_cache_hit=outcome.vision_cache_hit,
        vision_error=vision_error,
    )

    # Record bytes source so observability can confirm whether the
    # provider actually received an image.
    if image_bytes:
        outcome.image_bytes_source = "caller"
    elif row_image_bytes:
        outcome.image_bytes_source = "db"

    return outcome


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _call_provider(
    *,
    image_bytes: Optional[bytes],
    row_image_bytes: Optional[bytes],
    row_mime: Optional[str],
    image_mime_type: Optional[str],
    question: str,
    ocr_text: str,
    context_hint: str,
    max_bytes: int,
) -> tuple[Optional[VisionResult], Optional[str], int]:
    """Call the Vision provider if justified + available.

    Returns (result, error_message, latency_ms). Exactly one of
    result / error_message may be non-None. When neither is set,
    Vision was simply not invoked (no provider configured, etc.).
    """
    try:
        provider = get_vision_provider()
    except VisionProviderError as exc:
        return None, f"provider_unavailable: {str(exc)[:200]}", 0
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"provider_factory_error: {str(exc)[:200]}", 0

    if provider is None:
        # VISION_ENABLED is false — caller should not have routed
        # here, but degrade safely.
        return None, "vision_disabled_at_runtime", 0

    payload = image_bytes or row_image_bytes
    if not payload:
        return None, "no_image_bytes_available", 0

    if len(payload) > max_bytes:
        return (
            None,
            f"image_too_large_for_provider ({len(payload)} > {max_bytes})",
            0,
        )

    mime = (image_mime_type or row_mime or "image/png").strip()

    prompt = _build_provider_prompt(question, context_hint)

    try:
        result = provider.analyze_image(
            image_bytes=payload,
            mime_type=mime,
            prompt=prompt,
            ocr_text=ocr_text,
            context_hint=context_hint,
            max_tokens=600,
        )
    except VisionProviderError as exc:
        return None, f"provider_error: {str(exc)[:200]}", 0
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"provider_unexpected_error: {str(exc)[:200]}", 0

    if not result.is_successful():
        return None, "provider_returned_empty_result", int(
            result.processing_time_ms or 0
        )

    return result, None, int(result.processing_time_ms or 0)


def _build_provider_prompt(question: str, context_hint: str) -> str:
    q = (question or "").strip()
    if context_hint:
        return f"{context_hint}\n\nQuestion: {q or 'Describe the image.'}"
    return q or "Describe the image."

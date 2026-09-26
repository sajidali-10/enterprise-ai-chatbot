"""Phase 34D — Advanced Visual Reasoning Orchestrator.

Single entry point for "run advanced visual reasoning on an image".
Chains:

    1. Kill switch + router on/off checks.
    2. Task classification (deterministic regex, no LLM).
    3. Image count + size cap validation.
    4. RBAC gate #1 — ``can_access_document`` BEFORE MinIO fetch.
    5. Authorized image-bytes fetch via :mod:`_image_fetch`.
    6. Cache lookup via :mod:`cache` (Redis first, optional in-process LRU).
    7. RBAC gate #2 — re-authorize BEFORE provider call (defense in depth,
       especially for historical images that the chat endpoint did not
       directly authorize).
    8. Provider call — Phase 34B ``VisionProvider.analyze_image`` with
       the new optional ``task`` + ``images`` parameters.
    9. Parse result into a bounded :class:`VisualReasoningResult`.
   10. Cache the result (Redis + optional in-process LRU).
   11. Return an :class:`AdvancedOrchestratorOutcome`.

The orchestrator NEVER raises. Failures degrade to a safe empty
result with explicit ``error`` and ``disabled`` flags so the chat
pipeline can fall back to OCR-only / Phase 34B evidence.

RBAC semantics:

* A missing image row, an unauthorized image, an oversized image,
  or a MinIO failure all collapse to ``outcome.skipped_reason``
  (one of ``unauthorized``, ``not_found``, ``too_large``,
  ``fetch_failed``). The orchestrator does NOT distinguish those
  states externally — an unauthorized caller must not be able to
  learn whether the image exists.
* Both RBAC gates are independent and both MUST pass. Gate #2
  protects against the case where the document became inaccessible
  between Gate #1 and the provider call (e.g. RBAC change mid-flight).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.security.auth import AuthContext
from app.security.permissions import can_access_document

from app.services.advanced_vision import cache as adv_cache
from app.services.advanced_vision._image_fetch import (
    AuthorizedImage,
    fetch_authorized_image_bytes,
)
from app.services.advanced_vision.base import (
    AdvancedVisualTask,
    VisualReasoningResult,
    build_advanced_vision_cache_key,
    empty_result,
    image_content_hash,
    safe_parse_provider_payload,
)
from app.services.advanced_vision.router import (
    TaskClassification,
    classify_visual_task,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


@dataclass
class AdvancedOrchestratorOutcome:
    """Result of running the advanced orchestrator.

    The chat pipeline reads ``result`` to render prompt context and
    ``metadata`` (via ``to_dict``) to surface
    ``advanced_vision_*`` fields into the response / LangSmith trace.

    ``ran`` distinguishes "advanced vision was not invoked because
    the router decided it was not needed" (kill switch / no task
    signal) from "advanced vision ran but the provider failed /
    returned nothing" (so the LLM still sees a safe empty result).

    ``skipped_reason`` is ``None`` when ``ran`` is True. When ``ran``
    is False, ``skipped_reason`` explains why (kill_switch,
    router_disabled, no_task_signal, identifier_lookup,
    product_meaning, too_many_images, unauthorized, not_found,
    too_large, fetch_failed, provider_unavailable, no_provider,
    cache_disabled_no_provider, parse_failure, disabled, etc.).
    """

    ran: bool = False
    result: Optional[VisualReasoningResult] = None
    classification: Optional[TaskClassification] = None
    skipped_reason: Optional[str] = None
    error: Optional[str] = None
    cache_hit: bool = False
    cache_source: str = ""
    provider_called: bool = False
    provider_name: str = ""
    provider_model: str = ""
    latency_ms: int = 0
    image_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.metadata or {})
        out.update({
            "advanced_vision_ran": bool(self.ran),
            "advanced_vision_task_type": (
                (self.result.task_type if self.result is not None else None)
                or (self.classification.task_type if self.classification else None)
                or AdvancedVisualTask.GENERAL_VISUAL.value
            ),
            "advanced_vision_called": bool(self.provider_called),
            "advanced_vision_cache_hit": bool(self.cache_hit),
            "advanced_vision_cache_source": self.cache_source or "",
            "advanced_vision_provider": self.provider_name or "",
            "advanced_vision_model": self.provider_model or "",
            "advanced_vision_latency_ms": int(self.latency_ms or 0),
            "advanced_vision_image_count": len(self.image_ids or []),
            "advanced_vision_image_ids": list(self.image_ids or []),
            "advanced_vision_skipped_reason": self.skipped_reason or "",
            "advanced_vision_error": self.error or "",
            "advanced_vision_summary": (
                self.result.summary if self.result is not None else ""
            ),
            "advanced_vision_confidence": (
                self.result.confidence if self.result is not None else None
            ),
        })
        return out


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------


def _is_advanced_vision_enabled() -> bool:
    if not bool(getattr(settings, "ADVANCED_VISION_ENABLED", False)):
        return False
    if not bool(getattr(settings, "ADVANCED_VISION_ROUTER_ENABLED", True)):
        return False
    return True


def _effective_max_image_bytes() -> int:
    cap = int(getattr(settings, "ADVANCED_VISION_MAX_IMAGE_BYTES", 0) or 0)
    if cap <= 0:
        cap = int(getattr(settings, "VISION_MAX_IMAGE_BYTES", 8 * 1024 * 1024) or 0)
    if cap <= 0:
        cap = 8 * 1024 * 1024
    return cap


def _effective_timeout() -> Optional[float]:
    timeout = getattr(settings, "ADVANCED_VISION_TIMEOUT_SECONDS", 0) or 0
    if timeout and float(timeout) > 0:
        return float(timeout)
    fallback = getattr(settings, "VISION_TIMEOUT_SECONDS", 30) or 30
    return float(fallback) if float(fallback) > 0 else None


def _max_images_per_request() -> int:
    return int(getattr(settings, "ADVANCED_VISION_MAX_IMAGES_PER_REQUEST", 2) or 2)


# ---------------------------------------------------------------------------
# Provider resolution (reuses Phase 34B factory)
# ---------------------------------------------------------------------------


def _resolve_provider():
    """Return a Phase 34B VisionProvider instance, or None.

    Centralised so the orchestrator can be tested by monkeypatching
    ``get_vision_provider``. The Phase 34B factory honours
    ``VISION_ENABLED`` itself — when Vision is disabled, the
    factory returns None and the orchestrator degrades to skip.
    """
    try:
        from app.services.vision.factory import get_vision_provider
        return get_vision_provider()
    except Exception as exc:
        logger.debug("advanced_vision: provider factory raised: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point: single image
# ---------------------------------------------------------------------------


def run_advanced_visual_reasoning(
    db: Any,
    *,
    auth: Optional[AuthContext],
    question: str,
    image_id: int,
    ocr_text: str = "",
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    forced_task: Optional[str] = None,
    context_hint: str = "",
    provider: Optional[Any] = None,
) -> AdvancedOrchestratorOutcome:
    """Run advanced visual reasoning on a single authorized image.

    Args:
        db: SQLAlchemy session.
        auth: AuthContext. ``None`` is treated as unauthorized.
        question: The user question (used for task classification).
        image_id: ``DocumentImage.id``. RBAC is enforced on this id.
        ocr_text: Pre-extracted OCR body. Passed to the provider as
            ground truth for visible text.
        image_bytes: Optional pre-fetched bytes. When supplied, the
            orchestrator skips the MinIO fetch but STILL enforces
            the RBAC gate via :func:`_gate_authorize`.
        image_mime_type: MIME type for the supplied bytes. Defaults
            to ``image/png`` when unknown.
        forced_task: Optional task override. Used by tests and by
            callers that already classified the question.
        context_hint: Free-text context forwarded to the provider.
        provider: Optional Phase 34B ``VisionProvider`` instance.
            When supplied, it is used directly (dependency injection)
            so tests and the live application resolve the SAME seam
            and a forced provider failure is observable through the
            real chat path. When ``None`` (default) the orchestrator
            falls back to :func:`_resolve_provider`, which reuses the
            Phase 34B factory (``app.services.vision.factory
            .get_vision_provider``).

    Returns:
        :class:`AdvancedOrchestratorOutcome`. The function never
        raises. When advanced vision is disabled or not justified,
        ``ran=False`` and ``result`` may be ``None``. When
        advanced vision ran but failed, ``ran=True`` and
        ``result`` is a safe empty result.
    """
    outcome = AdvancedOrchestratorOutcome()

    if not _is_advanced_vision_enabled():
        outcome.skipped_reason = "kill_switch"
        return outcome

    # --- Task classification ----------------------------------------------
    classification = (
        _forced_classification(forced_task)
        if forced_task
        else classify_visual_task(question or "", image_count=1)
    )
    outcome.classification = classification

    if not classification.should_run_advanced_vision:
        if classification.is_identifier_lookup:
            outcome.skipped_reason = "identifier_lookup"
        elif classification.is_product_meaning:
            outcome.skipped_reason = "product_meaning"
        else:
            outcome.skipped_reason = "no_task_signal"
        return outcome

    # --- RBAC gate #1 (BEFORE MinIO fetch / bytes use) ------------------
    if image_bytes is None:
        authorized = _gate_authorize(db, image_id, auth)
        if authorized is None:
            outcome.skipped_reason = "unauthorized"
            return outcome
        authorized_image: AuthorizedImage = authorized
    else:
        # Even when bytes are supplied, the image_id MUST still be
        # authorized — defense in depth. The caller might have an
        # in-memory cache from an earlier authorized request, but we
        # cannot trust that the same document_id is still accessible.
        authorized_image = _gate_authorize(db, image_id, auth)
        if authorized_image is None:
            outcome.skipped_reason = "unauthorized"
            return outcome
        # Override bytes with the caller's payload when supplied.
        if authorized_image.bytes_ is None:
            authorized_image.bytes_ = bytes(image_bytes)
        if not authorized_image.mime_type and image_mime_type:
            authorized_image.mime_type = str(image_mime_type)

    bytes_ = authorized_image.bytes_ or b""
    if not bytes_:
        outcome.skipped_reason = "fetch_failed"
        outcome.error = "image_bytes_empty"
        return outcome

    if len(bytes_) > _effective_max_image_bytes():
        outcome.skipped_reason = "too_large"
        outcome.error = f"image_too_large_for_provider ({len(bytes_)} > {_effective_max_image_bytes()})"
        return outcome

    outcome.image_ids = [int(authorized_image.image_id)]
    # NOTE: outcome.ran is set AFTER RBAC gate #2 + provider-availability
    # below. The tests assert ran=False on every failure path (gate #2,
    # no_provider, oversized, fetch_failed, unauthorized_pre_provider).

    # --- Provider resolution (for a consistent cache key) --------------
    # Resolve the provider name BEFORE the cache lookup so the READ and
    # WRITE cache keys use the SAME provider identifier. Prior to this
    # the read used ``settings.VISION_PROVIDER`` (= 'mock') while the
    # write used ``outcome.provider_name`` (= the provider's actual
    # ``.name``, e.g. 'phase34d-live'), so the two keys never matched and
    # the cache could never return a hit.
    #
    # DI seam (Phase 34D): an injected ``provider`` is used verbatim (so
    # tests and the live app resolve the SAME provider abstraction);
    # otherwise we fall back to the Phase 34B factory, which already
    # reuses ``app.services.vision.factory.get_vision_provider``.
    provider = provider if provider is not None else _resolve_provider()
    if provider is not None:
        outcome.provider_name = str(getattr(provider, "name", ""))
        outcome.provider_model = str(getattr(provider, "model", ""))

    # --- Cache lookup -----------------------------------------------------
    cache_key = _build_cache_key(
        provider_name=outcome.provider_name or None,
        task_type=classification.task_type,
        image_hashes=[image_content_hash(bytes_)],
    )
    read = adv_cache.get_cached_result(
        cache_key,
        expect_schema_version=int(
            getattr(settings, "ADVANCED_VISION_SCHEMA_VERSION", 1)
        ),
    )
    if read.hit and read.result is not None and read.result.is_successful():
        outcome.cache_hit = True
        outcome.cache_source = read.source or ""
        read.result.image_ids = list(outcome.image_ids)
        outcome.result = read.result
        outcome.latency_ms = 0
        outcome.ran = True  # cache hit produced a usable result
        return outcome

    # --- RBAC gate #2 (BEFORE provider call) ------------------------------
    # Re-check immediately before sending bytes to the provider. This
    # is the defense-in-depth gate: the document might have become
    # inaccessible between gate #1 and now.
    if not _reauthorize(db, authorized_image.document_id, auth):
        outcome.skipped_reason = "unauthorized_pre_provider"
        return outcome

    # All early gates passed (RBAC #1, image fetch, size cap, RBAC #2).
    # We are now in the "execute the orchestrator" phase. Set ran=True
    # so every downstream failure path (no_provider, provider_timeout,
    # provider_unavailable, parse_failure) still records ran=True —
    # we attempted to run, we just couldn't produce a usable result.
    outcome.ran = True

    # --- Provider call ----------------------------------------------------
    provider = _resolve_provider()
    if provider is None:
        outcome.skipped_reason = "no_provider"
        outcome.error = "vision_provider_unavailable"
        outcome.result = empty_result(
            task_type=classification.task_type,
            provider="",
            model="",
            summary="advanced vision provider unavailable",
            image_ids=list(outcome.image_ids),
        )
        return outcome

    outcome.provider_called = True
    outcome.provider_name = str(getattr(provider, "name", ""))
    outcome.provider_model = str(getattr(provider, "model", ""))

    # Rebuild the cache key with the resolved provider name + model so a
    # cache hit from a different provider is correctly avoided.
    cache_key = _build_cache_key(
        provider_name=outcome.provider_name,
        task_type=classification.task_type,
        image_hashes=[image_content_hash(bytes_)],
    )

    # --- OCR input alignment (Phase 34D) -------------------------------
    # The chat path supplies ``ocr_text`` from the retrieved chunks, which
    # is often empty for visual-observation questions (the retriever does
    # not surface the image's OCR chunk for a "what looks wrong?" query).
    # For diagram / UI / table tasks the provider must still see the
    # persisted OCR labels so it can combine layout with exact text.
    #
    # The self-fetched persisted OCR is AUTHORITATIVE whenever it exists:
    # for an authorized image we prefer the already-indexed OCR chunks
    # (no re-OCR — mirrors knowledge_record._load_ocr_text_for_image).
    # The caller-supplied ``ocr_text`` (often partial or stale) is used
    # ONLY as a fallback when there is no persisted OCR to read.
    effective_ocr_text = ""
    if db is not None and authorized_image is not None:
        effective_ocr_text = (
            _load_persisted_ocr_for_image(db, int(authorized_image.image_id)) or ""
        ).strip()
    if not effective_ocr_text:
        effective_ocr_text = (ocr_text or "").strip()


    start = time.time() * 1000.0
    try:
        from app.services.vision.base import (
            VisionProviderError,
            VisionProviderTimeoutError,
            VisionProviderUnavailableError,
        )
    except Exception:  # pragma: no cover - defensive
        VisionProviderError = Exception  # type: ignore[misc,assignment]
        VisionProviderTimeoutError = Exception  # type: ignore[misc,assignment]
        VisionProviderUnavailableError = Exception  # type: ignore[misc,assignment]

    try:
        provider_result = provider.analyze_image(
            image_bytes=bytes_,
            mime_type=authorized_image.mime_type or "image/png",
            prompt=question or "Describe the image.",
            ocr_text=effective_ocr_text or "",
            context_hint=context_hint or "",
            max_tokens=_max_tokens_for_task(classification.task_type),
            timeout_s=_effective_timeout(),
            task=classification.task_type,
        )
    except VisionProviderTimeoutError as exc:
        outcome.error = f"provider_timeout: {str(exc)[:200]}"
        outcome.skipped_reason = "provider_timeout"
        outcome.result = empty_result(
            task_type=classification.task_type,
            provider=outcome.provider_name,
            model=outcome.provider_model,
            summary="advanced vision provider timed out",
            image_ids=list(outcome.image_ids),
        )
        outcome.latency_ms = int(time.time() * 1000.0 - start)
        return outcome
    except VisionProviderUnavailableError as exc:
        outcome.error = f"provider_unavailable: {str(exc)[:200]}"
        outcome.skipped_reason = "provider_unavailable"
        outcome.result = empty_result(
            task_type=classification.task_type,
            provider=outcome.provider_name,
            model=outcome.provider_model,
            summary="advanced vision provider unavailable",
            image_ids=list(outcome.image_ids),
        )
        outcome.latency_ms = int(time.time() * 1000.0 - start)
        return outcome
    except VisionProviderError as exc:
        outcome.error = f"provider_error: {str(exc)[:200]}"
        outcome.skipped_reason = "provider_error"
        outcome.result = empty_result(
            task_type=classification.task_type,
            provider=outcome.provider_name,
            model=outcome.provider_model,
            summary="advanced vision provider failed",
            image_ids=list(outcome.image_ids),
        )
        outcome.latency_ms = int(time.time() * 1000.0 - start)
        return outcome
    except Exception as exc:  # pragma: no cover - defensive
        outcome.error = f"provider_unexpected_error: {str(exc)[:200]}"
        outcome.skipped_reason = "provider_unexpected_error"
        outcome.result = empty_result(
            task_type=classification.task_type,
            provider=outcome.provider_name,
            model=outcome.provider_model,
            summary="advanced vision provider raised an unexpected error",
            image_ids=list(outcome.image_ids),
        )
        outcome.latency_ms = int(time.time() * 1000.0 - start)
        return outcome

    outcome.latency_ms = int(time.time() * 1000.0 - start)

    # The provider returns a Phase 34B ``VisionResult``. We map it into a
    # ``VisualReasoningResult`` by reading its description / findings /
    # etc. as the summary / observations. Phase 34B's structured fields
    # are intentionally narrow; the mapping is the contract between the
    # two layers.
    parsed = _vision_result_to_advanced(
        provider_result=provider_result,
        task_type=classification.task_type,
        image_ids=list(outcome.image_ids),
    )
    if parsed is None or not parsed.is_successful():
        outcome.skipped_reason = "parse_failure"
        outcome.result = empty_result(
            task_type=classification.task_type,
            provider=outcome.provider_name,
            model=outcome.provider_model,
            summary="advanced vision returned unparseable response",
            image_ids=list(outcome.image_ids),
        )
        return outcome

    outcome.result = parsed

    # --- Cache write ------------------------------------------------------
    adv_cache.set_cached_result(cache_key, parsed)

    return outcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate_authorize(
    db: Any,
    image_id: int,
    auth: Optional[AuthContext],
) -> Optional[AuthorizedImage]:
    """Run RBAC gate #1.

    Returns the ``AuthorizedImage`` (with bytes already fetched)
    on success. Returns ``None`` for unauthorized / not-found /
    fetch-failed paths — the orchestrator must not be able to
    distinguish "no such image" from "unauthorized" externally.
    """
    return fetch_authorized_image_bytes(
        db, int(image_id), auth, require_bytes=True,
        max_bytes=_effective_max_image_bytes(),
    )


def _load_persisted_ocr_for_image(db: Any, image_id: int) -> str:
    """Fetch persisted OCR text already indexed for ``image_id``.

    Phase 34A stores OCR text in Qdrant chunk payloads (not on the
    ``DocumentImage`` row). This reads those already-indexed chunks — it
    does NOT re-run OCR. Mirrors
    ``app.services.multimodal.knowledge_record._load_ocr_text_for_image``
    so diagram / UI / table advanced-vision tasks receive exact OCR labels
    even when the chat retriever did not surface the image's OCR chunk.

    Never raises: on any failure it returns ``""`` so the caller falls
    back to whatever OCR text was already supplied.
    """
    if db is None or not image_id:
        return ""
    try:
        from app.services.vector.qdrant_service import (
            fetch_chunks_by_image_id,
        )
        chunks = fetch_chunks_by_image_id(int(image_id), limit=8)
    except Exception as exc:
        logger.debug(
            "advanced_vision: persisted OCR fetch failed for image_id=%s: %s",
            image_id, exc,
        )
        return ""
    if not chunks:
        return ""
    # Concatenate de-duplicated chunk content in stable order.
    try:
        chunks = sorted(
            chunks,
            key=lambda c: (
                int(c.get("chunk_index") or 0)
                if c.get("chunk_index") is not None
                else 0,
                str(c.get("chunk_id") or ""),
            ),
        )
    except Exception:
        pass
    seen: set = set()
    pieces: List[str] = []
    for chunk in chunks:
        cid = str(chunk.get("chunk_id") or "")
        if cid in seen:
            continue
        seen.add(cid)
        content = (chunk.get("content") or "").strip()
        if content:
            pieces.append(content)
    return "\n".join(pieces)


def _reauthorize(db: Any, document_id: int, auth: Optional[AuthContext]) -> bool:
    """Run RBAC gate #2 — final check immediately before the provider call."""
    if auth is None or not getattr(auth, "is_authenticated", False):
        return False
    try:
        return bool(can_access_document(auth, int(document_id), db=db))
    except Exception as exc:
        logger.debug("advanced_vision: pre-provider re-auth raised: %s", exc)
        return False


def _forced_classification(task_type: str) -> TaskClassification:
    """Build a TaskClassification from a caller-supplied task type.

    Used by tests and by callers that already classified the
    question. The classifier's identifier / product-meaning checks
    are SKIPPED — the caller is asserting the task explicitly.
    """
    from app.services.advanced_vision.router import TaskClassification

    return TaskClassification(
        task_type=str(task_type or AdvancedVisualTask.GENERAL_VISUAL.value),
        is_identifier_lookup=False,
        is_product_meaning=False,
        requires_image=True,
        requires_comparison=(task_type == AdvancedVisualTask.IMAGE_COMPARISON.value),
        matched_patterns=["forced"],
    )


def _build_cache_key(
    *,
    provider_name: Optional[str],
    task_type: str,
    image_hashes: List[str],
) -> str:
    """Build a Redis key from the canonical fields.

    When ``provider_name`` is ``None`` (cache lookup before the
    provider was resolved) the key uses the
    ``settings.VISION_PROVIDER`` value so a hit on a different
    provider does not bleed across.
    """
    prov = (provider_name or getattr(settings, "VISION_PROVIDER", "") or "").strip().lower()
    model = str(getattr(settings, "VISION_MODEL", "") or "").strip()
    schema_version = int(
        getattr(settings, "ADVANCED_VISION_SCHEMA_VERSION", 1)
    )
    namespace = str(
        getattr(settings, "ADVANCED_VISION_CACHE_NAMESPACE", "advanced_vision") or "advanced_vision"
    )
    task = str(task_type or AdvancedVisualTask.GENERAL_VISUAL.value)
    # Compose the legacy key shape, then prepend the namespace.
    inner = build_advanced_vision_cache_key(
        schema_version=schema_version,
        provider=prov,
        model=model,
        task_type=task,
        image_hashes=list(image_hashes or []),
    )
    return f"{namespace}:{inner}"


def _max_tokens_for_task(task_type: str) -> int:
    """Task-aware max_tokens for the provider call."""
    try:
        from app.services.advanced_vision.prompts import build_advanced_vision_prompt
    except Exception:  # pragma: no cover - defensive
        return 800
    built = build_advanced_vision_prompt(
        task_type=task_type,
        question="",
        ocr_text="",
    )
    return int(built.max_tokens or 800)


def _vision_result_to_advanced(
    *,
    provider_result: Any,
    task_type: str,
    image_ids: List[int],
) -> Optional[VisualReasoningResult]:
    """Convert a Phase 34B ``VisionResult`` into a ``VisualReasoningResult``.

    The Phase 34B ``VisionResult`` is intentionally narrow (description
    / visual_findings / detected_entities / visual_states / tags /
    confidence). Phase 34D reads these as the generic / general
    summary + observations. The Phase 34D cache stores the parsed
    VisualReasoningResult; the openai-compatible provider is
    responsible for returning the full Phase 34D JSON schema when
    ``task`` is set, but the lenient parser will accept the narrower
    Phase 34B shape as a degraded but safe response.
    """
    try:
        # Phase 34B exposes its JSON via ``raw`` when the provider
        # returned strict JSON. If ``raw`` is present, prefer it.
        raw = getattr(provider_result, "raw", None) if provider_result else None
        if isinstance(raw, dict) and "task_type" in raw:
            return safe_parse_provider_payload(
                raw,
                provider=getattr(provider_result, "provider", "") or "",
                model=getattr(provider_result, "model", "") or "",
                processing_time_ms=int(getattr(provider_result, "processing_time_ms", 0) or 0),
                task_type=task_type,
                image_ids=list(image_ids or []),
            )
        # Fall back to the Phase 34B shape.
        description = str(getattr(provider_result, "description", "") or "")
        findings = list(getattr(provider_result, "visual_findings", []) or [])
        entities = list(getattr(provider_result, "detected_entities", []) or [])
        states = list(getattr(provider_result, "visual_states", []) or [])
        confidence = getattr(provider_result, "confidence", None)
        provider = getattr(provider_result, "provider", "") or ""
        model = getattr(provider_result, "model", "") or ""
        latency = int(getattr(provider_result, "processing_time_ms", 0) or 0)
        if not (description or findings or entities or states):
            return None
        return safe_parse_provider_payload(
            {
                "task_type": task_type,
                "summary": description[:280],
                "observations": findings + states,
                "entities": entities,
                "confidence": confidence,
            },
            provider=provider,
            model=model,
            processing_time_ms=latency,
            task_type=task_type,
            image_ids=list(image_ids or []),
        )
    except Exception as exc:
        logger.debug("advanced_vision: provider-result mapping failed: %s", exc)
        return None


__all__ = [
    "AdvancedOrchestratorOutcome",
    "run_advanced_visual_reasoning",
]

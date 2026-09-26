"""Phase 34D — Image-comparison helper.

Implements the IMAGE_COMPARISON task on top of the existing Phase 34B
VisionProvider and the Phase 34D orchestrator. Two authorized images
are compared A → B (B is the *after* state).

RBAC contract:

* Both images are independently authorized via ``can_access_document``
  BEFORE any bytes leave the backend (gate #1).
* Both images are independently re-authorized immediately before the
  provider call (gate #2, defense in depth).
* If either image fails either gate, the comparison returns a
  ``ComparisonOutcome`` with ``ran=False`` and ``skipped_reason``
  set. NO image bytes, filenames, comparison result, or existence
  information is returned to the unauthorized caller.

Cache contract:

* Cache key preserves A → B order (``hash_a`` then ``hash_b``). A
  swapped request produces a different key so a swapped cache hit
  cannot silently invert the comparison answer.
* Schema version + provider + model are embedded in the key. Bump
  ``ADVANCED_VISION_SCHEMA_VERSION`` to invalidate.

The module is non-throwing. Every failure path returns a
``ComparisonOutcome`` so the chat pipeline is always answerable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

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
from app.services.advanced_vision.orchestrator import AdvancedOrchestratorOutcome
from app.services.advanced_vision.router import TaskClassification


logger = logging.getLogger(__name__)


@dataclass
class ComparisonOutcome:
    """Result of running the image-comparison flow.

    Mirrors :class:`AdvancedOrchestratorOutcome` for the two-image
    case. ``image_ids`` always contains exactly two entries on a
    successful run (A, B in that order). On every failure path the
    list may be empty or partial — callers MUST NOT log partial
    image IDs to user-visible surfaces.
    """

    ran: bool = False
    result: Optional[VisualReasoningResult] = None
    skipped_reason: Optional[str] = None
    error: Optional[str] = None
    cache_hit: bool = False
    cache_source: str = ""
    provider_called: bool = False
    provider_name: str = ""
    provider_model: str = ""
    latency_ms: int = 0
    image_ids: List[int] = field(default_factory=list)
    image_a_filename: Optional[str] = None
    image_b_filename: Optional[str] = None

    def to_advanced_outcome(self) -> AdvancedOrchestratorOutcome:
        """Promote to :class:`AdvancedOrchestratorOutcome` for integration."""
        out = AdvancedOrchestratorOutcome(
            ran=self.ran,
            result=self.result,
            classification=TaskClassification(
                task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
                is_identifier_lookup=False,
                is_product_meaning=False,
                requires_image=True,
                requires_comparison=True,
                matched_patterns=["image_comparison"],
            ),
            skipped_reason=self.skipped_reason,
            error=self.error,
            cache_hit=self.cache_hit,
            cache_source=self.cache_source,
            provider_called=self.provider_called,
            provider_name=self.provider_name,
            provider_model=self.provider_model,
            latency_ms=self.latency_ms,
            image_ids=list(self.image_ids or []),
        )
        return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_advanced_visual_comparison(
    db: Any,
    *,
    auth: Optional[AuthContext],
    question: str,
    image_id_a: int,
    image_id_b: int,
    ocr_text_a: str = "",
    ocr_text_b: str = "",
    image_bytes_a: Optional[bytes] = None,
    image_bytes_b: Optional[bytes] = None,
    image_mime_a: Optional[str] = None,
    image_mime_b: Optional[str] = None,
    context_hint: str = "",
    provider: Optional[Any] = None,
) -> ComparisonOutcome:
    """Compare two authorized images (A → B).

    Args:
        db: SQLAlchemy session.
        auth: AuthContext. ``None`` is treated as unauthorized.
        question: The user question (used as the comparison prompt).
        image_id_a: The "before" image id.
        image_id_b: The "after" image id.
        ocr_text_a: Pre-extracted OCR for image A (optional).
        ocr_text_b: Pre-extracted OCR for image B (optional).
        image_bytes_a / image_bytes_b: Optional pre-fetched bytes
            (caller-supplied; RBAC still applies via
            ``image_id_a`` / ``image_id_b``).
        image_mime_a / image_mime_b: MIME types for the supplied
            bytes. Defaults to ``image/png``.
        context_hint: Free-text context forwarded to the provider.
        provider: Optional Phase 34B ``VisionProvider`` instance
            (dependency injection — same seam as the single-image
            orchestrator). When ``None``, falls back to the Phase 34B
            factory resolver.

    Returns:
        :class:`ComparisonOutcome`. Never raises.
    """
    outcome = ComparisonOutcome()

    if not _is_enabled():
        outcome.skipped_reason = "kill_switch"
        return outcome

    # Early auth gate: an unauthenticated caller never reaches the
    # image-fetch path, so the comparison fails with a single
    # 'unauthorized' reason (not 'unauthorized_pre_provider').
    if auth is None or not getattr(auth, "is_authenticated", False):
        outcome.skipped_reason = "unauthorized"
        return outcome

    if not int(image_id_a) or not int(image_id_b):
        outcome.skipped_reason = "invalid_image_ids"
        return outcome

    if int(image_id_a) == int(image_id_b):
        # Comparison of an image against itself is meaningless and
        # could mask a wiring bug. Refuse explicitly.
        outcome.skipped_reason = "identical_image_ids"
        return outcome

    if _max_images_per_request() < 2:
        outcome.skipped_reason = "comparison_disabled_by_cap"
        return outcome

    # --- Gate 1: authorize both BEFORE MinIO fetch ---------------------
    auth_a = _gate_authorize(db, image_id_a, auth, bytes_override=image_bytes_a)
    if auth_a is None:
        outcome.skipped_reason = "unauthorized"
        return outcome

    auth_b = _gate_authorize(db, image_id_b, auth, bytes_override=image_bytes_b)
    if auth_b is None:
        outcome.skipped_reason = "unauthorized"
        return outcome

    bytes_a = auth_a.bytes_ or b""
    bytes_b = auth_b.bytes_ or b""
    if not bytes_a or not bytes_b:
        outcome.skipped_reason = "fetch_failed"
        outcome.error = "image_bytes_empty"
        return outcome

    cap = _effective_max_image_bytes()
    if len(bytes_a) > cap or len(bytes_b) > cap:
        outcome.skipped_reason = "too_large"
        outcome.error = (
            f"image_too_large_for_provider "
            f"({len(bytes_a)},{len(bytes_b)} > {cap})"
        )
        return outcome

    outcome.image_ids = [int(auth_a.image_id), int(auth_b.image_id)]
    outcome.image_a_filename = auth_a.filename
    outcome.image_b_filename = auth_b.filename
    # NOTE: outcome.ran is set AFTER RBAC gate #2 + provider-availability
    # below so every failure path records ran=False.

    # --- Cache lookup (order-preserving) ---------------------------------
    hash_a = image_content_hash(bytes_a)
    hash_b = image_content_hash(bytes_b)
    cache_key = _build_comparison_cache_key(image_hashes=[hash_a, hash_b])
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
        outcome.ran = True  # cache hit produced a usable comparison result
        return outcome

    # --- Gate 2: re-authorize both IMMEDIATELY before provider call -----
    if not _reauthorize(db, auth_a.document_id, auth):
        outcome.skipped_reason = "unauthorized_pre_provider"
        return outcome
    if not _reauthorize(db, auth_b.document_id, auth):
        outcome.skipped_reason = "unauthorized_pre_provider"
        return outcome

    # All early gates passed. We are now in the "execute the comparison"
    # phase. Set ran=True so every downstream failure path
    # (no_provider, provider_timeout, parse_failure) still records
    # ran=True — we attempted to run, we just couldn't produce a
    # usable comparison result.
    outcome.ran = True

    # --- Provider call ----------------------------------------------------
    provider = provider if provider is not None else _resolve_provider()
    if provider is None:
        outcome.skipped_reason = "no_provider"
        outcome.error = "vision_provider_unavailable"
        outcome.result = empty_result(
            task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
            provider="",
            model="",
            summary="advanced vision provider unavailable",
            image_ids=list(outcome.image_ids),
        )
        return outcome

    outcome.provider_called = True
    outcome.provider_name = str(getattr(provider, "name", ""))
    outcome.provider_model = str(getattr(provider, "model", ""))

    # Rebuild the cache key with the resolved provider name + model.
    cache_key = _build_comparison_cache_key(
        image_hashes=[hash_a, hash_b],
        provider_name=outcome.provider_name,
        model_name=outcome.provider_model,
    )

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
            image_bytes=bytes_a,
            mime_type=auth_a.mime_type or "image/png",
            prompt=question or "Compare the two images and describe what changed.",
            ocr_text=(ocr_text_a or "") + ("\n" + ocr_text_b if ocr_text_b else ""),
            context_hint=context_hint or "",
            max_tokens=_max_tokens_for_comparison(),
            timeout_s=_effective_timeout(),
            task=AdvancedVisualTask.IMAGE_COMPARISON.value,
            images=[
                (bytes_a, auth_a.mime_type or "image/png"),
                (bytes_b, auth_b.mime_type or "image/png"),
            ],
        )
    except (VisionProviderTimeoutError, VisionProviderUnavailableError, VisionProviderError) as exc:
        outcome.error = f"provider_error: {str(exc)[:200]}"
        outcome.skipped_reason = "provider_error"
        outcome.result = empty_result(
            task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
            provider=outcome.provider_name,
            model=outcome.provider_model,
            summary="advanced vision comparison provider failed",
            image_ids=list(outcome.image_ids),
        )
        outcome.latency_ms = int(time.time() * 1000.0 - start)
        return outcome
    except Exception as exc:  # pragma: no cover - defensive
        outcome.error = f"provider_unexpected_error: {str(exc)[:200]}"
        outcome.skipped_reason = "provider_unexpected_error"
        outcome.result = empty_result(
            task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
            provider=outcome.provider_name,
            model=outcome.provider_model,
            summary="advanced vision comparison provider raised an unexpected error",
            image_ids=list(outcome.image_ids),
        )
        outcome.latency_ms = int(time.time() * 1000.0 - start)
        return outcome

    outcome.latency_ms = int(time.time() * 1000.0 - start)

    parsed = _vision_result_to_comparison(
        provider_result=provider_result,
        task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
        image_ids=list(outcome.image_ids),
    )
    if parsed is None or not parsed.is_successful():
        outcome.skipped_reason = "parse_failure"
        outcome.result = empty_result(
            task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
            provider=outcome.provider_name,
            model=outcome.provider_model,
            summary="advanced vision comparison returned unparseable response",
            image_ids=list(outcome.image_ids),
        )
        return outcome

    outcome.result = parsed
    adv_cache.set_cached_result(cache_key, parsed)
    return outcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_enabled() -> bool:
    if not bool(getattr(settings, "ADVANCED_VISION_ENABLED", False)):
        return False
    if not bool(getattr(settings, "ADVANCED_VISION_ROUTER_ENABLED", True)):
        return False
    return True


def _max_images_per_request() -> int:
    return int(getattr(settings, "ADVANCED_VISION_MAX_IMAGES_PER_REQUEST", 2) or 2)


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


def _resolve_provider():
    try:
        from app.services.vision.factory import get_vision_provider
        return get_vision_provider()
    except Exception as exc:
        logger.debug("advanced_vision.comparison: provider factory raised: %s", exc)
        return None


def _gate_authorize(
    db: Any,
    image_id: int,
    auth: Optional[AuthContext],
    *,
    bytes_override: Optional[bytes],
) -> Optional[AuthorizedImage]:
    """Run gate #1 for one image.

    Returns the ``AuthorizedImage`` (with bytes already fetched) on
    success, ``None`` for unauthorized / not-found / fetch-failed.
    The ``bytes_override`` lets callers inject pre-fetched bytes
    (RBAC is still enforced via the image_id).
    """
    auth_img = fetch_authorized_image_bytes(
        db, int(image_id), auth, require_bytes=True,
        max_bytes=_effective_max_image_bytes(),
    )
    if auth_img is None:
        return None
    if bytes_override is not None:
        if not auth_img.bytes_:
            auth_img.bytes_ = bytes(bytes_override)
    return auth_img


def _reauthorize(db: Any, document_id: int, auth: Optional[AuthContext]) -> bool:
    if auth is None or not getattr(auth, "is_authenticated", False):
        return False
    try:
        return bool(can_access_document(auth, int(document_id), db=db))
    except Exception as exc:
        logger.debug("advanced_vision.comparison: re-auth raised: %s", exc)
        return False


def _build_comparison_cache_key(
    *,
    image_hashes: List[str],
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Build an order-preserving Redis key for IMAGE_COMPARISON.

    ``image_hashes`` MUST be ``[hash_a, hash_b]``. Order is preserved
    so a swapped request produces a different cache key.
    """
    prov = (provider_name or getattr(settings, "VISION_PROVIDER", "") or "").strip().lower()
    model = str(model_name or getattr(settings, "VISION_MODEL", "") or "").strip()
    schema_version = int(getattr(settings, "ADVANCED_VISION_SCHEMA_VERSION", 1))
    namespace = str(
        getattr(settings, "ADVANCED_VISION_CACHE_NAMESPACE", "advanced_vision") or "advanced_vision"
    )
    task = AdvancedVisualTask.IMAGE_COMPARISON.value
    inner = build_advanced_vision_cache_key(
        schema_version=schema_version,
        provider=prov,
        model=model,
        task_type=task,
        image_hashes=list(image_hashes or []),
    )
    return f"{namespace}:{inner}"


def _max_tokens_for_comparison() -> int:
    try:
        from app.services.advanced_vision.prompts import build_advanced_vision_prompt
    except Exception:  # pragma: no cover - defensive
        return 1200
    built = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
        question="",
        ocr_text="",
    )
    return int(built.max_tokens or 1200)


def _vision_result_to_comparison(
    *,
    provider_result: Any,
    task_type: str,
    image_ids: List[int],
) -> Optional[VisualReasoningResult]:
    """Convert a Phase 34B ``VisionResult`` (or its raw JSON) into a
    ``VisualReasoningResult`` for IMAGE_COMPARISON.

    Same lenient mapping as the single-image orchestrator. If the
    provider returned strict JSON via ``raw``, prefer that path; if
    it returned the narrower Phase 34B shape, degrade safely.
    """
    try:
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
        description = str(getattr(provider_result, "description", "") or "")
        findings = list(getattr(provider_result, "visual_findings", []) or [])
        if not (description or findings):
            return None
        return safe_parse_provider_payload(
            {
                "task_type": task_type,
                "summary": description[:280],
                "observations": findings,
                "comparison_changes": [
                    {"subject": "(provider-summary)", "kind": "visual-state-changed", "before": "", "after": description[:200]}
                ],
            },
            provider=getattr(provider_result, "provider", "") or "",
            model=getattr(provider_result, "model", "") or "",
            processing_time_ms=int(getattr(provider_result, "processing_time_ms", 0) or 0),
            task_type=task_type,
            image_ids=list(image_ids or []),
        )
    except Exception as exc:
        logger.debug("advanced_vision.comparison: result mapping failed: %s", exc)
        return None


__all__ = [
    "ComparisonOutcome",
    "run_advanced_visual_comparison",
]

"""Phase 34B — Vision persistence + cache helpers.

Persists ``VisionResult`` rows on ``document_images`` and
implements the cache-reuse policy:

    * cache hit when the same ``document_image_id`` already has a
      successful Vision row from the SAME ``provider`` and
      ``model`` with the SAME ``schema_version``,
    * cache miss otherwise (the provider is called, the row is
      updated),
    * failed Vision rows are NOT cache hits — the next request
      may succeed.

The module intentionally depends on SQLAlchemy + the existing
``DocumentImage`` model. RBAC is enforced by the caller (the chat
endpoint already gates on the resolved ``document_id``); the
helper only operates on IDs the caller is authorized to use.

All write helpers are non-throwing — a persistence error is
returned as ``PersistenceResult(error=...)`` so the chat pipeline
can continue with OCR-only evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services.vision.base import (
    VisionResult,
    VISION_SCHEMA_VERSION,
    build_cache_key,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PersistenceResult:
    """Outcome of a cache lookup / write."""

    cache_hit: bool = False
    vision_result: Optional[VisionResult] = None
    error: Optional[str] = None


def _safe_json(value) -> Optional[List[Any]]:
    """Coerce a value to a JSON-serialisable list / None.

    The DB column is JSONB. The provider returns Python lists; we
    defensively coerce here so a future provider returning
    something unexpected cannot break Alembic-upgraded rows.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [v for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return None


def lookup_cached_vision(
    db: Session,
    *,
    document_image_id: int,
    provider: str,
    model: str,
    schema_version: int = VISION_SCHEMA_VERSION,
) -> PersistenceResult:
    """Return a cached ``VisionResult`` if one exists for this image.

    The lookup is strict: the cached row must have
    ``vision_status='success'`` AND match ``provider``, ``model``,
    and ``schema_version``. Stale rows are ignored.
    """
    try:
        from app.models.document import DocumentImage

        row = (
            db.query(DocumentImage)
            .filter(DocumentImage.id == int(document_image_id))
            .first()
        )
    except Exception as exc:
        return PersistenceResult(
            cache_hit=False,
            error=f"cache_lookup_db_error: {str(exc)[:200]}",
        )

    if row is None:
        return PersistenceResult(cache_hit=False, error="image_row_not_found")

    if getattr(row, "vision_status", None) != "success":
        return PersistenceResult(cache_hit=False)

    if (getattr(row, "vision_provider", None) or "") != (provider or ""):
        return PersistenceResult(cache_hit=False)
    if (getattr(row, "vision_model", None) or "") != (model or ""):
        return PersistenceResult(cache_hit=False)

    row_schema_version = getattr(row, "vision_cache_key", None) or ""
    expected_prefix = build_cache_key(
        document_image_id=int(document_image_id),
        provider=provider,
        model=model,
        schema_version=schema_version,
    )
    if row_schema_version != expected_prefix:
        # Stored cache key does not match the current schema version.
        return PersistenceResult(cache_hit=False)

    vision_result = VisionResult(
        description=row.vision_description or "",
        image_type=row.vision_image_type or "unknown",
        visual_findings=_safe_json(row.vision_findings) or [],
        detected_entities=_safe_json(row.vision_entities) or [],
        # NOTE: DocumentImage maps the DB column ``vision_states``
        # onto the Python attribute ``visual_states`` (see
        # ``app/models/document.py`` — the Column declaration uses
        # ``Column("vision_states", JSON, ...),``). Reading
        # ``row.vision_states`` raises AttributeError, which the
        # orchestrator's broad except then silently turns into a
        # cache miss. Always use the Python attribute name.
        visual_states=_safe_json(row.visual_states) or [],
        tags=_safe_json(row.vision_tags) or [],
        confidence=row.vision_confidence,
        provider=provider,
        model=model,
        processing_time_ms=0,  # cached — we don't replay provider latency
        schema_version=schema_version,
        raw=None,
    )

    if not vision_result.is_successful():
        return PersistenceResult(cache_hit=False)

    return PersistenceResult(cache_hit=True, vision_result=vision_result)


def persist_vision_result(
    db: Session,
    *,
    document_image_id: int,
    vision_result: VisionResult,
) -> PersistenceResult:
    """Write a successful ``VisionResult`` onto the DocumentImage row.

    Returns ``cache_hit=False`` because the call did write — the
    caller distinguishes cache hits BEFORE calling the provider.
    """
    try:
        from app.models.document import DocumentImage

        row = (
            db.query(DocumentImage)
            .filter(DocumentImage.id == int(document_image_id))
            .first()
        )
    except Exception as exc:
        return PersistenceResult(
            cache_hit=False,
            error=f"persist_lookup_db_error: {str(exc)[:200]}",
        )

    if row is None:
        return PersistenceResult(cache_hit=False, error="image_row_not_found")

    cache_key = build_cache_key(
        document_image_id=int(document_image_id),
        provider=vision_result.provider,
        model=vision_result.model,
        schema_version=vision_result.schema_version,
    )

    try:
        row.vision_status = "success"
        row.vision_provider = vision_result.provider
        row.vision_model = vision_result.model
        row.vision_description = vision_result.description
        row.vision_image_type = vision_result.image_type
        row.vision_findings = list(vision_result.visual_findings or [])
        row.vision_entities = list(vision_result.detected_entities or [])
        # Python attribute name is ``visual_states``; the DB column
        # name is ``vision_states`` (see DocumentImage mapping).
        row.visual_states = list(vision_result.visual_states or [])
        row.vision_tags = list(vision_result.tags or [])
        row.vision_confidence = vision_result.confidence
        row.vision_processed_at = _now()
        row.vision_error = None
        row.vision_cache_key = cache_key
        db.add(row)
        db.commit()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return PersistenceResult(
            cache_hit=False,
            error=f"persist_write_error: {str(exc)[:200]}",
        )

    # Phase 34C -- Persistent Multimodal Knowledge. After a successful
    # Vision write, refresh the image knowledge point so the OCR-only
    # point is upgraded to OCR+Vision in place. The point id is
    # deterministic so this is an in-place overwrite -- no duplicates.
    # The hook is fault-tolerant: failures here MUST NOT break the
    # Vision persistence contract. Indexing failures are logged but
    # not returned to the caller.
    try:
        from app.services.multimodal.lifecycle import refresh_image_knowledge

        refresh_image_knowledge(db, int(document_image_id))
    except Exception as exc:  # pragma: no cover - defensive guard
        try:
            logger.warning(
                "vision.persist: multimodal refresh failed for image_id=%s: %s",
                document_image_id, exc,
            )
        except Exception:
            pass

    return PersistenceResult(cache_hit=False, vision_result=vision_result)


def persist_vision_failure(
    db: Session,
    *,
    document_image_id: int,
    provider: str,
    model: str,
    error: str,
) -> PersistenceResult:
    """Record a failed Vision attempt so we don't lose the signal.

    The row is NOT a cache hit. We persist only minimal metadata so
    ops can inspect later.
    """
    try:
        from app.models.document import DocumentImage

        row = (
            db.query(DocumentImage)
            .filter(DocumentImage.id == int(document_image_id))
            .first()
        )
        if row is None:
            return PersistenceResult(cache_hit=False, error="image_row_not_found")
        row.vision_status = "failed"
        row.vision_provider = provider
        row.vision_model = model
        row.vision_error = (error or "")[:500]
        row.vision_processed_at = _now()
        # Clear any stale cache key so the next request retries.
        row.vision_cache_key = None
        db.add(row)
        db.commit()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return PersistenceResult(
            cache_hit=False,
            error=f"persist_failure_db_error: {str(exc)[:200]}",
        )
    return PersistenceResult(cache_hit=False, error=(error or "")[:200])


def _now():
    from datetime import datetime
    return datetime.utcnow()

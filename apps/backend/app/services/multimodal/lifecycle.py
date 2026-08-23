"""Phase 34C -- Multimodal lifecycle hooks.

This module is the integration surface for image / document / Vision
lifecycle events. Every function here is **safe to call from request
handlers** -- failures are caught and logged, never raised, so a
broken Qdrant / embedding call cannot break upload, OCR, or Vision
flow.

Hook points (who calls what):

* ``app.vision.persistence.persist_vision_result``  -> ``refresh_image_knowledge``
  After a successful Vision write, rebuild the image knowledge record
  so the OCR-only point is upgraded to OCR+Vision in place.

* ``app.api.documents.py`` image delete path         -> ``delete_image_knowledge_for_image``
  When a ``DocumentImage`` row is removed, remove the matching
  Qdrant point.

* ``app.api.documents.py`` document reindex path    -> ``reindex_document_images``
  After re-ingestion, rebuild image knowledge points for the
  document.

* ``app.api.documents.py`` document delete path     -> ``delete_image_knowledge_for_document``
  Called alongside ``qdrant_service.delete_vectors_by_document_id``
  as a defensive explicit cleanup. The same effect is achieved by
  the existing document-id filter, but the explicit call makes the
  intent visible in audit logs and protects against future
  source-type-aware filters that might exclude image knowledge.

None of the functions call Vision or LLM providers. They only READ
persisted Phase 34B columns and call into the indexer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.multimodal.indexer import (
    DeleteResult,
    IndexResult,
    delete_image_knowledge,
    delete_image_knowledge_by_document_id,
    upsert_image_knowledge,
)
from app.services.multimodal.knowledge_record import (
    ImageKnowledgeRecord,
    load_image_record,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def _settings():
    from app.core.config import settings

    return settings


def _schema_version() -> int:
    try:
        return int(_settings().MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION)
    except Exception:
        return 1


def _is_enabled() -> bool:
    try:
        return bool(_settings().MULTIMODAL_KNOWLEDGE_ENABLED)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Single-image refresh (called after Vision persistence)
# ---------------------------------------------------------------------------


def refresh_image_knowledge(db, document_image_id: int) -> IndexResult:
    """Rebuild + upsert the image knowledge point for one image.

    Called from ``app.vision.persistence.persist_vision_result``
    AFTER a successful Vision row is written. Re-running on an OCR-only
    image with no Vision still produces an OCR-only knowledge point.
    Never raises; always returns an ``IndexResult``.
    """
    if not _is_enabled():
        return IndexResult(status="disabled", error="multimodal_disabled")
    if not document_image_id:
        return IndexResult(status="skipped", error="invalid_image_id")

    try:
        record = load_image_record(
            db,
            int(document_image_id),
            schema_version=_schema_version(),
        )
    except Exception as exc:
        logger.warning(
            "multimodal: refresh load failed for image_id=%s: %s",
            document_image_id, exc,
        )
        return IndexResult(status="error", error=f"load_failed: {str(exc)[:160]}")

    if record is None:
        # Image row missing or no usable content -> remove any stale
        # knowledge point so it doesn't linger.
        delete_image_knowledge(int(document_image_id))
        return IndexResult(status="skipped", error="no_record")

    try:
        return upsert_image_knowledge(record)
    except Exception as exc:
        logger.warning(
            "multimodal: refresh upsert failed for image_id=%s: %s",
            document_image_id, exc,
        )
        return IndexResult(status="error", error=f"upsert_failed: {str(exc)[:160]}")


# ---------------------------------------------------------------------------
# Image delete
# ---------------------------------------------------------------------------


def delete_image_knowledge_for_image(image_id: int) -> DeleteResult:
    """Remove the knowledge point for a single image (image-delete hook)."""
    if not _is_enabled():
        return DeleteResult(deleted=0)
    if not image_id:
        return DeleteResult(deleted=0, error="invalid_image_id")
    return delete_image_knowledge(int(image_id))


# ---------------------------------------------------------------------------
# Document delete
# ---------------------------------------------------------------------------


def delete_image_knowledge_for_document(document_id: int) -> DeleteResult:
    """Remove every knowledge point for a document (document-delete hook).

    Defensive explicit cleanup called alongside the existing
    ``qdrant_service.delete_vectors_by_document_id``. The latter
    already removes all Qdrant points matching ``document_id``; this
    call exists so the image knowledge layer is self-contained even
    if the broader filter ever changes.
    """
    if not _is_enabled():
        return DeleteResult(deleted=0)
    if not document_id:
        return DeleteResult(deleted=0, error="invalid_document_id")
    return delete_image_knowledge_by_document_id(int(document_id))


# ---------------------------------------------------------------------------
# Document reindex
# ---------------------------------------------------------------------------


def reindex_document_images(db, document_id: int, *, batch_size: int = 50) -> Dict[str, Any]:
    """Rebuild image knowledge points for every ``DocumentImage`` of a document.

    Idempotent: re-running overwrites existing points in place (the
    point id is deterministic). Safe to call after a Vision enrichment
    round -- the second pass updates the same points with the new
    Vision content instead of creating duplicates.

    Returns a summary dict:
        {
            "document_id": int,
            "scanned": int,        # DocumentImage rows examined
            "indexed": int,        # successful upserts
            "skipped": int,        # no usable content (no OCR success, no Vision)
            "errors": int,
            "error_samples": list[str],  # capped at 5 for observability
        }

    Never raises -- partial failures are recorded in the summary.
    """
    summary: Dict[str, Any] = {
        "document_id": int(document_id) if document_id else None,
        "scanned": 0,
        "indexed": 0,
        "skipped": 0,
        "errors": 0,
        "error_samples": [],
    }
    if not _is_enabled():
        summary["disabled"] = True
        return summary
    if not document_id:
        summary["skipped"] += 1
        return summary

    image_ids = _list_image_ids_for_document(db, int(document_id))
    summary["scanned"] = len(image_ids)
    if not image_ids:
        return summary

    schema_version = _schema_version()
    for image_id in image_ids:
        try:
            record = load_image_record(db, int(image_id), schema_version=schema_version)
        except Exception as exc:
            summary["errors"] += 1
            if len(summary["error_samples"]) < 5:
                summary["error_samples"].append(f"image_id={image_id} load_failed: {str(exc)[:120]}")
            continue
        if record is None:
            # No usable content -- drop any stale point.
            try:
                delete_image_knowledge(int(image_id))
            except Exception:
                pass
            summary["skipped"] += 1
            continue
        try:
            result = upsert_image_knowledge(record)
            if result.status == "upserted":
                summary["indexed"] += 1
            elif result.status == "error":
                summary["errors"] += 1
                if len(summary["error_samples"]) < 5:
                    summary["error_samples"].append(f"image_id={image_id} {result.error or 'error'}")
            else:
                summary["skipped"] += 1
        except Exception as exc:
            summary["errors"] += 1
            if len(summary["error_samples"]) < 5:
                summary["error_samples"].append(f"image_id={image_id} upsert_failed: {str(exc)[:120]}")

    return summary


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _list_image_ids_for_document(db, document_id: int) -> List[int]:
    """Return all ``DocumentImage.id`` for a document, ordered by id.

    Works with the test SQLite DB (no JSON filters) and the production
    PostgreSQL DB identically because we only read the integer primary
    key.
    """
    try:
        from app.models.document import DocumentImage

        rows = (
            db.query(DocumentImage.id)
            .filter(DocumentImage.document_id == int(document_id))
            .order_by(DocumentImage.id.asc())
            .all()
        )
        return [int(r[0]) for r in rows]
    except Exception as exc:
        logger.warning(
            "multimodal: failed to list image_ids for document_id=%s: %s",
            document_id, exc,
        )
        return []


# ---------------------------------------------------------------------------
# Whole-image rebuild (used by backfill CLI)
# ---------------------------------------------------------------------------


def reindex_all_images(db, *, batch_size: int = 50, document_id: Optional[int] = None) -> Dict[str, Any]:
    """Rebuild image knowledge points across documents.

    Parameters:
        batch_size: Reserved for future chunking; the current
            implementation indexes one image at a time.
        document_id: If supplied, restrict to a single document.

    Returns a summary dict with the same shape as ``reindex_document_images``.
    """
    summary: Dict[str, Any] = {
        "scanned": 0,
        "indexed": 0,
        "skipped": 0,
        "errors": 0,
        "error_samples": [],
        "disabled": not _is_enabled(),
    }
    if not _is_enabled():
        return summary

    document_ids: List[int] = []
    if document_id is not None:
        document_ids = [int(document_id)]
    else:
        try:
            from app.models.document import Document

            document_ids = [
                int(d[0]) for d in db.query(Document.id).order_by(Document.id.asc()).all()
            ]
        except Exception as exc:
            summary["errors"] += 1
            summary["error_samples"].append(f"list_documents_failed: {str(exc)[:120]}")
            return summary

    for doc_id in document_ids:
        sub = reindex_document_images(db, doc_id, batch_size=batch_size)
        summary["scanned"] += sub.get("scanned", 0)
        summary["indexed"] += sub.get("indexed", 0)
        summary["skipped"] += sub.get("skipped", 0)
        summary["errors"] += sub.get("errors", 0)
        for sample in sub.get("error_samples", []):
            if len(summary["error_samples"]) < 5:
                summary["error_samples"].append(sample)
    return summary


__all__ = [
    "refresh_image_knowledge",
    "delete_image_knowledge_for_image",
    "delete_image_knowledge_for_document",
    "reindex_document_images",
    "reindex_all_images",
]

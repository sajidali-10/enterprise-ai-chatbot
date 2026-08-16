"""
Phase 34A.1.2 — Authenticated Image Resolver

Resolves the image/document that the user is implicitly asking about
when their chat question is an ``image_content`` question but no
explicit ``image_context`` was supplied by the frontend.

The frontend (Phase 34A.1.x) does not yet attach ``image_context`` to
chat requests, so the backend has to determine the "image in scope"
itself. We do this by:

  1. Looking at the most recently indexed OCR-derived document that
     the authenticated user can access, using the same RBAC rules
     that drive the rest of the system (Document.visibility,
     ownership, DocumentPermission rows, role-shared docs).
  2. Preferring ``DocumentImage`` rows whose OCR actually succeeded
     (``ocr_status == 'success'``) and skipping any image that the
     resolver cannot positively confirm is accessible to the user.

The resolver NEVER searches arbitrary historical images. It returns
the single most-recently-uploaded accessible image document for the
authenticated user, and only when an image-content question is in
flight. Other question types (general / error_lookup) bypass this
module entirely.

The resolver returns a small ``ResolvedImage`` namedtuple describing
what was found, plus a structured diagnostic dict suitable for the
observability layer. It does not touch Qdrant directly — that is the
job of ``qdrant_service.fetch_chunks_by_document_id`` /
``fetch_chunks_by_image_id`` which the answer generator calls once
the resolver has produced a target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class ResolvedImage:
    """Result of resolving an image for an image-content question.

    ``document_id`` is always populated when ``resolved`` is True;
    ``image_id`` is populated when a specific DocumentImage row was
    resolved (preferred), otherwise None and the answer generator
    fetches by ``document_id`` alone.
    """

    resolved: bool
    document_id: Optional[int] = None
    image_id: Optional[int] = None
    document_filename: Optional[str] = None
    image_filename: Optional[str] = None
    ocr_status: Optional[str] = None
    created_at: Optional[datetime] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def resolve_recent_image(
    db: Session,
    auth: Optional[Any],
    *,
    limit: int = 25,
) -> ResolvedImage:
    """Resolve the most recently accessible OCR-derived document for ``auth``.

    Strategy:
        1. Pull the most recent OCR-bearing document rows
           (``DocumentImage.ocr_status == 'success'``) ordered by
           ``DocumentImage.created_at desc``.
        2. Apply the existing RBAC filter
           (``filter_documents_by_permission``) to that candidate
           set. The FIRST candidate that survives RBAC filtering is
           the resolved image.
        3. If RBAC denies every candidate, return ``ResolvedImage``
           with ``resolved=False`` so the caller can produce the
           standard grounded "not enough information" response.

    Args:
        db: Active SQLAlchemy session (caller-owned).
        auth: AuthContext from the chat request. May be None for
            unauthenticated calls (in which case nothing resolves).
        limit: How many recent candidates to consider before stopping.
            Default 25 — covers typical "what did I just upload"
            flows without scanning the full history.
    """
    empty = ResolvedImage(resolved=False, diagnostics={
        "image_resolution_mode": "none",
        "candidate_count": 0,
    })

    if db is None:
        empty.diagnostics["image_resolution_mode"] = "skipped_no_session"
        return empty

    if auth is None or not getattr(auth, "is_authenticated", False):
        empty.diagnostics["image_resolution_mode"] = "skipped_unauthenticated"
        return empty

    # Local imports keep this module importable in test contexts where
    # models/permissions may not be wired yet.
    from app.models.document import Document, DocumentImage

    # Lazy import so test stubs can override permissions module.
    try:
        from app.security.permissions import filter_documents_by_permission
    except Exception as exc:  # pragma: no cover - defensive
        empty.diagnostics["image_resolution_mode"] = "skipped_no_permissions"
        empty.diagnostics["error"] = str(exc)[:200]
        return empty

    # Pull the most recent OCR-derived images.
    try:
        candidates: List[Tuple[DocumentImage, Document]] = (
            db.query(DocumentImage, Document)
            .join(Document, DocumentImage.document_id == Document.id)
            .filter(DocumentImage.ocr_status == "success")
            .filter(DocumentImage.created_at.isnot(None))
            .order_by(desc(DocumentImage.created_at))
            .limit(max(1, int(limit)))
            .all()
        )
    except Exception as exc:
        empty.diagnostics["image_resolution_mode"] = "skipped_db_error"
        empty.diagnostics["error"] = str(exc)[:200]
        logger.warning("image_resolver: query failed: %s", exc)
        return empty

    if not candidates:
        empty.diagnostics["image_resolution_mode"] = "no_ocr_candidates"
        return empty

    # Apply RBAC. Admin/sysadmin/dev-bypass paths will get every
    # candidate ID; non-admin users will only get the ones they own
    # or have been shared.
    candidate_doc_ids = [int(doc.id) for _img, doc in candidates]
    accessible_doc_ids = filter_documents_by_permission(
        auth, candidate_doc_ids, db=db
    )
    accessible_set = set(int(d) for d in (accessible_doc_ids or []))

    chosen_doc_id: Optional[int] = None
    chosen_image_id: Optional[int] = None
    chosen_filename: Optional[str] = None
    chosen_image_filename: Optional[str] = None
    chosen_ocr_status: Optional[str] = None
    chosen_created_at: Optional[datetime] = None
    considered_count = 0

    for image, document in candidates:
        considered_count += 1
        if int(document.id) not in accessible_set:
            continue
        chosen_doc_id = int(document.id)
        chosen_image_id = int(image.id)
        chosen_filename = document.original_name or document.filename
        chosen_image_filename = image.original_filename
        chosen_ocr_status = image.ocr_status
        chosen_created_at = image.created_at
        break

    if chosen_doc_id is None:
        return ResolvedImage(
            resolved=False,
            diagnostics={
                "image_resolution_mode": "rbac_denied",
                "candidate_count": len(candidates),
                "considered_count": considered_count,
            },
        )

    return ResolvedImage(
        resolved=True,
        document_id=chosen_doc_id,
        image_id=chosen_image_id,
        document_filename=chosen_filename,
        image_filename=chosen_image_filename,
        ocr_status=chosen_ocr_status,
        created_at=chosen_created_at,
        diagnostics={
            "image_resolution_mode": "recent",
            "candidate_count": len(candidates),
            "considered_count": considered_count,
            "resolved_document_id": chosen_doc_id,
            "resolved_image_id": chosen_image_id,
            "resolved_filename": chosen_filename,
        },
    )


def extract_explicit_target(image_context: Optional[Dict[str, Any]]) -> ResolvedImage:
    """Build a ResolvedImage from an explicit ``image_context`` dict
    supplied by the frontend or another caller.

    Returns ``resolved=False`` when the dict is empty / missing / has
    no usable IDs. The caller treats this as "no explicit target;
    fall back to recent-image resolution".
    """
    if not image_context or not isinstance(image_context, dict):
        return ResolvedImage(resolved=False, diagnostics={
            "image_resolution_mode": "none",
        })

    # Prefer document_id over image_id when both exist; this matches
    # the wider codebase's convention of scoping citations to the
    # document the user is in.
    doc_id = image_context.get("document_id")
    img_id = image_context.get("image_id")
    version_id = image_context.get("document_version_id")

    try:
        doc_id_int = int(doc_id) if doc_id is not None else None
    except Exception:
        doc_id_int = None
    try:
        img_id_int = int(img_id) if img_id is not None else None
    except Exception:
        img_id_int = None
    try:
        version_id_int = int(version_id) if version_id is not None else None
    except Exception:
        version_id_int = None

    if doc_id_int is None and img_id_int is None:
        return ResolvedImage(resolved=False, diagnostics={
            "image_resolution_mode": "none",
        })

    return ResolvedImage(
        resolved=True,
        document_id=doc_id_int,
        image_id=img_id_int,
        diagnostics={
            "image_resolution_mode": "explicit",
            "resolved_document_id": doc_id_int,
            "resolved_image_id": img_id_int,
            "resolved_document_version_id": version_id_int,
        },
    )


def recent_targets_from_image_context(
    image_context: Optional[Dict[str, Any]],
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Pull ``(document_id, image_id)`` tuples from
    ``image_context.recent_images``. Used by the routing layer as a
    preferred fallback when an explicit image context was supplied
    but the candidate chunks don't match it.
    """
    if not image_context or not isinstance(image_context, dict):
        return []
    recent = image_context.get("recent_images")
    if not isinstance(recent, list):
        return []
    out: List[Tuple[Optional[int], Optional[int]]] = []
    for entry in recent:
        if not isinstance(entry, dict):
            continue
        try:
            d = int(entry.get("document_id")) if entry.get("document_id") is not None else None
        except Exception:
            d = None
        try:
            i = int(entry.get("image_id")) if entry.get("image_id") is not None else None
        except Exception:
            i = None
        out.append((d, i))
    return out

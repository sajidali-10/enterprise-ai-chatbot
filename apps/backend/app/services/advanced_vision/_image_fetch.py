"""Phase 34D — Authorized image-bytes fetch helper.

Centralises the MinIO fetch path used by the advanced orchestrator
and the comparison helper. RBAC is enforced *before* any network
call to MinIO so an unauthorized image never produces a presigned
URL or a downloaded blob.

The helper is small and deliberately narrow. It is the only module
in Phase 34D that knows how to fetch image bytes — both the
advanced orchestrator and the comparison helper go through it.

The module is import-safe: every MinIO / DB call is wrapped, and
importing does not require the MinIO bucket to exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.core.config import settings
from app.security.permissions import can_access_document
from app.security.auth import AuthContext


logger = logging.getLogger(__name__)


@dataclass
class AuthorizedImage:
    """Result of an authorized image fetch.

    ``image_id`` is always set on success. ``document_id`` is the
    resolved parent document. ``bytes_`` is the raw image payload
    (PNG / JPEG / WEBP / TIFF / BMP / GIF) when the caller asks
    for bytes. ``mime_type`` mirrors the DocumentImage row.
    ``storage_key`` is the MinIO object key; it is NEVER returned
    on an unauthorized path.

    The helper deliberately does not carry an "error reason"
    outward. An unauthorized caller receives a falsy result; the
    only signal is ``None``. RBAC must never leak the existence of
    an unauthorized image.
    """

    image_id: int
    document_id: int
    mime_type: str
    storage_key: str
    bytes_: Optional[bytes] = None
    filename: Optional[str] = None


def fetch_authorized_image_bytes(
    db: Any,
    image_id: int,
    auth: Optional[AuthContext],
    *,
    require_bytes: bool = True,
    max_bytes: Optional[int] = None,
) -> Optional[AuthorizedImage]:
    """Resolve, authorize, and (optionally) fetch image bytes.

    Args:
        db: SQLAlchemy session (caller-owned).
        image_id: The ``DocumentImage.id`` to fetch.
        auth: AuthContext. ``None`` is treated as unauthorized.
        require_bytes: When True (default) the helper also pulls
            the bytes from MinIO. When False, only metadata is
            loaded (used for pre-flight checks / observability).
        max_bytes: Hard cap on downloaded bytes. Defaults to
            ``settings.ADVANCED_VISION_MAX_IMAGE_BYTES`` or
            ``settings.VISION_MAX_IMAGE_BYTES``.

    Returns:
        ``AuthorizedImage`` on success; ``None`` on any failure
        (image not found, unauthorized, MinIO error, oversize).
        The helper does NOT raise; callers can treat ``None`` as
        "skip this image" without inspecting a reason.

    RBAC guarantee:
        ``None`` is returned for both "image not found" and
        "image unauthorized". This prevents the existence of a
        private image from leaking to a non-owner.
    """
    if db is None or image_id is None or int(image_id) <= 0:
        return None
    if auth is None or not getattr(auth, "is_authenticated", False):
        return None

    # --- Gate 1: load DocumentImage row + check authorization ---
    try:
        from app.models.document import DocumentImage
        row = (
            db.query(DocumentImage)
            .filter(DocumentImage.id == int(image_id))
            .first()
        )
    except Exception as exc:
        logger.debug("advanced_vision: DocumentImage load failed for image_id=%s: %s", image_id, exc)
        return None

    if row is None:
        return None

    document_id = getattr(row, "document_id", None)
    if document_id is None:
        return None

    # Authorize BEFORE MinIO fetch. ``can_access_document`` returns
    # False for both "no row" and "no permission", which is exactly
    # the semantics we want — no existence leak.
    try:
        authorized = bool(can_access_document(auth, int(document_id), db=db))
    except Exception as exc:
        logger.debug("advanced_vision: can_access_document raised: %s", exc)
        return None
    if not authorized:
        return None

    mime_type = getattr(row, "mime_type", None) or "image/png"
    storage_key = getattr(row, "storage_key", None) or ""
    filename = getattr(row, "original_filename", None)

    if not require_bytes:
        return AuthorizedImage(
            image_id=int(image_id),
            document_id=int(document_id),
            mime_type=str(mime_type),
            storage_key=str(storage_key),
            bytes_=None,
            filename=filename,
        )

    cap = int(
        max_bytes
        if max_bytes is not None
        else (
            getattr(settings, "ADVANCED_VISION_MAX_IMAGE_BYTES", 0)
            or getattr(settings, "VISION_MAX_IMAGE_BYTES", 8 * 1024 * 1024)
        )
    )
    if cap <= 0:
        cap = 8 * 1024 * 1024

    # --- MinIO fetch ---
    try:
        from app.core.minio_client import get_minio_client
        client = get_minio_client()
    except Exception as exc:
        logger.debug("advanced_vision: minio client unavailable: %s", exc)
        return None

    try:
        response = client.get_object(settings.MINIO_BUCKET, storage_key)
        try:
            payload = response.read()
        finally:
            try:
                response.close()
            except Exception:
                pass
            try:
                response.release_conn()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("advanced_vision: minio get_object failed for image_id=%s: %s", image_id, exc)
        return None

    if not payload:
        return None
    if len(payload) > cap:
        # Refuse oversized downloads rather than truncating — silent
        # truncation could mislead the model into "seeing" a smaller
        # image than the user uploaded.
        return None

    return AuthorizedImage(
        image_id=int(image_id),
        document_id=int(document_id),
        mime_type=str(mime_type),
        storage_key=str(storage_key),
        bytes_=bytes(payload),
        filename=filename,
    )


def resolve_authorized_image_metadata(
    db: Any,
    image_id: int,
    auth: Optional[AuthContext],
) -> Optional[AuthorizedImage]:
    """Load authorized image metadata WITHOUT downloading bytes.

    Convenience wrapper around :func:`fetch_authorized_image_bytes`
    for callers that only need ``document_id`` / ``mime_type`` /
    ``storage_key`` (e.g. to compute a cache key).
    """
    return fetch_authorized_image_bytes(
        db, image_id, auth, require_bytes=False
    )


__all__ = [
    "AuthorizedImage",
    "fetch_authorized_image_bytes",
    "resolve_authorized_image_metadata",
]

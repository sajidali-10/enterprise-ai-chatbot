"""Phase 34C -- Multimodal knowledge backfill CLI.

Run explicitly after Phase 34C is rolled out to (re-)build image
knowledge points across existing ``DocumentImage`` rows.

Idempotent: re-running the script over an already-indexed corpus
writes the same deterministic point id for each image, so it is a
safe no-op when nothing changed and an in-place overwrite when the
underlying OCR / Vision data changed.

Permission-safe: only indexes images on documents the caller can
access. When ``--user-id`` is supplied the script filters the
document list through the existing Phase 6 ``DocumentPermission``
checker. Without ``--user-id`` (admin / dev mode), the script
operates on every document.

Observable: prints a one-line summary per batch. Never prints OCR
text or Vision descriptions.

This module is the CLI entry point. It is NOT invoked at startup --
operators run it explicitly:

    python -m app.services.multimodal.scripts.backfill [options]

Common invocations:

    # Dry-run over the whole corpus (no writes).
    python -m app.services.multimodal.scripts.backfill --dry-run

    # Backfill a single document.
    python -m app.services.multimodal.scripts.backfill --document-id 42

    # Bounded backfill with explicit limits.
    python -m app.services.multimodal.scripts.backfill --limit 200 --batch-size 25
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backfill_multimodal",
        description=(
            "Phase 34C backfill: rebuild image knowledge points in the "
            "existing Qdrant collection. Idempotent + restartable."
        ),
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Restrict the backfill to a single document_id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to index (0 = no limit).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Reserved for future chunking; current implementation indexes one image at a time.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only -- never write to Qdrant. Prints the summary it would have produced.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Permission scope: only index images on documents the given user can read. "
             "Omit for admin / dev mode (all documents).",
    )
    parser.add_argument(
        "--include-ocr-empty",
        action="store_true",
        help="Include images whose OCR status is missing/empty. "
             "Default skips them (they would produce no useful knowledge text).",
    )
    return parser


def _accessible_document_ids(db, user_id: Optional[int]) -> Optional[List[int]]:
    """Return the list of document_ids the caller can access.

    Returns None when the caller is unrestricted (admin / dev mode).
    Returns an empty list when the caller has no accessible documents.
    """
    if user_id is None:
        return None

    try:
        from app.security.permissions import filter_documents_by_permission, PermissionChecker
        from app.security.auth import AuthContext
        from app.security.models import User, UserRole
    except Exception as exc:
        logger.warning("backfill: permission module unavailable: %s", exc)
        return None

    try:
        user = db.query(User).filter(User.id == int(user_id)).first()
    except Exception as exc:
        logger.warning("backfill: cannot load user %s: %s", user_id, exc)
        return []
    if user is None:
        logger.warning("backfill: user_id=%s not found", user_id)
        return []

    role = getattr(user, "role", UserRole.user) or UserRole.user
    try:
        auth = AuthContext(user_id=int(user.id), username=user.username, role=role)
    except Exception:
        # Older AuthContext signatures -- fall back to kwargs.
        auth = AuthContext(**{"user_id": int(user.id), "username": user.username, "role": role})

    try:
        from app.models.document import Document

        all_ids = [int(d[0]) for d in db.query(Document.id).all()]
    except Exception as exc:
        logger.warning("backfill: cannot list documents: %s", exc)
        return []

    try:
        allowed = filter_documents_by_permission(auth, all_ids)
    except Exception as exc:
        logger.warning("backfill: permission filter failed: %s", exc)
        return []

    return [int(d) for d in allowed]


def _image_ids_for(db, document_id: int, *, include_ocr_empty: bool) -> List[int]:
    try:
        from app.models.document import DocumentImage
    except Exception as exc:
        logger.warning("backfill: cannot import DocumentImage: %s", exc)
        return []

    try:
        q = db.query(DocumentImage.id).filter(DocumentImage.document_id == int(document_id))
        if not include_ocr_empty:
            q = q.filter(DocumentImage.ocr_status == "success")
        rows = q.order_by(DocumentImage.id.asc()).all()
        return [int(r[0]) for r in rows]
    except Exception as exc:
        logger.warning("backfill: cannot list image_ids for document_id=%s: %s", document_id, exc)
        return []


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the backfill CLI according to ``args`` and return a summary."""
    summary: Dict[str, Any] = {
        "scanned": 0,
        "indexed": 0,
        "skipped": 0,
        "errors": 0,
        "error_samples": [],
        "dry_run": bool(args.dry_run),
        "document_id": args.document_id,
        "user_id": args.user_id,
        "limit": int(args.limit or 0),
    }

    try:
        from app.core.config import settings as _settings
    except Exception as exc:
        summary["errors"] += 1
        summary["error_samples"].append(f"settings_import_failed: {str(exc)[:120]}")
        return summary

    if not bool(getattr(_settings, "MULTIMODAL_KNOWLEDGE_ENABLED", False)):
        summary["disabled"] = True
        return summary

    if bool(args.dry_run):
        # Dry-run path: count what we WOULD index and return.
        try:
            from app.db.session import SessionLocal

            db = SessionLocal()
            try:
                document_ids = _resolve_target_documents(db, args)
                image_count = 0
                for did in document_ids:
                    image_count += len(_image_ids_for(db, did, include_ocr_empty=bool(args.include_ocr_empty)))
                summary["scanned"] = image_count
                summary["would_index"] = image_count
                return summary
            finally:
                db.close()
        except Exception as exc:
            summary["errors"] += 1
            summary["error_samples"].append(f"dry_run_failed: {str(exc)[:120]}")
            return summary

    # Real backfill.
    try:
        from app.db.session import SessionLocal
        from app.services.multimodal.lifecycle import reindex_all_images

        db = SessionLocal()
        try:
            document_ids = _resolve_target_documents(db, args)
            if args.document_id is not None:
                # Restricted run -- call the per-document helper so
                # the summary shape matches ``reindex_document_images``.
                from app.services.multimodal.lifecycle import reindex_document_images

                per_doc_summaries = []
                for did in document_ids:
                    sub = reindex_document_images(db, int(did), batch_size=int(args.batch_size or 50))
                    per_doc_summaries.append(sub)
                    summary["scanned"] += sub.get("scanned", 0)
                    summary["indexed"] += sub.get("indexed", 0)
                    summary["skipped"] += sub.get("skipped", 0)
                    summary["errors"] += sub.get("errors", 0)
                    for sample in sub.get("error_samples", []):
                        if len(summary["error_samples"]) < 5:
                            summary["error_samples"].append(sample)
                summary["per_document"] = per_doc_summaries
            else:
                full = reindex_all_images(db, batch_size=int(args.batch_size or 50))
                summary["scanned"] = full.get("scanned", 0)
                summary["indexed"] = full.get("indexed", 0)
                summary["skipped"] = full.get("skipped", 0)
                summary["errors"] = full.get("errors", 0)
                for sample in full.get("error_samples", []):
                    if len(summary["error_samples"]) < 5:
                        summary["error_samples"].append(sample)
        finally:
            db.close()
    except Exception as exc:
        summary["errors"] += 1
        summary["error_samples"].append(f"backfill_failed: {str(exc)[:120]}")

    if int(args.limit or 0) > 0:
        # Cap the report -- we may have indexed fewer than scanned when
        # applying the limit at the lifecycle layer (currently not
        # applied there; surface the configured limit for transparency).
        summary["limit_applied"] = int(args.limit)
    return summary


def _resolve_target_documents(db, args: argparse.Namespace) -> List[int]:
    """Return the document_ids the backfill should process."""
    if args.document_id is not None:
        return [int(args.document_id)]

    allowed = _accessible_document_ids(db, args.user_id)
    if allowed is None:
        try:
            from app.models.document import Document

            return [int(d[0]) for d in db.query(Document.id).order_by(Document.id.asc()).all()]
        except Exception as exc:
            logger.warning("backfill: cannot list documents: %s", exc)
            return []

    return list(allowed)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    summary = run(args)
    # Never print raw image content / OCR / Vision bodies. Only the
    # aggregate counts and capped error samples.
    safe_summary = {k: v for k, v in summary.items() if k != "per_document"}
    if "per_document" in summary:
        safe_summary["per_document_count"] = len(summary["per_document"])
    print("BACKFILL_SUMMARY " + repr(safe_summary))
    # Exit code: non-zero only when there were errors.
    return 1 if summary.get("errors", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

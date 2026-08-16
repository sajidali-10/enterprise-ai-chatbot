"""
Phase 34A.1 — Image-Aware Query Routing

Distinguishes between two intents expressed by the user:

A. IMAGE CONTENT QUESTION
   "What error code is shown in the image I uploaded?"
   "What does the screenshot say?"
   For these, retrieval is scoped to OCR-derived chunks (image_ocr,
   pdf_page_ocr, docx_image_ocr) associated with the current or most
   recently referenced upload. Unrelated KB sources are excluded.

B. TROUBLESHOOTING / KNOWLEDGE QUESTION
   "What does error 902 mean?"
   "How do I troubleshoot the error shown in the screenshot?"
   For these, retrieval broadens to the full knowledge base but the
   OCR chunk containing the identifier is strongly boosted so it
   surfaces in citations alongside the KB documentation.

No vision model is used. No external API calls. Routing decisions are
made from the query text alone (via `query_analysis`) and from any
`image_context` the chat endpoint forwards into the pipeline.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.rag.query_analysis import QueryAnalysis


# OCR / image-derived source_type values used by Phase 34A ingestion.
# The values come from `document_images.source_type` and are mirrored
# on each chunk's metadata via `content_type` / `source_type`.
IMAGE_SOURCE_TYPES = frozenset({
    "image_ocr",
    "pdf_ocr",
    "pdf_page_ocr",
    "docx_image_ocr",
    # Legacy aliases used by earlier ingestion paths.
    "image",
    "pdf_page",
    "docx_image",
})


# Image file extensions we recognise when classifying legacy chunks
# from their `source_file_name` (legacy chunks do not have a
# `source_type` payload field). Conservative — only the most common.
_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif")

# Phase 34A legacy content prefixes produced by the image/PDF/DOCX
# parsers. These are stable strings and are the safest signal that a
# chunk was OCR-derived. We deliberately do NOT match arbitrary
# "OCR Text:" text inside native documents — only when it appears as
# the structured prefix produced by the parsers.
_LEGACY_OCR_CONTENT_PREFIXES = (
    "Source Type: Image\nOCR Text:",
    "Source Type: PDF Image\nOCR Text:",
    "Source Type: DOCX Image\nOCR Text:",
)

# Some legacy ingestion paths prefix the file name (not just the body)
# with the same kind of marker. We treat those consistently.
_LEGACY_OCR_TITLE_PREFIXES = (
    "[Image OCR]",
    "[PDF OCR]",
    "[DOCX OCR]",
)


def _looks_like_legacy_ocr_chunk(chunk: dict) -> bool:
    """Conservative fallback for chunks indexed BEFORE Phase 34A.1.1.

    Chunks indexed after this fix carry an explicit ``source_type`` in
    their Qdrant payload (see qdrant_service.upsert_chunks). Chunks
    indexed earlier do not — they may still be OCR-derived but
    only carry an OCR-text body prefixed with ``Source Type: Image``
    or an image-extension filename.

    Returns True only for clearly identifiable image-derived chunks.
    MUST NOT classify arbitrary text documents as images.
    """
    if not chunk:
        return False

    content = chunk.get("content") or ""
    if isinstance(content, str):
        for prefix in _LEGACY_OCR_CONTENT_PREFIXES:
            if content.startswith(prefix):
                return True

    title = chunk.get("title") or ""
    if isinstance(title, str):
        for prefix in _LEGACY_OCR_TITLE_PREFIXES:
            if title.startswith(prefix):
                return True

    fname = (chunk.get("source_file_name") or "").lower()
    if fname and fname.endswith(_IMAGE_FILE_EXTENSIONS):
        # Filename alone is suggestive but not conclusive — the body
        # must look OCR-shaped (either prefix above, or substantive
        # OCR text). Without any other signal we leave the chunk
        # un-classified so text documents that happen to be named
        # ``screenshot.txt`` do not get pulled into image routing.
        body = (chunk.get("content") or "")
        if isinstance(body, str) and body.lstrip().startswith(("Source Type:", "OCR Text:", "Failed Reason:", "Error Code:", "Message:")):
            return True

    # Section headings written by some PDF parsers (e.g. "Page 3 - image")
    section = chunk.get("section_heading") or ""
    if isinstance(section, str) and re.search(r"\bimage\b", section, re.IGNORECASE):
        if isinstance(content, str) and "OCR" in content:
            return True

    return False


def is_image_source_chunk(chunk: dict) -> bool:
    """Return True if a chunk's metadata indicates OCR/image origin.

    Phase 34A.1.1 — combines two signals:

    1. Structured metadata (``source_type`` / ``content_type`` payload
       field set by ``qdrant_service.upsert_chunks``).
    2. Legacy fallback for chunks indexed before this fix
       (``_looks_like_legacy_ocr_chunk``).
    """
    if not chunk:
        return False
    candidates = [
        chunk.get("source_type"),
        chunk.get("content_type"),
        chunk.get("metadata", {}).get("source_type") if isinstance(chunk.get("metadata"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, str) and c.lower() in IMAGE_SOURCE_TYPES:
            return True
    return _looks_like_legacy_ocr_chunk(chunk)


# Toggle to disable the legacy fallback without touching call sites.
# Defaults to enabled so existing OCR content is immediately
# routable; operators can flip this off once all OCR documents have
# been reindexed (see docs/34a1_1_reindex.md).
LEGACY_OCR_FALLBACK_ENABLED = os.getenv("RAG_LEGACY_OCR_FALLBACK_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}


def _is_image_source_with_fallback(chunk: dict) -> bool:
    """Wrapper that respects the runtime toggle for the legacy fallback."""
    if not chunk:
        return False
    candidates = [
        chunk.get("source_type"),
        chunk.get("content_type"),
        chunk.get("metadata", {}).get("source_type") if isinstance(chunk.get("metadata"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, str) and c.lower() in IMAGE_SOURCE_TYPES:
            return True
    if LEGACY_OCR_FALLBACK_ENABLED:
        return _looks_like_legacy_ocr_chunk(chunk)
    return False


def references_uploaded_image(query: str) -> bool:
    """Thin convenience wrapper — kept here for backward-compat with
    callers that import `references_uploaded_image` from this module."""
    from app.rag.query_analysis import references_uploaded_image as _r
    return _r(query)


def select_image_aware_chunks(
    chunks: List[dict],
    analysis: Optional[QueryAnalysis],
    image_context: Optional[Dict[str, Any]] = None,
    image_source_types: Optional[set] = None,
) -> Tuple[List[dict], dict]:
    """Apply image-aware routing to a chunk list.

    Args:
        chunks: Candidate chunks from retrieval. Not mutated.
        analysis: The `QueryAnalysis` for the user query.
        image_context: Optional dict supplied by the chat endpoint
            describing the uploaded image in play. Recognised keys:
              - image_id (int|str)
              - document_id (int|str)
              - document_version_id (int|str)
              - recent_images (list[dict]): most-recently-uploaded images,
                used as a fallback when image_id is unspecified.
            When this dict is absent or empty, image-source scoping is
            still attempted if the query looks image-content related.
        image_source_types: Override set of source_types considered
            image-derived. Defaults to `IMAGE_SOURCE_TYPES`.

    Returns:
        Tuple of (selected chunks, routing metadata dict).

    Routing rules:

    * If the query references an uploaded image AND no diagnostic
      identifier is present (analysis.query_type == "image_content"):
      - Return ONLY image-source chunks.
      - Prefer chunks whose image_id / document_id matches the
        supplied image_context. If none match, fall back to any
        image-source chunks.

    * If the query references an uploaded image AND an identifier is
      present (analysis.query_type == "error_lookup" with image
      context):
      - Keep all chunks. The hybrid retriever already boosts
        exact-match chunks. This function records the routing
        decision for diagnostics but does NOT drop chunks.

    * If the query does not reference an uploaded image:
      - Pass-through. No scoping applied.
    """
    if not chunks:
        # Distinguish two empty-list cases for observability:
        #   * no image reference in the query → no routing attempted
        #   * the user asked an image-content question but retrieval
        #     produced zero chunks (likely the OCR/text index is empty).
        if analysis is not None and analysis.references_uploaded_image:
            return [], {
                "routing_mode": "image_content_no_ocr_available",
                "image_context_used": bool(image_context),
                "scoped_count": 0,
                "kept_count": 0,
                "dropped_count": 0,
            }
        return [], {
            "routing_mode": "passthrough",
            "image_context_used": False,
            "scoped_count": 0,
            "kept_count": 0,
            "dropped_count": 0,
        }

    source_types = image_source_types or IMAGE_SOURCE_TYPES
    references_image = bool(analysis and analysis.references_uploaded_image)
    # Phase 34A.1.1 — branch on the canonical `query_type` set by
    # `app.rag.query_analysis`. `query_type == "image_content"` is the
    # ONLY signal that the user is asking about the contents of an
    # uploaded image without any other identifier or troubleshooting
    # framing. Branching on `has_identifiers()` alone misclassified
    # questions like "How do I troubleshoot the error shown in the
    # image?" as image_content (no extracted code/term) when they
    # should broaden to the KB.
    query_type = (analysis.query_type if analysis else "general") or "general"
    is_image_content = query_type == "image_content"

    if not references_image:
        return list(chunks), {
            "routing_mode": "passthrough",
            "image_context_used": False,
            "scoped_count": 0,
            "kept_count": len(chunks),
            "dropped_count": 0,
        }

    # Case A — image-content question.
    if is_image_content:
        # Phase 34A.1.1: use the legacy-fallback-aware classifier so
        # chunks indexed before this fix are still scoped correctly.
        image_chunks = [c for c in chunks if _is_image_source_with_fallback(c)]
        if not image_chunks:
            # No image chunks present — leave the original list alone
            # but record the routing decision so observability can see
            # the user's intent was image-content.
            return list(chunks), {
                "routing_mode": "image_content_no_ocr_available",
                "image_context_used": bool(image_context),
                "scoped_count": 0,
                "kept_count": len(chunks),
                "dropped_count": 0,
            }

        # Phase 34A.1.1 — strict scoping when explicit image context is
        # supplied. If the caller (the chat endpoint or the frontend)
        # knows which image/document the user is referring to, we
        # restrict citations to chunks from THAT image/document.
        # Unrelated image chunks from previous uploads are dropped to
        # avoid pulling in stale historical images when the user asks
        # about the one they just uploaded.
        target_ids = _extract_target_ids(image_context)
        has_explicit_target = bool(target_ids)

        if has_explicit_target:
            preferred = [c for c in image_chunks if _chunk_matches_targets(c, target_ids)]
            if preferred:
                selected = preferred
                routing_mode = "image_content_scoped_explicit"
            else:
                # Explicit context was supplied but no chunk matches
                # it. This can happen if the user's uploaded image was
                # OCR'd but the resulting chunk did not carry the
                # image_id/document_id we expected. Fall back to the
                # most-recently-uploaded image from image_context so we
                # never serve a stale image from a previous upload.
                recent_target_ids = _extract_recent_image_targets(image_context)
                if recent_target_ids:
                    preferred_recent = [
                        c for c in image_chunks
                        if _chunk_matches_targets(c, recent_target_ids)
                    ]
                    if preferred_recent:
                        selected = preferred_recent
                        routing_mode = "image_content_scoped_recent"
                    else:
                        # Last-resort safety net: scope to the single
                        # most recent image-derived chunk in this
                        # retrieval batch. Better to answer with the
                        # closest available match than with an
                        # arbitrary historical image.
                        selected = image_chunks[:1]
                        routing_mode = "image_content_scoped_fallback"
                else:
                    selected = image_chunks[:1]
                    routing_mode = "image_content_scoped_fallback"
        else:
            # No explicit image context — keep all image-derived
            # chunks so the user sees every relevant OCR they uploaded.
            selected = image_chunks
            routing_mode = "image_content_scoped"

        return selected, {
            "routing_mode": routing_mode,
            "image_context_used": bool(image_context),
            "scoped_count": len(selected),
            "kept_count": len(selected),
            "dropped_count": len(chunks) - len(selected),
            "target_ids": list(target_ids),
        }

    # Case B — troubleshooting question with image context.
    # Keep all chunks. The exact-match booster inside the hybrid
    # retriever will rank the OCR chunk appropriately.
    return list(chunks), {
        "routing_mode": "troubleshooting_with_image_context",
        "image_context_used": bool(image_context),
        "scoped_count": 0,
        "kept_count": len(chunks),
        "dropped_count": 0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_in_sources(chunk: dict, source_types: set) -> bool:
    if not chunk:
        return False
    candidates = [
        chunk.get("source_type"),
        chunk.get("content_type"),
        chunk.get("metadata", {}).get("source_type") if isinstance(chunk.get("metadata"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, str) and c.lower() in source_types:
            return True
    return False


def _extract_target_ids(image_context: Optional[Dict[str, Any]]) -> set:
    if not image_context:
        return set()
    ids: set = set()
    # Phase 34A.1.1 — when the frontend supplies an explicit
    # `document_id` / `image_id` for the in-scope image, prefer that.
    # The legacy order preserved `image_id` first; we keep that.
    for key in ("image_id", "document_id", "document_version_id"):
        v = image_context.get(key)
        if v is not None:
            ids.add(str(v))
    return ids


def _extract_recent_image_targets(image_context: Optional[Dict[str, Any]]) -> set:
    """Pull id targets from `recent_images` entries, if the frontend
    supplied them. Used as a safer fallback than ``image_chunks[:1]``
    when an explicit target is missing.
    """
    if not image_context:
        return set()
    recent = image_context.get("recent_images")
    if not isinstance(recent, list):
        return set()
    ids: set = set()
    for entry in recent:
        if not isinstance(entry, dict):
            continue
        for key in ("image_id", "document_id", "document_version_id"):
            v = entry.get(key)
            if v is not None:
                ids.add(str(v))
    return ids


def _chunk_matches_targets(chunk: dict, target_ids: set) -> bool:
    if not target_ids:
        return False
    candidates = [
        chunk.get("image_id"),
        chunk.get("document_id"),
        chunk.get("document_version_id"),
        chunk.get("metadata", {}).get("image_id") if isinstance(chunk.get("metadata"), dict) else None,
        chunk.get("metadata", {}).get("document_id") if isinstance(chunk.get("metadata"), dict) else None,
        chunk.get("metadata", {}).get("document_version_id") if isinstance(chunk.get("metadata"), dict) else None,
    ]
    for c in candidates:
        if c is None:
            continue
        if str(c) in target_ids:
            return True
    return False

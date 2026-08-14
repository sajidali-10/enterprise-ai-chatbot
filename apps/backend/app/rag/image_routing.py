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

from typing import Any, Dict, List, Optional, Tuple

from app.rag.query_analysis import QueryAnalysis


# OCR / image-derived source_type values used by Phase 34A ingestion.
# The values come from `document_images.source_type` and are mirrored
# on each chunk's metadata via `content_type` / `source_type`.
IMAGE_SOURCE_TYPES = frozenset({
    "image_ocr",
    "pdf_page_ocr",
    "docx_image_ocr",
    # Some ingestion paths set content_type differently. We also accept
    # the broader Phase 34A identifiers.
    "image",
    "pdf_page",
    "docx_image",
})


def is_image_source_chunk(chunk: dict) -> bool:
    """Return True if a chunk's metadata indicates OCR/image origin."""
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
    has_identifier = bool(analysis and analysis.has_identifiers())

    if not references_image:
        return list(chunks), {
            "routing_mode": "passthrough",
            "image_context_used": False,
            "scoped_count": 0,
            "kept_count": len(chunks),
            "dropped_count": 0,
        }

    # Case A — image-content question.
    if not has_identifier:
        image_chunks = [c for c in chunks if _chunk_in_sources(c, source_types)]
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

        # Prefer image_context-matching chunks when available, but keep
        # ALL image chunks (not just the matching ones). This is the
        # behaviour the spec calls for: an image-content question is
        # scoped to image-source chunks only; the user's "current"
        # image_context is used to boost ordering rather than to
        # exclude other image chunks from the same session.
        target_ids = _extract_target_ids(image_context)
        preferred = [c for c in image_chunks if _chunk_matches_targets(c, target_ids)]
        non_preferred = [c for c in image_chunks if c not in preferred]
        selected = preferred + non_preferred

        return selected, {
            "routing_mode": "image_content_scoped",
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
    for key in ("image_id", "document_id", "document_version_id"):
        v = image_context.get(key)
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
    ]
    for c in candidates:
        if c is None:
            continue
        if str(c) in target_ids:
            return True
    return False

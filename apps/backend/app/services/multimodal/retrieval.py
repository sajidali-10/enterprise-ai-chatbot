"""Phase 34C -- Multimodal retrieval overlay.

Adds an optional second pass over the existing Qdrant collection that
selects ``source_type=image_knowledge`` candidates and merges them
into the retrieval result. The pass is a transparent overlay -- it
NEVER bypasses the existing Phase 34A / 34B / hybrid retrieval. It
runs AFTER relevance thresholding and BEFORE the RBAC permission
filter (which still removes image knowledge for documents the user
cannot access).

Public entry point:

    integrate_image_knowledge(
        query, base_chunks, analysis, ...,
    ) -> (chunks, metadata)

Behaviour matrix:

    enabled=False                -> returns base_chunks unchanged
    base_chunks empty             -> returns base_chunks unchanged
    no image_knowledge candidates -> returns base_chunks unchanged
    query has image-intent        -> image candidates get a boost
    query is product-meaning      -> image candidates sorted below KB
                                       (explicit demote + cap)
    RBAC:                         -> relies on the existing
                                       `filter_documents_by_permission`
                                       helper applied later

The function is intentionally pure -- no DB writes, no provider calls
beyond the existing embedding provider + Qdrant client.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.multimodal.config import MULTIMODAL_SOURCE_TYPE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Historical image-intent phrases
# ---------------------------------------------------------------------------
#
# These phrases signal that the user wants to find a previously
# uploaded image (NOT the image attached to the current chat). When
# they match, we treat the query as image-intent for the purposes of
# the image-knowledge retrieval boost (NOT for the OCR-scoping path
# used by Phase 34A.1.2 -- that path still requires an explicit
# image_context or the existing image-content phrase list).
#
# Keeping this list separate from
# ``app.rag.query_analysis._IMAGE_INTENT_PATTERNS`` is deliberate: we
# want it to ONLY boost historical image retrieval, not flip the
# query into ``query_type=="image_content"`` (which would scope the
# OCR routing and hide KB answers).

_HISTORICAL_IMAGE_INTENT_PATTERNS = [
    re.compile(r"\bwhich\s+(?:screenshot|image|diagram|picture|photo|attachment|figure)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:screenshot|image|diagram|picture)\s+(?:showed|shows|show|contained|contains|displayed)\b", re.IGNORECASE),
    # "Show me the screenshot", "Find me an image", "Locate the picture"
    re.compile(r"\b(?:find|show|search|locate|see)\s+(?:me\s+)?(?:the\s+|a\s+|an\s+|any\s+)?(?:screenshot|image|diagram|picture|photo)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+we\s+have\s+(?:a|an|any)\s+(?:screenshot|image|diagram|picture)\b", re.IGNORECASE),
    re.compile(r"\bprevious(?:ly)?\s+(?:uploaded\s+)?(?:screenshot|image|diagram|picture|photo)\b", re.IGNORECASE),
    re.compile(r"\b(?:a|the)\s+dashboard\s+(?:showing|with|of|where)\b", re.IGNORECASE),
    re.compile(r"\b(?:a|the)\s+diagram\s+(?:showing|with|of|where)\b", re.IGNORECASE),
    re.compile(r"\b(?:image|screenshot|screen|photo)\s+where\b", re.IGNORECASE),
    re.compile(r"\bany\s+(?:image|screenshot)\s+(?:showing|with|of)\b", re.IGNORECASE),
]


def has_historical_image_intent(query: str) -> bool:
    """Return True if the query asks about previously uploaded images.

    Pure helper -- no side effects. Used by the retrieval overlay to
    decide whether to boost ``image_knowledge`` candidates. Returns
    True when one of the historical phrases matches.
    """
    if not query:
        return False
    for pat in _HISTORICAL_IMAGE_INTENT_PATTERNS:
        if pat.search(query):
            return True
    return False


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def _settings():
    from app.core.config import settings

    return settings


def _is_enabled() -> bool:
    try:
        return bool(_settings().MULTIMODAL_KNOWLEDGE_ENABLED)
    except Exception:
        return False


def _intent_boost() -> float:
    try:
        return float(_settings().MULTIMODAL_IMAGE_INTENT_BOOST)
    except Exception:
        return 0.25


def _max_image_sources() -> int:
    try:
        return int(_settings().MULTIMODAL_MAX_IMAGE_SOURCES)
    except Exception:
        return 2


# ---------------------------------------------------------------------------
# Query analysis glue
# ---------------------------------------------------------------------------


def _is_product_meaning_query(analysis: Any) -> bool:
    """Return True for queries that want authoritative product knowledge.

    Examples: "What does error 902 mean?", "How do I fix the failure?".

    These queries should keep image knowledge available as supporting
    evidence but the authoritative KB chunk must outrank them. The
    retrieval layer applies an explicit demote + the
    ``MULTIMODAL_MAX_IMAGE_SOURCES`` cap.
    """
    if analysis is None:
        return False
    # Image-intent or image-content queries are explicitly NOT product
    # meaning -- the user wants the image.
    if getattr(analysis, "references_uploaded_image", False):
        return False
    if getattr(analysis, "query_type", "") == "image_content":
        return False
    has_id = bool(getattr(analysis, "error_codes", None) or getattr(analysis, "technical_terms", None))
    strong_troubleshooting = bool(getattr(analysis, "raw_signals", {}).get("strong_troubleshooting_matched"))
    return has_id or strong_troubleshooting


# ---------------------------------------------------------------------------
# Qdrant search
# ---------------------------------------------------------------------------


def _search_image_knowledge(query: str, *, limit: int) -> List[Dict[str, Any]]:
    """Vector search Qdrant for ``source_type=image_knowledge`` points only.

    Reuses the existing embedding provider. Returns chunks in the
    same shape as the base hybrid retrieval but with extra fields
    populated for the citation layer.
    """
    if not query or limit <= 0:
        return []
    try:
        from app.services.embeddings import get_embedding_provider
    except Exception as exc:
        logger.debug("multimodal: embedding import failed: %s", exc)
        return []
    try:
        provider = get_embedding_provider()
        embedding = provider.embed([query])[0]
    except Exception as exc:
        logger.warning("multimodal: embedding failed in retrieval overlay: %s", exc)
        return []

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from app.services.vector.qdrant_service import get_qdrant_client
    except Exception as exc:
        logger.debug("multimodal: qdrant import failed: %s", exc)
        return []

    collection = str(_settings().QDRANT_COLLECTION)
    flt = Filter(must=[
        FieldCondition(key="source_type", match=MatchValue(value=MULTIMODAL_SOURCE_TYPE)),
    ])
    try:
        client = get_qdrant_client()
        results = client.search(
            collection_name=collection,
            query_vector=embedding,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        logger.warning("multimodal: Qdrant search failed in retrieval overlay: %s", exc)
        return []

    chunks: List[Dict[str, Any]] = []
    for hit in results or []:
        payload = getattr(hit, "payload", None) or {}
        chunks.append({
            "chunk_id": str(getattr(hit, "id", "")),
            "point_id": getattr(hit, "id", None),
            "document_id": payload.get("document_id"),
            "document_version_id": payload.get("document_version_id"),
            "image_id": payload.get("image_id"),
            "chunk_index": payload.get("chunk_index"),
            "content": payload.get("content") or payload.get("knowledge_text") or "",
            "knowledge_text": payload.get("knowledge_text") or "",
            "source_file_name": payload.get("source_file_name") or "",
            "title": payload.get("title") or payload.get("source_file_name") or "",
            "source_type": payload.get("source_type"),
            "content_type": payload.get("content_type"),
            "mime_type": payload.get("mime_type"),
            "image_type": payload.get("image_type"),
            "vision_provider": payload.get("vision_provider"),
            "vision_model": payload.get("vision_model"),
            "vision_processed_at": payload.get("vision_processed_at"),
            "owner_user_id": payload.get("owner_user_id"),
            "visibility": payload.get("visibility"),
            "is_ocr": payload.get("is_ocr"),
            "has_vision": payload.get("has_vision"),
            "knowledge_schema_version": payload.get("knowledge_schema_version"),
            "score": float(getattr(hit, "score", 0.0) or 0.0),
            "_retrieval_channel": "image_knowledge_vector",
            "_knowledge_kind": "image",
        })
    return chunks


# ---------------------------------------------------------------------------
# Boosting & sorting
# ---------------------------------------------------------------------------


def _apply_intent_boost(
    chunks: List[Dict[str, Any]],
    *,
    boost: float,
    image_intent: bool,
) -> List[Dict[str, Any]]:
    """Apply the multiplicative image-intent boost to image-knowledge chunks."""
    if not image_intent or boost <= 0:
        return chunks
    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("_knowledge_kind") == "image":
            new_chunk = dict(chunk)
            base = float(new_chunk.get("score") or 0.0)
            new_chunk["score"] = base * (1.0 + float(boost))
            new_chunk["_image_intent_boost_applied"] = float(boost)
            out.append(new_chunk)
        else:
            out.append(chunk)
    return out


def _apply_authority_demote(
    chunks: List[Dict[str, Any]],
    *,
    enabled: bool,
) -> List[Dict[str, Any]]:
    """Sort image-knowledge chunks below KB chunks for product-meaning queries.

    The demote is a stable secondary sort by ``_knowledge_kind``: KB
    chunks (``_knowledge_kind != "image"``) come first, image
    knowledge last. Within each group the original order (by ``score``
    desc) is preserved.
    """
    if not enabled:
        return chunks
    # Stable sort by (is_image, -score) preserves the score ordering
    # inside each group while pushing image knowledge to the bottom.
    def _sort_key(c: Dict[str, Any]) -> Tuple[int, float]:
        is_image = 1 if c.get("_knowledge_kind") == "image" else 0
        return (is_image, -float(c.get("score") or 0.0))

    return sorted(chunks, key=_sort_key)


def _apply_image_cap(
    chunks: List[Dict[str, Any]],
    *,
    max_image_sources: int,
) -> List[Dict[str, Any]]:
    """Drop excess image-knowledge chunks beyond ``max_image_sources``.

    The cap applies to the FINAL list. Image-knowledge chunks are
    removed from the tail (lowest score) first so the strongest
    image evidence survives.
    """
    if max_image_sources < 0:
        return chunks
    image_idxs = [i for i, c in enumerate(chunks) if c.get("_knowledge_kind") == "image"]
    if len(image_idxs) <= max_image_sources:
        return chunks
    keep = set(image_idxs[:max_image_sources])
    # Drop excess image-knowledge chunks from the tail of the list
    # (the demote pass pushed them to the bottom already; if not, we
    # still keep the highest-scoring ones).
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks):
        if c.get("_knowledge_kind") == "image" and i not in keep:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def integrate_image_knowledge(
    query: str,
    base_chunks: Sequence[Dict[str, Any]],
    analysis: Any,
    *,
    image_context: Optional[Dict[str, Any]] = None,
    candidate_limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Overlay image knowledge candidates onto an existing retrieval result.

    Args:
        query: The user query (already rewritten by the retriever).
        base_chunks: The chunks returned by the hybrid retriever after
            relevance thresholding. Not mutated.
        analysis: The ``QueryAnalysis`` produced by
            ``app.rag.query_analysis.analyze_query``.
        image_context: Optional frontend-supplied image context. Not
            used for source scoping here (Phase 34A.1.2 already does
            that); used only as an intent hint.
        candidate_limit: Number of image-knowledge candidates to
            fetch. Defaults to ``settings.RETRIEVAL_VECTOR_TOP_K`` or
            ``5`` whichever is smaller.

    Returns:
        Tuple ``(merged_chunks, metadata)``. ``merged_chunks`` is a new
        list with image knowledge appended / boosted / capped /
        demoted. ``metadata`` is a diagnostics dict for LangSmith +
        the response debug payload.

    The function NEVER raises on transient errors -- it falls back to
    the base chunks and records ``enabled=False`` or
    ``error=...`` in the metadata.
    """
    metadata: Dict[str, Any] = {
        "enabled": _is_enabled(),
        "knowledge_kind": MULTIMODAL_SOURCE_TYPE,
        "multimodal_candidates_retrieved": 0,
        "image_knowledge_candidates": 0,
        "image_knowledge_selected": 0,
        "image_knowledge_boost_applied": 0.0,
        "image_knowledge_authority_demote_applied": False,
        "image_knowledge_source_ids": [],
        "multimodal_index_version": _settings().MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION,
        "image_intent_matched": False,
        "product_meaning_matched": _is_product_meaning_query(analysis),
    }

    if not _is_enabled():
        return list(base_chunks), metadata

    # NOTE: an empty base_chunks is NOT an early exit. Phase 34C's
    # primary use case is historical image retrieval -- the user
    # asks "Which screenshot showed Server B failing?" with no
    # attached image and no KB context. The hybrid retriever may
    # return zero text chunks; we still want to search Qdrant for
    # image knowledge. Skipping the search here would defeat the
    # whole feature.

    # Decide the intent: in-scope image reference (analysis), explicit
    # image_context, OR a historical phrase in the query.
    image_intent = bool(getattr(analysis, "references_uploaded_image", False)) or bool(image_context) or has_historical_image_intent(query)
    metadata["image_intent_matched"] = bool(image_intent)

    # Pull candidates from Qdrant. Limit respects the configured
    # vector top_k but is capped to keep the overlay cheap.
    try:
        if candidate_limit is None:
            try:
                topk = int(_settings().RETRIEVAL_VECTOR_TOP_K)
            except Exception:
                topk = 15
            candidate_limit = min(max(5, topk), 20)
        candidates = _search_image_knowledge(query, limit=int(candidate_limit))
    except Exception as exc:
        metadata["error"] = f"overlay_search_failed: {str(exc)[:120]}"
        logger.warning("multimodal: overlay search failed: %s", exc)
        return list(base_chunks), metadata

    metadata["multimodal_candidates_retrieved"] = len(candidates)
    metadata["image_knowledge_candidates"] = len(candidates)
    metadata["image_knowledge_source_ids"] = [
        c.get("image_id") for c in candidates if c.get("image_id") is not None
    ]
    if not candidates:
        return list(base_chunks), metadata

    # Merge: image-knowledge candidates that duplicate an existing
    # base chunk (same document_id + image_id) are dropped to avoid
    # double-citing the same image.
    merged: List[Dict[str, Any]] = list(base_chunks)
    existing_pairs = set()
    for c in base_chunks:
        key = (c.get("document_id"), c.get("image_id"))
        if key != (None, None):
            existing_pairs.add(key)
    for cand in candidates:
        key = (cand.get("document_id"), cand.get("image_id"))
        if key in existing_pairs:
            continue
        merged.append(cand)

    # Boost when image-intent phrases are present.
    boost = _intent_boost()
    if image_intent:
        merged = _apply_intent_boost(merged, boost=boost, image_intent=True)
        metadata["image_knowledge_boost_applied"] = float(boost)

    # Authority demote: product-meaning queries keep KB on top.
    product_meaning = bool(metadata.get("product_meaning_matched"))
    if product_meaning:
        merged = _apply_authority_demote(merged, enabled=True)
        metadata["image_knowledge_authority_demote_applied"] = True

    # Hard cap on image sources.
    cap = _max_image_sources()
    pre_cap_count = sum(1 for c in merged if c.get("_knowledge_kind") == "image")
    merged = _apply_image_cap(merged, max_image_sources=cap)
    post_cap_count = sum(1 for c in merged if c.get("_knowledge_kind") == "image")
    metadata["image_knowledge_selected"] = post_cap_count
    metadata["image_knowledge_dropped_by_cap"] = max(0, pre_cap_count - post_cap_count)

    return merged, metadata


__all__ = [
    "integrate_image_knowledge",
    "has_historical_image_intent",
]

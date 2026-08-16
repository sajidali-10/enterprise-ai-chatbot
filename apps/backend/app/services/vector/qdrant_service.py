from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Optional
from app.core.config import settings

_client = None

def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
    return _client


def delete_collection():
    """Delete the Qdrant collection (for reindexing with new embeddings)."""
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION in collections:
        client.delete_collection(collection_name=settings.QDRANT_COLLECTION)


def ensure_collection(dimension: int):
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )


def delete_vectors_by_document_id(document_id: int) -> int:
    """
    Delete all vectors for a specific document.
    Returns the number of vectors deleted.
    """
    client = get_qdrant_client()
    # First, find all points for this document
    try:
        results = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            limit=10000,
        )
        if results[0]:
            point_ids = [p.id for p in results[0]]
            client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points=point_ids,
            )
            return len(point_ids)
    except Exception:
        pass
    return 0

def upsert_chunks(chunks_with_embeddings: List[tuple], metadata: List[dict]):
    """
    chunks_with_embeddings: list of (chunk_content, embedding_vector)
    metadata: list of dicts with chunk metadata (id, document_id, chunk_index, source_file_name, etc.)

    Phase 34A.1.1 — structured OCR/source-type metadata.

    The payload now carries the full set of OCR / image-derived fields
    the upload endpoint passes in, so that downstream retrieval can
    distinguish image-derived chunks from native text. Fields written:

        - source_type           (canonical: image_ocr | pdf_ocr | docx_image_ocr | native_text)
        - content_type          (legacy alias for source_type)
        - image_id              (DocumentImage.id for OCR-derived chunks)
        - ocr_provider          (e.g. tesseract)
        - ocr_confidence        (0-100 int)
        - page_number           (PDF/DOCX page)
        - mime_type             (chunk / document MIME)
        - is_ocr                (bool shortcut for image-derived chunks)

    Existing chunks (indexed before this change) keep working — they
    just don't carry these fields, and the legacy fallback in
    `app.rag.image_routing` will classify them from their content.
    """
    client = get_qdrant_client()
    points = []
    for i, ((chunk_content, embedding), meta) in enumerate(zip(chunks_with_embeddings, metadata)):
        # Use integer ID to avoid Qdrant 1.18+ parsing issues with underscore-separated strings
        point_id = meta['chunk_id']

        # Phase 34A.1.1: derive canonical source_type. Trust the
        # caller-supplied `source_type` if present, otherwise fall
        # back to `content_type`, otherwise infer from `image_id`,
        # otherwise default to native_text.
        source_type = (
            meta.get("source_type")
            or meta.get("content_type")
            or ("image_ocr" if meta.get("image_id") else "native_text")
        )
        content_type = meta.get("content_type") or source_type
        is_ocr = source_type in {
            "image_ocr",
            "pdf_ocr",
            "pdf_page_ocr",
            "docx_image_ocr",
        }

        # Resolve page_number — accept either `page_number` or `page`
        # (legacy alias used by the upload endpoint).
        page_number = meta.get("page_number")
        if page_number is None:
            page_number = meta.get("page")

        payload = {
            "chunk_id": meta["chunk_id"],
            "document_id": meta["document_id"],
            "document_version_id": meta["document_version_id"],
            "chunk_index": meta["chunk_index"],
            "content": chunk_content,
            "content_hash": meta["content_hash"],
            "source_file_name": meta.get("source_file_name", ""),
            "title": meta.get("title"),
            "section_heading": meta.get("section_heading"),
            # Phase 34A.1.1 — structured source / OCR metadata
            "source_type": source_type,
            "content_type": content_type,
            "image_id": meta.get("image_id"),
            "ocr_provider": meta.get("ocr_provider"),
            "ocr_confidence": meta.get("ocr_confidence"),
            "page_number": page_number,
            "mime_type": meta.get("mime_type"),
            "is_ocr": is_ocr,
        }

        # Strip None values so legacy payloads stay clean and Qdrant
        # doesn't store a bunch of explicit-null fields. Fields that
        # downstream code looks up via .get() handle None gracefully
        # already; this just keeps payloads tidy.
        payload = {k: v for k, v in payload.items() if v is not None}

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        ))

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=points,
    )

def search(query_embedding: list[float], limit: int = 5) -> list:
    client = get_qdrant_client()
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=limit,
    )
    return results


def get_point(point_id: int) -> Optional[dict]:
    """Fetch a single Qdrant point by ID (for diagnostics / reindex helpers).

    Returns the raw payload dict, or None if the point does not exist.
    """
    try:
        client = get_qdrant_client()
        points = client.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        p = points[0]
        return getattr(p, "payload", None) or {}
    except Exception:
        return None


def fetch_chunks_by_document_id(
    document_id: int,
    image_id: Optional[int] = None,
    limit: int = 32,
) -> List[dict]:
    """Phase 34A.1.2 — fetch chunks for a specific document_id (and optionally
    a specific image_id) directly from Qdrant using payload filters.

    This bypasses semantic similarity and is the safe retrieval path for
    explicit / recently-resolved image-content queries, where the source
    relationship itself is the relevance signal.

    Returns:
        A list of chunk dicts in the same shape as `search()` results,
        each containing at minimum: chunk_id, document_id, chunk_index,
        content, source_file_name, title, score (None — by-id lookup
        has no semantic score), and any payload fields the indexer
        stored (source_type, image_id, ocr_provider, etc.).
    """
    if not document_id:
        return []
    try:
        client = get_qdrant_client()
        must = [FieldCondition(key="document_id", match=MatchValue(value=int(document_id)))]
        if image_id is not None:
            must.append(FieldCondition(key="image_id", match=MatchValue(value=int(image_id))))
        results = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(must=must),
            limit=max(1, min(int(limit or 32), 256)),
            with_payload=True,
            with_vectors=False,
        )
        points = results[0] if results else []
    except Exception:
        return []

    chunks: List[dict] = []
    for point in points or []:
        payload = getattr(point, "payload", None) or {}
        chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "document_id": payload.get("document_id"),
            "document_version_id": payload.get("document_version_id"),
            "chunk_index": payload.get("chunk_index"),
            "content": payload.get("content", ""),
            "source_file_name": payload.get("source_file_name", ""),
            "title": payload.get("title"),
            "section_heading": payload.get("section_heading"),
            "score": None,  # by-id lookup has no semantic score
            "source_type": payload.get("source_type"),
            "content_type": payload.get("content_type"),
            "image_id": payload.get("image_id"),
            "ocr_provider": payload.get("ocr_provider"),
            "ocr_confidence": payload.get("ocr_confidence"),
            "page_number": payload.get("page_number"),
            "mime_type": payload.get("mime_type"),
            "is_ocr": payload.get("is_ocr"),
            "_retrieval_channel": "qdrant_direct_by_id",
        })
    # Stable order: by chunk_index then chunk_id (deterministic citation order).
    chunks.sort(key=lambda c: (
        int(c.get("chunk_index") or 0) if c.get("chunk_index") is not None else 0,
        str(c.get("chunk_id") or ""),
    ))
    return chunks


def fetch_chunks_by_image_id(image_id: int, limit: int = 32) -> List[dict]:
    """Phase 34A.1.2 — fetch chunks for a specific DocumentImage.id.

    Thin wrapper over `fetch_chunks_by_document_id` for callers that
    only know the image_id (e.g. when DocumentImage is the first
    anchor the resolver has access to).
    """
    if not image_id:
        return []
    try:
        client = get_qdrant_client()
        results = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="image_id", match=MatchValue(value=int(image_id)))
            ]),
            limit=max(1, min(int(limit or 32), 256)),
            with_payload=True,
            with_vectors=False,
        )
        points = results[0] if results else []
    except Exception:
        return []

    chunks: List[dict] = []
    for point in points or []:
        payload = getattr(point, "payload", None) or {}
        chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "document_id": payload.get("document_id"),
            "document_version_id": payload.get("document_version_id"),
            "chunk_index": payload.get("chunk_index"),
            "content": payload.get("content", ""),
            "source_file_name": payload.get("source_file_name", ""),
            "title": payload.get("title"),
            "section_heading": payload.get("section_heading"),
            "score": None,
            "source_type": payload.get("source_type"),
            "content_type": payload.get("content_type"),
            "image_id": payload.get("image_id"),
            "ocr_provider": payload.get("ocr_provider"),
            "ocr_confidence": payload.get("ocr_confidence"),
            "page_number": payload.get("page_number"),
            "mime_type": payload.get("mime_type"),
            "is_ocr": payload.get("is_ocr"),
            "_retrieval_channel": "qdrant_direct_by_image_id",
        })
    chunks.sort(key=lambda c: (
        int(c.get("chunk_index") or 0) if c.get("chunk_index") is not None else 0,
        str(c.get("chunk_id") or ""),
    ))
    return chunks
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
    """
    client = get_qdrant_client()
    points = []
    for i, ((chunk_content, embedding), meta) in enumerate(zip(chunks_with_embeddings, metadata)):
        # Use integer ID to avoid Qdrant 1.18+ parsing issues with underscore-separated strings
        point_id = meta['chunk_id']
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "chunk_id": meta["chunk_id"],
                "document_id": meta["document_id"],
                "document_version_id": meta["document_version_id"],
                "chunk_index": meta["chunk_index"],
                "content": chunk_content,
                "content_hash": meta["content_hash"],
                "source_file_name": meta.get("source_file_name", ""),
                "title": meta.get("title"),
                "section_heading": meta.get("section_heading"),
            }
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
"""
Custom Vector Store Adapter

Wraps the existing Qdrant service from app/services/vector/qdrant_service.py.
No logic duplication — delegates to qdrant_service functions.
"""

from app.services.vector import qdrant_service as qsvc
from app.providers.base import VectorStoreProvider, ChunkDict, SearchResult


class CustomVectorStoreProvider(VectorStoreProvider):
    """
    Phase 22 active provider (qdrant): wraps qdrant_service.py.

    Provides index, search, and delete operations over the configured Qdrant
    collection using settings.QDRANT_COLLECTION.
    """

    def index(self, chunks_with_embeddings: list[tuple], metadata: list[dict]) -> None:
        """
        Wrap qsvc.upsert_chunks.

        Args:
            chunks_with_embeddings: List of (text_content, embedding_vector) tuples.
            metadata: List of chunk metadata dicts with keys: chunk_id, document_id,
                document_version_id, chunk_index, content_hash, source_file_name,
                title, section_heading.
        """
        qsvc.upsert_chunks(chunks_with_embeddings, metadata)

    def search(
        self,
        query_embedding: list[float],
        limit: int,
        score_threshold: float,
    ) -> list[SearchResult]:
        raw_results = qsvc.search(query_embedding=query_embedding, limit=limit)
        results: list[SearchResult] = []
        for result in raw_results:
            if hasattr(result, "score") and result.score < score_threshold:
                continue
            payload = result.payload
            results.append({
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "content": payload.get("content", ""),
                "source_file_name": payload.get("source_file_name", ""),
                "title": payload.get("title", ""),
                "score": getattr(result, "score", None),
            })
        return results

    def delete_by_document_id(self, document_id: int) -> int:
        return qsvc.delete_vectors_by_document_id(document_id)
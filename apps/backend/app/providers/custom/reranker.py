"""
Custom Reranker Adapter — No-Op (Pass-Through)

Wraps the existing noop reranker from app/rag/reranker.py.
Phase 22 only supports 'none' (identity pass-through).
Future phases will add Cohere, BGE, and LangChain rerankers.
"""

from app.rag.reranker import NoOpReranker as _NoOpReranker
from app.providers.base import RerankerProvider, ChunkDict


class NoOpRerankerProvider(RerankerProvider):
    """
    Phase 22 only active reranker: identity pass-through.

    The underlying NoOpReranker returns chunks unchanged with their
    original scores preserved as rerank_score.
    """

    def __init__(self):
        self._impl = _NoOpReranker()

    def rerank(
        self,
        query: str,
        chunks: list[ChunkDict],
        top_n: int | None = None,
    ) -> list[ChunkDict]:
        # NoOpReranker returns List[RerankResult]; convert back to ChunkDict
        results = self._impl.rerank(query, chunks, top_n=top_n)
        return [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "content": r.content,
                "source_file_name": r.source_file_name,
                "title": r.title,
                "score": r.rerank_score,
                "rerank_score": r.rerank_score,
            }
            for r in results
        ]
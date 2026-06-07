"""
Reranker Abstraction

Provides a unified interface for reranking retrieval results.
Supports mock reranker (default) and placeholders for Cohere/BGE rerankers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class RerankResult:
    """A single reranked result with its new score."""
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    source_file_name: str
    title: str
    original_score: float
    rerank_score: float


class RerankerBase(ABC):
    """Abstract base class for reranking strategies."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank a list of chunks for a given query.
        
        Args:
            query: The search query.
            chunks: List of chunk dicts with keys like chunk_id, document_id,
                    chunk_index, content, source_file_name, title, score.
            top_n: Return only top N results. If None, return all.
            
        Returns:
            List of RerankResult objects sorted by rerank_score descending.
        """
        ...


class MockReranker(RerankerBase):
    """
    Mock reranker that returns chunks sorted by a simple heuristic.
    
    In a real implementation, this would call a reranking API like:
    - Cohere Rerank (cohere.com)
    - BGE Reranker (moka-ai.github.io/bge-reranker)
    """

    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank by combining original score with query-chunk text overlap.
        Lowercase word overlap is used as a simple relevance signal.
        """
        query_terms = set(query.lower().split())
        
        results = []
        for chunk in chunks:
            content_lower = chunk.get("content", "").lower()
            content_terms = set(content_lower.split())
            
            # Calculate simple overlap score
            overlap = len(query_terms & content_terms)
            overlap_ratio = overlap / max(len(query_terms), 1)
            
            # Combine with original score (weighted average)
            original_score = chunk.get("score", 0.5)
            # Boost if query terms appear in title or section heading
            title_text = (chunk.get("title") or "").lower()
            heading_text = (chunk.get("section_heading") or "").lower()
            title_boost = 0.1 if any(qt in title_text or qt in heading_text for qt in query_terms) else 0.0
            
            # Final score: 70% original + 20% overlap + 10% title boost
            rerank_score = (0.7 * original_score) + (0.2 * overlap_ratio) + title_boost
            
            results.append(RerankResult(
                chunk_id=chunk.get("chunk_id", ""),
                document_id=chunk.get("document_id", ""),
                chunk_index=chunk.get("chunk_index", 0),
                content=chunk.get("content", ""),
                source_file_name=chunk.get("source_file_name", ""),
                title=chunk.get("title", ""),
                original_score=original_score,
                rerank_score=rerank_score,
            ))
        
        # Sort by rerank score descending
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_n is not None:
            results = results[:top_n]
        
        return results


class NoOpReranker(RerankerBase):
    """
    No-op reranker that returns chunks unchanged.
    Used when reranking is disabled.
    """

    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        results = []
        for chunk in chunks:
            results.append(RerankResult(
                chunk_id=chunk.get("chunk_id", ""),
                document_id=chunk.get("document_id", ""),
                chunk_index=chunk.get("chunk_index", 0),
                content=chunk.get("content", ""),
                source_file_name=chunk.get("source_file_name", ""),
                title=chunk.get("title", ""),
                original_score=chunk.get("score", 0.0),
                rerank_score=chunk.get("score", 0.0),
            ))
        
        if top_n is not None:
            results = results[:top_n]
        return results


# Map of reranker type names to classes
RERANKER_REGISTRY = {
    "mock": MockReranker,
    "noop": NoOpReranker,
    "passthrough": NoOpReranker,
}


def get_reranker(reranker_type: Optional[str] = None) -> RerankerBase:
    """
    Factory to get a reranker by name.
    
    Args:
        reranker_type: One of "mock", "noop", "passthrough", "cohere", "bge", or None.
        
    Returns:
        RerankerBase instance.
    """
    if reranker_type is None:
        return NoOpReranker()
    
    reranker_type = reranker_type.lower()
    
    if reranker_type in RERANKER_REGISTRY:
        return RERANKER_REGISTRY[reranker_type]()
    elif reranker_type in ("cohere", "bge"):
        raise NotImplementedError(
            f"Reranker '{reranker_type}' is not yet implemented. "
            f"Use 'mock', 'noop', or 'passthrough' for now. "
            f"Cohere and BGE reranker integration coming soon."
        )
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
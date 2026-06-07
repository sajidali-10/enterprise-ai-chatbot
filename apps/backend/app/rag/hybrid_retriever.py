"""
Hybrid Retrieval Service

Combines vector search (Qdrant) and keyword search (PostgreSQL)
with configurable fusion and reranking.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from app.services.vector.qdrant_service import search as qdrant_search
from app.services.search.keyword_search import search_chunks_keyword
from app.services.embeddings import get_embedding_provider
from app.rag.reranker import get_reranker, RerankerBase, RerankResult


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    vector_top_k: int = 10
    keyword_top_k: int = 10
    final_top_k: int = 5
    min_score: float = 0.0
    reranker_type: str = "noop"
    rerank_final_k: Optional[int] = None
    keyword_weight: float = 0.3
    vector_weight: float = 0.7


@dataclass
class ScoredChunk:
    """A chunk with its retrieval score from hybrid fusion."""
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    source_file_name: str
    title: str
    vector_score: Optional[float]
    keyword_score: Optional[float]
    fused_score: float
    is_from_vector: bool
    is_from_keyword: bool


def _normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max normalize scores to [0, 1] range.
    Handles empty lists and constant scores.
    """
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _fuse_scores(
    vector_results: List[dict],
    keyword_results: List[dict],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[ScoredChunk]:
    """
    Fuse vector and keyword search results using weighted score fusion.
    
    Uses reciprocal rank fusion (RRF) as a fallback/complement to score fusion.
    """
    # Build lookup maps
    chunk_map: dict[str, ScoredChunk] = {}
    
    # Process vector results
    vector_scores = [r.get("score", 0.0) for r in vector_results]
    normalized_vector = _normalize_scores(vector_scores)
    
    for i, result in enumerate(vector_results):
        chunk_id = str(result.get("chunk_id", f"v_{i}"))
        norm_score = normalized_vector[i] if i < len(normalized_vector) else 0.0
        chunk_map[chunk_id] = ScoredChunk(
            chunk_id=chunk_id,
            document_id=str(result.get("document_id", "")),
            chunk_index=result.get("chunk_index", 0),
            content=result.get("content", ""),
            source_file_name=result.get("source_file_name", ""),
            title=result.get("title", ""),
            vector_score=norm_score,
            keyword_score=None,
            fused_score=norm_score * vector_weight,
            is_from_vector=True,
            is_from_keyword=False,
        )
    
    # Process keyword results
    keyword_scores = [r.get("score", 0.0) for r in keyword_results]
    normalized_keyword = _normalize_scores(keyword_scores)
    
    for i, result in enumerate(keyword_results):
        chunk_id = str(result.get("chunk_id", f"k_{i}"))
        norm_score = normalized_keyword[i] if i < len(normalized_keyword) else 0.0
        
        if chunk_id in chunk_map:
            # Chunk already exists from vector search
            existing = chunk_map[chunk_id]
            existing.keyword_score = norm_score
            existing.fused_score = (
                (existing.vector_score or 0.0) * vector_weight +
                norm_score * keyword_weight
            )
            existing.is_from_keyword = True
        else:
            chunk_map[chunk_id] = ScoredChunk(
                chunk_id=chunk_id,
                document_id=str(result.get("document_id", "")),
                chunk_index=result.get("chunk_index", 0),
                content=result.get("content", ""),
                source_file_name=result.get("source_file_name", ""),
                title=result.get("title", ""),
                vector_score=None,
                keyword_score=norm_score,
                fused_score=norm_score * keyword_weight,
                is_from_vector=False,
                is_from_keyword=True,
            )
    
    # Sort by fused score
    sorted_chunks = sorted(chunk_map.values(), key=lambda x: x.fused_score, reverse=True)
    return sorted_chunks


def retrieve_chunks_hybrid(
    query: str,
    config: Optional[RetrievalConfig] = None,
    reranker: Optional[RerankerBase] = None,
) -> Tuple[List[dict], dict]:
    """
    Perform hybrid retrieval combining vector and keyword search.
    
    Args:
        query: User's search query.
        config: Retrieval configuration with top_k, weights, and reranker settings.
        reranker: Optional reranker to apply after fusion. If None, uses config.reranker_type.
        
    Returns:
        Tuple of (list of chunk dicts, retrieval metadata dict with scores info).
    """
    if config is None:
        config = RetrievalConfig()
    
    if reranker is None:
        reranker = get_reranker(config.reranker_type)
    
    # 1. Vector search (Qdrant)
    provider = get_embedding_provider()
    query_embedding = provider.embed([query])[0]
    vector_results = qdrant_search(query_embedding=query_embedding, limit=config.vector_top_k)
    
    vector_chunks = []
    for result in vector_results:
        payload = result.payload if hasattr(result, 'payload') else result
        vector_chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "document_id": payload.get("document_id"),
            "document_version_id": payload.get("document_version_id"),
            "chunk_index": payload.get("chunk_index"),
            "content": payload.get("content", ""),
            "source_file_name": payload.get("source_file_name", ""),
            "title": payload.get("title", ""),
            "section_heading": payload.get("section_heading"),
            "score": getattr(result, 'score', 0.0) if hasattr(result, 'score') else 0.0,
        })
    
    # 2. Keyword search (PostgreSQL)
    keyword_chunks = search_chunks_keyword(
        query=query,
        limit=config.keyword_top_k,
        min_score=config.min_score,
    )
    
    # 3. Fuse results
    fused_chunks = _fuse_scores(
        vector_results=vector_chunks,
        keyword_results=keyword_chunks,
        vector_weight=config.vector_weight,
        keyword_weight=config.keyword_weight,
    )
    
    # 4. Apply min_score filter
    if config.min_score > 0:
        fused_chunks = [c for c in fused_chunks if c.fused_score >= config.min_score]
    
    # 5. Convert to dict format for reranker
    chunk_dicts = []
    for chunk in fused_chunks:
        chunk_dicts.append({
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "source_file_name": chunk.source_file_name,
            "title": chunk.title,
            "score": chunk.fused_score,
        })
    
    # 6. Apply reranking
    if config.rerank_final_k is not None and config.rerank_final_k > 0:
        top_chunks_for_rerank = chunk_dicts[:config.rerank_final_k]
    else:
        top_chunks_for_rerank = chunk_dicts
    
    rerank_results = reranker.rerank(query, top_chunks_for_rerank, top_n=config.final_top_k)
    
    # 7. Build final result with all required fields
    final_chunks = []
    for rr in rerank_results:
        final_chunks.append({
            "chunk_id": rr.chunk_id,
            "document_id": rr.document_id,
            "chunk_index": rr.chunk_index,
            "content": rr.content,
            "source_file_name": rr.source_file_name,
            "title": rr.title,
            "score": rr.rerank_score,
            "original_score": rr.original_score,
            "vector_score": chunk_dicts[[c["chunk_id"] for c in chunk_dicts].index(rr.chunk_id)]["score"] if rr.chunk_id in [c["chunk_id"] for c in chunk_dicts] else None,
        })
    
    # Build metadata about retrieval
    metadata = {
        "vector_results_count": len(vector_chunks),
        "keyword_results_count": len(keyword_chunks),
        "fused_results_count": len(fused_chunks),
        "final_results_count": len(final_chunks),
        "reranker_type": config.reranker_type,
        "vector_weight": config.vector_weight,
        "keyword_weight": config.keyword_weight,
    }
    
    return final_chunks, metadata
"""
RAG Retrieval Module

Provides retrieval functions for RAG chat:
- retrieve_chunks: Original vector-only retrieval (backward compatible)
- retrieve_chunks_hybrid: New hybrid retrieval combining vector + keyword search
"""

from typing import Optional
from app.services.vector.qdrant_service import search as qdrant_search
from app.services.embeddings import get_embedding_provider
from app.rag.hybrid_retriever import (
    retrieve_chunks_hybrid,
    RetrievalConfig,
)
from app.rag.query_rewriter import get_query_rewriter
from app.core.config import settings


def retrieve_chunks(query: str, limit: int = 5, score_threshold: float = 0.5) -> list[dict]:
    """
    Embed query, search Qdrant, return top chunks above score threshold.
    Returns list of chunk dicts with metadata.
    
    Note: This is the original vector-only retrieval for backward compatibility.
    For hybrid retrieval with keyword search and reranking, use retrieve_chunks_hybrid().
    """
    provider = get_embedding_provider()
    query_embedding = provider.embed([query])[0]
    
    results = qdrant_search(query_embedding=query_embedding, limit=limit)
    
    chunks = []
    for result in results:
        if hasattr(result, 'score') and result.score < score_threshold:
            continue
        payload = result.payload
        chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "document_id": payload.get("document_id"),
            "chunk_index": payload.get("chunk_index"),
            "content": payload.get("content", ""),
            "source_file_name": payload.get("source_file_name", ""),
            "title": payload.get("title", ""),
            "score": getattr(result, 'score', None),
        })
    
    return chunks


def retrieve_chunks_with_settings(query: str, debug: bool = False) -> tuple[list[dict], dict]:
    """
    Retrieve chunks using hybrid retrieval with configurable settings.
    
    Uses settings from config:
    - RETRIEVAL_VECTOR_TOP_K: Number of vector results to fetch
    - RETRIEVAL_KEYWORD_TOP_K: Number of keyword results to fetch
    - RETRIEVAL_FINAL_TOP_K: Final number of results after reranking
    - RETRIEVAL_MIN_SCORE: Minimum score threshold
    - RETRIEVAL_RERANKER_TYPE: Reranker to use (noop, mock)
    - RETRIEVAL_VECTOR_WEIGHT: Weight for vector scores in fusion
    - RETRIEVAL_KEYWORD_WEIGHT: Weight for keyword scores in fusion
    - RETRIEVAL_QUERY_REWRITER: Query rewriter to use
    
    Args:
        query: User's search query.
        debug: If True, return debug metadata about retrieval process.
        
    Returns:
        Tuple of (chunks list, metadata dict).
    """
    # Apply query rewriting if configured
    rewriter = get_query_rewriter(settings.RETRIEVAL_QUERY_REWRITER)
    rewritten_query = rewriter.rewrite(query)
    
    # Build retrieval config from settings
    config = RetrievalConfig(
        vector_top_k=settings.RETRIEVAL_VECTOR_TOP_K,
        keyword_top_k=settings.RETRIEVAL_KEYWORD_TOP_K,
        final_top_k=settings.RETRIEVAL_FINAL_TOP_K,
        min_score=settings.RETRIEVAL_MIN_SCORE,
        reranker_type=settings.RETRIEVAL_RERANKER_TYPE,
        rerank_final_k=settings.RETRIEVAL_RERANK_FINAL_K,
        vector_weight=settings.RETRIEVAL_VECTOR_WEIGHT,
        keyword_weight=settings.RETRIEVAL_KEYWORD_WEIGHT,
    )
    
    chunks, metadata = retrieve_chunks_hybrid(
        query=rewritten_query,
        config=config,
    )
    
    # Add debug info if requested
    if debug or settings.RETRIEVAL_SHOW_DEBUG:
        metadata["debug"] = True
        metadata["original_query"] = query
        metadata["rewritten_query"] = rewritten_query
        metadata["config"] = {
            "vector_top_k": config.vector_top_k,
            "keyword_top_k": config.keyword_top_k,
            "final_top_k": config.final_top_k,
            "min_score": config.min_score,
            "reranker_type": config.reranker_type,
            "vector_weight": config.vector_weight,
            "keyword_weight": config.keyword_weight,
        }
        # Add score details for each chunk
        for i, chunk in enumerate(chunks):
            chunk["_debug_index"] = i + 1
            if "original_score" in chunk:
                chunk["_original_score"] = chunk.pop("original_score")
    
    return chunks, metadata
"""
RAG Retrieval Module

Provides retrieval functions for RAG chat:
- retrieve_chunks: Original vector-only retrieval (backward compatible)
- retrieve_chunks_hybrid: New hybrid retrieval combining vector + keyword search
- retrieve_chunks_with_auth: Permission-aware retrieval (Phase 6)
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

# Import security modules for permission filtering (Phase 6)
try:
    from app.security.auth import AuthContext
    from app.security.permissions import PermissionChecker, filter_documents_by_permission
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    AuthContext = None


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
        from app.services.embeddings import get_embedding_provider_info
        metadata["debug"] = True
        metadata["original_query"] = query
        metadata["rewritten_query"] = rewritten_query
        metadata["embedding_provider_info"] = get_embedding_provider_info()
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


def retrieve_chunks_with_auth(
    query: str,
    auth: AuthContext,
    debug: bool = False,
) -> tuple[list[dict], dict]:
    """
    Retrieve chunks with permission filtering (Phase 6).
    
    This function:
    1. Performs hybrid retrieval
    2. Filters out chunks from documents the user cannot access
    3. Returns only authorized chunks with their document IDs
    
    Admin users bypass permission filtering (see permissions.py).
    Dev users without database records cannot access any documents.
    
    Args:
        query: User's search query.
        auth: Authentication context with user info and role.
        debug: If True, return debug metadata about retrieval process.
        
    Returns:
        Tuple of (filtered chunks list, metadata dict with permission info).
    """
    if not HAS_SECURITY:
        # Security module not available, fall back to unfiltered retrieval
        return retrieve_chunks_with_settings(query, debug)
    
    # Get all chunks from hybrid retrieval
    chunks, metadata = retrieve_chunks_with_settings(query, debug)
    
    if not chunks:
        metadata["permission_filtered"] = False
        metadata["accessible_count"] = 0
        metadata["total_count"] = 0
        return chunks, metadata
    
    # Extract document IDs from chunks
    document_ids = list(set(c.get("document_id") for c in chunks if c.get("document_id")))
    
    # Filter to only accessible documents
    accessible_doc_ids = filter_documents_by_permission(auth, document_ids)
    accessible_doc_ids_set = set(accessible_doc_ids)
    
    # Filter chunks to only those from accessible documents
    original_count = len(chunks)
    filtered_chunks = [
        c for c in chunks 
        if c.get("document_id") in accessible_doc_ids_set
    ]
    final_count = len(filtered_chunks)
    
    # Update metadata
    metadata["permission_filtered"] = True
    metadata["accessible_count"] = final_count
    metadata["total_count"] = original_count
    metadata["filtered_count"] = original_count - final_count
    metadata["user_id"] = auth.user_id
    metadata["username"] = auth.username
    metadata["user_role"] = auth.role.value if hasattr(auth.role, 'value') else str(auth.role)
    
    if debug or settings.RETRIEVAL_SHOW_DEBUG:
        metadata["debug"] = True
        metadata["original_query"] = query
        metadata["accessible_document_ids"] = list(accessible_doc_ids_set)
    
    return filtered_chunks, metadata


def get_retrievable_document_ids(auth: AuthContext) -> list[int]:
    """
    Get all document IDs that a user can access.
    
    This is useful for displaying document lists in the UI.
    Admin users get all document IDs.
    """
    if not HAS_SECURITY:
        return []
    
    # Admin can access all documents
    if auth.is_admin():
        return []  # Empty list means "all" in the permission checker
    
    # Dev users without user_id can't access any documents
    if auth.user_id is None:
        return []
    
    from app.db.session import SessionLocal
    from app.security.models import DocumentPermission
    
    db = SessionLocal()
    try:
        permissions = db.query(DocumentPermission).filter(
            DocumentPermission.user_id == auth.user_id,
            DocumentPermission.can_read == True,
        ).all()
        return [p.document_id for p in permissions]
    finally:
        db.close()
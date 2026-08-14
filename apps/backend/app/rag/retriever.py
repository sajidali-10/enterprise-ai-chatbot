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
    retrieve_with_strategy,
    RetrievalConfig,
)
from app.rag.query_rewriter import get_query_rewriter
from app.rag.query_analysis import analyze_query
from app.rag.relevance_filter import apply_relevance_threshold
from app.rag.image_routing import select_image_aware_chunks
from app.core.config import settings
from app.services.langsmith_tracing import trace_span, redact_filenames, safe_chunk_content, get_current_span

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
    # Phase 31A — instrumentation: open a `rag_retrieval` span so that
    # strategy, chunk count, top score, and selected file names are
    # visible in LangSmith alongside the rest of the request. Inputs and
    # outputs are scrubbed before they leave the process.
    with trace_span(
        "rag_retrieval",
        metadata={
            "phase": "rag_retrieval",
            "query_length": len(query or ""),
        },
    ) as retrieval_span:
        # Apply query rewriting if configured
        rewriter = get_query_rewriter(settings.RETRIEVAL_QUERY_REWRITER)
        rewritten_query = rewriter.rewrite(query)

        # Phase 34A.1 — deterministic query analysis. Used to drive
        # exact-match reranking and image-aware routing. Never makes
        # an LLM call.
        query_analysis = analyze_query(rewritten_query)

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

        # Phase 30E: Use strategy dispatch (hybrid_mmr by default)
        # Falls back to hybrid if the configured strategy is invalid.
        strategy = getattr(settings, "RAG_RETRIEVAL_STRATEGY", "hybrid_mmr")
        mmr_lambda = getattr(settings, "RAG_MMR_LAMBDA", 0.7)
        chunks, metadata = retrieve_with_strategy(
            query=rewritten_query,
            strategy=strategy,
            config=config,
            mmr_lambda=mmr_lambda,
            query_analysis=query_analysis,
            exact_match_boost=getattr(settings, "RAG_EXACT_MATCH_BOOST", 0.30),
            exact_term_boost=getattr(settings, "RAG_EXACT_TERM_BOOST", 0.15),
        )

        # Phase 34A.1 — relevance floor + max sources cap. Applied AFTER
        # hybrid scoring + exact-match reranking so the threshold acts
        # on the most informative score we have.
        try:
            min_floor = float(getattr(settings, "RAG_MIN_RELEVANCE_FLOOR", 0.10) or 0.0)
        except Exception:
            min_floor = 0.0
        try:
            max_sources = int(getattr(settings, "RAG_MAX_FINAL_SOURCES", 6) or 6)
        except Exception:
            max_sources = 6

        chunks, filter_meta = apply_relevance_threshold(
            chunks, min_score=min_floor, max_results=max_sources
        )
        metadata["relevance_filter"] = filter_meta
        metadata["candidate_count"] = filter_meta.get("candidate_count", 0)
        metadata["filtered_count"] = filter_meta.get("filtered_count", 0)
        metadata["final_source_count"] = filter_meta.get("final_source_count", len(chunks))

        # Surface Phase 34A.1 diagnostics.
        metadata["query_type"] = query_analysis.query_type
        metadata["query_analysis"] = query_analysis.to_dict()
        metadata["exact_match_count"] = metadata.get("exact_match_count", 0)
        metadata["retrieval_mode"] = "image_aware" if (
            getattr(settings, "RAG_IMAGE_AWARE_ROUTING_ENABLED", True)
            and query_analysis.references_uploaded_image
        ) else "standard"

        # Phase 31A — fill retrieval span metadata now that we know the outcome.
        if retrieval_span is not None:
            try:
                retrieval_span.set_meta(
                    "retrieval_strategy",
                    metadata.get("strategy", strategy),
                )
                retrieval_span.set_meta("hybrid_applied", bool(metadata.get("hybrid_applied")))
                retrieval_span.set_meta("mmr_applied", bool(metadata.get("mmr_applied")))
                retrieval_span.set_meta("vector_results_count", metadata.get("vector_results_count"))
                retrieval_span.set_meta("keyword_results_count", metadata.get("keyword_results_count"))
                retrieval_span.set_meta("selected_chunk_count", len(chunks))
                retrieval_span.set_meta(
                    "source_file_names",
                    redact_filenames(list({c.get("source_file_name") for c in chunks if c.get("source_file_name")})),
                )
                if chunks:
                    scores = [
                        float(c.get("score") or 0.0)
                        for c in chunks
                        if c.get("score") is not None
                    ]
                    if scores:
                        retrieval_span.set_meta("top_score", round(max(scores), 4))
                        retrieval_span.set_meta("min_score", round(min(scores), 4))
            except Exception:
                # Never let tracing break retrieval
                pass

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
                "retrieval_strategy": strategy,
                "mmr_lambda": mmr_lambda,
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
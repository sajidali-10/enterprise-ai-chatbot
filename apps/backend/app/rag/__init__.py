"""
RAG (Retrieval-Augmented Generation) Module

This module provides components for RAG-based chat:
- retriever: Chunk retrieval (vector, keyword, and hybrid)
- answer_generator: Full RAG pipeline with LLM generation
- query_rewriter: Query rewriting for improved retrieval (Phase 5)
- reranker: Reranking abstraction for result refinement (Phase 5)
- hybrid_retriever: Hybrid retrieval combining vector and keyword search (Phase 5)
- prompt_builder: RAG prompt construction
- citations: Citation formatting for RAG responses
"""

from app.rag.retriever import retrieve_chunks, retrieve_chunks_with_settings
from app.rag.answer_generator import generate_answer_with_rag, generate_answer_without_rag
from app.rag.query_rewriter import (
    QueryRewriterBase,
    PassthroughRewriter,
    MockQueryRewriter,
    get_query_rewriter,
)
from app.rag.reranker import (
    RerankerBase,
    RerankResult,
    MockReranker,
    NoOpReranker,
    get_reranker,
)
from app.rag.hybrid_retriever import (
    retrieve_chunks_hybrid,
    RetrievalConfig,
    ScoredChunk,
)

__all__ = [
    # Retrieval
    "retrieve_chunks",
    "retrieve_chunks_with_settings",
    "retrieve_chunks_hybrid",
    "RetrievalConfig",
    "ScoredChunk",
    # Answer generation
    "generate_answer_with_rag",
    "generate_answer_without_rag",
    # Query rewriting
    "QueryRewriterBase",
    "PassthroughRewriter",
    "MockQueryRewriter",
    "get_query_rewriter",
    # Reranking
    "RerankerBase",
    "RerankResult",
    "MockReranker",
    "NoOpReranker",
    "get_reranker",
]
"""
RAG Configuration Service

Provides a safe, read-only summary of current RAG settings.
No secrets, API keys, tokens, or raw .env values are exposed.
"""

import os

from app.core.config import settings
from app.providers.factory import get_provider_status, _is_ragas_available


def _langsmith_endpoint_host_safe() -> str:
    """Return hostname from LANGSMITH_ENDPOINT for safe display."""
    try:
        from urllib.parse import urlparse
        return urlparse(settings.LANGSMITH_ENDPOINT).hostname or "unknown"
    except Exception:
        return "unknown"


def get_rag_config() -> dict:
    """
    Return a safe, admin-readable RAG configuration summary.

    Fields are deliberately chosen to be informative without exposing
    any sensitive values (no API keys, tokens, passwords, or secrets).
    """
    return {
        # Pipeline providers
        "rag_pipeline_provider": settings.RAG_PIPELINE_PROVIDER,
        "document_loader_provider": settings.DOCUMENT_LOADER_PROVIDER,
        "text_splitter_provider": settings.TEXT_SPLITTER_PROVIDER,
        "retriever_provider": settings.RETRIEVER_PROVIDER,
        "reranker_provider": settings.RERANKER_PROVIDER,
        # Embedding settings
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        # Vector store
        "vector_store_provider": settings.VECTOR_STORE_PROVIDER,
        # Retrieval behavior
        "top_k": settings.RAG_TOP_K,
        "score_threshold": settings.RAG_SCORE_THRESHOLD,
        # Chunking
        "chunk_size": settings.RAG_CHUNK_SIZE,
        "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
        # Context
        "context_max_chunks": settings.RAG_CONTEXT_MAX_CHUNKS,
        "context_max_characters": settings.RAG_CONTEXT_MAX_CHARACTERS,
        # Conversation context
        "conversation_context_enabled": settings.RAG_CONVERSATION_CONTEXT_ENABLED,
        "conversation_context_max_messages": settings.RAG_CONVERSATION_CONTEXT_MAX_MESSAGES,
        "conversation_context_max_characters": settings.RAG_CONVERSATION_CONTEXT_MAX_CHARACTERS,
        # Feature flags (no secrets — these are booleans only)
        "langchain_enabled": False,
        "reranker_enabled": settings.RERANKER_ENABLED,
        # Phase 22: Provider interface foundation status
        "provider_status": get_provider_status(),
        # Phase 25: Retrieval & Reranker upgrade foundation
        "retrieval_status": {
            "retrieval_mode": settings.RETRIEVAL_MODE,
            "top_k": settings.RAG_TOP_K,
            "score_threshold": settings.RAG_SCORE_THRESHOLD,
            "candidate_k": settings.RETRIEVAL_CANDIDATE_K,
            "hybrid_enabled": settings.HYBRID_SEARCH_ENABLED,
            "hybrid_keyword_weight": settings.RETRIEVAL_KEYWORD_WEIGHT,
            "hybrid_vector_weight": settings.RETRIEVAL_VECTOR_WEIGHT,
            "reranker_provider": settings.RERANKER_PROVIDER,
            "reranker_enabled": settings.RERANKER_ENABLED,
            "reranker_top_n": settings.RERANKER_TOP_N,
            "reranker_model": settings.RERANKER_MODEL,
        },
        # Phase 26: RAGAS evaluation foundation
        "ragas_status": {
            "ragas_enabled": settings.RAGAS_ENABLED,
            "ragas_available": _is_ragas_available(),
            "evaluator_provider": settings.RAGAS_EVALUATOR_PROVIDER,
            "evaluator_model": settings.RAGAS_EVALUATOR_MODEL,
            "report_dir": settings.RAGAS_REPORT_DIR,
        },
        # Phase 27: LangSmith observability foundation
        "langsmith_status": {
            "langsmith_tracing": settings.LANGSMITH_TRACING,
            "langsmith_available": True,
            "project": settings.LANGSMITH_PROJECT,
            "endpoint_host": _langsmith_endpoint_host_safe(),
            "log_full_prompt": settings.LANGSMITH_LOG_FULL_PROMPT,
            "log_document_text": settings.LANGSMITH_LOG_DOCUMENT_TEXT,
            "log_user_input": settings.LANGSMITH_LOG_USER_INPUT,
            "log_retrieved_context": settings.LANGSMITH_LOG_RETRIEVED_CONTEXT,
            "sample_rate": settings.LANGSMITH_SAMPLE_RATE,
            "has_tracing_key": bool(os.environ.get("LANGSMITH_API_KEY", "").strip()),
        },
        # Phase 34B — Automatic Vision Intelligence status. Safe
        # summary for admin consumption: NO API keys, NO base URL
        # secrets, NO image bytes are exposed.
        "vision_status": {
            "vision_enabled": settings.VISION_ENABLED,
            "vision_provider": settings.VISION_PROVIDER,
            "vision_model": settings.VISION_MODEL,
            "vision_router_enabled": settings.VISION_ROUTER_ENABLED,
            "vision_ocr_confidence_threshold": settings.VISION_OCR_CONFIDENCE_THRESHOLD,
            "vision_min_ocr_text_length": settings.VISION_MIN_OCR_TEXT_LENGTH,
            "vision_timeout_seconds": settings.VISION_TIMEOUT_SECONDS,
            "vision_max_retries": settings.VISION_MAX_RETRIES,
            "vision_cache_schema_version": settings.VISION_CACHE_SCHEMA_VERSION,
            "has_vision_api_key": bool(os.environ.get("VISION_API_KEY", "").strip()),
            "has_vision_base_url": bool((settings.VISION_BASE_URL or "").strip()),
            "warning": (
                "VISION_ENABLED is true but VISION_PROVIDER is set to an unknown value."
                if settings.VISION_ENABLED
                and settings.VISION_PROVIDER not in ("mock", "openai-compatible")
                else None
            ),
        },
    }
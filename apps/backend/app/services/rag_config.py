"""
RAG Configuration Service

Provides a safe, read-only summary of current RAG settings.
No secrets, API keys, tokens, or raw .env values are exposed.
"""

from app.core.config import settings


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
        "reranker_enabled": settings.RERANKER_PROVIDER.lower() not in ("none", ""),
    }
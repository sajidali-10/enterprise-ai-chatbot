"""
Provider Factory

Factory functions that resolve provider names (from environment settings) to
provider instances. Raises ValueError on unknown provider names — fail-fast
is intentional since a misconfigured provider is a deployment error.

Phase 22/24 supported providers:
  - custom        → CustomDocumentLoaderProvider, CustomTextSplitterProvider,
                    CustomRetrieverProvider, CustomRagPipelineProvider
  - local         → CustomEmbeddingProvider  (sentence-transformers)
  - qdrant        → CustomVectorStoreProvider
  - none          → NoOpRerankerProvider
  - langchain     → LangChainDocumentLoaderProvider, LangChainTextSplitterProvider

Phase 23 adds embedding upgrade foundation:
  - Future embedding providers (openai, cohere, voyage, bge, e5) listed as "planned"
  - Embedding reindex status with Qdrant dimension safety check
  - Phase 23 new settings: EMBEDDING_NORMALIZE, EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE

Phase 25 adds retriever/reranker upgrade foundation:
  - Future reranker providers (cohere, bge, cross_encoder, langchain) listed as "planned" / "disabled"
  - Retrieval strategy status with retrieval mode, top_k, score threshold, candidate K,
    hybrid search, reranker settings, and future reranker placeholders
"""

from app.core.config import settings

# ---------------------------------------------------------------------------
# Supported providers per module
# ---------------------------------------------------------------------------

AVAILABLE_DOCUMENT_LOADER_PROVIDERS = ["custom", "langchain"]
AVAILABLE_TEXT_SPLITTER_PROVIDERS = ["custom", "langchain"]
AVAILABLE_EMBEDDING_PROVIDERS = ["local"]
AVAILABLE_VECTOR_STORE_PROVIDERS = ["qdrant"]
AVAILABLE_RETRIEVER_PROVIDERS = ["custom"]
AVAILABLE_RERANKER_PROVIDERS = ["none"]
AVAILABLE_RAG_PIPELINE_PROVIDERS = ["custom"]

# Future embedding providers — planned but not yet implemented/available
FUTURE_EMBEDDING_PROVIDERS = {
    "openai": "planned",
    "cohere": "planned",
    "voyage": "planned",
    "bge": "planned",
    "e5": "planned",
}

# Future reranker providers — planned but not yet implemented/available (Phase 25)
# Status values: "planned" + "disabled" (requires API key / external dependency)
FUTURE_RERANKER_PROVIDERS = {
    "none": "available",  # active default — no-op reranker
    "cohere": "planned",
    "bge": "planned",
    "cross_encoder": "planned",
    "langchain": "planned",
}


def _is_langchain_available() -> bool:
    """Detect whether langchain and langchain-community are importable."""
    try:
        import langchain  # noqa: F401
        import langchain_community  # noqa: F401
        return True
    except ImportError:
        return False


def _is_ragas_available() -> bool:
    """Detect whether ragas is importable and its top-level init succeeds."""
    try:
        import ragas  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve(name: str, available: list[str], provider_kind: str) -> str:
    """
    Validate a provider name against the available list.

    Raises ValueError if the name is not in the available list.
    """
    if name not in available:
        raise ValueError(
            f"Unknown {provider_kind} provider: '{name}'. "
            f"Available in this phase: {available}. "
            f"See provider interface documentation for adding new providers."
        )
    return name


# ---------------------------------------------------------------------------
# Factory functions  (lazy imports to avoid loading heavy deps at import time)
# ---------------------------------------------------------------------------


def get_document_loader_provider():
    """Resolve DOCUMENT_LOADER_PROVIDER to a DocumentLoaderProvider instance."""
    name = _resolve(
        settings.DOCUMENT_LOADER_PROVIDER.strip().lower(),
        AVAILABLE_DOCUMENT_LOADER_PROVIDERS,
        "document loader",
    )
    if name == "custom":
        from app.providers.custom.document_loader import CustomDocumentLoaderProvider
        return CustomDocumentLoaderProvider()
    if name == "langchain":
        if not _is_langchain_available():
            raise ValueError(
                "DOCUMENT_LOADER_PROVIDER=langchain is set but langchain or "
                "langchain-community is not installed. Run: "
                "pip install langchain>=0.4.0 langchain-community>=0.4.0"
            )
        from app.providers.langchain import LangChainDocumentLoaderProvider
        return LangChainDocumentLoaderProvider()
    raise ValueError(f"Document loader provider '{name}' not implemented.")


def get_text_splitter_provider():
    """Resolve TEXT_SPLITTER_PROVIDER to a TextSplitterProvider instance."""
    name = _resolve(
        settings.TEXT_SPLITTER_PROVIDER.strip().lower(),
        AVAILABLE_TEXT_SPLITTER_PROVIDERS,
        "text splitter",
    )
    if name == "custom":
        from app.providers.custom.text_splitter import CustomTextSplitterProvider
        return CustomTextSplitterProvider()
    if name == "langchain":
        if not _is_langchain_available():
            raise ValueError(
                "TEXT_SPLITTER_PROVIDER=langchain is set but langchain or "
                "langchain-community is not installed. Run: "
                "pip install langchain>=0.4.0 langchain-community>=0.4.0"
            )
        from app.providers.langchain import LangChainTextSplitterProvider
        return LangChainTextSplitterProvider()
    raise ValueError(f"Text splitter provider '{name}' not implemented.")


def get_embedding_provider():
    """Resolve EMBEDDING_PROVIDER to an EmbeddingProvider instance."""
    name = _resolve(
        settings.EMBEDDING_PROVIDER.strip().lower(),
        AVAILABLE_EMBEDDING_PROVIDERS,
        "embedding",
    )
    if name == "local":
        from app.providers.custom.embeddings import CustomEmbeddingProvider
        return CustomEmbeddingProvider()
    raise ValueError(f"Embedding provider '{name}' not implemented.")


def get_vector_store_provider():
    """Resolve VECTOR_STORE_PROVIDER to a VectorStoreProvider instance."""
    name = _resolve(
        settings.VECTOR_STORE_PROVIDER.strip().lower(),
        AVAILABLE_VECTOR_STORE_PROVIDERS,
        "vector store",
    )
    if name == "qdrant":
        from app.providers.custom.vector_store import CustomVectorStoreProvider
        return CustomVectorStoreProvider()
    raise ValueError(f"Vector store provider '{name}' not implemented.")


def get_retriever_provider():
    """Resolve RETRIEVER_PROVIDER to a RetrieverProvider instance."""
    name = _resolve(
        settings.RETRIEVER_PROVIDER.strip().lower(),
        AVAILABLE_RETRIEVER_PROVIDERS,
        "retriever",
    )
    if name == "custom":
        from app.providers.custom.retriever import CustomRetrieverProvider
        return CustomRetrieverProvider()
    raise ValueError(f"Retriever provider '{name}' not implemented.")


def get_reranker_provider():
    """Resolve RERANKER_PROVIDER to a RerankerProvider instance."""
    name = _resolve(
        settings.RERANKER_PROVIDER.strip().lower(),
        AVAILABLE_RERANKER_PROVIDERS,
        "reranker",
    )
    if name == "none":
        from app.providers.custom.reranker import NoOpRerankerProvider
        return NoOpRerankerProvider()
    raise ValueError(f"Reranker provider '{name}' not implemented.")


def get_rag_pipeline_provider():
    """Resolve RAG_PIPELINE_PROVIDER to a RagPipelineProvider instance."""
    name = _resolve(
        settings.RAG_PIPELINE_PROVIDER.strip().lower(),
        AVAILABLE_RAG_PIPELINE_PROVIDERS,
        "RAG pipeline",
    )
    if name == "custom":
        from app.providers.custom.rag_pipeline import CustomRagPipelineProvider
        return CustomRagPipelineProvider()
    raise ValueError(f"RAG pipeline provider '{name}' not implemented.")


# ---------------------------------------------------------------------------
# Embedding reindex status  (Phase 23)
# ---------------------------------------------------------------------------

def get_embedding_reindex_status() -> dict:
    """
    Safely inspect the Qdrant collection to determine if a reindex is required.

    Returns a dict with:
      - collection_name: Qdrant collection name from settings
      - collection_dimension: int from Qdrant, or "unknown" if inspection fails
      - configured_dimension: int from settings.EMBEDDING_DIMENSION
      - reindex_required: bool — True if dimensions mismatch
      - error: str or None — error message if Qdrant inspection failed

    This function NEVER raises — all failures are caught and reported as
    safe "unknown" values so the admin config endpoint remains functional.
    """
    result = {
        "collection_name": settings.QDRANT_COLLECTION,
        "collection_dimension": "unknown",
        "configured_dimension": settings.EMBEDDING_DIMENSION,
        "reindex_required": "unknown",
        "error": None,
    }

    try:
        from app.services.vector.qdrant_service import get_qdrant_client
        client = get_qdrant_client()
        collection_info = client.get_collection(collection_name=settings.QDRANT_COLLECTION)
        vectors_config = collection_info.config.params.vectors
        if vectors_config and hasattr(vectors_config, "size"):
            result["collection_dimension"] = vectors_config.size
            result["reindex_required"] = vectors_config.size != settings.EMBEDDING_DIMENSION
    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Provider status summary (safe, no secrets)
# ---------------------------------------------------------------------------

def get_provider_status() -> dict:
    """
    Return a safe, admin-readable provider status summary.

    No API keys, tokens, or secrets are included.
    """
    reindex_status = get_embedding_reindex_status()
    langchain_available = _is_langchain_available()
    ragas_available = _is_ragas_available()

    return {
        "available_providers": {
            "document_loader": AVAILABLE_DOCUMENT_LOADER_PROVIDERS,
            "text_splitter": AVAILABLE_TEXT_SPLITTER_PROVIDERS,
            "embedding": AVAILABLE_EMBEDDING_PROVIDERS,
            "vector_store": AVAILABLE_VECTOR_STORE_PROVIDERS,
            "retriever": AVAILABLE_RETRIEVER_PROVIDERS,
            "reranker": AVAILABLE_RERANKER_PROVIDERS,
            "rag_pipeline": AVAILABLE_RAG_PIPELINE_PROVIDERS,
        },
        "active_providers": {
            "document_loader": settings.DOCUMENT_LOADER_PROVIDER,
            "text_splitter": settings.TEXT_SPLITTER_PROVIDER,
            "embedding": settings.EMBEDDING_PROVIDER,
            "vector_store": settings.VECTOR_STORE_PROVIDER,
            "retriever": settings.RETRIEVER_PROVIDER,
            "reranker": settings.RERANKER_PROVIDER,
            "rag_pipeline": settings.RAG_PIPELINE_PROVIDER,
        },
        "langchain_available": langchain_available,
        "langchain_enabled": langchain_available
        and (
            settings.DOCUMENT_LOADER_PROVIDER.strip().lower() == "langchain"
            or settings.TEXT_SPLITTER_PROVIDER.strip().lower() == "langchain"
        ),
        "provider_switching_ready": True,
        "unsupported_providers_disabled": True,
        # Phase 23: Embedding upgrade foundation
        "embedding_status": {
            "active_provider": settings.EMBEDDING_PROVIDER,
            "active_model": settings.EMBEDDING_MODEL,
            "active_dimension": settings.EMBEDDING_DIMENSION,
            "normalize": settings.EMBEDDING_NORMALIZE,
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
            "device": settings.EMBEDDING_DEVICE,
            "collection_name": reindex_status["collection_name"],
            "collection_dimension": reindex_status["collection_dimension"],
            "reindex_required": reindex_status["reindex_required"],
            "future_providers": dict(FUTURE_EMBEDDING_PROVIDERS),
        },
        # Phase 25: Retriever & Reranker upgrade foundation
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
            "future_rerankers": dict(FUTURE_RERANKER_PROVIDERS),
        },
        # Phase 26: RAGAS evaluation foundation
        "ragas_status": {
            "ragas_enabled": settings.RAGAS_ENABLED,
            "ragas_available": ragas_available,
            "evaluator_provider": settings.RAGAS_EVALUATOR_PROVIDER,
            "evaluator_model": settings.RAGAS_EVALUATOR_MODEL,
            "report_dir": settings.RAGAS_REPORT_DIR,
        },
    }
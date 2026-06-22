"""
Provider Factory

Factory functions that resolve provider names (from environment settings) to
provider instances. Raises ValueError on unknown provider names — fail-fast
is intentional since a misconfigured provider is a deployment error.

Phase 22 supported providers:
  - custom        → CustomDocumentLoaderProvider, CustomTextSplitterProvider,
                    CustomRetrieverProvider, CustomRagPipelineProvider
  - local         → CustomEmbeddingProvider  (sentence-transformers)
  - qdrant        → CustomVectorStoreProvider
  - none          → NoOpRerankerProvider

Future phases will add LangChain adapters and additional provider variants.
"""

from app.core.config import settings

# ---------------------------------------------------------------------------
# Supported providers per module (Phase 22 foundation only)
# ---------------------------------------------------------------------------

AVAILABLE_DOCUMENT_LOADER_PROVIDERS = ["custom"]
AVAILABLE_TEXT_SPLITTER_PROVIDERS = ["custom"]
AVAILABLE_EMBEDDING_PROVIDERS = ["local"]          # "openai" coming in future phase
AVAILABLE_VECTOR_STORE_PROVIDERS = ["qdrant"]
AVAILABLE_RETRIEVER_PROVIDERS = ["custom"]
AVAILABLE_RERANKER_PROVIDERS = ["none"]
AVAILABLE_RAG_PIPELINE_PROVIDERS = ["custom"]


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
# Factory functions  (lazy imports to avoid loading sentence-transformers at import time)
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
# Provider status summary (safe, no secrets)
# ---------------------------------------------------------------------------

def get_provider_status() -> dict:
    """
    Return a safe, admin-readable provider status summary.

    No API keys, tokens, or secrets are included.
    """
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
        "langchain_available": False,
        "langchain_enabled": False,
        "provider_switching_ready": True,   # foundation ready; UI switching in future phase
        "unsupported_providers_disabled": True,
    }
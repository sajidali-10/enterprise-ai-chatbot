import os
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.mock_provider import MockEmbeddingProvider
from app.services.embeddings.openai_compatible_provider import OpenAICompatibleEmbeddingProvider

# Lazy import for optional dependencies
_sentence_transformers_provider = None


def _get_sentence_transformers_provider():
    global _sentence_transformers_provider
    if _sentence_transformers_provider is None:
        from app.services.embeddings.sentence_transformers_provider import (
            SentenceTransformersEmbeddingProvider,
        )
        _sentence_transformers_provider = SentenceTransformersEmbeddingProvider()
    return _sentence_transformers_provider


def get_embedding_provider() -> EmbeddingProvider:
    provider_name = os.getenv("EMBEDDING_PROVIDER", "mock").lower().strip()
    if provider_name == "openai":
        return OpenAICompatibleEmbeddingProvider()
    elif provider_name == "local":
        return _get_sentence_transformers_provider()
    return MockEmbeddingProvider()


def get_embedding_provider_info() -> dict:
    """Return info about the current embedding provider for debugging."""
    provider_name = os.getenv("EMBEDDING_PROVIDER", "mock").lower().strip()
    if provider_name == "openai":
        return {"provider": "openai", "model": os.getenv("OPENAI_EMBEDDING_MODEL", "unknown")}
    elif provider_name == "local":
        try:
            p = _get_sentence_transformers_provider()
            return {"provider": "local", "model": p.model_name, "dimension": p.dimension}
        except ImportError as e:
            return {"provider": "local", "error": str(e)}
    return {"provider": "mock", "dimension": int(os.getenv("EMBEDDING_DIMENSION", "384"))}
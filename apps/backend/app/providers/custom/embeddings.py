"""
Custom Embeddings Adapter

Wraps the existing sentence-transformers local embedding provider.
Delegates to app.services.embeddings for model loading and embedding.
"""

from app.services.embeddings import get_embedding_provider
from app.providers.base import EmbeddingProvider


class CustomEmbeddingProvider(EmbeddingProvider):
    """
    Phase 22 active provider (local): wraps sentence-transformers/all-MiniLM-L6-v2.

    Exposes the same interface as the base EmbeddingProvider ABC.
    The underlying get_embedding_provider() is called once lazily on first use.
    """

    def __init__(self):
        self._provider = get_embedding_provider()

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    @property
    def model_name(self) -> str | None:
        """Extended property — available on sentence-transformers provider."""
        return getattr(self._provider, "model_name", None)
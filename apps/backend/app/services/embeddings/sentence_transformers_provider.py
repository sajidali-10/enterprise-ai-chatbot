"""
Local Sentence Transformers Embedding Provider

Uses sentence-transformers/all-MiniLM-L6-v2 for high-quality local embeddings.
Dimension: 384, no API key required.
"""

import os
from typing import Optional
from app.services.embeddings.base import EmbeddingProvider
from app.core.config import settings

# Lazy import to avoid hard dependency when not using local embeddings
_sentence_transformers = None
_transformers_model = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _sentence_transformers, _transformers_model
    if _sentence_transformers is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv(
                "LOCAL_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            _sentence_transformers = True
            _transformers_model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
    return _transformers_model


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._model_name = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = _get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
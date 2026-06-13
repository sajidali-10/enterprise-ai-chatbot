"""
Local Sentence Transformers Embedding Provider

Uses sentence-transformers/all-MiniLM-L6-v2 for high-quality local embeddings.
Dimension: 384, no API key required.
"""

import logging
import os
from typing import List
from app.services.embeddings.base import EmbeddingProvider
from app.core.config import settings

logger = logging.getLogger(__name__)

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

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Defensive validation
        if not isinstance(texts, list):
            logger.error(f"Embedding input must be a list, got {type(texts)}")
            raise ValueError("Embedding input must be a list of strings")

        # Validate and classify each input; preserve ordering with zero vectors for invalid inputs
        valid_texts: list[str] = []
        valid_indices: list[int] = []
        for i, text in enumerate(texts):
            if text is None:
                logger.warning(f"Skipping None value at index {i} in embedding input")
                continue
            if not isinstance(text, str):
                logger.warning(f"Skipping non-string value at index {i}: {type(text)}")
                continue
            if text.strip() == "":
                logger.warning(f"Skipping empty string at index {i} in embedding input")
                continue
            valid_texts.append(text)
            valid_indices.append(i)

        # If no valid texts, return all zero vectors to preserve output shape
        if not valid_texts:
            logger.warning("All texts are invalid (None/empty/non-string), returning zero vectors")
            return [[0.0] * self._dimension for _ in texts]

        model = _get_model()
        try:
            embeddings = model.encode(valid_texts, normalize_embeddings=True)
            embeddings_list = embeddings.tolist()
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            raise

        # Reconstruct output preserving input order: zero vectors for invalid inputs
        output: list[list[float]] = [[0.0] * self._dimension for _ in texts]
        for idx, emb in zip(valid_indices, embeddings_list):
            output[idx] = emb

        return output

"""Tests for embedding providers."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.services.embeddings.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)


class FakeModel:
    """Fake sentence-transformers model for testing."""
    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def encode(self, texts, normalize_embeddings=True):
        # Return deterministic fake embeddings
        return np.array([[float(i + 1)] * self._dimension for i in range(len(texts))])


@pytest.fixture
def provider():
    return SentenceTransformersEmbeddingProvider()


def test_embed_valid_text_returns_embeddings(provider, monkeypatch):
    """Valid text still embeds successfully."""
    monkeypatch.setattr(
        "app.services.embeddings.sentence_transformers_provider._get_model",
        lambda: FakeModel(),
    )
    result = provider.embed(["hello world", "another sentence"])
    assert len(result) == 2
    assert len(result[0]) == provider.dimension
    assert len(result[1]) == provider.dimension
    assert result[0][0] == 1.0
    assert result[1][0] == 2.0


def test_embed_rejects_none_safely(provider, monkeypatch):
    """Local embedding provider rejects None safely (returns zero vector)."""
    monkeypatch.setattr(
        "app.services.embeddings.sentence_transformers_provider._get_model",
        lambda: FakeModel(),
    )
    result = provider.embed([None])
    assert len(result) == 1
    assert len(result[0]) == provider.dimension
    assert all(v == 0.0 for v in result[0])


def test_embed_handles_empty_string_safely(provider, monkeypatch):
    """Local embedding provider handles empty string safely (returns zero vector)."""
    monkeypatch.setattr(
        "app.services.embeddings.sentence_transformers_provider._get_model",
        lambda: FakeModel(),
    )
    result = provider.embed([""])
    assert len(result) == 1
    assert len(result[0]) == provider.dimension
    assert all(v == 0.0 for v in result[0])


def test_embed_preserves_order_with_mixed_inputs(provider, monkeypatch):
    """Mixed valid/invalid inputs preserve output order."""
    monkeypatch.setattr(
        "app.services.embeddings.sentence_transformers_provider._get_model",
        lambda: FakeModel(),
    )
    result = provider.embed(["hello", None, "", "world"])
    assert len(result) == 4
    # "hello" -> first embedding from fake model
    assert result[0][0] == 1.0
    # None -> zero vector
    assert all(v == 0.0 for v in result[1])
    # "" -> zero vector
    assert all(v == 0.0 for v in result[2])
    # "world" -> second embedding from fake model
    assert result[3][0] == 2.0


def test_embed_rejects_non_list_input(provider):
    """Non-list input raises ValueError."""
    with pytest.raises(ValueError, match="list of strings"):
        provider.embed("not a list")


def test_embed_all_invalid_returns_zero_vectors(provider, monkeypatch):
    """All invalid inputs return zero vectors preserving count."""
    monkeypatch.setattr(
        "app.services.embeddings.sentence_transformers_provider._get_model",
        lambda: FakeModel(),
    )
    result = provider.embed([None, "", None])
    assert len(result) == 3
    for emb in result:
        assert len(emb) == provider.dimension
        assert all(v == 0.0 for v in emb)

"""
Tests for Provider Factory

Validates:
- Default providers resolve to correct custom/local/qdrant/none implementations
- Unknown provider names raise ValueError with a clear message
- Provider status summary is safe (no secrets, no API keys exposed)
- Existing RAG config endpoint still works and includes provider_status
- All existing admin RAG config tests still pass

Note: imports go directly to individual module files (not custom/__init__.py)
to avoid triggering eager adapter imports that load heavy dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestProviderFactoryDefaults:
    """Factory resolves configured providers to the correct Phase 22 implementations."""

    def test_document_loader_provider_resolves_to_custom(self):
        from app.providers.factory import get_document_loader_provider
        from app.providers.custom.document_loader import CustomDocumentLoaderProvider

        provider = get_document_loader_provider()
        assert isinstance(provider, CustomDocumentLoaderProvider)

    def test_text_splitter_provider_resolves_to_custom(self):
        from app.providers.factory import get_text_splitter_provider
        from app.providers.custom.text_splitter import CustomTextSplitterProvider

        provider = get_text_splitter_provider()
        assert isinstance(provider, CustomTextSplitterProvider)

    def test_embedding_provider_resolves_to_local(self):
        from app.providers.factory import get_embedding_provider
        from app.providers.custom.embeddings import CustomEmbeddingProvider

        provider = get_embedding_provider()
        assert isinstance(provider, CustomEmbeddingProvider)

    def test_vector_store_provider_resolves_to_qdrant(self):
        from app.providers.factory import get_vector_store_provider
        from app.providers.custom.vector_store import CustomVectorStoreProvider

        provider = get_vector_store_provider()
        assert isinstance(provider, CustomVectorStoreProvider)

    def test_retriever_provider_resolves_to_custom(self):
        from app.providers.factory import get_retriever_provider
        from app.providers.custom.retriever import CustomRetrieverProvider

        provider = get_retriever_provider()
        assert isinstance(provider, CustomRetrieverProvider)

    def test_reranker_provider_resolves_to_noop(self):
        from app.providers.factory import get_reranker_provider
        from app.providers.custom.reranker import NoOpRerankerProvider

        provider = get_reranker_provider()
        assert isinstance(provider, NoOpRerankerProvider)

    def test_rag_pipeline_provider_resolves_to_custom(self):
        from app.providers.factory import get_rag_pipeline_provider
        from app.providers.custom.rag_pipeline import CustomRagPipelineProvider

        provider = get_rag_pipeline_provider()
        assert isinstance(provider, CustomRagPipelineProvider)


class TestProviderFactoryErrors:
    """Factory raises ValueError on unknown provider names."""

    def test_unknown_document_loader_raises(self, monkeypatch):
        monkeypatch.setenv("DOCUMENT_LOADER_PROVIDER", "langchain")
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_document_loader_provider

        with pytest.raises(ValueError, match="Unknown document loader provider"):
            get_document_loader_provider()

        monkeypatch.delenv("DOCUMENT_LOADER_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)

    def test_unknown_text_splitter_raises(self, monkeypatch):
        monkeypatch.setenv("TEXT_SPLITTER_PROVIDER", "langchain")
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_text_splitter_provider

        with pytest.raises(ValueError, match="Unknown text splitter provider"):
            get_text_splitter_provider()

        monkeypatch.delenv("TEXT_SPLITTER_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)

    def test_unknown_embedding_raises(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_embedding_provider

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider()

        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)

    def test_unknown_vector_store_raises(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "chroma")
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_vector_store_provider

        with pytest.raises(ValueError, match="Unknown vector store provider"):
            get_vector_store_provider()

        monkeypatch.delenv("VECTOR_STORE_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)

    def test_unknown_retriever_raises(self, monkeypatch):
        monkeypatch.setenv("RETRIEVER_PROVIDER", "langchain")
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_retriever_provider

        with pytest.raises(ValueError, match="Unknown retriever provider"):
            get_retriever_provider()

        monkeypatch.delenv("RETRIEVER_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)

    def test_unknown_reranker_raises(self, monkeypatch):
        monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_reranker_provider

        with pytest.raises(ValueError, match="Unknown reranker provider"):
            get_reranker_provider()

        monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)


class TestProviderStatus:
    """Provider status summary is safe — no secrets or API keys exposed."""

    def test_provider_status_returns_dict(self):
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        assert isinstance(status, dict)

    def test_provider_status_has_expected_keys(self):
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        assert "available_providers" in status
        assert "active_providers" in status
        assert "langchain_available" in status
        assert "langchain_enabled" in status
        assert "provider_switching_ready" in status
        assert "unsupported_providers_disabled" in status

    def test_provider_status_no_secrets(self):
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        raw = str(status).lower()

        secret_keywords = ["secret", "password", "token", "api_key", "key", "credential"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Potential secret keyword(s) found in provider_status: {found}"

    def test_provider_status_langchain_not_enabled(self):
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        assert status["langchain_available"] is False
        assert status["langchain_enabled"] is False

    def test_provider_status_switching_ready(self):
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        assert status["provider_switching_ready"] is True


class TestRagConfigIntegration:
    """RAG config endpoint includes provider_status and remains backward-compatible."""

    def test_rag_config_includes_provider_status(self, jwt_admin_client):
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()
        assert "provider_status" in data

    def test_rag_config_provider_status_no_secrets(self, jwt_admin_client):
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()
        status = data["provider_status"]
        raw = str(status).lower()

        secret_keywords = ["secret", "password", "token", "api_key", "key", "credential"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Potential secret keyword(s) found in provider_status: {found}"

    def test_rag_config_existing_fields_unchanged(self, jwt_admin_client):
        """All existing fields from Phase 21 are still present."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()

        expected = [
            "rag_pipeline_provider", "document_loader_provider",
            "text_splitter_provider", "retriever_provider", "reranker_provider",
            "embedding_provider", "embedding_model", "embedding_dimension",
            "vector_store_provider", "top_k", "score_threshold",
            "chunk_size", "chunk_overlap",
            "context_max_chunks", "context_max_characters",
            "conversation_context_enabled",
            "conversation_context_max_messages",
            "conversation_context_max_characters",
            "langchain_enabled", "reranker_enabled",
            "provider_status",
        ]
        for field in expected:
            assert field in data, f"Missing field: {field}"
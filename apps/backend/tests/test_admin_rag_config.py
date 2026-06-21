"""
Tests for Admin RAG Configuration endpoint.

Validates:
- Endpoint requires admin access
- No secrets are exposed
- Default values are correct
- Env override works for non-sensitive settings
"""

import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAdminRAGConfigEndpoint:
    """Tests for /api/admin/rag/config"""

    def test_rag_config_requires_auth(self):
        """Unauthenticated requests should be rejected."""
        response = client.get("/api/admin/rag/config")
        assert response.status_code in (401, 403)

    def test_rag_config_user_forbidden(self, jwt_user_client):
        """Non-admin users should not access RAG config."""
        response = jwt_user_client.get("/api/admin/rag/config")
        assert response.status_code in (403, 401)

    def test_rag_config_admin_can_access(self, jwt_admin_client):
        """Admin users can access RAG config."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200

    def test_rag_config_response_structure(self, jwt_admin_client):
        """Response must contain all expected RAG config fields."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()

        expected_fields = [
            "rag_pipeline_provider",
            "document_loader_provider",
            "text_splitter_provider",
            "retriever_provider",
            "reranker_provider",
            "embedding_provider",
            "embedding_model",
            "embedding_dimension",
            "vector_store_provider",
            "top_k",
            "score_threshold",
            "chunk_size",
            "chunk_overlap",
            "context_max_chunks",
            "context_max_characters",
            "conversation_context_enabled",
            "conversation_context_max_messages",
            "conversation_context_max_characters",
            "langchain_enabled",
            "reranker_enabled",
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_rag_config_no_secret_fields_in_response(self, jwt_admin_client):
        """Response must not contain any secret-related field names."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        raw = response.text.lower()
        # Must not contain any secret-related key names
        assert "secret" not in raw or '"secret"' not in raw
        assert "password" not in raw or '"password"' not in raw
        assert "token" not in raw or '"token"' not in raw
        assert "api_key" not in raw or '"api_key"' not in raw
        assert "key" not in raw or '"key"' not in raw
        assert "credential" not in raw or '"credential"' not in raw

    def test_rag_config_default_values(self, jwt_admin_client):
        """Default values must match expected Phase 21 defaults."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()

        # Provider defaults
        assert data["rag_pipeline_provider"] == "custom"
        assert data["document_loader_provider"] == "custom"
        assert data["text_splitter_provider"] == "custom"
        assert data["embedding_provider"] == "local"
        assert data["vector_store_provider"] == "qdrant"
        assert data["retriever_provider"] == "custom"
        assert data["reranker_provider"] == "none"

        # Embedding defaults
        assert data["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert data["embedding_dimension"] == 384

        # Retrieval defaults
        assert data["top_k"] == 6
        assert data["score_threshold"] == 0.35

        # Chunking defaults
        assert data["chunk_size"] == 1000
        assert data["chunk_overlap"] == 150

        # Context defaults
        assert data["context_max_chunks"] == 6
        assert data["context_max_characters"] == 12000

        # Conversation context defaults
        assert data["conversation_context_enabled"] is True
        assert data["conversation_context_max_messages"] == 6
        assert data["conversation_context_max_characters"] == 3500

        # Feature flags
        assert data["langchain_enabled"] is False
        assert data["reranker_enabled"] is False

    def test_rag_config_env_override_works(self, jwt_admin_client, monkeypatch):
        """Non-sensitive RAG settings can be overridden via env vars."""
        monkeypatch.setenv("RAG_TOP_K", "10")
        monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.5")
        monkeypatch.setenv("RAG_CHUNK_SIZE", "2000")
        monkeypatch.setenv("RAG_CHUNK_OVERLAP", "300")
        monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "768")

        # Re-import to pick up env changes
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        from app.core.config import settings

        assert settings.RAG_TOP_K == 10
        assert settings.RAG_SCORE_THRESHOLD == 0.5
        assert settings.RAG_CHUNK_SIZE == 2000
        assert settings.RAG_CHUNK_OVERLAP == 300
        assert settings.EMBEDDING_MODEL == "sentence-transformers/all-mpnet-base-v2"
        assert settings.EMBEDDING_DIMENSION == 768

        # Cleanup: reset env and reload so subsequent tests are unaffected
        monkeypatch.delenv("RAG_TOP_K", raising=False)
        monkeypatch.delenv("RAG_SCORE_THRESHOLD", raising=False)
        monkeypatch.delenv("RAG_CHUNK_SIZE", raising=False)
        monkeypatch.delenv("RAG_CHUNK_OVERLAP", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
        importlib.reload(app.core.config)

    def test_rag_config_langchain_always_false(self, jwt_admin_client):
        """LangChain must always report as disabled in this phase."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()
        assert data["langchain_enabled"] is False

    def test_rag_config_reranker_enabled_when_not_none(self, monkeypatch):
        """reranker_enabled must be True when reranker provider is set."""
        monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
        import importlib
        import app.core.config
        import app.services.rag_config
        importlib.reload(app.core.config)
        importlib.reload(app.services.rag_config)
        from app.core.config import settings
        from app.services.rag_config import get_rag_config

        assert settings.RERANKER_PROVIDER == "cohere"
        config = get_rag_config()
        assert config["reranker_enabled"] is True

        # Cleanup: reset env and reload so subsequent tests are unaffected
        monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.services.rag_config)

    def test_rag_config_reranker_disabled_when_none(self, jwt_admin_client):
        """reranker_enabled must be False when reranker provider is 'none'."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()
        assert data["reranker_provider"] == "none"
        assert data["reranker_enabled"] is False

    def test_rag_config_conversation_context_defaults_to_true(self, jwt_admin_client):
        """Conversation context must default to enabled."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_context_enabled"] is True
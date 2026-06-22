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

    def test_langchain_document_loader_resolves_when_available(self, monkeypatch):
        """Phase 24: langchain is now a known document loader provider.

        Verifies that when DOCUMENT_LOADER_PROVIDER=langchain and langchain
        packages are installed, the factory returns LangChainDocumentLoaderProvider.
        """
        monkeypatch.setenv("DOCUMENT_LOADER_PROVIDER", "langchain")
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_document_loader_provider
        from app.providers.langchain.document_loader import LangChainDocumentLoaderProvider

        provider = get_document_loader_provider()
        assert isinstance(provider, LangChainDocumentLoaderProvider)

        monkeypatch.delenv("DOCUMENT_LOADER_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)

    def test_langchain_document_loader_missing_deps_raises(self, monkeypatch):
        """Phase 24: if langchain is set but packages are missing → ValueError with install hint."""
        import importlib
        import app.core.config
        import app.providers.factory

        monkeypatch.setenv("DOCUMENT_LOADER_PROVIDER", "langchain")
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)
        # Apply patch AFTER reload, THEN import the function (so function __globals__ sees patched name)
        original_fn = app.providers.factory._is_langchain_available
        app.providers.factory._is_langchain_available = lambda: False
        try:
            from app.providers.factory import get_document_loader_provider
            with pytest.raises(ValueError, match="langchain.*not installed"):
                get_document_loader_provider()
        finally:
            app.providers.factory._is_langchain_available = original_fn
            monkeypatch.delenv("DOCUMENT_LOADER_PROVIDER", raising=False)
            importlib.reload(app.core.config)
            importlib.reload(app.providers.factory)

    def test_unknown_document_loader_raises(self, monkeypatch):
        """An arbitrary unknown provider name still raises ValueError."""
        monkeypatch.setenv("DOCUMENT_LOADER_PROVIDER", "nonexistent_provider")
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

    def test_langchain_text_splitter_resolves_when_available(self, monkeypatch):
        """Phase 24: langchain is now a known text splitter provider.

        Verifies that when TEXT_SPLITTER_PROVIDER=langchain and langchain
        packages are installed, the factory returns LangChainTextSplitterProvider.
        """
        monkeypatch.setenv("TEXT_SPLITTER_PROVIDER", "langchain")
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_text_splitter_provider
        from app.providers.langchain.text_splitter import LangChainTextSplitterProvider

        provider = get_text_splitter_provider()
        assert isinstance(provider, LangChainTextSplitterProvider)

        monkeypatch.delenv("TEXT_SPLITTER_PROVIDER", raising=False)
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)

    def test_langchain_text_splitter_missing_deps_raises(self, monkeypatch):
        """Phase 24: if langchain is set but packages are missing → ValueError with install hint."""
        import importlib
        import app.core.config
        import app.providers.factory

        monkeypatch.setenv("TEXT_SPLITTER_PROVIDER", "langchain")
        importlib.reload(app.core.config)
        importlib.reload(app.providers.factory)
        original_fn = app.providers.factory._is_langchain_available
        app.providers.factory._is_langchain_available = lambda: False
        try:
            from app.providers.factory import get_text_splitter_provider
            with pytest.raises(ValueError, match="langchain.*not installed"):
                get_text_splitter_provider()
        finally:
            app.providers.factory._is_langchain_available = original_fn
            monkeypatch.delenv("TEXT_SPLITTER_PROVIDER", raising=False)
            importlib.reload(app.core.config)
            importlib.reload(app.providers.factory)

    def test_unknown_text_splitter_raises(self, monkeypatch):
        """An arbitrary unknown provider name still raises ValueError."""
        monkeypatch.setenv("TEXT_SPLITTER_PROVIDER", "nonexistent_provider")
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

        secret_keywords = ["secret", "password", "token", "api_key", "credential"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Potential secret keyword(s) found in provider_status: {found}"

    def test_provider_status_langchain_fields_present(self):
        """Phase 24: langchain_available and langchain_enabled must be in status."""
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        # Both fields must exist (values depend on whether langchain is installed)
        assert "langchain_available" in status
        assert "langchain_enabled" in status
        # langchain_enabled must be a bool
        assert isinstance(status["langchain_available"], bool)
        assert isinstance(status["langchain_enabled"], bool)

    def test_provider_status_langchain_not_enabled_when_custom_default(self):
        """When both providers are 'custom', langchain_enabled must be False."""
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        active = status["active_providers"]
        if active["document_loader"] == "custom" and active["text_splitter"] == "custom":
            assert status["langchain_enabled"] is False

    def test_provider_status_switching_ready(self):
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        assert status["provider_switching_ready"] is True

    def test_provider_status_embedding_status_present(self):
        """embedding_status dict must be present in provider_status."""
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        assert "embedding_status" in status

    def test_provider_status_embedding_status_has_required_keys(self):
        """embedding_status must contain all required Phase 23 fields."""
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        es = status["embedding_status"]
        required = [
            "active_provider", "active_model", "active_dimension",
            "normalize", "batch_size", "device",
            "collection_name", "collection_dimension",
            "reindex_required", "future_providers",
        ]
        for key in required:
            assert key in es, f"Missing embedding_status key: {key}"

    def test_provider_status_embedding_status_future_providers(self):
        """Future embedding providers must be listed as 'planned'."""
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        fp = status["embedding_status"]["future_providers"]
        for provider in ["openai", "cohere", "voyage", "bge", "e5"]:
            assert provider in fp, f"Missing future provider: {provider}"
            assert fp[provider] == "planned", f"Future provider {provider} should be 'planned', got {fp[provider]}"

    def test_provider_status_embedding_status_no_secrets(self):
        """embedding_status must not expose API keys or secrets."""
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        es = status["embedding_status"]
        raw = str(es).lower()
        secret_keywords = ["secret", "password", "token", "api_key", "credential"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Potential secret keyword(s) found in embedding_status: {found}"


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

        secret_keywords = ["secret", "password", "token", "api_key", "credential"]
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

    def test_rag_config_embedding_status_in_provider_status(self, jwt_admin_client):
        """RAG config endpoint must expose embedding_status via provider_status."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()
        es = data["provider_status"]["embedding_status"]
        assert es["active_provider"] == "local"
        assert es["active_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert es["active_dimension"] == 384
        assert "collection_name" in es
        assert "future_providers" in es


class TestEmbeddingReindexStatus:
    """Tests for get_embedding_reindex_status()."""

    def test_get_embedding_reindex_status_returns_dict(self):
        from app.providers.factory import get_embedding_reindex_status

        status = get_embedding_reindex_status()
        assert isinstance(status, dict)
        assert "collection_name" in status
        assert "collection_dimension" in status
        assert "configured_dimension" in status
        assert "reindex_required" in status
        assert "error" in status

    def test_get_embedding_reindex_status_safe_on_qdrant_error(self, monkeypatch):
        """get_embedding_reindex_status must never raise — returns safe values on Qdrant failure."""
        import importlib
        import app.services.vector.qdrant_service as qsvc

        # Simulate Qdrant being unavailable: set nonexistent host, clear client cache,
        # and reload config so settings pick up the new env var
        monkeypatch.setenv("QDRANT_HOST", "nonexistent-host")
        qsvc._client = None
        import app.core.config
        importlib.reload(app.core.config)
        importlib.reload(qsvc)
        import app.providers.factory
        importlib.reload(app.providers.factory)
        from app.providers.factory import get_embedding_reindex_status

        # Must not raise — returns safe "unknown" values on connection failure
        status = get_embedding_reindex_status()
        assert status["collection_dimension"] == "unknown"
        assert status["reindex_required"] == "unknown"
        assert status["error"] is not None

        monkeypatch.delenv("QDRANT_HOST", raising=False)
        qsvc._client = None
        importlib.reload(app.core.config)
        importlib.reload(qsvc)
        importlib.reload(app.providers.factory)

    def test_get_embedding_reindex_status_local_dimensions_match(self):
        """When local provider is active and dimensions match, reindex not required."""
        from app.providers.factory import get_embedding_reindex_status
        from app.core.config import settings

        status = get_embedding_reindex_status()
        # Local/sentence-transformers dimension must match collection when no change made
        if isinstance(status["collection_dimension"], int):
            assert status["configured_dimension"] == settings.EMBEDDING_DIMENSION
            assert status["reindex_required"] is False


# ---------------------------------------------------------------------------
# Phase 25: Retriever & Reranker Upgrade Foundation
# ---------------------------------------------------------------------------


class TestNoOpRerankerProvider:
    """Tests for NoOpRerankerProvider — Phase 25."""

    def test_noop_reranker_returns_chunks_unchanged(self):
        """NoOpRerankerProvider must return chunks with original scores preserved."""
        from app.providers.custom.reranker import NoOpRerankerProvider

        provider = NoOpRerankerProvider()
        chunks = [
            {"chunk_id": "a", "document_id": 1, "chunk_index": 0, "content": "hello world", "source_file_name": "doc.txt", "title": "Doc", "score": 0.9},
            {"chunk_id": "b", "document_id": 1, "chunk_index": 1, "content": "another chunk", "source_file_name": "doc.txt", "title": "Doc", "score": 0.7},
        ]
        result = provider.rerank("hello", chunks)

        # Scores must be preserved (rerank_score == original score)
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["rerank_score"] == 0.7
        # Content must not change
        assert result[0]["content"] == "hello world"
        assert result[1]["content"] == "another chunk"
        # chunk_ids preserved in original order
        assert result[0]["chunk_id"] == "a"
        assert result[1]["chunk_id"] == "b"

    def test_noop_reranker_truncates_to_top_n(self):
        """NoOpRerankerProvider with top_n set returns at most top_n chunks in original order."""
        from app.providers.custom.reranker import NoOpRerankerProvider

        provider = NoOpRerankerProvider()
        chunks = [
            {"chunk_id": str(i), "document_id": 1, "chunk_index": i, "content": f"chunk {i}", "source_file_name": "doc.txt", "title": "Doc", "score": 0.1 * i}
            for i in range(1, 6)
        ]
        result = provider.rerank("query", chunks, top_n=3)

        # No-op preserves original order; top_n truncates
        assert len(result) == 3
        assert result[0]["chunk_id"] == "1"  # original order, first 3
        assert result[1]["chunk_id"] == "2"
        assert result[2]["chunk_id"] == "3"

    def test_noop_reranker_exposes_provider_name(self):
        """NoOpRerankerProvider must expose provider_name for status visibility."""
        from app.providers.custom.reranker import NoOpRerankerProvider

        provider = NoOpRerankerProvider()
        assert hasattr(provider, "provider_name")
        assert provider.provider_name == "none"


class TestFutureRerankerProviders:
    """Tests for FUTURE_RERANKER_PROVIDERS — Phase 25."""

    def test_future_reranker_providers_exists(self):
        from app.providers.factory import FUTURE_RERANKER_PROVIDERS

        assert isinstance(FUTURE_RERANKER_PROVIDERS, dict)
        assert len(FUTURE_RERANKER_PROVIDERS) >= 5

    def test_future_reranker_providers_structure(self):
        from app.providers.factory import FUTURE_RERANKER_PROVIDERS

        # none must be available and active
        assert FUTURE_RERANKER_PROVIDERS["none"] == "available"
        # cohere, bge, cross_encoder, langchain must be planned
        for name in ["cohere", "bge", "cross_encoder", "langchain"]:
            assert name in FUTURE_RERANKER_PROVIDERS, f"{name} missing from FUTURE_RERANKER_PROVIDERS"
            assert FUTURE_RERANKER_PROVIDERS[name] == "planned", f"{name} should be 'planned', got {FUTURE_RERANKER_PROVIDERS[name]}"

    def test_future_rerankers_in_provider_status(self):
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        rs = status["retrieval_status"]
        fr = rs.get("future_rerankers", {})
        assert fr.get("none") == "available"
        for name in ["cohere", "bge", "cross_encoder", "langchain"]:
            assert name in fr, f"{name} missing from future_rerankers in provider_status"
            assert fr[name] == "planned"


class TestRetrievalStatus:
    """Tests for retrieval_status in provider_status and rag_config — Phase 25."""

    def test_retrieval_status_has_required_keys(self):
        from app.providers.factory import get_provider_status

        rs = get_provider_status()["retrieval_status"]
        required = [
            "retrieval_mode", "top_k", "score_threshold", "candidate_k",
            "hybrid_enabled", "hybrid_keyword_weight", "hybrid_vector_weight",
            "reranker_provider", "reranker_enabled", "reranker_top_n", "reranker_model",
            "future_rerankers",
        ]
        for key in required:
            assert key in rs, f"Missing retrieval_status key: {key}"

    def test_retrieval_status_default_values(self):
        from app.providers.factory import get_provider_status

        rs = get_provider_status()["retrieval_status"]
        assert rs["retrieval_mode"] == "vector"
        assert rs["top_k"] == 6
        assert rs["candidate_k"] == 15
        assert rs["hybrid_enabled"] is False
        assert rs["reranker_enabled"] is False
        assert rs["reranker_provider"] == "none"
        assert rs["reranker_model"] == "none"
        assert rs["reranker_top_n"] == 6

    def test_retrieval_status_no_secrets(self):
        from app.providers.factory import get_provider_status

        rs = get_provider_status()["retrieval_status"]
        raw = str(rs).lower()
        # Check only high-specificity secret keywords to avoid false positives
        # from field names like hybrid_keyword_weight (contains "key")
        secret_keywords = ["secret", "password", "token", "api_key", "credential"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Potential secret(s) in retrieval_status: {found}"

    def test_hybrid_disabled_by_default(self):
        from app.core.config import settings

        assert settings.HYBRID_SEARCH_ENABLED is False

    def test_retrieval_status_in_rag_config_endpoint(self, jwt_admin_client):
        """RAG config endpoint must expose retrieval_status at top level."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()
        assert "retrieval_status" in data
        rs = data["retrieval_status"]
        assert rs["retrieval_mode"] == "vector"
        assert rs["hybrid_enabled"] is False
        assert rs["reranker_enabled"] is False
        assert rs["reranker_provider"] == "none"

    def test_retrieval_status_no_secrets_in_rag_config(self, jwt_admin_client):
        """RAG config endpoint retrieval_status must not expose secrets."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        rs = response.json()["retrieval_status"]
        raw = str(rs).lower()
        secret_keywords = ["secret", "password", "token", "api_key", "credential"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Potential secret(s) in rag_config retrieval_status: {found}"


class TestInvalidRerankerProvider:
    """Phase 25: invalid reranker provider name raises ValueError."""

    def test_unknown_reranker_raises_clear_error(self, monkeypatch):
        monkeypatch.setenv("RERANKER_PROVIDER", "nonexistent_reranker")
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
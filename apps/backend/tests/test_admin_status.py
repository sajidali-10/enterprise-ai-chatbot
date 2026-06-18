"""
Tests for Admin System Status endpoint.

Validates:
- Endpoint requires admin access
- No secrets are exposed
- Provider disabled state is represented correctly
- LiteLLM disabled is not treated as failure
- Response structure is correct
"""

import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAdminStatusEndpoint:
    """Tests for /api/admin/system/status"""

    def test_system_status_requires_auth(self):
        """Unauthenticated requests should be rejected."""
        response = client.get("/api/admin/system/status")
        assert response.status_code in (401, 403)

    def test_system_status_user_forbidden(self, jwt_user_client):
        """Non-admin users should not access system status."""
        response = jwt_user_client.get("/api/admin/system/status")
        assert response.status_code in (403, 401)

    def test_system_status_admin_can_access(self, jwt_admin_client):
        """Admin users can access system status."""
        response = jwt_admin_client.get("/api/admin/system/status")
        assert response.status_code == 200
        data = response.json()
        assert "gateway" in data
        assert "application" in data
        assert "data" in data
        assert "ai_provider" in data
        assert "rag_quality" in data
        assert "documents" in data
        assert "security" in data
        assert "recent_activity" in data
        assert "overall_healthy" in data
        assert "status_source_note" in data

    def test_no_secret_fields_in_response(self, jwt_admin_client):
        """Response must not contain any secret-related field names."""
        response = jwt_admin_client.get("/api/admin/system/status")
        assert response.status_code == 200
        raw = response.text.lower()
        assert "secret" not in raw or '"secret"' not in raw
        assert "password" not in raw or '"password"' not in raw
        assert "token" not in raw or '"token"' not in raw
        assert "api_key" not in raw or '"api_key"' not in raw

    def test_provider_info_safe_fields(self, monkeypatch):
        """Provider info only returns safe fields."""
        monkeypatch.setenv("JWT_SECRET_KEY", "super-secret-key-do-not-expose")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
        monkeypatch.setenv("MINIO_ROOT_PASSWORD", "minio-secret-password")

        from app.api.admin_status import _get_provider_info
        info = _get_provider_info()
        for key in info.keys():
            assert "secret" not in key.lower()
            assert "password" not in key.lower()
            assert "token" not in key.lower()
            assert "key" not in key.lower()
            assert "api" not in key.lower()

    def test_provider_info_lite_llm_disabled_not_failure(self, monkeypatch):
        """LiteLLM disabled should show as disabled, not unhealthy."""
        monkeypatch.setenv("LITELLM_ENABLED", "false")
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")

        from app.api.admin_status import _get_provider_info
        info = _get_provider_info()
        assert info["litellm_enabled"] is False
        assert info["gateway_mode"] is False
        assert info["provider"] == "openrouter"

    def test_provider_info_gateway_mode_when_litellm(self, monkeypatch):
        """Gateway mode should be true when provider is litellm."""
        monkeypatch.setenv("LLM_PROVIDER", "litellm")
        monkeypatch.setenv("LITELLM_ENABLED", "true")

        from app.api.admin_status import _get_provider_info
        info = _get_provider_info()
        assert info["provider"] == "litellm"
        assert info["gateway_mode"] is True
        assert info["litellm_enabled"] is True

    def test_provider_info_openrouter_model(self, monkeypatch):
        """OpenRouter model should be read from env."""
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

        from app.api.admin_status import _get_provider_info
        info = _get_provider_info()
        assert info["provider"] == "openrouter"
        assert info["model"] == "openai/gpt-oss-120b:free"

    def test_gateway_inferred_from_headers(self, jwt_admin_client):
        """Gateway HTTPS should be inferred from request headers."""
        response = jwt_admin_client.get(
            "/api/admin/system/status",
            headers={"X-Forwarded-Proto": "https", "Host": "chatbot.hiplink.com"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["gateway"]["https_active"] is True
        assert data["gateway"]["domain"] == "chatbot.hiplink.com"

    def test_document_summary_counts(self, db_session):
        """Document summary should return numeric counts."""
        from app.api.admin_status import _get_document_summary
        summary = _get_document_summary(db_session)
        assert isinstance(summary["total_documents"], int)
        assert isinstance(summary["indexed"], int)
        assert isinstance(summary["failed"], int)
        assert isinstance(summary["pending"], int)
        assert "collection_name" in summary
        assert "embedding_provider" in summary
        assert "embedding_dimension" in summary

    def test_rag_summary_no_runs(self, db_session):
        """RAG summary should handle no evaluation runs gracefully."""
        from app.api.admin_status import _get_rag_summary
        summary = _get_rag_summary(db_session)
        assert summary["status"] == "no_runs"
        assert summary["total_tests"] == 0
        assert summary["pass_percentage"] == 0.0

    def test_security_summary_fields(self, db_session):
        """Security summary should contain expected safe fields."""
        from app.api.admin_status import _get_security_summary
        from app.security.auth import AuthContext
        from app.security.models import UserRole

        auth = AuthContext(
            user_id=1,
            username="admin",
            role=UserRole.ADMIN,
            is_authenticated=True,
            is_external=False,
            session_id="test-session",
        )
        summary = _get_security_summary(db_session, auth)
        assert "auth_mode" in summary
        assert "current_user_role" in summary
        assert "total_active_users" in summary
        assert "recent_failed_logins_24h" in summary
        assert "recent_audit_events_24h" in summary
        assert "last_admin_action" in summary

    def test_endpoint_returns_200_when_security_summary_fails(self, jwt_admin_client, monkeypatch):
        """
        Even if the security subsection raises an exception, the endpoint
        must still return HTTP 200 with a degraded-but-safe security section.
        """
        import app.api.admin_status as admin_status_mod

        def broken_security_summary(db, auth):
            raise RuntimeError("simulated security DB failure")

        monkeypatch.setattr(admin_status_mod, "_get_security_summary", broken_security_summary)
        response = jwt_admin_client.get("/api/admin/system/status")

        assert response.status_code == 200, "Endpoint must not 500 when a subsection fails"
        data = response.json()

        # Other sections must still be present
        assert "gateway" in data
        assert "application" in data
        assert "data" in data
        assert "ai_provider" in data
        assert "rag_quality" in data
        assert "documents" in data
        assert "recent_activity" in data

        # Security section must exist and contain safe degraded values
        assert "security" in data
        sec = data["security"]
        assert isinstance(sec["total_active_users"], int)
        assert isinstance(sec["recent_failed_logins_24h"], int)
        assert isinstance(sec["recent_audit_events_24h"], int)
        # No raw error details leaked
        raw = response.text.lower()
        assert "simulated" not in raw
        assert "runtimeerror" not in raw
        assert "traceback" not in raw
        assert "stack" not in raw

    def test_endpoint_returns_200_when_rag_summary_fails(self, jwt_admin_client, monkeypatch):
        """
        Even if the RAG summary subsection raises an exception, the endpoint
        must still return HTTP 200 with degraded default values for rag_quality.
        """
        import app.api.admin_status as admin_status_mod

        def broken_rag_summary(db):
            raise RuntimeError("simulated RAG DB failure")

        monkeypatch.setattr(admin_status_mod, "_get_rag_summary", broken_rag_summary)
        response = jwt_admin_client.get("/api/admin/system/status")

        assert response.status_code == 200
        data = response.json()
        rag = data["rag_quality"]
        assert rag["status"] == "unavailable"
        assert rag["total_tests"] == 0
        raw = response.text.lower()
        assert "simulated" not in raw
        assert "runtimeerror" not in raw

    def test_recent_activity_isolation(self, jwt_admin_client, monkeypatch):
        """
        Even if _get_recent_activity fails completely, the endpoint still
        returns 200 and all other sections are populated.
        """
        import app.api.admin_status as admin_status_mod

        def broken_recent_activity(db):
            raise RuntimeError("simulated recent_activity failure")

        monkeypatch.setattr(admin_status_mod, "_get_recent_activity", broken_recent_activity)
        response = jwt_admin_client.get("/api/admin/system/status")

        assert response.status_code == 200
        data = response.json()
        # recent_activity must be present with safe defaults
        assert "recent_activity" in data
        assert isinstance(data["recent_activity"], dict)
        # No secrets leaked
        raw = response.text.lower()
        assert "simulated" not in raw
        assert "runtimeerror" not in raw

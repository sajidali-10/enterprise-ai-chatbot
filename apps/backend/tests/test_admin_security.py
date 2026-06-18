"""
Tests for Admin Security Operations endpoint.

Validates:
- Endpoint requires admin access
- Non-admin and unauthenticated requests are rejected
- Response contains expected safe summary sections
- No secrets are exposed
- Failed subsection returns degraded defaults, not HTTP 500
- Export endpoints work
- Invalid time range parameters are handled
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.security.models import AuditAction, AuditLog, UserRole

client = TestClient(app)


class TestAdminSecurityOverview:
    """Tests for GET /api/admin/security/overview"""

    def test_overview_requires_auth(self):
        """Unauthenticated requests should be rejected."""
        response = client.get("/api/admin/security/overview")
        assert response.status_code in (401, 403)

    def test_overview_user_forbidden(self, jwt_user_client):
        """Non-admin users should not access security overview."""
        response = jwt_user_client.get("/api/admin/security/overview")
        assert response.status_code in (403, 401)

    def test_overview_admin_can_access(self, jwt_admin_client):
        """Admin users can access security overview."""
        response = jwt_admin_client.get("/api/admin/security/overview")
        assert response.status_code == 200
        data = response.json()

        assert "summary_cards" in data
        assert "failed_login_activity" in data
        assert "admin_actions" in data
        assert "document_security" in data
        assert "rag_safety" in data
        assert "risky_activity" in data
        assert "export_options" in data
        assert "query_window_hours" in data
        assert "queried_since" in data

    def test_overview_default_hours_24(self, jwt_admin_client):
        """Default time window is 24 hours."""
        response = jwt_admin_client.get("/api/admin/security/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["query_window_hours"] == 24

    def test_overview_hours_param(self, jwt_admin_client):
        """Time window hours parameter is respected."""
        response = jwt_admin_client.get("/api/admin/security/overview?hours=168")
        assert response.status_code == 200
        data = response.json()
        assert data["query_window_hours"] == 168

    def test_overview_hours_param_bounds(self, jwt_admin_client):
        """Hours parameter is bounded (1-720)."""
        # Too low
        r = jwt_admin_client.get("/api/admin/security/overview?hours=0")
        assert r.status_code == 422
        # Too high
        r = jwt_admin_client.get("/api/admin/security/overview?hours=1000")
        assert r.status_code == 422

    def test_summary_cards_types(self, jwt_admin_client):
        """Summary cards contain integer counts."""
        response = jwt_admin_client.get("/api/admin/security/overview")
        assert response.status_code == 200
        cards = response.json()["summary_cards"]
        for key in [
            "failed_logins_24h",
            "successful_logins_24h",
            "admin_actions_24h",
            "permission_denied_24h",
            "document_access_changes_24h",
            "rag_fallbacks_24h",
        ]:
            assert key in cards
            assert isinstance(cards[key], int)
            assert cards[key] >= 0

    def test_no_secret_fields_in_response(self, jwt_admin_client):
        """Response must not contain secret-related field names."""
        response = jwt_admin_client.get("/api/admin/security/overview")
        assert response.status_code == 200
        raw = response.text.lower()
        assert '"password"' not in raw
        assert '"secret"' not in raw
        assert '"token"' not in raw
        assert '"api_key"' not in raw

    def test_risky_activity_structure(self, jwt_admin_client):
        """Risky activity section has the expected structure."""
        response = jwt_admin_client.get("/api/admin/security/overview")
        assert response.status_code == 200
        risky = response.json()["risky_activity"]
        assert "repeated_failed_login_ips" in risky
        assert "repeated_permission_denied_users" in risky
        assert isinstance(risky["repeated_failed_login_ips"], list)
        assert isinstance(risky["repeated_permission_denied_users"], list)

    def test_endpoint_returns_200_when_subsection_fails(self, jwt_admin_client, monkeypatch, db_session):
        """If a subsection raises an exception, the endpoint still returns 200 with degraded data."""
        import app.api.admin_security as mod

        def broken_summary(db, since):
            raise RuntimeError("simulated security summary failure")

        monkeypatch.setattr(mod, "_build_summary_cards", broken_summary)
        response = jwt_admin_client.get("/api/admin/security/overview")

        assert response.status_code == 200, "Endpoint must not 500 when a subsection fails"
        data = response.json()

        # Failed section should have safe defaults
        cards = data["summary_cards"]
        for key in [
            "failed_logins_24h",
            "successful_logins_24h",
            "admin_actions_24h",
            "permission_denied_24h",
            "document_access_changes_24h",
            "rag_fallbacks_24h",
        ]:
            assert cards[key] == 0, f"{key} should be 0 after degradation"

        # Other sections should still be present
        assert "failed_login_activity" in data
        assert "admin_actions" in data
        assert "document_security" in data
        assert "rag_safety" in data
        assert "risky_activity" in data

        # No secret leakage
        raw = response.text.lower()
        assert "simulated" not in raw
        assert "runtimeerror" not in raw
        assert "traceback" not in raw

    def test_export_json_requires_admin(self, jwt_user_client):
        """Non-admin users cannot export JSON."""
        response = jwt_user_client.get("/api/admin/security/export.json")
        assert response.status_code in (403, 401)

    def test_export_csv_requires_admin(self, jwt_user_client):
        """Non-admin users cannot export CSV."""
        response = jwt_user_client.get("/api/admin/security/export.csv")
        assert response.status_code in (403, 401)

    def test_export_json_admin_can_access(self, jwt_admin_client):
        """Admin users can export JSON."""
        response = jwt_admin_client.get("/api/admin/security/export.json")
        assert response.status_code == 200
        data = response.json()
        assert "exported_at" in data
        assert "count" in data
        assert isinstance(data["count"], int)
        assert "records" in data
        assert isinstance(data["records"], list)

    def test_export_csv_admin_can_access(self, jwt_admin_client):
        """Admin users can export CSV."""
        response = jwt_admin_client.get("/api/admin/security/export.csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")
        content = response.text
        assert "id,timestamp,username,action,status" in content
        assert "audit-export" in response.headers.get("content-disposition", "")

    def test_export_csv_with_filter(self, jwt_admin_client):
        """CSV export respects action filter."""
        response = jwt_admin_client.get("/api/admin/security/export.csv?action=LOGIN_FAILURE&status=failure")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    def test_export_json_with_filter(self, jwt_admin_client):
        """JSON export respects event_type filter."""
        response = jwt_admin_client.get("/api/admin/security/export.json?event_type=login&status=success")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["count"], int)

    def test_overview_with_audit_data(self, jwt_admin_client, db_session):
        """Overview returns correct counts when audit logs exist."""
        from datetime import datetime, timedelta
        from app.security.models import AuditLog

        now = datetime.utcnow()
        log = AuditLog(
            user_id=None,
            username="testuser",
            action=AuditAction.LOGIN_FAILURE,
            request_ip="192.168.1.1",
            request_user_agent="Mozilla/5.0 Test",
            status="failure",
            created_at=now - timedelta(hours=1),
        )
        db_session.add(log)
        db_session.commit()

        response = jwt_admin_client.get("/api/admin/security/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["summary_cards"]["failed_logins_24h"] >= 1

        # Verify failed login event is in the list
        events = data["failed_login_activity"]["events"]
        assert len(events) >= 1
        assert events[0]["username"] == "testuser"
        assert events[0]["request_ip"] == "192.168.1.1"

        # Verify IP count
        ip_counts = data["failed_login_activity"]["login_counts_by_ip"]
        assert len(ip_counts) >= 1

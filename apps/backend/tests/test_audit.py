"""
Tests for Phase 14 — Audit Logging Hardening.
"""

from fastapi.testclient import TestClient

from app.core.config import settings
from app.security.models import User, UserRole, AuditLog, AuditAction
from app.security.password import hash_password


def _create_user(db_session, username: str, email: str, password: str, role: UserRole = UserRole.USER, is_active: bool = True) -> User:
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role,
        is_active=is_active,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


class TestAuditLogin:
    def test_login_success_logged(self, client: TestClient, db_session):
        _create_user(db_session, "auditlogin", "auditlogin@example.com", "UserPass123!")
        res = client.post("/api/auth/login", json={
            "username_or_email": "auditlogin",
            "password": "UserPass123!",
        })
        assert res.status_code == 200

        # Check audit log exists
        log = db_session.query(AuditLog).filter(
            AuditLog.action == AuditAction.LOGIN_SUCCESS,
            AuditLog.username == "auditlogin",
        ).first()
        assert log is not None
        assert log.status == "success"

    def test_login_failure_logged(self, client: TestClient, db_session):
        _create_user(db_session, "auditfail", "auditfail@example.com", "UserPass123!")
        res = client.post("/api/auth/login", json={
            "username_or_email": "auditfail",
            "password": "wrongpassword",
        })
        assert res.status_code == 401

        log = db_session.query(AuditLog).filter(
            AuditLog.action == AuditAction.LOGIN_FAILURE,
            AuditLog.username == "auditfail",
        ).first()
        assert log is not None
        assert log.status == "failure"


class TestAuditAdminActions:
    def test_user_created_logged(self, client: TestClient, db_session):
        admin = _create_user(db_session, "auditcreate", "auditcreate@example.com", "AdminPass123!", role=UserRole.ADMIN)
        login = client.post("/api/auth/login", json={
            "username_or_email": "auditcreate",
            "password": "AdminPass123!",
        })
        token = login.json()["access_token"]

        res = client.post(
            "/api/admin/users",
            json={
                "username": "newauditeduser",
                "email": "newaudited@example.com",
                "password": "Str0ngP@ssw0rd!",
                "role": "user",
                "is_active": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 201

        log = db_session.query(AuditLog).filter(
            AuditLog.action == AuditAction.USER_CREATED,
        ).order_by(AuditLog.id.desc()).first()
        assert log is not None
        assert log.status == "success"
        assert "newauditeduser" in (log.details or "")

    def test_password_reset_logged(self, client: TestClient, db_session):
        admin = _create_user(db_session, "auditreset", "auditreset@example.com", "AdminPass123!", role=UserRole.ADMIN)
        user = _create_user(db_session, "resettarget", "resettarget@example.com", "UserPass123!", role=UserRole.USER)

        login = client.post("/api/auth/login", json={
            "username_or_email": "auditreset",
            "password": "AdminPass123!",
        })
        token = login.json()["access_token"]

        res = client.post(
            f"/api/admin/users/{user.id}/reset-password",
            json={"new_password": "NewStr0ngP@ss!"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200

        log = db_session.query(AuditLog).filter(
            AuditLog.action == AuditAction.PASSWORD_RESET,
        ).order_by(AuditLog.id.desc()).first()
        assert log is not None
        assert log.status == "success"


class TestAuditLogAPI:
    def test_admin_can_list_audit_logs(self, client: TestClient, db_session):
        admin = _create_user(db_session, "auditapi", "auditapi@example.com", "AdminPass123!", role=UserRole.ADMIN)
        login = client.post("/api/auth/login", json={
            "username_or_email": "auditapi",
            "password": "AdminPass123!",
        })
        token = login.json()["access_token"]

        res = client.get("/api/admin/audit-logs", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert "entries" in data
        assert "total" in data
        assert "total_pages" in data
        assert data["page"] == 1

    def test_non_admin_cannot_access_audit_logs(self, client: TestClient, db_session):
        user = _create_user(db_session, "audituser", "audituser@example.com", "UserPass123!", role=UserRole.USER)
        login = client.post("/api/auth/login", json={
            "username_or_email": "audituser",
            "password": "UserPass123!",
        })
        token = login.json()["access_token"]

        res = client.get("/api/admin/audit-logs", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_audit_logs_filter_by_action(self, client: TestClient, db_session):
        admin = _create_user(db_session, "auditfilter", "auditfilter@example.com", "AdminPass123!", role=UserRole.ADMIN)
        _create_user(db_session, "filtertarget", "filtertarget@example.com", "UserPass123!", role=UserRole.USER)

        login = client.post("/api/auth/login", json={
            "username_or_email": "auditfilter",
            "password": "AdminPass123!",
        })
        token = login.json()["access_token"]

        # Create a user to generate audit log
        client.post(
            "/api/admin/users",
            json={
                "username": "filtercreated",
                "email": "filtercreated@example.com",
                "password": "Str0ngP@ssw0rd!",
                "role": "user",
                "is_active": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        # Filter by USER_CREATED
        res = client.get(
            "/api/admin/audit-logs?action=USER_CREATED",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert any(e["action"] == "user_created" for e in data["entries"])

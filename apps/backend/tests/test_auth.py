"""
Tests for Phase 12 — Production Authentication, JWT, and Backend-Enforced Authorization.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.security.models import User, UserRole
from app.security.password import hash_password, verify_password
from app.db.base import Base


# Ensure JWT is usable in tests
settings.JWT_SECRET_KEY = "test-secret-key-for-jwt"
settings.JWT_ALGORITHM = "HS256"
settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60


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


class TestPasswordHashing:
    def test_password_is_hashed(self):
        plain = "supersecret"
        hashed = hash_password(plain)
        assert hashed != plain
        assert hashed.startswith("$2")

    def test_verify_password_success(self):
        plain = "supersecret"
        hashed = hash_password(plain)
        assert verify_password(plain, hashed) is True

    def test_verify_password_failure(self):
        plain = "supersecret"
        hashed = hash_password(plain)
        assert verify_password("wrongpassword", hashed) is False


class TestLogin:
    def test_login_success(self, client: TestClient, db_session):
        _create_user(db_session, "testuser", "test@example.com", "password123")
        res = client.post("/api/auth/login", json={
            "username_or_email": "testuser",
            "password": "password123",
        })
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["role"] == "user"
        assert data["user"]["permissions"]["can_use_general_chat"] is True

    def test_login_with_email(self, client: TestClient, db_session):
        _create_user(db_session, "emailuser", "email@example.com", "password123")
        res = client.post("/api/auth/login", json={
            "username_or_email": "email@example.com",
            "password": "password123",
        })
        assert res.status_code == 200
        assert res.json()["access_token"]

    def test_login_wrong_password(self, client: TestClient, db_session):
        _create_user(db_session, "wrongpass", "wrong@example.com", "password123")
        res = client.post("/api/auth/login", json={
            "username_or_email": "wrongpass",
            "password": "badpassword",
        })
        assert res.status_code == 401

    def test_login_inactive_user(self, client: TestClient, db_session):
        _create_user(db_session, "inactive", "inactive@example.com", "password123", is_active=False)
        res = client.post("/api/auth/login", json={
            "username_or_email": "inactive",
            "password": "password123",
        })
        assert res.status_code == 401


class TestMe:
    def test_me_with_valid_token(self, client: TestClient, db_session):
        _create_user(db_session, "meuser", "me@example.com", "password123", role=UserRole.USER)
        login_res = client.post("/api/auth/login", json={
            "username_or_email": "meuser",
            "password": "password123",
        })
        token = login_res.json()["access_token"]
        res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert data["username"] == "meuser"
        assert data["role"] == "user"
        assert "permissions" in data

    def test_me_invalid_token(self, client: TestClient):
        res = client.get("/api/auth/me", headers={"Authorization": "Bearer invalidtoken"})
        assert res.status_code == 401

    def test_me_no_token(self, client: TestClient):
        res = client.get("/api/auth/me")
        assert res.status_code == 401

    def test_me_expired_token(self, client: TestClient, db_session, monkeypatch):
        monkeypatch.setattr(settings, "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", -1)
        _create_user(db_session, "expired", "expired@example.com", "password123")
        login_res = client.post("/api/auth/login", json={
            "username_or_email": "expired",
            "password": "password123",
        })
        token = login_res.json()["access_token"]
        res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 401


class TestAuthorization:
    def test_admin_can_access_observability(self, client: TestClient, db_session):
        _create_user(db_session, "adminonly", "admin@example.com", "adminpass", role=UserRole.ADMIN)
        login = client.post("/api/auth/login", json={
            "username_or_email": "adminonly",
            "password": "adminpass",
        })
        token = login.json()["access_token"]
        res = client.get("/api/admin/observability/summary", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200

    def test_user_cannot_access_observability(self, client: TestClient, db_session):
        _create_user(db_session, "regular", "regular@example.com", "userpass", role=UserRole.USER)
        login = client.post("/api/auth/login", json={
            "username_or_email": "regular",
            "password": "userpass",
        })
        token = login.json()["access_token"]
        res = client.get("/api/admin/observability/summary", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_viewer_cannot_access_observability(self, client: TestClient, db_session):
        _create_user(db_session, "viewer", "viewer@example.com", "viewerpass", role=UserRole.VIEWER)
        login = client.post("/api/auth/login", json={
            "username_or_email": "viewer",
            "password": "viewerpass",
        })
        token = login.json()["access_token"]
        res = client.get("/api/admin/observability/summary", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_viewer_cannot_use_general_chat(self, client: TestClient, db_session):
        _create_user(db_session, "viewerc", "viewerc@example.com", "viewerpass", role=UserRole.VIEWER)
        login = client.post("/api/auth/login", json={
            "username_or_email": "viewerc",
            "password": "viewerpass",
        })
        token = login.json()["access_token"]
        res = client.post("/api/chat", json={
            "message": "hello",
            "mode": "general_chat",
        }, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_non_admin_cannot_use_debug_chat(self, client: TestClient, db_session):
        _create_user(db_session, "userdebug", "userdebug@example.com", "userpass", role=UserRole.USER)
        login = client.post("/api/auth/login", json={
            "username_or_email": "userdebug",
            "password": "userpass",
        })
        token = login.json()["access_token"]
        res = client.post("/api/chat", json={
            "message": "hello",
            "mode": "debug",
        }, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_admin_can_use_debug_chat(self, client: TestClient, db_session):
        _create_user(db_session, "admindebug", "admindebug@example.com", "adminpass", role=UserRole.ADMIN)
        login = client.post("/api/auth/login", json={
            "username_or_email": "admindebug",
            "password": "adminpass",
        })
        token = login.json()["access_token"]
        res = client.post("/api/chat", json={
            "message": "hello",
            "mode": "debug",
        }, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200

    def test_document_delete_requires_permission(self, client: TestClient, db_session):
        _create_user(db_session, "nodelete", "nodelete@example.com", "userpass", role=UserRole.USER)
        login = client.post("/api/auth/login", json={
            "username_or_email": "nodelete",
            "password": "userpass",
        })
        token = login.json()["access_token"]
        res = client.delete("/api/documents/1", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_document_reindex_requires_permission(self, client: TestClient, db_session):
        _create_user(db_session, "noreindex", "noreindex@example.com", "userpass", role=UserRole.USER)
        login = client.post("/api/auth/login", json={
            "username_or_email": "noreindex",
            "password": "userpass",
        })
        token = login.json()["access_token"]
        res = client.post("/api/documents/1/index", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_document_upload_requires_permission(self, client: TestClient, db_session):
        import io
        _create_user(db_session, "noupload", "noupload@example.com", "viewerpass", role=UserRole.VIEWER)
        login = client.post("/api/auth/login", json={
            "username_or_email": "noupload",
            "password": "viewerpass",
        })
        token = login.json()["access_token"]
        res = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 403

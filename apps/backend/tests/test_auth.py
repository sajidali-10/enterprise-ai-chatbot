"""
Tests for Phase 12 — Production Authentication, JWT, and Backend-Enforced Authorization.
Tests for Phase 14 — JWT token_version invalidation and password policy.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.security.models import User, UserRole
from app.security.password import hash_password, verify_password, validate_password_policy
from app.db.base import Base


# Ensure JWT is usable in tests
settings.JWT_SECRET_KEY = "test-secret-key-for-jwt"
settings.JWT_ALGORITHM = "HS256"
settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60


def _create_user(db_session, username: str, email: str, password: str, role: UserRole = UserRole.user, is_active: bool = True) -> User:
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
        _create_user(db_session, "meuser", "me@example.com", "password123", role=UserRole.user)
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
        _create_user(db_session, "adminonly", "admin@example.com", "adminpass", role=UserRole.admin)
        login = client.post("/api/auth/login", json={
            "username_or_email": "adminonly",
            "password": "adminpass",
        })
        token = login.json()["access_token"]
        res = client.get("/api/admin/observability/summary", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200

    def test_user_cannot_access_observability(self, client: TestClient, db_session):
        _create_user(db_session, "regular", "regular@example.com", "userpass", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "regular",
            "password": "userpass",
        })
        token = login.json()["access_token"]
        res = client.get("/api/admin/observability/summary", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_viewer_cannot_access_observability(self, client: TestClient, db_session):
        _create_user(db_session, "viewer", "viewer@example.com", "viewerpass", role=UserRole.viewer)
        login = client.post("/api/auth/login", json={
            "username_or_email": "viewer",
            "password": "viewerpass",
        })
        token = login.json()["access_token"]
        res = client.get("/api/admin/observability/summary", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_viewer_cannot_use_general_chat(self, client: TestClient, db_session):
        _create_user(db_session, "viewerc", "viewerc@example.com", "viewerpass", role=UserRole.viewer)
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
        _create_user(db_session, "userdebug", "userdebug@example.com", "userpass", role=UserRole.user)
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
        _create_user(db_session, "admindebug", "admindebug@example.com", "adminpass", role=UserRole.admin)
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
        _create_user(db_session, "nodelete", "nodelete@example.com", "userpass", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "nodelete",
            "password": "userpass",
        })
        token = login.json()["access_token"]
        res = client.delete("/api/documents/1", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_document_reindex_requires_permission(self, client: TestClient, db_session):
        _create_user(db_session, "noreindex", "noreindex@example.com", "userpass", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "noreindex",
            "password": "userpass",
        })
        token = login.json()["access_token"]
        res = client.post("/api/documents/1/index", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 403

    def test_document_upload_requires_permission(self, client: TestClient, db_session):
        import io
        _create_user(db_session, "noupload", "noupload@example.com", "viewerpass", role=UserRole.viewer)
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


class TestTokenVersionInvalidation:
    def test_password_reset_invalidates_old_token(self, client: TestClient, db_session):
        # Create admin and regular user
        admin = _create_user(db_session, "tokenadmin", "tokenadmin@example.com", "AdminPass123!", role=UserRole.admin)
        user = _create_user(db_session, "tokenuser", "tokenuser@example.com", "UserPass123!", role=UserRole.user)
        db_session.commit()

        # Login as user and get token
        login = client.post("/api/auth/login", json={
            "username_or_email": "tokenuser",
            "password": "UserPass123!",
        })
        assert login.status_code == 200
        token = login.json()["access_token"]

        # Token should work for /api/auth/me
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200

        # Admin resets user password
        admin_login = client.post("/api/auth/login", json={
            "username_or_email": "tokenadmin",
            "password": "AdminPass123!",
        })
        admin_token = admin_login.json()["access_token"]
        reset = client.post(
            "/api/admin/users/2/reset-password",
            json={"new_password": "NewStr0ngP@ss!"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert reset.status_code == 200

        # Old token should now be rejected
        me_after = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_after.status_code == 401

    def test_deactivation_invalidates_old_token(self, client: TestClient, db_session):
        # Create admin and user
        admin = _create_user(db_session, "deactadmin", "deactadmin@example.com", "AdminPass123!", role=UserRole.admin)
        user = _create_user(db_session, "deactuser", "deactuser@example.com", "UserPass123!", role=UserRole.user)

        # Login as user
        login = client.post("/api/auth/login", json={
            "username_or_email": "deactuser",
            "password": "UserPass123!",
        })
        assert login.status_code == 200
        token = login.json()["access_token"]

        # Token works initially
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200

        # Admin deactivates user
        admin_login = client.post("/api/auth/login", json={
            "username_or_email": "deactadmin",
            "password": "AdminPass123!",
        })
        admin_token = admin_login.json()["access_token"]
        patch = client.patch(
            "/api/admin/users/2",
            json={"is_active": False},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert patch.status_code == 200

        # Old token should now be rejected
        me_after = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_after.status_code == 401

    def test_inactive_user_token_rejected(self, client: TestClient, db_session):
        # Create inactive user and try to use token
        user = _create_user(db_session, "inactiveuser", "inactiveuser@example.com", "UserPass123!", role=UserRole.user, is_active=True)

        # Login to get token
        login = client.post("/api/auth/login", json={
            "username_or_email": "inactiveuser",
            "password": "UserPass123!",
        })
        token = login.json()["access_token"]
        assert login.status_code == 200

        # Deactivate the user directly in DB
        user.is_active = False
        db_session.commit()

        # Token should be rejected
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 401


class TestPasswordPolicy:
    def test_password_policy_accepts_strong(self):
        assert validate_password_policy("MyStr0ng#Pass") is None
        assert validate_password_policy("ComplexP@ssw0rd!") is None
        assert validate_password_policy("A1b2C3d4E5f6!") is None

    def test_password_policy_rejects_too_short(self):
        error = validate_password_policy("Short1!")
        assert error is not None
        assert "at least 12" in error

    def test_password_policy_rejects_insufficient_classes(self):
        error = validate_password_policy("longbutnouppercase")
        assert error is not None
        assert "at least 3" in error

        error2 = validate_password_policy("LongButNoDigitsOrSymbols")
        assert error2 is not None
        assert "at least 3" in error2

    def test_password_policy_rejects_common_weak(self):
        error = validate_password_policy("Password123!")
        assert error is not None
        assert "too common" in error



    def test_create_user_enforces_password_policy(self, client: TestClient, db_session):
        admin = _create_user(db_session, "policyadmin", "policyadmin@example.com", "AdminPass123!", role=UserRole.admin)
        login = client.post("/api/auth/login", json={
            "username_or_email": "policyadmin",
            "password": "AdminPass123!",
        })
        admin_token = login.json()["access_token"]

        # Try to create user with weak password
        res = client.post(
            "/api/admin/users",
            json={
                "username": "weakuser",
                "email": "weakuser@example.com",
                "password": "short",
                "role": "user",
                "is_active": True,
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 400
        assert "at least 12" in res.json()["detail"]

        # Create user with strong password should succeed
        res2 = client.post(
            "/api/admin/users",
            json={
                "username": "stronguser",
                "email": "stronguser@example.com",
                "password": "Str0ngP@ssw0rd!",
                "role": "user",
                "is_active": True,
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert res2.status_code == 201, res2.json()

    def test_reset_password_enforces_password_policy(self, client: TestClient, db_session):
        admin = _create_user(db_session, "pwadmin", "pwadmin@example.com", "AdminPass123!", role=UserRole.admin)
        user = _create_user(db_session, "pwuser", "pwuser@example.com", "UserPass123!", role=UserRole.user)

        login = client.post("/api/auth/login", json={
            "username_or_email": "pwadmin",
            "password": "AdminPass123!",
        })
        admin_token = login.json()["access_token"]

        # Try weak password
        res = client.post(
            f"/api/admin/users/{user.id}/reset-password",
            json={"new_password": "weak"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 400
        assert "at least 12" in res.json()["detail"]

        # Strong password should succeed
        res2 = client.post(
            f"/api/admin/users/{user.id}/reset-password",
            json={"new_password": "NewStr0ngP@ss!"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert res2.status_code == 200

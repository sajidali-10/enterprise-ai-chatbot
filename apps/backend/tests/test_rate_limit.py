"""
Tests for Phase 14 — API Rate Limiting.
"""

import pytest
from fastapi.testclient import TestClient

from app.core import rate_limit as rate_limit_module
from app.core.config import settings
from app.security.models import User, UserRole
from app.security.password import hash_password


def _clear_rate_limit_store():
    """Clear the in-memory rate limit store and Redis keys between tests."""
    rate_limit_module._in_memory_store.clear()
    try:
        import redis as redis_lib
        r = redis_lib.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            socket_connect_timeout=2,
            decode_responses=True,
        )
        if r.ping():
            for key in r.scan_iter(match="*"):
                r.delete(key)
    except Exception:
        pass  # Redis not available, in-memory was cleared already


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


class TestLoginRateLimit:
    def test_repeated_login_attempts_return_429(self, client: TestClient, db_session, monkeypatch):
        _clear_rate_limit_store()
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
        _create_user(db_session, "ratelimituser", "ratelimit@example.com", "UserPass123!")

        # Make 5 failed login attempts (the limit)
        for i in range(5):
            res = client.post("/api/auth/login", json={
                "username_or_email": "ratelimituser",
                "password": "wrongpassword",
            })
            # First 5 should get 401 (invalid credentials)
            assert res.status_code == 401, f"Attempt {i+1}: expected 401, got {res.status_code}"

        # 6th attempt should be rate limited (429)
        res = client.post("/api/auth/login", json={
            "username_or_email": "ratelimituser",
            "password": "wrongpassword",
        })
        assert res.status_code == 429, f"Expected 429, got {res.status_code}: {res.text}"
        assert "Too Many Requests" in res.json()["detail"]

    def test_normal_login_works(self, client: TestClient, db_session, monkeypatch):
        _clear_rate_limit_store()
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
        _create_user(db_session, "normaluser", "normal@example.com", "UserPass123!")

        res = client.post("/api/auth/login", json={
            "username_or_email": "normaluser",
            "password": "UserPass123!",
        })
        assert res.status_code == 200
        assert "access_token" in res.json()

    def test_rate_limit_disabled_override(self, client: TestClient, db_session, monkeypatch):
        _clear_rate_limit_store()
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)

        _create_user(db_session, "disabledrate", "disabled@example.com", "UserPass123!")

        # Make many requests without hitting rate limit
        for i in range(10):
            res = client.post("/api/auth/login", json={
                "username_or_email": "disabledrate",
                "password": "wrongpassword",
            })
            # Should still get 401 (wrong password), never 429
            assert res.status_code == 401, f"Attempt {i+1}: expected 401, got {res.status_code}"


class TestChatRateLimit:
    def test_chat_rate_limit(self, client: TestClient, db_session, monkeypatch):
        _clear_rate_limit_store()
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)

        _create_user(db_session, "chatuser", "chat@example.com", "UserPass123!", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "chatuser",
            "password": "UserPass123!",
        })
        token = login.json()["access_token"]

        # Make 30 chat requests (the limit)
        for i in range(30):
            res = client.post("/api/chat", json={
                "message": "hello",
                "mode": "general_chat",
            }, headers={"Authorization": f"Bearer {token}"})
            # Should succeed or get 403 (viewer permission) but not 429 yet
            assert res.status_code in (200, 403), f"Attempt {i+1}: got {res.status_code}"

        # 31st request should be rate limited
        res = client.post("/api/chat", json={
            "message": "hello",
            "mode": "general_chat",
        }, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 429, f"Expected 429, got {res.status_code}"

"""
Tests for Phase 12.1 — Admin User Management API.

Covers:
- Admin can create/list/get/update/deactivate users
- Password is bcrypt-hashed, never returned in API responses
- Duplicate username/email rejected
- Deactivated users cannot login
- Password reset works and user can login with new password
- Non-admin (regular user) gets 403
- Viewer gets 403
- Unauthenticated requests get 401
- Last active admin cannot be deactivated/demoted/deleted
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

from app.core.config import settings
from app.security.models import User, UserRole
from app.security.password import hash_password, verify_password


settings.JWT_SECRET_KEY = "test-secret-key-for-jwt"
settings.JWT_ALGORITHM = "HS256"
settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60


def _make_user(db_session, username, email, password, role=UserRole.user, is_active=True) -> User:
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


def _login(client, username_or_email, password) -> str:
    res = client.post("/api/auth/login", json={
        "username_or_email": username_or_email,
        "password": password,
    })
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


@pytest.fixture
def admin_user(db_session):
    return _make_user(db_session, "admin1", "admin1@test.com", "AdminPass123!", role=UserRole.admin)


@pytest.fixture
def regular_user(db_session):
    return _make_user(db_session, "user1", "user1@test.com", "userpass123", role=UserRole.user)


@pytest.fixture
def viewer_user(db_session):
    return _make_user(db_session, "viewer1", "viewer1@test.com", "viewerpass123", role=UserRole.viewer)


@pytest.fixture
def admin_client(client, admin_user, db_session):
    """Separate TestClient with admin JWT headers — shares DB session via dependency overrides."""
    token = _login(client, "admin1", "AdminPass123!")
    # Ensure dependency overrides are set (client fixture already set them, but be safe)
    from app.db.session import get_db
    def override_get_db():
        db_session.expire_all()
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    admin_c = TestClient(app)
    admin_c.headers["Authorization"] = f"Bearer {token}"
    yield admin_c


@pytest.fixture
def user_client(client, regular_user, db_session):
    """Separate TestClient with regular-user JWT headers."""
    token = _login(client, "user1", "userpass123")
    from app.db.session import get_db
    def override_get_db():
        db_session.expire_all()
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    user_c = TestClient(app)
    user_c.headers["Authorization"] = f"Bearer {token}"
    yield user_c


@pytest.fixture
def viewer_client(client, viewer_user, db_session):
    """Separate TestClient with viewer JWT headers."""
    token = _login(client, "viewer1", "viewerpass123")
    from app.db.session import get_db
    def override_get_db():
        db_session.expire_all()
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    viewer_c = TestClient(app)
    viewer_c.headers["Authorization"] = f"Bearer {token}"
    yield viewer_c


# ==============================================================================
# Authorization tests
# ==============================================================================

class TestAuthorization:
    def test_unauthenticated_list_returns_401(self, client):
        res = client.get("/api/admin/users")
        assert res.status_code == 401

    def test_unauthenticated_create_returns_401(self, client):
        res = client.post("/api/admin/users", json={
            "username": "x", "email": "x@x.com", "password": "LongEnough123!"
        })
        assert res.status_code == 401

    def test_regular_user_list_returns_403(self, user_client):
        res = user_client.get("/api/admin/users")
        assert res.status_code == 403

    def test_regular_user_create_returns_403(self, user_client):
        res = user_client.post("/api/admin/users", json={
            "username": "x", "email": "x@x.com", "password": "LongEnough123!"
        })
        assert res.status_code == 403

    def test_viewer_list_returns_403(self, viewer_client):
        res = viewer_client.get("/api/admin/users")
        assert res.status_code == 403

    def test_viewer_create_returns_403(self, viewer_client):
        res = viewer_client.post("/api/admin/users", json={
            "username": "x", "email": "x@x.com", "password": "LongEnough123!"
        })
        assert res.status_code == 403

    def test_inactive_admin_cannot_access(self, client, db_session, admin_user):
        # Deactivate admin directly in DB
        admin_user.is_active = False
        db_session.commit()
        token = _login.__wrapped__(client, "admin1", "AdminPass123!") if False else None
        # Login should fail for inactive user
        res = client.post("/api/auth/login", json={
            "username_or_email": "admin1",
            "password": "AdminPass123!",
        })
        assert res.status_code == 401


# ==============================================================================
# List users
# ==============================================================================

class TestListUsers:
    def test_admin_lists_users(self, admin_client, admin_user, regular_user, viewer_user):
        res = admin_client.get("/api/admin/users")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) == 3
        usernames = {u["username"] for u in data}
        assert usernames == {"admin1", "user1", "viewer1"}

    def test_response_does_not_contain_password(self, admin_client, admin_user):
        res = admin_client.get("/api/admin/users")
        assert res.status_code == 200
        for user in res.json():
            assert "hashed_password" not in user
            assert "password" not in user

    def test_get_single_user_returns_permissions(self, admin_client, regular_user):
        res = admin_client.get(f"/api/admin/users/{regular_user.id}")
        assert res.status_code == 200
        data = res.json()
        assert data["username"] == "user1"
        assert "permissions" in data
        perms = data["permissions"]
        assert perms["can_use_general_chat"] is True
        assert perms["can_manage_users"] is False
        assert "hashed_password" not in data


# ==============================================================================
# Create user
# ==============================================================================

class TestCreateUser:
    def test_admin_creates_regular_user(self, admin_client, db_session):
        res = admin_client.post("/api/admin/users", json={
            "username": "newuser",
            "email": "newuser@test.com",
            "password": "ValidPass123!",
            "full_name": "New User",
            "role": "user",
        })
        assert res.status_code == 201
        data = res.json()
        assert data["username"] == "newuser"
        assert data["role"] == "user"
        assert data["is_active"] is True
        assert "hashed_password" not in data
        assert "password" not in data

    def test_admin_creates_viewer_user(self, admin_client):
        res = admin_client.post("/api/admin/users", json={
            "username": "newviewer",
            "email": "newviewer@test.com",
            "password": "ValidPass123!",
            "role": "viewer",
        })
        assert res.status_code == 201
        assert res.json()["role"] == "viewer"

    def test_admin_creates_admin_user(self, admin_client):
        res = admin_client.post("/api/admin/users", json={
            "username": "newadmin",
            "email": "newadmin@test.com",
            "password": "ValidPass123!",
            "role": "admin",
        })
        assert res.status_code == 201
        assert res.json()["role"] == "admin"

    def test_password_is_hashed(self, admin_client, db_session):
        admin_client.post("/api/admin/users", json={
            "username": "hashcheck",
            "email": "hashcheck@test.com",
            "password": "Plain_Pass123!",
            "role": "user",
        })
        user = db_session.query(User).filter(User.username == "hashcheck").first()
        assert user is not None
        assert user.hashed_password != "Plain_Pass123!"
        assert user.hashed_password.startswith("$2")
        assert verify_password("Plain_Pass123!", user.hashed_password)

    def test_duplicate_username_rejected(self, admin_client, regular_user):
        res = admin_client.post("/api/admin/users", json={
            "username": "user1",
            "email": "different@test.com",
            "password": "ValidPass123!",
            "role": "user",
        })
        assert res.status_code == 409

    def test_duplicate_email_rejected(self, admin_client, regular_user):
        res = admin_client.post("/api/admin/users", json={
            "username": "different",
            "email": "user1@test.com",
            "password": "ValidPass123!",
            "role": "user",
        })
        assert res.status_code == 409

    def test_short_password_rejected(self, admin_client):
        res = admin_client.post("/api/admin/users", json={
            "username": "shortpw",
            "email": "shortpw@test.com",
            "password": "short",
            "role": "user",
        })
        assert res.status_code == 400

    def test_invalid_email_rejected(self, admin_client):
        res = admin_client.post("/api/admin/users", json={
            "username": "bademail",
            "email": "not-an-email",
            "password": "ValidPass123!",
            "role": "user",
        })
        assert res.status_code == 422


# ==============================================================================
# Update user
# ==============================================================================

class TestUpdateUser:
    def test_admin_updates_role(self, admin_client, regular_user):
        res = admin_client.patch(f"/api/admin/users/{regular_user.id}", json={
            "role": "viewer",
        })
        assert res.status_code == 200
        assert res.json()["role"] == "viewer"

    def test_admin_updates_full_name(self, admin_client, regular_user):
        res = admin_client.patch(f"/api/admin/users/{regular_user.id}", json={
            "full_name": "Updated Name",
        })
        assert res.status_code == 200
        assert res.json()["full_name"] == "Updated Name"

    def test_admin_updates_email(self, admin_client, regular_user):
        res = admin_client.patch(f"/api/admin/users/{regular_user.id}", json={
            "email": "newemail@test.com",
        })
        assert res.status_code == 200
        assert res.json()["email"] == "newemail@test.com"

    def test_admin_deactivates_user(self, admin_client, regular_user):
        res = admin_client.patch(f"/api/admin/users/{regular_user.id}", json={
            "is_active": False,
        })
        assert res.status_code == 200
        assert res.json()["is_active"] is False

    def test_admin_cannot_demote_last_admin(self, admin_client, admin_user):
        # There's only one admin; demoting to user should fail
        res = admin_client.patch(f"/api/admin/users/{admin_user.id}", json={
            "role": "user",
        })
        assert res.status_code == 400
        assert "last active admin" in res.json()["detail"].lower()

    def test_admin_cannot_deactivate_last_admin(self, admin_client, admin_user):
        res = admin_client.patch(f"/api/admin/users/{admin_user.id}", json={
            "is_active": False,
        })
        assert res.status_code == 400
        assert "last active admin" in res.json()["detail"].lower()

    def test_admin_can_demote_when_another_admin_exists(self, admin_client, db_session):
        # Create a second admin
        admin_client.post("/api/admin/users", json={
            "username": "admin2",
            "email": "admin2@test.com",
            "password": "Admin2Pass123!",
            "role": "admin",
        })
        # Now demoting admin1 should succeed (admin2 still active)
        from app.db.session import SessionLocal
        admin1 = db_session.query(User).filter(User.username == "admin1").first()
        res = admin_client.patch(f"/api/admin/users/{admin1.id}", json={
            "role": "user",
        })
        assert res.status_code == 200


# ==============================================================================
# Deactivate (soft delete)
# ==============================================================================

class TestDeactivateUser:
    def test_admin_deactivates_user(self, admin_client, regular_user, db_session):
        res = admin_client.delete(f"/api/admin/users/{regular_user.id}")
        assert res.status_code == 200
        assert res.json()["is_active"] is False
        # Verify in DB
        db_session.refresh(regular_user)
        assert regular_user.is_active is False

    def test_deactivated_user_cannot_login(self, admin_client, client, regular_user, db_session):
        admin_client.delete(f"/api/admin/users/{regular_user.id}")
        res = client.post("/api/auth/login", json={
            "username_or_email": "user1",
            "password": "userpass123",
        })
        assert res.status_code == 401

    def test_admin_cannot_delete_last_admin(self, admin_client, admin_user):
        res = admin_client.delete(f"/api/admin/users/{admin_user.id}")
        assert res.status_code == 400


# ==============================================================================
# Password reset
# ==============================================================================

class TestPasswordReset:
    def test_admin_resets_password(self, admin_client, regular_user, client):
        res = admin_client.post(f"/api/admin/users/{regular_user.id}/reset-password", json={
            "new_password": "NewPassword123!",
        })
        assert res.status_code == 200
        assert res.json()["success"] is True
        assert "NewPassword123!" not in res.text

    def test_user_can_login_with_reset_password(self, admin_client, regular_user, client):
        admin_client.post(f"/api/admin/users/{regular_user.id}/reset-password", json={
            "new_password": "ResetPass123!",
        })
        # Old password should fail
        old_login = client.post("/api/auth/login", json={
            "username_or_email": "user1",
            "password": "userpass123",
        })
        assert old_login.status_code == 401
        # New password should work
        new_login = client.post("/api/auth/login", json={
            "username_or_email": "user1",
            "password": "ResetPass123!",
        })
        assert new_login.status_code == 200

    def test_password_reset_response_does_not_leak_password(self, admin_client, regular_user):
        res = admin_client.post(f"/api/admin/users/{regular_user.id}/reset-password", json={
            "new_password": "secret_reset_pw_123",
        })
        assert "secret_reset_pw_123" not in res.text
        assert "hashed_password" not in res.text

    def test_short_reset_password_rejected(self, admin_client, regular_user):
        res = admin_client.post(f"/api/admin/users/{regular_user.id}/reset-password", json={
            "new_password": "short",
        })
        assert res.status_code == 400

    def test_password_reset_nonexistent_user_returns_404(self, admin_client):
        res = admin_client.post("/api/admin/users/99999/reset-password", json={
            "new_password": "ValidPass123!",
        })
        assert res.status_code == 404


# ==============================================================================
# Inactive user cannot access protected endpoints with old token
# ==============================================================================

class TestInactiveUserTokenInvalidation:
    def test_deactivated_user_me_returns_401(self, admin_client, client, regular_user, db_session):
        # Get a valid token for regular_user
        user_token = _login(client, "user1", "userpass123")
        # /me should work before deactivation
        client.headers["Authorization"] = f"Bearer {user_token}"
        res = client.get("/api/auth/me")
        assert res.status_code == 200

        # Admin deactivates the user (use admin_client which has admin headers)
        admin_res = admin_client.delete(f"/api/admin/users/{regular_user.id}")
        assert admin_res.status_code == 200, f"Admin deactivate failed: {admin_res.text}"

        # /me should now fail with deactivated token
        client.headers["Authorization"] = f"Bearer {user_token}"
        res = client.get("/api/auth/me")
        assert res.status_code == 401
        client.headers.pop("Authorization", None)


# ==============================================================================
# Phase 33: SYSADMIN and Protected User Tests
# ==============================================================================

class TestSysadminAndProtectedUser:
    """Tests for SYSADMIN role and protected user safeguards (Phase 33)."""

    def test_sysadmin_can_be_created(self, admin_client, db_session):
        """SYSADMIN role can be assigned when creating a user."""
        res = admin_client.post("/api/admin/users", json={
            "username": "sysadmin1",
            "email": "sysadmin1@test.com",
            "password": "SysAdminPass123!",
            "role": "sysadmin",
        })
        assert res.status_code == 201, res.text
        data = res.json()
        assert data["role"] == "sysadmin"
        assert data["username"] == "sysadmin1"

    def test_sysadmin_has_full_permissions(self, admin_client, db_session):
        """SYSADMIN role grants all permissions including can_manage_users."""
        res = admin_client.post("/api/admin/users", json={
            "username": "sysadmin2",
            "email": "sysadmin2@test.com",
            "password": "SysAdminPass123!",
            "role": "sysadmin",
        })
        assert res.status_code == 201
        sysadmin_id = res.json()["id"]
        
        # Get user with permissions
        res = admin_client.get(f"/api/admin/users/{sysadmin_id}")
        assert res.status_code == 200
        perms = res.json()["permissions"]
        assert perms["can_manage_users"] is True
        assert perms["can_use_general_chat"] is True
        assert perms["can_access_observability"] is True
        assert perms["can_access_evaluations"] is True

    def test_admin_cannot_modify_protected_sysadmin(self, admin_client, db_session):
        """Regular admin cannot modify a protected SYSADMIN user."""
        # Create a protected sysadmin user directly in DB
        sysadmin = User(
            username="protected_sysadmin",
            email="protected@test.com",
            hashed_password=hash_password("ProtectedPass123!"),
            role=UserRole.sysadmin,
            is_active=True,
            is_protected=True,
        )
        db_session.add(sysadmin)
        db_session.commit()
        db_session.refresh(sysadmin)

        # Admin tries to deactivate protected sysadmin -> should fail
        res = admin_client.delete(f"/api/admin/users/{sysadmin.id}")
        assert res.status_code == 403
        assert "protected" in res.json()["detail"].lower()

        # Admin tries to demote protected sysadmin -> should fail
        res = admin_client.patch(f"/api/admin/users/{sysadmin.id}", json={
            "role": "user",
        })
        assert res.status_code == 403
        assert "protected" in res.json()["detail"].lower()

        # Admin tries to reset password of protected sysadmin -> should fail
        res = admin_client.post(f"/api/admin/users/{sysadmin.id}/reset-password", json={
            "new_password": "NewPassword123!",
        })
        assert res.status_code == 403
        assert "protected" in res.json()["detail"].lower()

        # Admin tries to reactivate protected sysadmin -> should fail
        res = admin_client.post(f"/api/admin/users/{sysadmin.id}/reactivate")
        assert res.status_code == 403
        assert "protected" in res.json()["detail"].lower()

    def test_sysadmin_can_modify_other_sysadmin(self, admin_client, db_session):
        """SYSADMIN can modify another SYSADMIN user."""
        # Create a protected sysadmin user directly in DB
        sysadmin1 = User(
            username="sysadmin1",
            email="sysadmin1@test.com",
            hashed_password=hash_password("Pass123!"),
            role=UserRole.sysadmin,
            is_active=True,
            is_protected=True,
        )
        sysadmin2 = User(
            username="sysadmin2",
            email="sysadmin2@test.com",
            hashed_password=hash_password("Pass123!"),
            role=UserRole.sysadmin,
            is_active=True,
            is_protected=True,
        )
        db_session.add_all([sysadmin1, sysadmin2])
        db_session.commit()
        db_session.refresh(sysadmin1)
        db_session.refresh(sysadmin2)

        # Login as sysadmin1
        sysadmin_client = TestClient(app)
        token = _login(sysadmin_client, "sysadmin1", "Pass123!")
        sysadmin_client.headers["Authorization"] = f"Bearer {token}"

        # sysadmin1 can deactivate sysadmin2
        res = sysadmin_client.delete(f"/api/admin/users/{sysadmin2.id}")
        assert res.status_code == 200
        assert res.json()["is_active"] is False

        # sysadmin1 can reactivate sysadmin2
        res = sysadmin_client.post(f"/api/admin/users/{sysadmin2.id}/reactivate")
        assert res.status_code == 200
        assert res.json()["is_active"] is True

        # sysadmin1 can demote sysadmin2
        res = sysadmin_client.patch(f"/api/admin/users/{sysadmin2.id}", json={
            "role": "admin",
        })
        assert res.status_code == 200
        assert res.json()["role"] == "admin"

    def test_admin_cannot_modify_protected_user_even_if_not_sysadmin(self, admin_client, db_session):
        """Protected flag works independently of role - admin cannot modify any protected user."""
        # Create a protected regular user
        protected_user = User(
            username="protected_regular",
            email="protected_regular@test.com",
            hashed_password=hash_password("Pass123!"),
            role=UserRole.user,
            is_active=True,
            is_protected=True,
        )
        db_session.add(protected_user)
        db_session.commit()
        db_session.refresh(protected_user)

        # Admin tries to deactivate protected regular user -> should fail
        res = admin_client.delete(f"/api/admin/users/{protected_user.id}")
        assert res.status_code == 403
        assert "protected" in res.json()["detail"].lower()

        # Admin tries to demote protected regular user -> should fail
        res = admin_client.patch(f"/api/admin/users/{protected_user.id}", json={
            "role": "viewer",
        })
        assert res.status_code == 403
        assert "protected" in res.json()["detail"].lower()

    def test_reactivate_user(self, admin_client, regular_user):
        """Admin can reactivate a deactivated user."""
        # First deactivate
        res = admin_client.delete(f"/api/admin/users/{regular_user.id}")
        assert res.status_code == 200
        assert res.json()["is_active"] is False

        # Then reactivate
        res = admin_client.post(f"/api/admin/users/{regular_user.id}/reactivate")
        assert res.status_code == 200
        assert res.json()["is_active"] is True

    def test_reactivate_already_active_user_fails(self, admin_client, regular_user):
        """Reactivating an already active user returns 400."""
        res = admin_client.post(f"/api/admin/users/{regular_user.id}/reactivate")
        assert res.status_code == 400
        assert "already active" in res.json()["detail"].lower()

    def test_protected_sysadmin_cannot_be_reactivated_by_admin(self, admin_client, db_session):
        """Admin cannot reactivate a deactivated protected sysadmin."""
        sysadmin = User(
            username="sysadmin_reactivate",
            email="sysadmin_reactivate@test.com",
            hashed_password=hash_password("Pass123!"),
            role=UserRole.sysadmin,
            is_active=False,
            is_protected=True,
        )
        db_session.add(sysadmin)
        db_session.commit()
        db_session.refresh(sysadmin)

        # Admin tries to reactivate -> should fail
        res = admin_client.post(f"/api/admin/users/{sysadmin.id}/reactivate")
        assert res.status_code == 403
        assert "protected" in res.json()["detail"].lower()

    def test_sysadmin_login_works(self, client, db_session):
        """SYSADMIN user can login successfully."""
        sysadmin = User(
            username="login_sysadmin",
            email="login_sysadmin@test.com",
            hashed_password=hash_password("LoginPass123!"),
            role=UserRole.sysadmin,
            is_active=True,
        )
        db_session.add(sysadmin)
        db_session.commit()

        res = client.post("/api/auth/login", json={
            "username_or_email": "login_sysadmin",
            "password": "LoginPass123!",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["user"]["role"] == "sysadmin"
        assert data["user"]["permissions"]["can_manage_users"] is True
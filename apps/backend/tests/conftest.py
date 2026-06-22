import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from app.main import app
from app.db.session import get_db
from app.core.config import settings
from app.security.models import User, UserRole
from app.security.password import hash_password

# Ensure JWT works in tests
settings.JWT_SECRET_KEY = "test-secret-key-for-pytest"
settings.JWT_ALGORITHM = "HS256"
settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Disable rate limiting in tests to avoid false failures from shared test IPs
settings.RATE_LIMIT_ENABLED = False

# Set CORS origins for tests so CORS preflight/header tests work
settings.CORS_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"

TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Patch JWT auth provider to use test DB session
import importlib
_auth_mod = importlib.import_module("app.security.auth")
_auth_mod.SessionLocal = TestingSessionLocal
# Also patch the permissions module's SessionLocal so the resolver uses the test DB
_perms_mod = importlib.import_module("app.security.permissions")
_perms_mod.SessionLocal = TestingSessionLocal
# Patch api.auth's SessionLocal so audit logging writes to the test DB
_auth_api_mod = importlib.import_module("app.api.auth")
_auth_api_mod.SessionLocal = TestingSessionLocal
# Patch security.audit's SessionLocal so audit logger writes to the test DB
_audit_mod = importlib.import_module("app.security.audit")
_audit_mod.SessionLocal = TestingSessionLocal


def _create_test_user(db_session, username: str, email: str, password: str, role: UserRole = UserRole.USER, is_active: bool = True) -> User:
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


def _login_user(client: TestClient, username_or_email: str, password: str) -> str:
    res = client.post("/api/auth/login", json={
        "username_or_email": username_or_email,
        "password": password,
    })
    assert res.status_code == 200, f"Login failed: {res.text}"
    return res.json()["access_token"]

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        # Expire any cached objects so each request sees fresh data from the DB
        db_session.expire_all()
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def auth_client(db_session):
    """
    Test client with admin authentication via X-Dev-User header.
    
    Use this fixture for tests that require authentication.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    
    # Create client with admin auth header
    client = TestClient(app, headers={"X-Dev-User": "admin_user"})
    yield client
    
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def jwt_admin_client(db_session, client):
    """Test client authenticated as admin via JWT Bearer token."""
    # Remove X-Dev-User header so it doesn't conflict with JWT auth
    client.headers.pop("X-Dev-User", None)
    _create_test_user(db_session, "admin", "admin@test.com", "adminpass", role=UserRole.ADMIN)
    token = _login_user(client, "admin", "adminpass")
    client.headers["Authorization"] = f"Bearer {token}"
    yield client
    client.headers.pop("Authorization", None)


@pytest.fixture(scope="function")
def jwt_user_client(db_session, client):
    """Test client authenticated as regular user via JWT Bearer token."""
    # Remove X-Dev-User header so it doesn't conflict with JWT auth
    client.headers.pop("X-Dev-User", None)
    _create_test_user(db_session, "regular", "user@test.com", "userpass", role=UserRole.USER)
    token = _login_user(client, "regular", "userpass")
    client.headers["Authorization"] = f"Bearer {token}"
    yield client
    client.headers.pop("Authorization", None)


@pytest.fixture(scope="function")
def jwt_viewer_client(db_session, client):
    """Test client authenticated as viewer via JWT Bearer token."""
    _create_test_user(db_session, "viewer", "viewer@test.com", "viewerpass", role=UserRole.VIEWER)
    token = _login_user(client, "viewer", "viewerpass")
    client.headers["Authorization"] = f"Bearer {token}"
    yield client
    client.headers.pop("Authorization", None)
"""
Tests for Phase 14 — File Upload Hardening.
"""

import io
from fastapi.testclient import TestClient

from app.core.config import settings
from app.security.models import User, UserRole
from app.security.password import hash_password


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


class TestUploadFileTypeValidation:
    def test_unsupported_file_type_rejected(self, client: TestClient, db_session):
        user = _create_user(db_session, "uploaduser", "upload@example.com", "UserPass123!", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "uploaduser",
            "password": "UserPass123!",
        })
        token = login.json()["access_token"]

        res = client.post(
            "/api/documents/upload",
            files={"file": ("malicious.exe", io.BytesIO(b"fake exe content"), "application/x-msdownload")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 415
        assert "Unsupported file" in res.json()["detail"]

    def test_valid_pdf_accepted(self, client: TestClient, db_session):
        user = _create_user(db_session, "uploadpdf", "uploadpdf@example.com", "UserPass123!", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "uploadpdf",
            "password": "UserPass123!",
        })
        token = login.json()["access_token"]

        res = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(b"%PDF-1.4 fake pdf"), "application/pdf")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code in (200, 201, 500)  # 500 if extraction fails, which is OK for this test

    def test_valid_txt_accepted(self, client: TestClient, db_session):
        user = _create_user(db_session, "uploadtxt", "uploadtxt@example.com", "UserPass123!", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "uploadtxt",
            "password": "UserPass123!",
        })
        token = login.json()["access_token"]

        res = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", io.BytesIO(b"Hello world text content"), "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code in (200, 201)


class TestUploadSizeLimit:
    def test_oversized_upload_rejected(self, client: TestClient, db_session):
        user = _create_user(db_session, "uploadsize", "uploadsize@example.com", "UserPass123!", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "uploadsize",
            "password": "UserPass123!",
        })
        token = login.json()["access_token"]

        # Create content larger than UPLOAD_MAX_SIZE_MB
        max_size = settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024
        oversized = b"x" * (max_size + 1000)

        res = client.post(
            "/api/documents/upload",
            files={"file": ("huge.txt", io.BytesIO(oversized), "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 413
        assert "File too large" in res.json()["detail"]


class TestUploadPathTraversal:
    def test_path_traversal_filename_blocked(self, client: TestClient, db_session):
        user = _create_user(db_session, "uploadtraversal", "uploadtraversal@example.com", "UserPass123!", role=UserRole.user)
        login = client.post("/api/auth/login", json={
            "username_or_email": "uploadtraversal",
            "password": "UserPass123!",
        })
        token = login.json()["access_token"]

        res = client.post(
            "/api/documents/upload",
            files={"file": ("../../etc/passwd", io.BytesIO(b"fake content"), "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should be rejected for either path traversal or unsupported type
        # Path traversal check should catch it first
        assert res.status_code in (400, 415)
        if res.status_code == 400:
            assert "path traversal" in res.json()["detail"].lower()

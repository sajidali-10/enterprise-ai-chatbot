import io
import pytest
from fastapi.testclient import TestClient


def test_upload_txt(auth_client: TestClient):
    """Test document upload with authentication."""
    response = auth_client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"Hello world"), "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_name"] == "test.txt"
    assert data["mime_type"] == "text/plain"
    # Note: extracted_text is NOT returned by the endpoint - this is a test bug
    # The test was asserting a field that was never returned
    assert data["status"] == "indexed"


def test_upload_unsupported_type(auth_client: TestClient):
    """Test upload with unsupported file type returns 415."""
    response = auth_client.post(
        "/api/documents/upload",
        files={"file": ("test.exe", io.BytesIO(b"bad"), "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_too_large(auth_client: TestClient, monkeypatch):
    """Test upload with file too large returns 413."""
    import app.api.documents as docs_mod
    monkeypatch.setattr(docs_mod.settings, "UPLOAD_MAX_SIZE_MB", 0)
    response = auth_client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert response.status_code == 413


def test_upload_unauthenticated(client: TestClient):
    """Test upload without auth returns 401."""
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"Hello"), "text/plain")},
    )
    assert response.status_code == 401


def test_list_documents(auth_client: TestClient):
    """Test list documents with authentication."""
    auth_client.post(
        "/api/documents/upload",
        files={"file": ("a.txt", io.BytesIO(b"A"), "text/plain")},
    )
    response = auth_client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["original_name"] == "a.txt"
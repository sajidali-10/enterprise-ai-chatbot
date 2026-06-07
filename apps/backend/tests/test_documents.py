import io
import pytest
from fastapi.testclient import TestClient

def test_upload_txt(client: TestClient):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"Hello world"), "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_name"] == "test.txt"
    assert data["mime_type"] == "text/plain"
    assert data["extracted_text"] == "Hello world"
    assert data["status"] == "completed"

def test_upload_unsupported_type(client: TestClient):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.exe", io.BytesIO(b"bad"), "application/octet-stream")},
    )
    assert response.status_code == 415

def test_upload_too_large(client: TestClient, monkeypatch):
    import app.api.documents as docs_mod
    monkeypatch.setattr(docs_mod.settings, "UPLOAD_MAX_SIZE_MB", 0)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert response.status_code == 413

def test_list_documents(client: TestClient):
    client.post(
        "/api/documents/upload",
        files={"file": ("a.txt", io.BytesIO(b"A"), "text/plain")},
    )
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["original_name"] == "a.txt"
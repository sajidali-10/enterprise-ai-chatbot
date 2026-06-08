import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.document import Document, DocumentVersion


def test_index_endpoint_requires_document(auth_client, db_session):
    """Test index endpoint with auth but no document returns 404."""
    response = auth_client.post("/api/documents/99999/index")
    assert response.status_code == 404


def test_index_endpoint_requires_extracted_text(auth_client, db_session):
    """Test index endpoint requires extracted text."""
    # Create document without extracted text
    doc = Document(
        filename="test.txt",
        original_name="test.txt",
        mime_type="text/plain",
        size_bytes=10,
        status="completed",
    )
    db_session.add(doc)
    db_session.commit()
    
    response = auth_client.post(f"/api/documents/{doc.id}/index")
    assert response.status_code == 400


def test_index_endpoint_unauthenticated(client, db_session):
    """Test index endpoint without auth returns 401."""
    response = client.post("/api/documents/1/index")
    assert response.status_code == 401
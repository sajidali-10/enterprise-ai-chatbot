import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.document import Document, DocumentVersion

client = TestClient(app)

def test_index_endpoint_requires_document(db_session):
    # Without a document, should return 404
    response = client.post("/api/documents/99999/index")
    assert response.status_code == 404

def test_index_endpoint_requires_extracted_text(db_session):
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
    
    response = client.post(f"/api/documents/{doc.id}/index")
    assert response.status_code == 400
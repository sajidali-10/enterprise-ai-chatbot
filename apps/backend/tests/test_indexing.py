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
from unittest.mock import MagicMock


def test_index_skips_chunks_with_none_content(auth_client, db_session, monkeypatch):
    """Indexing does not pass None/empty content to embedding provider."""
    from app.models.document import Document, DocumentVersion

    doc = Document(
        filename="test.txt",
        original_name="test.txt",
        mime_type="text/plain",
        size_bytes=10,
        status="completed",
    )
    db_session.add(doc)
    db_session.commit()

    version = DocumentVersion(
        document_id=doc.id,
        version_number=1,
        storage_key="test.txt",
        extractor_type="PlainTextParser",
        extracted_text="some text",
    )
    db_session.add(version)
    db_session.commit()

    class FakeChunk:
        def __init__(self, content, idx=0):
            self.content = content
            self.chunk_index = idx
            self.title = "T"
            self.section_heading = None
            self.page_number = None
            self.source_file_name = "f.txt"
            self.content_hash = "abc"

    import app.api.documents as docs_mod
    fake_chunker = MagicMock()
    fake_chunker.chunk.return_value = [
        FakeChunk("valid", 0),
        FakeChunk(None, 1),
        FakeChunk("", 2),
        FakeChunk("also valid", 3),
    ]
    monkeypatch.setattr(docs_mod, "RecursiveChunker", lambda: fake_chunker)

    mock_provider = MagicMock()
    mock_provider.dimension = 384
    mock_provider.embed.return_value = [
        [1.0] * 384,
        [2.0] * 384,
    ]
    monkeypatch.setattr(docs_mod, "get_embedding_provider", lambda: mock_provider)
    monkeypatch.setattr(docs_mod, "ensure_collection", lambda dim: None)
    monkeypatch.setattr(docs_mod, "upsert_chunks", lambda chunks, meta: None)

    response = auth_client.post(f"/api/documents/{doc.id}/index")
    assert response.status_code == 200
    assert response.json()["chunks_created"] == 2

    texts = mock_provider.embed.call_args[0][0]
    assert texts == ["valid", "also valid"]
    assert None not in texts
    assert "" not in texts


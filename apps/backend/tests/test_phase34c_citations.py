"""Phase 34C -- Citations tests.

Verifies that ``app.rag.citations.format_citations`` correctly
enriches image-knowledge citations with the Phase 34C metadata while
preserving the existing KB citation shape.

Mapping to acceptance criterion:
   18. Image citation resolves correctly.
"""

from __future__ import annotations

import os

import pytest


def _ensure_settings_env():
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_ENABLED", "true")
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION", "1")


_ensure_settings_env()


def _image_knowledge_chunk(**overrides):
    base = {
        "chunk_id": "9999",
        "point_id": 9999,
        "document_id": 42,
        "image_id": 7,
        "chunk_index": 0,
        "content": "Source image: server_b.png\nImage type: application_ui\n"
                   "OCR:\nServer A\nServer B\nFailed\n"
                   "Visual description:\nDashboard showing Server B failed.\n"
                   "Visual findings:\n- Server B is failed",
        "source_file_name": "server_b.png",
        "title": "server_b.png",
        "source_type": "image_knowledge",
        "content_type": "image_knowledge",
        "image_type": "application_ui",
        "vision_provider": "mock",
        "vision_model": "mock-v1",
        "vision_processed_at": "2026-08-23T10:00:00",
        "is_ocr": True,
        "has_vision": True,
        "knowledge_schema_version": 1,
        "score": 0.91,
    }
    base.update(overrides)
    return base


def _kb_chunk(**overrides):
    base = {
        "chunk_id": "100",
        "document_id": 1,
        "chunk_index": 0,
        "content": "Error 902 means message delivery failed.",
        "source_file_name": "kb.txt",
        "title": "kb.txt",
        "source_type": "native_text",
        "score": 0.8,
    }
    base.update(overrides)
    return base


def test_image_knowledge_citation_has_structured_metadata():
    """Image citations carry the Phase 34C metadata block."""
    from app.rag.citations import format_citations

    chunk = _image_knowledge_chunk()
    citations = format_citations([chunk])
    assert len(citations) == 1
    cit = citations[0]
    assert cit["citation_kind"] == "image_knowledge"
    assert cit["image_id"] == 7
    assert cit["image_type"] == "application_ui"
    assert cit["vision_provider"] == "mock"
    assert cit["vision_model"] == "mock-v1"
    assert cit["has_vision"] is True
    assert cit["knowledge_schema_version"] == 1
    assert cit["source_file_name"] == "server_b.png"
    assert cit["index"] == 1


def test_image_knowledge_citation_snippet_is_bounded():
    """The citation snippet for an image knowledge chunk is bounded."""
    from app.rag.citations import format_citations

    chunk = _image_knowledge_chunk()
    citations = format_citations([chunk])
    snippet = citations[0]["content_snippet"]
    assert snippet
    # We use clean_excerpt with a default max_length=200.
    assert len(snippet) <= 240  # clean_excerpt may add an ellipsis


def test_image_knowledge_does_not_leak_internal_point_id():
    """Internal Qdrant point IDs must NOT appear in user-facing citations."""
    from app.rag.citations import format_citations

    chunk = _image_knowledge_chunk()
    citations = format_citations([chunk])
    cit = citations[0]
    # We use ``chunk_id`` (which is the string representation of the
    # point id, but stable per-image). ``point_id`` is an internal
    # field that callers can use to trace but we don't expose it
    # via the citation dict.
    assert "point_id" not in cit


def test_kb_citation_unchanged():
    """18. KB citations keep the existing shape -- only image_knowledge chunks change."""
    from app.rag.citations import format_citations

    chunk = _kb_chunk()
    citations = format_citations([chunk])
    cit = citations[0]
    # No Phase 34C metadata on KB citations.
    assert "citation_kind" not in cit
    assert "image_id" not in cit
    assert cit["source_file_name"] == "kb.txt"


def test_mixed_kb_and_image_citations_keep_separation():
    """19. Mixed KB + image answer keeps source separation in the citation list."""
    from app.rag.citations import format_citations

    chunks = [_kb_chunk(), _image_knowledge_chunk()]
    citations = format_citations(chunks)
    assert len(citations) == 2
    # First citation is KB (the input order is preserved).
    assert "citation_kind" not in citations[0]
    # Second is image knowledge.
    assert citations[1]["citation_kind"] == "image_knowledge"
    # Each citation has a unique, sequential index.
    assert citations[0]["index"] == 1
    assert citations[1]["index"] == 2

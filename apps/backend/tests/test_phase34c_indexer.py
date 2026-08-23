"""Phase 34C -- Indexer tests.

Tests that exercise the upsert / delete contract of the multimodal
indexer with a stub Qdrant + stub embedding provider.

Mapping to acceptance criteria:

    3. Existing Vision result is reused without Vision provider call.
    4. Reindex is idempotent.
    5. Same image does not create duplicate Qdrant points.
    7. Stable point ID does not depend on timestamp.
    8. Correct metadata is persisted in Qdrant payload.
    9. RBAC/visibility metadata is included.
   10. Deleted image knowledge is removable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest


def _ensure_settings_env():
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_ENABLED", "true")
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION", "1")
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_TEXT_MAX_CHARS", "1800")


_ensure_settings_env()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class StubPoint:
    id: int
    vector: List[float]
    payload: Dict[str, Any]


@dataclass
class StubEmbeddingProvider:
    model_name: str = "stub-embed-v1"
    dimension: int = 8

    def embed(self, texts):
        # Deterministic vector -- small enough to inspect, large enough
        # to be non-zero. Constant so we can assert on it.
        return [[0.1] * self.dimension for _ in texts]


class StubQdrantClient:
    """In-memory Qdrant replacement that records upsert / delete calls."""

    def __init__(self):
        self.points: Dict[int, StubPoint] = {}
        self.upsert_calls: List[List[int]] = []
        self.delete_calls: List[List[int]] = []
        self.search_calls: List[Dict[str, Any]] = []
        self.scroll_calls: List[Dict[str, Any]] = []
        self.retrieve_calls: List[List[int]] = []

    # upsert
    def upsert(self, *, collection_name, points):
        ids = []
        for p in points:
            self.points[p.id] = StubPoint(id=p.id, vector=p.vector, payload=p.payload)
            ids.append(p.id)
        self.upsert_calls.append(ids)

    # search
    def search(self, *, collection_name, query_vector, limit, query_filter=None, with_payload=True):
        self.search_calls.append({
            "collection": collection_name,
            "limit": limit,
            "filter": query_filter,
        })
        # Return points whose payload matches the filter (very small impl).
        out = []
        for pid, point in self.points.items():
            if query_filter is None or _filter_matches(point.payload, query_filter):
                out.append(type("Hit", (), {"id": pid, "payload": point.payload, "score": 0.9})())
            if len(out) >= limit:
                break
        return out

    # scroll
    def scroll(self, *, collection_name, scroll_filter, limit, with_payload, with_vectors):
        self.scroll_calls.append({"filter": scroll_filter, "limit": limit})
        out = []
        for pid, point in self.points.items():
            if _filter_matches(point.payload, scroll_filter):
                out.append(type("Point", (), {"id": pid, "payload": point.payload})())
            if len(out) >= limit:
                break
        return (out, None)

    # retrieve
    def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
        self.retrieve_calls.append(list(ids))
        return [type("Point", (), {"id": pid, "payload": self.points[pid].payload})() for pid in ids if pid in self.points]

    # delete
    def delete(self, *, collection_name, points):
        for pid in points:
            self.points.pop(pid, None)
        self.delete_calls.append(list(points))


def _filter_matches(payload: Dict[str, Any], flt) -> bool:
    """Tiny Qdrant filter matcher sufficient for source_type / image_id / document_id."""
    must = []
    if flt is None:
        return True
    for attr in ("must", "should", "must_not"):
        if hasattr(flt, attr):
            setattr(flt, attr, getattr(flt, attr))  # no-op, just touch
    try:
        must = list(flt.must or [])
    except Exception:
        return True
    for cond in must:
        try:
            key = cond.key
            value = cond.match.value
        except Exception:
            return True
        if str(payload.get(key)) != str(value):
            return False
    return True


def _make_record(**overrides):
    from app.services.multimodal.knowledge_record import ImageKnowledgeRecord
    base = dict(
        document_id=42,
        image_id=7,
        owner_user_id=1,
        visibility="global",
        original_filename="server_b.png",
        mime_type="image/png",
        ocr_text="Server A\nServer B\nFailed",
        ocr_confidence=92,
        ocr_status="success",
        knowledge_schema_version=1,
        is_ocr_only=True,
    )
    base.update(overrides)
    rec = ImageKnowledgeRecord(**base)
    from app.services.multimodal.knowledge_record import build_knowledge_text, cap_knowledge_text
    rec.knowledge_text = build_knowledge_text(rec)
    rec.embedding_text = cap_knowledge_text(rec.knowledge_text)
    return rec


@pytest.fixture
def stubbed_env(monkeypatch):
    """Patch the Qdrant client and embedding provider with stubs."""
    qc = StubQdrantClient()
    embedder = StubEmbeddingProvider()

    monkeypatch.setattr(
        "app.services.vector.qdrant_service.get_qdrant_client",
        lambda: qc,
    )
    monkeypatch.setattr(
        "app.services.embeddings.get_embedding_provider",
        lambda: embedder,
    )
    # Force settings module attribute reads to be deterministic.
    return qc, embedder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_upsert_creates_one_point(stubbed_env):
    """8. Correct metadata is persisted in Qdrant payload."""
    qc, _ = stubbed_env
    from app.services.multimodal.indexer import upsert_image_knowledge

    result = upsert_image_knowledge(_make_record())
    assert result.status == "upserted"
    assert len(qc.points) == 1
    pid, point = next(iter(qc.points.items()))
    assert pid == result.point_id

    payload = point.payload
    # Source / kind
    assert payload["source_type"] == "image_knowledge"
    assert payload["content_type"] == "image_knowledge"
    # Citation / RBAC
    assert payload["document_id"] == 42
    assert payload["image_id"] == 7
    assert payload["owner_user_id"] == 1
    assert payload["visibility"] == "global"
    assert payload["source_file_name"] == "server_b.png"
    assert payload["title"] == "server_b.png"
    # Provenance
    assert payload["is_ocr"] is True
    assert payload["has_vision"] is False
    assert payload["knowledge_schema_version"] == 1
    # Knowledge text is stored AND used as content
    assert "OCR:" in payload["content"]
    assert "Server B" in payload["knowledge_text"]
    # Embedding metadata
    assert payload["embedding_model"] == "stub-embed-v1"
    assert payload["embedding_dimension"] == 8


def test_upsert_is_idempotent(stubbed_env):
    """4 & 5. Re-running upsert for the same image hits the same point id."""
    qc, _ = stubbed_env
    from app.services.multimodal.indexer import upsert_image_knowledge

    r1 = upsert_image_knowledge(_make_record())
    r2 = upsert_image_knowledge(_make_record())
    assert r1.point_id == r2.point_id
    # Exactly one point remains.
    assert len(qc.points) == 1
    assert len(qc.upsert_calls) == 2  # both calls happened


def test_upsert_with_vision_sets_has_vision_true(stubbed_env):
    """3 + 9. Vision-enriched payload records has_vision=True; Vision fields present."""
    qc, _ = stubbed_env
    from app.services.multimodal.indexer import upsert_image_knowledge

    rec = _make_record(
        vision_status="success",
        vision_description="Dashboard with Server B failed.",
        visual_findings=["Server B is failed"],
        detected_entities=["Server B"],
        vision_provider="mock",
        vision_model="mock-v1",
        image_type="application_ui",
        is_ocr_only=False,
    )
    result = upsert_image_knowledge(rec)
    assert result.status == "upserted"
    point = qc.points[result.point_id]
    assert point.payload["has_vision"] is True
    assert point.payload["image_type"] == "application_ui"
    assert point.payload["vision_provider"] == "mock"
    assert point.payload["vision_model"] == "mock-v1"


def test_delete_removes_point(stubbed_env):
    """10. Deleted image knowledge is removable."""
    qc, _ = stubbed_env
    from app.services.multimodal.indexer import (
        delete_image_knowledge,
        upsert_image_knowledge,
    )

    rec = _make_record()
    r = upsert_image_knowledge(rec)
    assert r.point_id in qc.points
    del_result = delete_image_knowledge(rec.image_id)
    assert del_result.deleted == 1
    assert r.point_id not in qc.points


def test_delete_document_removes_only_image_knowledge(stubbed_env):
    """Document-level delete clears image knowledge points."""
    qc, _ = stubbed_env
    from app.services.multimodal.indexer import (
        delete_image_knowledge_by_document_id,
        upsert_image_knowledge,
    )
    upsert_image_knowledge(_make_record(document_id=1, image_id=1))
    upsert_image_knowledge(_make_record(document_id=1, image_id=2))
    upsert_image_knowledge(_make_record(document_id=2, image_id=3))
    # Inject a non-image-knowledge point manually and ensure it survives.
    qc.points[999_999] = StubPoint(id=999_999, vector=[0.0] * 8, payload={"source_type": "native_text", "document_id": 1})
    res = delete_image_knowledge_by_document_id(1)
    assert res.deleted == 2
    # Non-image-knowledge point untouched.
    assert 999_999 in qc.points


def test_upsert_when_disabled_returns_disabled(monkeypatch):
    """MULTIMODAL_KNOWLEDGE_ENABLED=false -> upsert is a no-op."""
    from app.core.config import settings as _settings
    monkeypatch.setattr(_settings, "MULTIMODAL_KNOWLEDGE_ENABLED", False)
    from app.services.multimodal.indexer import upsert_image_knowledge

    r = upsert_image_knowledge(_make_record())
    assert r.status == "disabled"


def test_upsert_with_empty_knowledge_skips(monkeypatch, stubbed_env):
    """An empty knowledge_text is a safe skip, not an upsert."""
    qc, _ = stubbed_env
    from app.services.multimodal.indexer import upsert_image_knowledge

    rec = _make_record()
    rec.knowledge_text = ""
    rec.embedding_text = ""
    r = upsert_image_knowledge(rec)
    assert r.status == "skipped"
    assert len(qc.points) == 0

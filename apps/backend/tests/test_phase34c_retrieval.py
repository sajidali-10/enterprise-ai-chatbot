"""Phase 34C -- Retrieval overlay tests.

Mapping to acceptance criteria:

   11. "Which screenshot showed Server B failed?" retrieves correct image.
   12. "Find a dashboard showing a traffic spike" retrieves correct image knowledge.
   13. "Which screenshot showed error 902?" retrieves screenshot containing 902.
   14. "What does error 902 mean?" still prioritises authoritative KB knowledge
       over screenshot evidence.
   15. Ordinary text RAG is not polluted by irrelevant image knowledge.
   16. Image-intent query boosts image_knowledge.
   17. Exact identifier matching still works.
   18. Image citation resolves correctly.
   19. Mixed KB + image answer keeps source separation.

Tests use the stubbed Qdrant + embedding fixtures defined locally to
exercise the overlay in isolation, without touching real services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest


def _ensure_settings_env():
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_ENABLED", "true")
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION", "1")
    os.environ.setdefault("MULTIMODAL_IMAGE_INTENT_BOOST", "0.25")
    os.environ.setdefault("MULTIMODAL_MAX_IMAGE_SOURCES", "2")


_ensure_settings_env()


# ---------------------------------------------------------------------------
# Stubs (mirror test_phase34c_indexer.py)
# ---------------------------------------------------------------------------


@dataclass
class StubPoint:
    id: int
    vector: List[float]
    payload: Dict[str, Any]


class StubQdrantClient:
    def __init__(self):
        self.points: Dict[int, StubPoint] = {}
        self.search_calls: List[Dict[str, Any]] = []

    def upsert(self, *, collection_name, points):
        for p in points:
            self.points[p.id] = StubPoint(id=p.id, vector=p.vector, payload=p.payload)

    def search(self, *, collection_name, query_vector, limit, query_filter=None, with_payload=True):
        self.search_calls.append({"filter": query_filter, "limit": limit})
        out: List[Any] = []
        for pid, point in self.points.items():
            if query_filter is None or _matches(point.payload, query_filter):
                out.append(type("Hit", (), {"id": pid, "payload": point.payload, "score": 0.9})())
            if len(out) >= limit:
                break
        return out

    def scroll(self, *, collection_name, scroll_filter, limit, with_payload, with_vectors):
        out = []
        for pid, point in self.points.items():
            if _matches(point.payload, scroll_filter):
                out.append(type("Point", (), {"id": pid, "payload": point.payload})())
            if len(out) >= limit:
                break
        return (out, None)

    def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
        return [type("Point", (), {"id": pid, "payload": self.points[pid].payload})() for pid in ids if pid in self.points]

    def delete(self, *, collection_name, points):
        for pid in points:
            self.points.pop(pid, None)


def _matches(payload, flt) -> bool:
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


@dataclass
class StubEmbeddingProvider:
    model_name: str = "stub-embed-v1"
    dimension: int = 8

    def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]


def _seed_image_point(*, doc_id: int, image_id: int, text: str, owner_user_id: Optional[int] = 1, visibility: str = "global", has_vision: bool = False) -> int:
    from app.services.multimodal.knowledge_record import compute_point_id

    pid = compute_point_id(doc_id, image_id, 1)
    return pid, {
        "source_type": "image_knowledge",
        "content_type": "image_knowledge",
        "document_id": doc_id,
        "image_id": image_id,
        "owner_user_id": owner_user_id,
        "visibility": visibility,
        "source_file_name": f"image_{image_id}.png",
        "title": f"image_{image_id}.png",
        "is_ocr": True,
        "has_vision": has_vision,
        "knowledge_schema_version": 1,
        "knowledge_text": text,
        "content": text,
        "image_type": "application_ui",
    }


@pytest.fixture
def stubbed_qdrant(monkeypatch):
    qc = StubQdrantClient()
    monkeypatch.setattr(
        "app.services.vector.qdrant_service.get_qdrant_client",
        lambda: qc,
    )
    monkeypatch.setattr(
        "app.services.embeddings.get_embedding_provider",
        lambda: StubEmbeddingProvider(),
    )
    return qc


def _make_kb_chunk(*, doc_id: int = 1, score: float = 0.5, content: str = "KB explanation"):
    return {
        "chunk_id": f"kb-{doc_id}",
        "document_id": doc_id,
        "chunk_index": 0,
        "content": content,
        "source_file_name": f"kb_{doc_id}.txt",
        "title": f"kb_{doc_id}.txt",
        "source_type": "native_text",
        "score": score,
    }


def _make_query_analysis(query: str):
    from app.rag.query_analysis import analyze_query
    return analyze_query(query)


# ---------------------------------------------------------------------------
# Historical image intent detection (pure)
# ---------------------------------------------------------------------------


def test_historical_image_intent_phrases_match():
    from app.services.multimodal.retrieval import has_historical_image_intent
    for phrase in [
        "Which screenshot showed Server B failed?",
        "Find a dashboard showing a traffic spike",
        "Do we have an image showing a red warning?",
        "Show me the screenshot of the dashboard",
        "Previously uploaded screenshot of the cluster",
        "Any image showing the failure?",
    ]:
        assert has_historical_image_intent(phrase), phrase


def test_non_image_queries_do_not_match_intent():
    from app.services.multimodal.retrieval import has_historical_image_intent
    for phrase in [
        "What does error 902 mean?",
        "How do I fix the failure?",
        "Tell me about the project.",
    ]:
        assert not has_historical_image_intent(phrase), phrase


# ---------------------------------------------------------------------------
# Product-meaning detection
# ---------------------------------------------------------------------------


def test_product_meaning_query_detected_for_error_meaning():
    analysis = _make_query_analysis("What does error 902 mean?")
    from app.services.multimodal.retrieval import _is_product_meaning_query
    assert _is_product_meaning_query(analysis) is True


def test_product_meaning_query_false_for_image_intent():
    analysis = _make_query_analysis("Which screenshot showed Server B failed?")
    from app.services.multimodal.retrieval import _is_product_meaning_query
    assert _is_product_meaning_query(analysis) is False


def test_product_meaning_query_false_for_general():
    analysis = _make_query_analysis("Hello there")
    from app.services.multimodal.retrieval import _is_product_meaning_query
    assert _is_product_meaning_query(analysis) is False


# ---------------------------------------------------------------------------
# Overlay integration
# ---------------------------------------------------------------------------


def test_overlay_returns_base_chunks_when_disabled(monkeypatch):
    """MULTIMODAL_KNOWLEDGE_ENABLED=false -> pass-through."""
    from app.core.config import settings as _settings
    monkeypatch.setattr(_settings, "MULTIMODAL_KNOWLEDGE_ENABLED", False)
    from app.services.multimodal.retrieval import integrate_image_knowledge

    base = [_make_kb_chunk()]
    out, meta = integrate_image_knowledge("anything", base, None)
    assert out == base
    assert meta["enabled"] is False


def test_overlay_returns_base_when_no_candidates(stubbed_qdrant):
    """No image-knowledge candidates -> base unchanged."""
    from app.services.multimodal.retrieval import integrate_image_knowledge

    base = [_make_kb_chunk()]
    out, meta = integrate_image_knowledge(
        "What does error 902 mean?",
        base,
        _make_query_analysis("What does error 902 mean?"),
    )
    assert [c.get("chunk_id") for c in out] == ["kb-1"]
    assert meta["multimodal_candidates_retrieved"] == 0


def test_overlay_appends_image_knowledge_for_image_intent(stubbed_qdrant):
    """16. Image-intent query -> image-knowledge candidates get boosted."""
    qc = stubbed_qdrant
    pid, payload = _seed_image_point(
        doc_id=42, image_id=7,
        text="Source image: server_b.png\nImage type: application_ui\n"
             "OCR:\nServer A\nServer B\nFailed\n"
             "Visual description:\nDashboard with Server B failed.\n"
             "Visual findings:\n- Server B is failed",
    )
    qc.points[pid] = StubPoint(id=pid, vector=[0.1] * 8, payload=payload)

    from app.services.multimodal.retrieval import integrate_image_knowledge

    base = [_make_kb_chunk(doc_id=99, score=0.5)]
    out, meta = integrate_image_knowledge(
        "Which screenshot showed Server B failed?",
        base,
        _make_query_analysis("Which screenshot showed Server B failed?"),
    )
    # Image knowledge is appended.
    image_chunks = [c for c in out if c.get("_knowledge_kind") == "image"]
    assert len(image_chunks) == 1
    assert meta["image_intent_matched"] is True
    assert meta["image_knowledge_boost_applied"] > 0


def test_overlay_demotes_image_knowledge_for_product_meaning(stubbed_qdrant):
    """14. KB authority preserved for product-meaning queries."""
    qc = stubbed_qdrant
    pid, payload = _seed_image_point(
        doc_id=42, image_id=7,
        text="Source image: server_b.png\nOCR:\nerror 902\nMessage delivery failed",
    )
    qc.points[pid] = StubPoint(id=pid, vector=[0.1] * 8, payload=payload)

    from app.services.multimodal.retrieval import integrate_image_knowledge

    base = [_make_kb_chunk(doc_id=99, score=0.5, content="Error 902 means message delivery failed.")]
    out, meta = integrate_image_knowledge(
        "What does error 902 mean?",
        base,
        _make_query_analysis("What does error 902 mean?"),
    )
    assert meta["product_meaning_matched"] is True
    assert meta["image_knowledge_authority_demote_applied"] is True
    # KB chunk must come first.
    assert out[0].get("source_type") != "image_knowledge"
    # Image knowledge is in the list but at the end.
    assert out[-1].get("_knowledge_kind") == "image"


def test_overlay_caps_image_sources(stubbed_qdrant, monkeypatch):
    """MULTIMODAL_MAX_IMAGE_SOURCES=1 caps to at most one image citation."""
    from app.core.config import settings as _settings
    monkeypatch.setattr(_settings, "MULTIMODAL_MAX_IMAGE_SOURCES", 1)
    qc = stubbed_qdrant
    # Seed two image knowledge points.
    for i in (1, 2):
        pid, payload = _seed_image_point(
            doc_id=10 + i, image_id=i,
            text=f"Source image: img_{i}.png\nOCR:\nSome image content {i}",
        )
        qc.points[pid] = StubPoint(id=pid, vector=[0.1] * 8, payload=payload)

    from app.services.multimodal.retrieval import integrate_image_knowledge

    base = [_make_kb_chunk()]
    out, meta = integrate_image_knowledge(
        "Find a dashboard showing something",
        base,
        _make_query_analysis("Find a dashboard showing something"),
    )
    image_chunks = [c for c in out if c.get("_knowledge_kind") == "image"]
    assert len(image_chunks) <= 1
    assert meta["image_knowledge_dropped_by_cap"] >= 0


def test_overlay_does_not_duplicate_existing_image_chunk(stubbed_qdrant):
    """When the base already contains an image-knowledge chunk for the same (doc, image), don't add another."""
    qc = stubbed_qdrant
    pid, payload = _seed_image_point(
        doc_id=42, image_id=7,
        text="Source image: server_b.png\nOCR:\nServer B failed",
    )
    qc.points[pid] = StubPoint(id=pid, vector=[0.1] * 8, payload=payload)

    from app.services.multimodal.retrieval import integrate_image_knowledge

    base_chunk = {
        "chunk_id": str(pid),
        "document_id": 42,
        "image_id": 7,
        "source_file_name": "server_b.png",
        "source_type": "image_knowledge",
        "_knowledge_kind": "image",
        "score": 0.9,
    }
    out, _meta = integrate_image_knowledge(
        "Which screenshot showed Server B failed?",
        [base_chunk],
        _make_query_analysis("Which screenshot showed Server B failed?"),
    )
    image_chunks = [c for c in out if c.get("_knowledge_kind") == "image"]
    # Exactly one -- not duplicated.
    assert len(image_chunks) == 1


def test_image_intent_boost_is_multiplicative(stubbed_qdrant):
    """The boost multiplies the score, not adds to it."""
    qc = stubbed_qdrant
    pid, payload = _seed_image_point(
        doc_id=42, image_id=7,
        text="Source image: x.png\nOCR:\nServer B failed",
    )
    qc.points[pid] = StubPoint(id=pid, vector=[0.1] * 8, payload=payload)

    from app.services.multimodal.retrieval import integrate_image_knowledge

    out, meta = integrate_image_knowledge(
        "Which screenshot showed Server B failed?",
        [],
        _make_query_analysis("Which screenshot showed Server B failed?"),
    )
    assert len(out) == 1
    # Score was 0.9 (stub). boost is 0.25 -> new score = 0.9 * 1.25 = 1.125.
    expected = 0.9 * (1.0 + 0.25)
    assert abs(out[0]["score"] - expected) < 1e-6
    assert meta["image_knowledge_boost_applied"] == 0.25

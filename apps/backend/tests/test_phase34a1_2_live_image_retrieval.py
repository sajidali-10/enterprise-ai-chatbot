"""
Phase 34A.1.2 — Live image-content retrieval fix regression tests.

Covers the 15 spec scenarios. Tests are unit-level: they exercise the
new resolver, the Qdrant direct-by-id helper, the image-content routing
selection, the answer-generator pre-resolution branch, the grounding
bypass behaviour, and the citations module's None-score handling.

The tests run without a live Qdrant / PostgreSQL — the direct chunk
provider is injected so we can stub Qdrant entirely.

Spec scenarios:

  1. New uploaded image contains OCR "Error 902". Question: "What error
     code is shown in the image I uploaded?" → OCR chunk is returned
     even when its semantic score is below the normal relevance floor.
  2. image_content does not require whole-KB semantic retrieval before
     source resolution.
  3. Explicit image_context takes precedence over recent-image fallback.
  4. Without explicit context, most recent accessible image is selected.
  5. RBAC prevents selecting another user's inaccessible image.
  6. Image-content answer uses only image OCR source.
  7. admin_test.txt cannot appear for an image-content query.
  8. Programmer's Guide cannot appear for pure image-reading question.
  9. "How do I troubleshoot error 902?" still performs broader KB
     retrieval.
 10. "How do I troubleshoot the error shown in the image?" uses image
     OCR to identify the error, then broadens retrieval.
 11. If no KB troubleshooting evidence exists: no fabricated
     troubleshooting answer.
 12. Normal text RAG remains unchanged.
 13. Legacy OCR payload fallback remains functional.
 14. Phase 34A OCR regression tests remain green.
 15. Phase 34A.1 and Phase 34A.1.1 regression tests remain green.
"""

from unittest.mock import MagicMock

import pytest

from app.rag.image_routing import (
    select_image_aware_chunks,
    select_image_content_chunks,
)
from app.rag.image_resolver import (
    ResolvedImage,
    extract_explicit_target,
    resolve_recent_image,
)
from app.rag.query_analysis import analyze_query, QueryAnalysis
from app.rag.grounding import (
    apply_grounding_checks,
    decide_evidence_level,
)
from app.rag.citations import score_excerpt_relevance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    content: str,
    *,
    source_type: str = "native_text",
    source_file_name: str = "file.txt",
    document_id: str = "1",
    image_id=None,
    chunk_id: str | None = None,
    score: float | None = 0.5,
) -> dict:
    return {
        "chunk_id": chunk_id or f"chunk_{content[:20]}",
        "document_id": document_id,
        "content": content,
        "source_file_name": source_file_name,
        "title": "",
        "score": score,
        "source_type": source_type,
        "image_id": image_id,
    }


def _stub_resolver(resolved: ResolvedImage):
    """Return a stub for `resolve_recent_image` that always returns the
    given ResolvedImage without touching the DB."""
    return lambda db, auth: resolved


def _stub_direct_provider(chunks_by_doc_image: dict | None = None, default=None):
    """Return a stub direct_chunks_provider that returns a fixed chunk
    list keyed by (document_id, image_id)."""
    store = chunks_by_doc_image or {}
    if default is None:
        default = []

    def _provider(doc_id, image_id):
        if (doc_id, image_id) in store:
            return list(store[(doc_id, image_id)])
        return list(default)

    return _provider


# ---------------------------------------------------------------------------
# Scenarios 1-2: by-id OCR retrieval bypasses semantic threshold
# ---------------------------------------------------------------------------


def test_image_content_ocr_returned_below_relevance_floor():
    """Scenario 1: OCR chunk for an image-content question is returned
    even when its semantic similarity to the meta-question is below
    RAG_MIN_RELEVANCE_FLOOR. Source relationship IS the relevance
    signal."""
    ocr_chunk = _chunk(
        "Failed Reason: 902 - Message delivery failed: rejected-forbidden-country",
        source_type="image_ocr",
        source_file_name="Screenshot 2026-08-05.png",
        document_id="57",
        image_id=3,
        score=None,  # by-id lookup has no semantic score
    )
    a = analyze_query("What error code is shown in the image I uploaded?")
    assert a.query_type == "image_content"

    provider = _stub_direct_provider({(57, 3): [ocr_chunk]})
    out, meta = select_image_content_chunks(
        a,
        image_context=None,
        resolved_document_id=57,
        resolved_image_id=3,
        direct_chunks_provider=provider,
    )
    assert out == [ocr_chunk]
    assert meta["routing_mode"] == "image_content_recent"
    assert meta["scoped_count"] == 1


def test_image_content_no_whole_kb_semantic_retrieval_required():
    """Scenario 2: select_image_content_chunks does NOT consume any
    semantic search results; the direct provider is the only path."""
    call_log: list = []

    def _provider(doc_id, image_id):
        call_log.append(("by_id", doc_id, image_id))
        return [_chunk(
            "Error 902",
            source_type="image_ocr",
            document_id=str(doc_id),
            image_id=image_id,
            score=None,
        )]

    a = analyze_query("What error code is shown in the image I uploaded?")
    out, meta = select_image_content_chunks(
        a,
        image_context={"document_id": 99, "image_id": 7},
        direct_chunks_provider=_provider,
    )
    assert len(out) == 1
    assert call_log == [("by_id", 99, 7)]
    assert meta["routing_mode"] == "image_content_explicit"
    # No semantic search was triggered — confirm by the absence of any
    # non-by-id calls.
    assert all(call[0] == "by_id" for call in call_log)


# ---------------------------------------------------------------------------
# Scenario 3: explicit image_context takes precedence over recent fallback
# ---------------------------------------------------------------------------


def test_explicit_image_context_takes_precedence_over_recent():
    a = analyze_query("What error code is shown in the image I uploaded?")
    explicit_chunk = _chunk(
        "Error 902 (explicit)",
        source_type="image_ocr",
        document_id="100",
        image_id=200,
        score=None,
    )
    recent_chunk = _chunk(
        "Error 902 (recent fallback)",
        source_type="image_ocr",
        document_id="57",
        image_id=3,
        score=None,
    )
    provider = _stub_direct_provider({
        (100, 200): [explicit_chunk],
        (57, 3): [recent_chunk],
    })
    out, meta = select_image_content_chunks(
        a,
        image_context={"document_id": 100, "image_id": 200},
        resolved_document_id=57,  # recent fallback would be doc 57
        resolved_image_id=3,
        direct_chunks_provider=provider,
    )
    assert out == [explicit_chunk]
    assert meta["routing_mode"] == "image_content_explicit"
    assert meta["resolved_document_id"] == 100
    assert meta["resolved_image_id"] == 200


# ---------------------------------------------------------------------------
# Scenario 4: without explicit context, resolver-supplied recent image wins
# ---------------------------------------------------------------------------


def test_recent_image_resolution_selected_when_no_explicit_context():
    a = analyze_query("What error code is shown in the image I uploaded?")
    recent_chunk = _chunk(
        "Error 902 (recent)",
        source_type="image_ocr",
        document_id="57",
        image_id=3,
        score=None,
    )
    provider = _stub_direct_provider({(57, 3): [recent_chunk]})
    out, meta = select_image_content_chunks(
        a,
        image_context=None,
        resolved_document_id=57,
        resolved_image_id=3,
        direct_chunks_provider=provider,
    )
    assert out == [recent_chunk]
    assert meta["routing_mode"] == "image_content_recent"
    assert meta["resolved_document_id"] == 57


def test_extract_explicit_target_prefers_document_id():
    explicit = extract_explicit_target({"document_id": 7, "image_id": 99})
    assert explicit.resolved is True
    assert explicit.document_id == 7
    assert explicit.image_id == 99


def test_extract_explicit_target_handles_empty_dict():
    explicit = extract_explicit_target(None)
    assert explicit.resolved is False


# ---------------------------------------------------------------------------
# Scenario 5: RBAC enforcement
# ---------------------------------------------------------------------------


def test_resolve_recent_image_returns_empty_when_rbac_denies_all(monkeypatch):
    """Scenario 5: when the recent-image candidates exist but RBAC
    denies them all (e.g. another user's private image), the resolver
    must return ``resolved=False`` with ``rbac_denied`` mode so the
    caller can fall back to the grounded no-info response."""

    class _FakeDoc:
        id = 999
        original_name = "private.png"
        filename = "private.png"

    class _FakeImage:
        id = 7
        document_id = 999
        original_filename = "private.png"
        ocr_status = "success"
        created_at = None

    fake_db = MagicMock()
    fake_db.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
        (_FakeImage(), _FakeDoc())
    ]

    # RBAC filter denies everything for this user.
    import app.security.permissions as perms
    monkeypatch.setattr(perms, "filter_documents_by_permission", lambda *a, **kw: [])

    auth = MagicMock(is_authenticated=True)
    result = resolve_recent_image(fake_db, auth)
    assert result.resolved is False
    assert result.diagnostics["image_resolution_mode"] == "rbac_denied"


# ---------------------------------------------------------------------------
# Scenario 6: image-content answer uses only image OCR source
# ---------------------------------------------------------------------------


def test_image_content_only_image_chunks_pass_through_routing():
    """Scenario 6: the image_content source-relationship bypass should
    not pull in unrelated native-text chunks."""
    chunks = [
        _chunk("Error 902 OCR text", source_type="image_ocr", image_id=3),
        _chunk("Programmer's Guide section", source_type="native_text"),
        _chunk("admin_test.txt content", source_type="native_text"),
    ]
    a = analyze_query("What error code is shown in the image I uploaded?")
    out, meta = select_image_aware_chunks(chunks, a, image_context={"image_id": 3})
    assert meta["routing_mode"] in {"image_content_scoped", "image_content_scoped_explicit"}
    assert all(c["source_type"] == "image_ocr" for c in out)
    sources = {c["source_file_name"] for c in out}
    assert all("admin" not in s for s in sources)


# ---------------------------------------------------------------------------
# Scenarios 7 + 8: admin_test.txt / Programmer's Guide do NOT appear
# for image-content questions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "What error code is shown in the image I uploaded?",
    "What does the screenshot I uploaded say?",
    "Read the text from my image",
])
def test_image_content_questions_exclude_admin_and_programmers_guide(query):
    chunks = [
        _chunk("Error 902 OCR", source_type="image_ocr", image_id=3),
        _chunk("admin_test.txt secret data", source_type="native_text",
               source_file_name="admin_test.txt"),
        _chunk("Programmer's Guide API section", source_type="native_text",
               source_file_name="Programmers_Guide.md"),
        _chunk("unrelated KB content", source_type="native_text"),
    ]
    a = analyze_query(query)
    out, meta = select_image_aware_chunks(chunks, a, image_context={"image_id": 3})
    filenames = [c["source_file_name"] for c in out]
    assert "admin_test.txt" not in filenames
    assert not any("Programmer" in f or "programmers" in f for f in filenames)
    # And we should have the OCR chunk
    assert any("Error 902" in (c.get("content") or "") for c in out)


# ---------------------------------------------------------------------------
# Scenario 9: direct error lookup broadens to KB
# ---------------------------------------------------------------------------


def test_direct_error_lookup_broadens_to_kb():
    """Scenario 9: 'How do I troubleshoot error 902?' (no image
    reference) → image_aware_chunks passes through (no scoping)."""
    chunks = [
        _chunk("Error 902 troubleshooting page", source_type="native_text"),
        _chunk("some unrelated doc", source_type="native_text"),
        _chunk("Old OCR text", source_type="image_ocr", image_id=99),
    ]
    a = analyze_query("How do I troubleshoot error 902?")
    assert a.query_type == "error_lookup"
    out, meta = select_image_aware_chunks(chunks, a)
    assert meta["routing_mode"] == "passthrough"
    assert len(out) == 3  # everything kept


# ---------------------------------------------------------------------------
# Scenario 10: image-based troubleshooting broadens after OCR identification
# ---------------------------------------------------------------------------


def test_image_based_troubleshooting_keeps_all_chunks():
    """Scenario 10: 'How do I troubleshoot the error shown in the
    image?' uses image as identifier but broadens to KB."""
    chunks = [
        _chunk("Error 902 troubleshooting page", source_type="native_text"),
        _chunk("Error 902 - rejected-forbidden-country", source_type="image_ocr", image_id=42),
        _chunk("unrelated doc", source_type="native_text"),
    ]
    a = analyze_query("How do I troubleshoot the error shown in the image?")
    # error_lookup (with image reference) → broadens to KB.
    out, meta = select_image_aware_chunks(chunks, a, image_context={"image_id": 42})
    assert meta["routing_mode"] == "troubleshooting_with_image_context"
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Scenario 11: insufficient KB evidence returns grounded fallback
# ---------------------------------------------------------------------------


def test_no_evidence_returns_weak_evidence():
    """Scenario 11: when chunks lack support for the query and the
    source is NOT an explicitly-resolved image, decide_evidence_level
    returns WEAK with weak_evidence rationale."""
    chunks = [
        _chunk("apples and oranges", source_type="native_text"),
    ]
    # NB: NOT marked as resolved_image_content in retrieval_metadata.
    el, meta = decide_evidence_level(
        query="How do I troubleshoot error 902?",
        chunks=chunks,
        retrieval_metadata={"query_type": "error_lookup", "retrieval_mode": "hybrid_mmr"},
    )
    assert el.value == "weak"
    assert meta["decision"] == "weak"


def test_resolved_image_content_grants_strong_evidence_even_with_zero_score():
    """Scenario 1 (continued): decide_evidence_level must grant STRONG
    evidence for resolved image_content sources even when the OCR
    chunk has score=None (by-id lookup)."""
    chunks = [
        _chunk(
            "Failed Reason: 902 - Message delivery failed",
            source_type="image_ocr",
            score=None,
            image_id=3,
        )
    ]
    el, meta = decide_evidence_level(
        query="What error code is shown in the image I uploaded?",
        chunks=chunks,
        retrieval_metadata={
            "query_type": "image_content",
            "retrieval_mode": "image_content_recent",
            "image_routing": {"routing_mode": "image_content_recent"},
        },
    )
    assert el.value == "strong"
    assert meta["decision"] == "strong"
    assert "resolved_image_content_source" in meta.get("rationale", [])


# ---------------------------------------------------------------------------
# Scenario 12: normal text RAG remains unchanged
# ---------------------------------------------------------------------------


def test_normal_text_rag_passes_through_image_routing():
    """Scenario 12: a normal text RAG question (no image reference)
    passes through image_routing unchanged."""
    chunks = [
        _chunk("API documentation", source_type="native_text"),
        _chunk("setup guide", source_type="native_text"),
    ]
    a = analyze_query("How do I configure the alert system?")
    assert a.query_type == "general"
    assert a.references_uploaded_image is False
    out, meta = select_image_aware_chunks(chunks, a)
    assert meta["routing_mode"] == "passthrough"
    assert len(out) == 2


def test_normal_text_rag_grounding_unchanged():
    """Scenario 12 (continued): apply_grounding_checks with the standard
    path (no bypass) still runs topic relevance as before."""
    chunks = [
        _chunk("completely unrelated", source_type="native_text"),
    ]
    should_block, fb, meta = apply_grounding_checks(
        chunks=chunks,
        answer=None,
        threshold=0.10,
        require_citations=False,
        query="How do I troubleshoot error 902?",
    )
    # Topic relevance blocks because the chunk doesn't relate.
    assert should_block is True
    assert meta.get("blocked_reason") in {"topic_not_relevant", "topic_not_relevant_single_keyword"}


# ---------------------------------------------------------------------------
# Scenario 13: legacy OCR payload fallback still works
# ---------------------------------------------------------------------------


def test_legacy_ocr_payload_fallback_still_works():
    """Scenario 13: chunks indexed before Phase 34A.1.1 don't carry
    structured source_type; the legacy fallback in image_routing must
    still classify them correctly."""
    chunks = [
        _chunk(
            "Source Type: Image\nOCR Text:\nFailed Reason: 902",
            source_type=None,  # legacy chunk without structured metadata
            source_file_name="Screenshot 2026-08-05.png",
        )
    ]
    # is_image_source_chunk should classify this as image-derived
    from app.rag.image_routing import is_image_source_chunk
    assert is_image_source_chunk(chunks[0]) is True


# ---------------------------------------------------------------------------
# Scenario 14: Phase 34A OCR regression tests still green
# (Covered by the existing test_phase34a_ocr.py suite which we
#  re-run via the regression runner.)
# ---------------------------------------------------------------------------


def test_phase34a_ocr_legacy_chunks_classifiable():
    """Smoke test that the OCR classification chain still recognises
    legacy PDF OCR and DOCX image OCR chunks."""
    from app.rag.image_routing import is_image_source_chunk
    chunks = [
        {
            "chunk_id": "c1", "document_id": "1", "score": None,
            "content": "Source Type: PDF Image\nOCR Text:\nPage 3",
            "source_file_name": "doc.pdf",
            "title": "",
            "source_type": None,
        },
        {
            "chunk_id": "c2", "document_id": "2", "score": None,
            "content": "Source Type: DOCX Image\nOCR Text:\nFigure 2",
            "source_file_name": "report.docx",
            "title": "",
            "source_type": None,
        },
        {
            "chunk_id": "c3", "document_id": "3", "score": None,
            "content": "extracted text",
            "source_file_name": "scan.png",
            "title": "[Image OCR] extracted text",
            "source_type": None,
        },
    ]
    for c in chunks:
        assert is_image_source_chunk(c) is True, c


# ---------------------------------------------------------------------------
# Scenario 15: Phase 34A.1 / 34A.1.1 regression coverage
# (Covered by test_phase34a1_*.py suites. We add a smoke test that
#  the new module imports don't break the existing routing API.)
# ---------------------------------------------------------------------------


def test_existing_select_image_aware_chunks_signature_preserved():
    """Scenario 15: the existing Phase 34A.1.1 select_image_aware_chunks
    signature remains intact and continues to work for non-image_content
    queries."""
    chunks = [
        _chunk("Error 902 documentation", source_type="native_text"),
    ]
    a = analyze_query("How do I troubleshoot error 902?")
    out, meta = select_image_aware_chunks(chunks, a)
    assert meta["routing_mode"] == "passthrough"
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Citations: by-id chunks with score=None no longer crash
# ---------------------------------------------------------------------------


def test_citations_score_excerpt_handles_none_relevance():
    """By-id chunks have ``relevance_score=None`` (no semantic
    similarity score). The citations module must treat that as a
    high baseline rather than crashing with TypeError."""
    score = score_excerpt_relevance(
        excerpt="Failed Reason: 902 - rejected",
        question="What error code is shown in the image I uploaded?",
        answer="The error code is 902.",
        relevance_score=None,
    )
    assert isinstance(score, float)
    assert score >= 0.0


# ---------------------------------------------------------------------------
# Recent-image resolver: no candidates / unauthenticated
# ---------------------------------------------------------------------------


def test_resolve_recent_image_unauthenticated_returns_empty():
    auth = MagicMock(is_authenticated=False)
    result = resolve_recent_image(MagicMock(), auth)
    assert result.resolved is False
    assert result.diagnostics["image_resolution_mode"] == "skipped_unauthenticated"


def test_resolve_recent_image_no_ocr_candidates_returns_empty(monkeypatch):
    fake_db = MagicMock()
    fake_db.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
    auth = MagicMock(is_authenticated=True)
    result = resolve_recent_image(fake_db, auth)
    assert result.resolved is False
    assert result.diagnostics["image_resolution_mode"] == "no_ocr_candidates"

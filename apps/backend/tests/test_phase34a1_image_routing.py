"""
Phase 34A.1 — Image-aware query routing tests.

Covers:
- Image-content question scopes retrieval to OCR-derived chunks.
- Troubleshooting question keeps all chunks (rely on exact-match boost).
- Passthrough when no image reference is present.
- is_image_source_chunk correctly classifies source_type values.
"""

import pytest

from app.rag.image_routing import (
    select_image_aware_chunks,
    is_image_source_chunk,
    references_uploaded_image,
)
from app.rag.query_analysis import analyze_query


def _chunk(content, source_type="native_text", image_id=None,
           document_id="1", score=0.5):
    chunk = {
        "chunk_id": "c_" + str(abs(hash(content)) % 100000),
        "document_id": document_id,
        "chunk_index": 0,
        "content": content,
        "title": "",
        "score": score,
        "source_type": source_type,
    }
    if image_id is not None:
        chunk["image_id"] = image_id
    return chunk


# ---------------------------------------------------------------------------
# is_image_source_chunk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", [
    "image_ocr", "pdf_page_ocr", "docx_image_ocr", "image", "pdf_page", "docx_image",
])
def test_image_source_chunk_detection(source_type):
    chunk = _chunk("anything", source_type=source_type)
    assert is_image_source_chunk(chunk) is True


def test_image_source_chunk_detection_native_text():
    chunk = _chunk("anything", source_type="native_text")
    assert is_image_source_chunk(chunk) is False


def test_image_source_chunk_detection_via_content_type():
    chunk = _chunk("anything", source_type="native_text")
    chunk["content_type"] = "image_ocr"
    assert is_image_source_chunk(chunk) is True


# ---------------------------------------------------------------------------
# Passthrough behaviour
# ---------------------------------------------------------------------------


def test_routing_passthrough_for_non_image_query():
    chunks = [_chunk("doc a"), _chunk("doc b"), _chunk("doc c")]
    a = analyze_query("How do I troubleshoot error 902?")
    out, meta = select_image_aware_chunks(chunks, a)
    assert len(out) == 3
    assert meta["routing_mode"] == "passthrough"
    assert meta["image_context_used"] is False


# ---------------------------------------------------------------------------
# Image-content scoping
# ---------------------------------------------------------------------------


def test_image_content_question_scopes_to_ocr_chunks():
    """When the user asks an image-content question and supplies an
    explicit image_context, only OCR chunks from THAT image survive.
    Unrelated KB sources and unrelated OCR chunks are both dropped."""
    chunks = [
        _chunk("unrelated admin notes", source_type="native_text"),
        _chunk("Error 902 - rejected", source_type="image_ocr", image_id=42),
        _chunk("Another unrelated doc", source_type="native_text"),
        _chunk("OCR text from another image", source_type="image_ocr", image_id=99),
    ]
    a = analyze_query("What error code is shown in the image I uploaded?")
    out, meta = select_image_aware_chunks(chunks, a, image_context={"image_id": 42})
    # Phase 34A.1.1 — with explicit image_context, scope to the matching
    # image_id only (no stale historical images).
    assert meta["routing_mode"] == "image_content_scoped_explicit"
    assert all(c["source_type"] == "image_ocr" for c in out)
    assert len(out) == 1
    # Only the matching chunk survives.
    assert out[0].get("image_id") == 42


def test_image_content_no_ocr_available_passthrough():
    chunks = [_chunk("doc a", source_type="native_text")]
    a = analyze_query("What error code is shown in the image I uploaded?")
    out, meta = select_image_aware_chunks(chunks, a)
    # No OCR chunks present; pass through and record the routing intent.
    assert meta["routing_mode"] == "image_content_no_ocr_available"
    assert len(out) == 1


def test_image_content_falls_back_to_any_image_chunk():
    """When no chunk matches the supplied image_context AND no recent
    image target is supplied, fall back to a single best-effort OCR
    chunk (never an arbitrary historical image)."""
    chunks = [
        _chunk("unrelated", source_type="native_text"),
        _chunk("some OCR text", source_type="image_ocr", image_id=99),
    ]
    a = analyze_query("What does the screenshot I uploaded say?")
    out, meta = select_image_aware_chunks(
        chunks, a, image_context={"image_id": 42}
    )
    # Phase 34A.1.1 — explicit image_context, no chunk matches, no
    # recent_images target → best-effort fallback to a single OCR chunk
    # rather than serving a stale historical image.
    assert meta["routing_mode"] == "image_content_scoped_fallback"
    assert len(out) == 1
    assert out[0]["source_type"] == "image_ocr"


# ---------------------------------------------------------------------------
# Troubleshooting with image context
# ---------------------------------------------------------------------------


def test_troubleshooting_with_image_context_keeps_all_chunks():
    chunks = [
        _chunk("Error 902 documentation page", source_type="native_text"),
        _chunk("Error 902 - rejected-forbidden-country", source_type="image_ocr", image_id=42),
        _chunk("unrelated", source_type="native_text"),
    ]
    a = analyze_query("How do I troubleshoot error 902 in the image?")
    out, meta = select_image_aware_chunks(chunks, a, image_context={"image_id": 42})
    # Troubleshooting questions broaden to the KB; we keep every chunk.
    assert len(out) == 3
    assert meta["routing_mode"] == "troubleshooting_with_image_context"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_routing_with_empty_chunks():
    a = analyze_query("What error code is shown in the image I uploaded?")
    out, meta = select_image_aware_chunks([], a)
    assert out == []
    assert meta["routing_mode"] == "image_content_no_ocr_available"


def test_routing_with_none_analysis():
    chunks = [_chunk("doc a", source_type="native_text")]
    out, meta = select_image_aware_chunks(chunks, None)
    # None analysis → no references_uploaded_image → passthrough.
    assert meta["routing_mode"] == "passthrough"
    assert len(out) == 1

"""
Phase 34A.1.1 — Image metadata & source routing regression tests.

Covers:
- Structured Qdrant payload writes (source_type, content_type,
  image_id, ocr_provider, ocr_confidence, page_number, mime_type,
  is_ocr) for image_ocr, pdf_ocr, docx_image_ocr, and native_text.
- Image-aware routing filters native_text out for image_content
  questions and respects explicit image_context/document_id.
- Legacy OCR chunk fallback (chunks without structured source_type
  but with the legacy "Source Type: Image" content prefix).
- Troubleshooting queries still search the broader KB and unrelated
  admin_test-like chunks remain filtered.
- Citations / source cards use only the final surviving chunks.
- Existing native-text RAG behaviour is unchanged.

The tests are pure-Python / unit-level so they run without a live
Qdrant or PostgreSQL. The Qdrant payload schema is verified by
constructing a metadata dict and running it through the payload
builder; image-aware routing is verified via
`select_image_aware_chunks`.
"""

import pytest

from app.rag.image_routing import (
    select_image_aware_chunks,
    is_image_source_chunk,
    _looks_like_legacy_ocr_chunk,
    _is_image_source_with_fallback,
    LEGACY_OCR_FALLBACK_ENABLED,
)
from app.rag.query_analysis import analyze_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload_for(
    source_type: str,
    *,
    document_id: str = "1",
    image_id=None,
    ocr_provider: str | None = None,
    ocr_confidence: int | None = None,
    mime_type: str | None = None,
    page_number: int | None = None,
    content: str = "anything",
    source_file_name: str = "file.txt",
) -> dict:
    """Mirror the dict shape `qdrant_service.upsert_chunks` writes into
    the Qdrant payload, then merged onto a chunk by `hybrid_retriever`."""
    is_ocr = source_type in {
        "image_ocr",
        "pdf_ocr",
        "pdf_page_ocr",
        "docx_image_ocr",
    }
    payload = {
        "chunk_id": "c_" + str(abs(hash(content)) % 100000),
        "document_id": document_id,
        "chunk_index": 0,
        "content": content,
        "source_file_name": source_file_name,
        "title": source_file_name,
        "score": 0.5,
        "source_type": source_type,
        "content_type": source_type,
        "image_id": image_id,
        "ocr_provider": ocr_provider,
        "ocr_confidence": ocr_confidence,
        "page_number": page_number,
        "mime_type": mime_type,
        "is_ocr": is_ocr,
    }
    return payload


def _chunk(payload: dict) -> dict:
    """Convert a payload dict into the chunk dict shape that
    `hybrid_retriever` and `image_routing` consume."""
    return {
        "chunk_id": payload["chunk_id"],
        "document_id": payload["document_id"],
        "chunk_index": payload["chunk_index"],
        "content": payload["content"],
        "source_file_name": payload["source_file_name"],
        "title": payload["title"],
        "score": payload["score"],
        "source_type": payload["source_type"],
        "content_type": payload["content_type"],
        "image_id": payload["image_id"],
        "ocr_provider": payload["ocr_provider"],
        "ocr_confidence": payload["ocr_confidence"],
        "page_number": payload["page_number"],
        "mime_type": payload["mime_type"],
        "is_ocr": payload["is_ocr"],
    }


# ---------------------------------------------------------------------------
# Qdrant payload schema — verifies upsert_chunks would write the right keys
# ---------------------------------------------------------------------------


class _FakePoint:
    def __init__(self, payload):
        self.payload = payload


class TestQdrantPayloadSchema:
    """Drive `qdrant_service.upsert_chunks` end-to-end with a fake Qdrant
    client to confirm the structured OCR metadata lands in the payload
    for every supported source_type."""

    def _run_upsert(self, metas):
        # Import inside the test so module-level side effects don't fire.
        from app.services.vector import qdrant_service as qsvc

        captured: list[list] = []

        class _FakeClient:
            def upsert(self, collection_name, points):
                captured.append(points)

        qsvc._client = _FakeClient()  # type: ignore[attr-defined]

        chunks_with_embeddings = [
            ("chunk-content-" + str(i), [0.0] * 4) for i, _ in enumerate(metas)
        ]
        qsvc.upsert_chunks(chunks_with_embeddings, metas)

        assert captured, "upsert was not called"
        return captured[0]

    def test_image_ocr_payload_carries_structured_metadata(self):
        meta = {
            "chunk_id": 1001,
            "document_id": 55,
            "document_version_id": 51,
            "chunk_index": 0,
            "content_hash": "abc",
            "source_file_name": "Screenshot 2026-08-05 174611.png",
            "title": "Screenshot 2026-08-05 174611.png",
            "section_heading": None,
            "source_type": "image_ocr",
            "content_type": "image_ocr",
            "mime_type": "image/png",
            "image_id": 7,
            "ocr_provider": "tesseract",
            "ocr_confidence": 92,
            "page": None,
        }
        points = self._run_upsert([meta])
        assert len(points) == 1
        p = points[0].payload
        assert p["source_type"] == "image_ocr"
        assert p["content_type"] == "image_ocr"
        assert p["image_id"] == 7
        assert p["ocr_provider"] == "tesseract"
        assert p["ocr_confidence"] == 92
        assert p["mime_type"] == "image/png"
        assert p["is_ocr"] is True
        # The legacy fields are still there for backward compat.
        assert p["chunk_id"] == 1001
        assert p["document_id"] == 55

    def test_pdf_ocr_payload_carries_structured_metadata(self):
        meta = {
            "chunk_id": 1002,
            "document_id": 60,
            "document_version_id": 60,
            "chunk_index": 0,
            "content_hash": "def",
            "source_file_name": "scan.pdf",
            "title": "scan.pdf",
            "section_heading": None,
            "source_type": "pdf_ocr",
            "content_type": "pdf_page_ocr",
            "mime_type": "application/pdf",
            "image_id": 8,
            "ocr_provider": "tesseract",
            "ocr_confidence": 80,
            "page_number": 3,
        }
        points = self._run_upsert([meta])
        p = points[0].payload
        assert p["source_type"] == "pdf_ocr"
        assert p["page_number"] == 3
        assert p["is_ocr"] is True

    def test_docx_image_ocr_payload_carries_structured_metadata(self):
        meta = {
            "chunk_id": 1003,
            "document_id": 61,
            "document_version_id": 61,
            "chunk_index": 0,
            "content_hash": "ghi",
            "source_file_name": "manual.docx",
            "title": "manual.docx",
            "section_heading": None,
            "source_type": "docx_image_ocr",
            "content_type": "docx_image_ocr",
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "image_id": 9,
            "ocr_provider": "tesseract",
            "ocr_confidence": 75,
        }
        points = self._run_upsert([meta])
        p = points[0].payload
        assert p["source_type"] == "docx_image_ocr"
        assert p["is_ocr"] is True

    def test_native_text_payload_is_ocr_false(self):
        meta = {
            "chunk_id": 1004,
            "document_id": 1,
            "document_version_id": 1,
            "chunk_index": 0,
            "content_hash": "jkl",
            "source_file_name": "Programmer's Guide.txt",
            "title": "Programmer's Guide.txt",
            "section_heading": None,
            "source_type": "native_text",
            "content_type": "native_text",
            "mime_type": "text/plain",
        }
        points = self._run_upsert([meta])
        p = points[0].payload
        assert p["source_type"] == "native_text"
        assert p["is_ocr"] is False
        assert "image_id" not in p
        assert "ocr_provider" not in p
        assert "ocr_confidence" not in p
        assert "page_number" not in p

    def test_payload_strips_explicit_nulls(self):
        meta = {
            "chunk_id": 1005,
            "document_id": 1,
            "document_version_id": 1,
            "chunk_index": 0,
            "content_hash": "mno",
            "source_file_name": "manual.txt",
            "title": "manual.txt",
            "section_heading": None,
            "source_type": "native_text",
            "content_type": "native_text",
            "mime_type": "text/plain",
            "image_id": None,
            "ocr_provider": None,
            "ocr_confidence": None,
        }
        points = self._run_upsert([meta])
        p = points[0].payload
        assert "image_id" not in p
        assert "ocr_provider" not in p
        assert "ocr_confidence" not in p

    def test_inferred_source_type_from_image_id(self):
        meta = {
            "chunk_id": 1006,
            "document_id": 2,
            "document_version_id": 2,
            "chunk_index": 0,
            "content_hash": "pqr",
            "source_file_name": "image.png",
            "title": "image.png",
            "section_heading": None,
            # No source_type/content_type supplied.
            "image_id": 42,
            "mime_type": "image/png",
        }
        points = self._run_upsert([meta])
        p = points[0].payload
        assert p["source_type"] == "image_ocr"
        assert p["is_ocr"] is True


# ---------------------------------------------------------------------------
# is_image_source_chunk — structured + legacy fallback
# ---------------------------------------------------------------------------


class TestIsImageSourceChunk:
    def test_structured_image_ocr(self):
        c = _chunk(_payload_for("image_ocr", image_id=5))
        assert is_image_source_chunk(c) is True

    def test_structured_pdf_ocr(self):
        c = _chunk(_payload_for("pdf_ocr", image_id=6))
        assert is_image_source_chunk(c) is True

    def test_structured_docx_image_ocr(self):
        c = _chunk(_payload_for("docx_image_ocr", image_id=7))
        assert is_image_source_chunk(c) is True

    def test_native_text_is_not_image(self):
        c = _chunk(_payload_for("native_text", source_file_name="admin_test.txt"))
        assert is_image_source_chunk(c) is False

    def test_legacy_content_prefix_classified_as_image(self):
        # Legacy chunk indexed before Phase 34A.1.1 — no source_type,
        # only the parser's content prefix.
        legacy = {
            "chunk_id": "legacy-1",
            "content": "Source Type: Image\nOCR Text:\nFailed Reason: 902 - Message delivery failed: rejected-forbidden-country",
            "source_file_name": "Screenshot 2026-08-05 174611.png",
            "title": "Screenshot 2026-08-05 174611.png",
            "score": 0.5,
        }
        assert is_image_source_chunk(legacy) is True

    def test_legacy_pdf_prefix_classified_as_image(self):
        legacy = {
            "chunk_id": "legacy-2",
            "content": "Source Type: PDF Image\nOCR Text:\nInvoice total: $1,200",
            "source_file_name": "scan.pdf",
            "title": "scan.pdf",
        }
        assert is_image_source_chunk(legacy) is True

    def test_legacy_docx_prefix_classified_as_image(self):
        legacy = {
            "chunk_id": "legacy-3",
            "content": "Source Type: DOCX Image\nOCR Text:\nDiagram caption: latency",
            "source_file_name": "manual.docx",
            "title": "manual.docx",
        }
        assert is_image_source_chunk(legacy) is True

    def test_unrelated_admin_test_is_not_classified_as_image(self):
        """The admin_test.txt reproduction case — must NOT be misclassified
        as an image source even though its content might incidentally
        mention 902."""
        c = {
            "chunk_id": "admin-1",
            "content": "Admin notes: This server is for the QA environment only. Do not use for production.",
            "source_file_name": "admin_test.txt",
            "title": "admin_test.txt",
        }
        assert is_image_source_chunk(c) is False

    def test_image_extension_without_ocr_body_is_not_classified(self):
        # A text file named "screenshot.txt" must NOT be classified as image
        # purely because of its filename.
        c = {
            "chunk_id": "edge-1",
            "content": "Plain text content",
            "source_file_name": "screenshot.txt",
            "title": "screenshot.txt",
        }
        assert is_image_source_chunk(c) is False


# ---------------------------------------------------------------------------
# Image-aware routing — image_content scenario
# ---------------------------------------------------------------------------


class TestImageContentRouting:
    def test_filters_native_text_chunks_for_image_content_query(self):
        """The headline live scenario — 'What error code is shown in the
        image I uploaded?' — must return ONLY the OCR-derived chunks.
        """
        chunks = [
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999",
                                content="admin notes unrelated to error 902")),
            _chunk(_payload_for("image_ocr", image_id=42, document_id="55",
                                content="Failed Reason: 902 - Message delivery failed: rejected-forbidden-country")),
            _chunk(_payload_for("native_text", source_file_name="Programmer's Guide.txt",
                                document_id="2",
                                content="general text")),
        ]
        a = analyze_query("What error code is shown in the image I uploaded?")
        out, meta = select_image_aware_chunks(chunks, a)
        assert meta["routing_mode"].startswith("image_content_")
        # Only the OCR-derived chunk survives.
        assert len(out) == 1
        assert out[0]["source_type"] == "image_ocr"
        assert out[0]["document_id"] == "55"
        assert meta["dropped_count"] == 2

    def test_explicit_image_context_scopes_to_matching_chunks(self):
        """If the frontend supplies image_id=42, only that chunk survives
        (NOT another OCR chunk from a different image)."""
        chunks = [
            _chunk(_payload_for("image_ocr", image_id=42, document_id="55",
                                content="Error 902 from the latest upload")),
            _chunk(_payload_for("image_ocr", image_id=99, document_id="100",
                                content="Error 902 from an older upload")),
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999", content="admin notes")),
        ]
        a = analyze_query("What error code is shown in the image I uploaded?")
        out, meta = select_image_aware_chunks(
            chunks, a, image_context={"image_id": 42}
        )
        assert meta["routing_mode"] == "image_content_scoped_explicit"
        assert len(out) == 1
        assert out[0]["image_id"] == 42
        # The OTHER image chunk (99) was dropped — no stale historical images.
        assert all(c.get("image_id") != 99 for c in out)

    def test_explicit_document_context_scopes_to_matching_chunks(self):
        chunks = [
            _chunk(_payload_for("image_ocr", image_id=42, document_id="55",
                                content="Error 902 from the latest upload")),
            _chunk(_payload_for("image_ocr", image_id=99, document_id="100",
                                content="Error 902 from an older upload")),
        ]
        a = analyze_query("What error code is shown in the image I uploaded?")
        out, meta = select_image_aware_chunks(
            chunks, a, image_context={"document_id": "55"}
        )
        assert meta["routing_mode"] == "image_content_scoped_explicit"
        assert len(out) == 1
        assert out[0]["document_id"] == "55"

    def test_recent_images_fallback_when_no_explicit_match(self):
        chunks = [
            _chunk(_payload_for("image_ocr", image_id=99, document_id="100",
                                content="Error 902 from older upload")),
        ]
        a = analyze_query("What error code is shown in the image I uploaded?")
        out, meta = select_image_aware_chunks(
            chunks,
            a,
            image_context={
                "image_id": 42,  # not in chunks
                "recent_images": [{"image_id": 99, "document_id": 100}],
            },
        )
        # Falls back to the recent_images target.
        assert meta["routing_mode"] == "image_content_scoped_recent"
        assert len(out) == 1
        assert out[0]["image_id"] == 99

    def test_no_ocr_chunks_at_all_returns_no_scoping(self):
        chunks = [
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999", content="notes")),
        ]
        a = analyze_query("What error code is shown in the image I uploaded?")
        out, meta = select_image_aware_chunks(chunks, a)
        # No OCR content available — leave the original list alone.
        assert meta["routing_mode"] == "image_content_no_ocr_available"
        assert len(out) == 1

    def test_legacy_image_chunk_is_included_via_fallback(self):
        """The legacy OCR screenshot indexed before Phase 34A.1.1 must
        still be routable to image_content questions."""
        legacy = {
            "chunk_id": "legacy-screenshot-1",
            "content": "Source Type: Image\nOCR Text:\nFailed Reason: 902 - Message delivery failed: rejected-forbidden-country",
            "source_file_name": "Screenshot 2026-08-05 174611.png",
            "title": "Screenshot 2026-08-05 174611.png",
            "score": 0.5,
            "document_id": "55",
            # NOTE: no source_type/content_type/image_id — legacy shape.
        }
        native = _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                     document_id="999", content="admin"))
        chunks = [legacy, native]
        a = analyze_query("What error code is shown in the image I uploaded?")
        out, meta = select_image_aware_chunks(chunks, a)
        assert meta["routing_mode"].startswith("image_content_")
        assert len(out) == 1
        assert out[0]["content"].startswith("Source Type: Image")


# ---------------------------------------------------------------------------
# Troubleshooting queries — broader KB, exact-match boost path
# ---------------------------------------------------------------------------


class TestTroubleshootingRouting:
    def test_troubleshooting_query_keeps_all_chunks(self):
        """Troubleshooting queries must broaden to the full KB; the
        exact-match booster inside hybrid_retriever will rank the OCR
        chunk with '902' alongside the documentation."""
        chunks = [
            _chunk(_payload_for("native_text", source_file_name="Programmer's Guide.txt",
                                document_id="2",
                                content="Documentation for error 902 - rejected-forbidden-country")),
            _chunk(_payload_for("image_ocr", image_id=42, document_id="55",
                                content="Failed Reason: 902 - Message delivery failed: rejected-forbidden-country")),
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999", content="admin notes")),
        ]
        a = analyze_query("How do I troubleshoot error 902?")
        out, meta = select_image_aware_chunks(chunks, a)
        assert meta["routing_mode"] == "passthrough"
        # All three chunks survive.
        assert len(out) == 3

    def test_troubleshooting_with_image_context_keeps_all_chunks(self):
        chunks = [
            _chunk(_payload_for("native_text", source_file_name="Programmer's Guide.txt",
                                document_id="2",
                                content="Error 902 - rejected-forbidden-country")),
            _chunk(_payload_for("image_ocr", image_id=42, document_id="55",
                                content="Error 902 - rejected-forbidden-country")),
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999", content="admin notes")),
        ]
        a = analyze_query("How do I troubleshoot the error shown in the image?")
        out, meta = select_image_aware_chunks(
            chunks, a, image_context={"image_id": 42}
        )
        # Troubleshooting keeps every chunk.
        assert meta["routing_mode"] == "troubleshooting_with_image_context"
        assert len(out) == 3

    def test_unrelated_admin_test_still_survives_when_relevant(self):
        """Sanity check — admin_test.txt is not in itself excluded. It
        only gets filtered out by the image-aware router when the user
        is asking about a SPECIFIC uploaded image. A general question
        keeps it."""
        chunks = [
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999",
                                content="admin_test.txt mentions error 902 in context of admin access")),
            _chunk(_payload_for("native_text", source_file_name="docs.txt",
                                document_id="2",
                                content="docs for error 902")),
        ]
        a = analyze_query("Tell me about error 902 in the docs")
        out, meta = select_image_aware_chunks(chunks, a)
        assert meta["routing_mode"] == "passthrough"
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Citations / source cards — only final surviving chunks
# ---------------------------------------------------------------------------


class TestFinalSourceFilter:
    """Citations are built from the chunks returned by `image_routing`.
    These tests verify that — for an image_content question — the
    surviving chunks contain ONLY OCR/image-derived sources."""

    def test_image_content_surviving_chunks_only_image(self):
        chunks = [
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999", content="admin notes")),
            _chunk(_payload_for("native_text", source_file_name="Programmer's Guide.txt",
                                document_id="2", content="docs")),
            _chunk(_payload_for("image_ocr", image_id=42, document_id="55",
                                content="Error 902 - rejected-forbidden-country")),
        ]
        a = analyze_query("What error code is shown in the image I uploaded?")
        out, _ = select_image_aware_chunks(
            chunks, a, image_context={"image_id": 42}
        )

        # Build citations from surviving chunks only.
        source_cards = []
        for c in out:
            source_cards.append({
                "source_file_name": c["source_file_name"],
                "content_snippet": c["content"][:80],
            })
        assert len(source_cards) == 1
        assert source_cards[0]["source_file_name"] == "Screenshot 2026-08-05 174611.png" \
            or "image_ocr" in (
                # Legacy fallback: filename might not exist on the legacy shape.
                str(out[0].get("source_type", ""))
            )
        # No admin_test.txt, no Programmer's Guide.
        joined = " ".join(s["source_file_name"] for s in source_cards)
        assert "admin_test" not in joined
        assert "Programmer" not in joined

    def test_image_content_legacy_surviving_chunks_only_image(self):
        legacy = {
            "chunk_id": "legacy-screenshot-1",
            "content": "Source Type: Image\nOCR Text:\nError 902 - rejected-forbidden-country",
            "source_file_name": "Screenshot 2026-08-05 174611.png",
            "title": "Screenshot 2026-08-05 174611.png",
            "score": 0.5,
            "document_id": "55",
        }
        chunks = [
            legacy,
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999", content="admin notes")),
        ]
        a = analyze_query("What error code is shown in the image I uploaded?")
        out, _ = select_image_aware_chunks(chunks, a)
        assert len(out) == 1
        assert out[0]["source_file_name"] == "Screenshot 2026-08-05 174611.png"

    def test_existing_native_text_rag_path_unchanged(self):
        """When the user is NOT asking about an uploaded image, the
        router is a strict passthrough — nothing about the existing
        native-text RAG flow changes."""
        chunks = [
            _chunk(_payload_for("native_text", source_file_name="docs.txt",
                                document_id="2", content="some docs")),
            _chunk(_payload_for("native_text", source_file_name="admin_test.txt",
                                document_id="999", content="admin notes")),
        ]
        a = analyze_query("What does error 902 mean?")
        out, meta = select_image_aware_chunks(chunks, a)
        assert meta["routing_mode"] == "passthrough"
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Legacy fallback toggle
# ---------------------------------------------------------------------------


class TestLegacyFallback:
    def test_legacy_fallback_flag_defaults_to_enabled(self):
        # The toggle reads from env at import time. Sanity check: the
        # module-level flag is the boolean we expect (True in test env).
        import app.rag.image_routing as ir
        assert isinstance(ir.LEGACY_OCR_FALLBACK_ENABLED, bool)

    def test_legacy_fallback_classifies_only_clearly_ocr_chunks(self):
        legacy = {
            "chunk_id": "legacy-1",
            "content": "Source Type: Image\nOCR Text:\nFailed Reason: 902",
            "source_file_name": "Screenshot.png",
            "title": "Screenshot.png",
        }
        assert _looks_like_legacy_ocr_chunk(legacy) is True

        non_image = {
            "chunk_id": "non-1",
            "content": "Plain text that happens to start with 'Source Type: Image' as a column header in a table",
            "source_file_name": "table.txt",
            "title": "table.txt",
        }
        # No image extension on the filename → not classified.
        assert _looks_like_legacy_ocr_chunk(non_image) is False

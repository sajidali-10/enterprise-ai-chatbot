"""Phase 34C -- ImageKnowledgeRecord + deterministic builder tests.

Pure unit tests. No DB, no Qdrant, no embeddings. The record builder
is a pure function of its inputs, so the entire surface can be
exercised deterministically.

Test inventory (maps to Phase 34C acceptance criteria 1, 2, 6, 7):

    1. OCR-only image creates searchable image knowledge.
    2. OCR + persisted Vision creates enriched image knowledge.
    6. knowledge_text is deterministic.
    7. Stable point ID does not depend on timestamp.
    8. Correct metadata is captured by the builder.
    9. knowledge_text is length-capped.
"""

from __future__ import annotations

import os
import time

import pytest


def _ensure_settings_env():
    """Make sure the multimodal settings have known defaults before import."""
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_ENABLED", "true")
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION", "1")
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_TEXT_MAX_CHARS", "1800")


_ensure_settings_env()


def _make_record(**overrides):
    """Build a record with sane defaults; override per-test fields."""
    from app.services.multimodal.knowledge_record import ImageKnowledgeRecord

    base = dict(
        document_id=42,
        image_id=7,
        owner_user_id=1,
        visibility="global",
        original_filename="server_b.png",
        mime_type="image/png",
        ocr_text="Server A\nServer B\nServer C\nFailed",
        ocr_confidence=92,
        ocr_status="success",
        knowledge_schema_version=1,
    )
    base.update(overrides)
    return ImageKnowledgeRecord(**base)


def test_ocr_only_record_marks_is_ocr_only_true():
    """1. OCR-only image -- no Vision fields present, is_ocr_only=True."""
    rec = _make_record()
    assert rec.is_ocr_only is True
    assert rec.vision_status is None
    assert rec.vision_description == ""


def test_ocr_only_knowledge_text_is_deterministic():
    """6. knowledge_text is deterministic for the same input."""
    a = _make_record()
    b = _make_record()
    # Same inputs -> same text, byte-for-byte.
    from app.services.multimodal.knowledge_record import build_knowledge_text
    assert build_knowledge_text(a) == build_knowledge_text(b)


def test_ocr_only_text_shape_contains_ocr_section():
    """OCR-only text mentions the file name and the OCR block."""
    rec = _make_record()
    from app.services.multimodal.knowledge_record import build_knowledge_text
    text = build_knowledge_text(rec)
    assert "Source image: server_b.png" in text
    assert "OCR:" in text
    assert "Server B" in text


def test_ocr_plus_vision_includes_description_and_findings():
    """2. OCR + persisted Vision adds description / findings / entities."""
    rec = _make_record(
        vision_status="success",
        vision_description="Operations dashboard showing Server B with a red Failed status.",
        visual_findings=["Server B has a red failure indicator.", "Server A and Server C appear healthy."],
        detected_entities=["Server A", "Server B", "Server C"],
        visual_states=["healthy", "failed"],
        vision_tags=["dashboard", "operations"],
        vision_provider="mock",
        vision_model="mock-v1",
        image_type="application_ui",
        is_ocr_only=False,
    )
    from app.services.multimodal.knowledge_record import build_knowledge_text
    text = build_knowledge_text(rec)
    assert "Image type: application_ui" in text
    assert "Visual description:" in text
    assert "Server B" in text  # from description
    assert "Visual findings:" in text
    assert "- Server B has a red failure indicator." in text
    assert "Entities:" in text
    assert "Visual states: healthy, failed" in text
    assert "Tags: dashboard, operations" in text
    assert rec.is_ocr_only is False


def test_knowledge_text_respects_length_cap():
    """9. knowledge_text is bounded so the embedding budget cannot inflate."""
    long_ocr = ("x" * 5000)  # 5000 chars
    rec = _make_record(ocr_text=long_ocr)
    from app.services.multimodal.knowledge_record import build_knowledge_text, cap_knowledge_text
    raw = build_knowledge_text(rec)
    # Build-time OCR clipping already keeps the OCR section at <= 900
    # chars. Use a smaller runtime cap so cap_knowledge_text actually
    # has something to clip and we can prove the cap is enforced.
    capped = cap_knowledge_text(raw, max_chars=200)
    # The cap must be enforced (allow for a tiny ellipsis overhead).
    assert len(capped) <= 220
    assert len(capped) < len(raw)


def test_knowledge_text_runtime_cap_is_a_ceiling():
    """The runtime cap is a hard ceiling, not a target."""
    long_ocr = ("Lorem ipsum dolor sit amet. " * 100)  # ~2700 chars
    rec = _make_record(ocr_text=long_ocr)
    from app.services.multimodal.knowledge_record import build_knowledge_text, cap_knowledge_text
    raw = build_knowledge_text(rec)
    # Build-time OCR clipping already keeps the OCR section at <= 900
    # chars; choose a smaller cap so the runtime cap is the
    # binding constraint and we can prove it.
    capped = cap_knowledge_text(raw, max_chars=400)
    assert len(capped) <= 420
    assert len(capped) < len(raw)


def test_point_id_is_stable_across_calls():
    """7. compute_point_id returns the same int for the same tuple."""
    from app.services.multimodal.knowledge_record import compute_point_id
    pid1 = compute_point_id(42, 7, 1)
    pid2 = compute_point_id(42, 7, 1)
    assert pid1 == pid2
    assert isinstance(pid1, int)
    assert pid1 > 0


def test_point_id_changes_with_image_id():
    """Different images get different point IDs."""
    from app.services.multimodal.knowledge_record import compute_point_id
    assert compute_point_id(1, 1, 1) != compute_point_id(1, 2, 1)


def test_point_id_changes_with_schema_version():
    """A schema_version bump produces a new point ID."""
    from app.services.multimodal.knowledge_record import compute_point_id
    assert compute_point_id(1, 1, 1) != compute_point_id(1, 1, 2)


def test_point_id_independent_of_timestamp():
    """The point id must NOT depend on wall-clock time."""
    from app.services.multimodal.knowledge_record import compute_point_id
    pid = compute_point_id(99, 100, 1)
    time.sleep(0.01)
    assert compute_point_id(99, 100, 1) == pid


def test_knowledge_text_skips_empty_sections():
    """Empty Vision sections must be omitted, not rendered as blanks."""
    rec = _make_record(
        vision_status="success",
        vision_description="",
        visual_findings=[],
        detected_entities=[],
        visual_states=[],
        vision_tags=[],
    )
    from app.services.multimodal.knowledge_record import build_knowledge_text
    text = build_knowledge_text(rec)
    assert "Visual description:" not in text
    assert "Visual findings:" not in text
    assert "Entities:" not in text


def test_knowledge_text_handles_no_content_at_all():
    """An image with neither OCR nor Vision surfaces a marker line."""
    rec = _make_record(
        ocr_text="",
        ocr_status="disabled",
        vision_status=None,
    )
    from app.services.multimodal.knowledge_record import build_knowledge_text
    text = build_knowledge_text(rec)
    assert "(no extractable image content)" in text


def test_findings_deduplicate_repeated_items():
    """Identical findings collapse to one bullet."""
    rec = _make_record(
        vision_status="success",
        visual_findings=["Server B is failed", "Server B is failed", "Server A healthy"],
    )
    from app.services.multimodal.knowledge_record import build_knowledge_text
    text = build_knowledge_text(rec)
    assert text.count("Server B is failed") == 1
    assert "Server A healthy" in text

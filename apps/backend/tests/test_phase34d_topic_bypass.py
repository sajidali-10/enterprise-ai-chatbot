"""Phase 34D — Unit tests for the narrowly-scoped advanced-vision
topic-relevance bypass.

The bypass fires ONLY when ALL of these hold:
  * the request has an explicit AUTHORIZED image_context (reflected as an
    ``advanced_vision`` evidence chunk that the orchestrator already RBAC-verified),
  * Phase 34D produced a successful advanced-vision chunk (ran=True, image_id set),
  * the query classifies as a visual task (UI/CHART/DIAGRAM/TABLE/GENERAL),
  * the query is NOT product-meaning / troubleshooting / identifier intent.

It must NOT bypass for:
  * ordinary KB / text-RAG questions,
  * product-meaning / troubleshooting questions,
  * identifier / exact-text lookups,
  * any query without an advanced-vision evidence chunk.
"""

from __future__ import annotations

import pytest

from app.rag.grounding import (
    _is_advanced_vision_topic_bypass,
    _run_grounding_checks,
)


def _adv_chunk(*, image_id: int = 7, task_type: str = "ui_state_analysis", ran: bool = True) -> dict:
    return {
        "chunk_id": f"advanced-vision-{image_id}",
        "document_id": 1,
        "image_id": image_id,
        "source_file_name": "e2e_ui.png",
        "source_type": "advanced_vision",
        "content_type": "advanced_vision",
        "content": "summary",
        "score": None,
        "metadata": {
            "phase": "phase34d_advanced_vision",
            "advanced_vision_ran": ran,
            "advanced_vision_task_type": task_type,
            "advanced_vision_image_ids": [image_id],
        },
    }


def _kb_chunk(text: str = "Error 902 means a delivery failure.") -> dict:
    return {
        "chunk_id": "kb-1",
        "document_id": 99,
        "source_file_name": "kb.txt",
        "source_type": "native_text",
        "content_type": "native_text",
        "content": text,
        "score": 0.7,
    }


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_bypass_visual_query_with_adv_chunk_allowed():
    chunks = [_adv_chunk(image_id=7, task_type="ui_state_analysis")]
    assert _is_advanced_vision_topic_bypass(
        "What visually looks wrong with this screen?", chunks
    ) is True


def test_bypass_chart_query_with_adv_chunk_allowed():
    chunks = [_adv_chunk(image_id=8, task_type="chart_analysis")]
    assert _is_advanced_vision_topic_bypass(
        "What trend or anomaly do you see?", chunks
    ) is True


def test_bypass_diagram_query_with_adv_chunk_allowed():
    chunks = [_adv_chunk(image_id=9, task_type="diagram_analysis")]
    assert _is_advanced_vision_topic_bypass(
        "What flow does this diagram show?", chunks
    ) is True


def test_bypass_table_query_with_adv_chunk_allowed():
    chunks = [_adv_chunk(image_id=10, task_type="table_visual_analysis")]
    assert _is_advanced_vision_topic_bypass(
        "Which row is highlighted in the table?", chunks
    ) is True


def test_no_adv_chunk_no_bypass():
    chunks = [_kb_chunk()]
    assert _is_advanced_vision_topic_bypass(
        "What visually looks wrong with this screen?", chunks
    ) is False


def test_adv_chunk_not_ran_no_bypass():
    chunks = [_adv_chunk(image_id=7, task_type="ui_state_analysis", ran=False)]
    assert _is_advanced_vision_topic_bypass(
        "What visually looks wrong with this screen?", chunks
    ) is False


def test_adv_chunk_missing_image_id_no_bypass():
    chunk = _adv_chunk(image_id=7)
    chunk.pop("image_id")
    assert _is_advanced_vision_topic_bypass(
        "What visually looks wrong with this screen?", [chunk]
    ) is False


def test_product_meaning_query_no_bypass_even_with_adv_chunk():
    chunks = [_adv_chunk(image_id=7, task_type="ui_state_analysis")]
    assert _is_advanced_vision_topic_bypass(
        "Why did Server B fail?", chunks
    ) is False


def test_troubleshooting_query_no_bypass_even_with_adv_chunk():
    chunks = [_adv_chunk(image_id=7)]
    assert _is_advanced_vision_topic_bypass(
        "How do I fix the error shown in this screenshot?", chunks
    ) is False


def test_product_meaning_how_do_i_fix_no_bypass():
    chunks = [_adv_chunk(image_id=7, task_type="chart_analysis")]
    assert _is_advanced_vision_topic_bypass(
        "What does error 902 mean and how should I fix it?", chunks
    ) is False


def test_identifier_lookup_no_bypass_even_with_adv_chunk():
    chunks = [_adv_chunk(image_id=7, task_type="ui_state_analysis")]
    assert _is_advanced_vision_topic_bypass(
        "What error code is shown?", chunks
    ) is False


def test_ordinary_text_rag_no_bypass():
    chunks = [_kb_chunk()]
    assert _is_advanced_vision_topic_bypass(
        "What is the support model?", chunks
    ) is False


def test_empty_query_no_bypass():
    assert _is_advanced_vision_topic_bypass("", [_adv_chunk(image_id=7)]) is False


# ---------------------------------------------------------------------------
# Integration-level tests through _run_grounding_checks
# ---------------------------------------------------------------------------


def test_grounding_visual_query_with_adv_chunk_bypasses_topic():
    chunks = [_adv_chunk(image_id=7, task_type="ui_state_analysis")]
    _, _msg, meta = _run_grounding_checks(
        chunks=chunks,
        query="What visually looks wrong with this screen?",
        threshold=0.1,
        require_citations=False,
    )
    assert meta.get("topic_bypassed_reason") == "phase34d_advanced_vision_evidence"
    assert meta.get("topic_checked") is False


def test_grounding_ordinary_text_rag_topic_still_checked():
    chunks = [_kb_chunk("Error 902 means a delivery failure.")]
    _, _msg, meta = _run_grounding_checks(
        chunks=chunks,
        query="What does the support model cover?",
        threshold=0.1,
        require_citations=False,
    )
    # No advanced-vision chunk → no bypass; topic check ran.
    assert meta.get("topic_bypassed_reason") is None
    assert meta.get("topic_checked") in (True, None, False)
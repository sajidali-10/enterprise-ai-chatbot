"""
Phase 34A.1 — Minimum-relevance filtering tests.

Covers:
- Threshold drops chunks below min_score.
- Cap truncates to max_results.
- All-results / one-result / zero-result scenarios.
- Unrelated low-score content is excluded.
- Exact-match chunk survives the threshold.
"""

import pytest

from app.rag.relevance_filter import apply_relevance_threshold


def _chunk(content, score, source_type="native_text"):
    return {
        "chunk_id": "c_" + str(abs(hash(content)) % 100000),
        "document_id": "1",
        "chunk_index": 0,
        "content": content,
        "title": "",
        "score": score,
        "source_type": source_type,
    }


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


def test_threshold_drops_below_min_score():
    chunks = [
        _chunk("high relevance", 0.85),
        _chunk("low relevance noise", 0.05),
        _chunk("medium relevance", 0.30),
    ]
    out, meta = apply_relevance_threshold(chunks, min_score=0.20, max_results=10)
    assert len(out) == 2
    assert meta["threshold_dropped"] == 1
    assert meta["final_source_count"] == 2
    # Sorted by score descending
    assert out[0]["score"] == 0.85
    assert out[1]["score"] == 0.30


def test_threshold_zero_keeps_all_but_still_caps():
    chunks = [_chunk(f"chunk{i}", 0.5 - i * 0.05) for i in range(5)]
    out, meta = apply_relevance_threshold(chunks, min_score=0.0, max_results=3)
    assert len(out) == 3
    assert meta["threshold_dropped"] == 0
    assert meta["cap_dropped"] == 2


def test_cap_only_no_threshold():
    chunks = [_chunk(f"chunk{i}", float(10 - i)) for i in range(10)]
    out, meta = apply_relevance_threshold(chunks, min_score=0.0, max_results=4)
    assert len(out) == 4
    assert meta["cap_dropped"] == 6


# ---------------------------------------------------------------------------
# Scenarios required by the spec
# ---------------------------------------------------------------------------


def test_scenario_all_results_relevant():
    chunks = [
        _chunk("chunk a", 0.90),
        _chunk("chunk b", 0.75),
        _chunk("chunk c", 0.55),
    ]
    out, meta = apply_relevance_threshold(chunks, min_score=0.30, max_results=5)
    assert len(out) == 3
    assert meta["filtered_count"] == 0


def test_scenario_one_result_relevant():
    chunks = [
        _chunk("relevant doc about 902", 0.80),
        _chunk("admin_test.txt unrelated", 0.05),
        _chunk("another unrelated doc", 0.10),
    ]
    out, meta = apply_relevance_threshold(chunks, min_score=0.30, max_results=5)
    assert len(out) == 1
    assert "902" in out[0]["content"]


def test_scenario_no_results_relevant():
    chunks = [
        _chunk("admin_test.txt unrelated", 0.05),
        _chunk("another unrelated doc", 0.10),
    ]
    out, meta = apply_relevance_threshold(chunks, min_score=0.30, max_results=5)
    assert out == []
    assert meta["final_source_count"] == 0


def test_scenario_exact_match_survives_threshold():
    """A chunk with a strong exact-match boost must not be dropped by
    the relevance threshold."""
    from app.rag.exact_match import apply_exact_match_boost
    from app.rag.query_analysis import analyze_query

    chunks = [
        _chunk("unrelated noise", 0.40),
        _chunk("totally unrelated admin stuff", 0.40),
        # The '902' chunk starts equal to the noise but is boosted
        # above it by exact-match (0.40 * 1.30 = 0.52).
        _chunk("Error 902 - message delivery failed", 0.40),
    ]
    a = analyze_query("How do I troubleshoot error 902?")
    boosted = apply_exact_match_boost(chunks, a)
    out, meta = apply_relevance_threshold(boosted, min_score=0.30, max_results=5)
    # The 902 chunk should now rank on top because of the boost.
    assert any("902" in c["content"] for c in out)
    assert out[0]["content"].startswith("Error 902")


def test_scenario_unrelated_admin_test_excluded():
    chunks = [
        _chunk("admin_test.txt internal scratchpad", 0.05),
        _chunk("Error 902 - rejected-forbidden-country", 0.10),
        _chunk("Other admin scratch notes", 0.07),
    ]
    out, meta = apply_relevance_threshold(chunks, min_score=0.30, max_results=5)
    # The admin_test chunk is dropped by the threshold.
    for c in out:
        assert "admin_test" not in c["content"]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_records_filtering_actions():
    chunks = [
        _chunk("high", 0.9),
        _chunk("medium", 0.5),
        _chunk("low", 0.1),
    ]
    out, meta = apply_relevance_threshold(chunks, min_score=0.30, max_results=1)
    assert meta["candidate_count"] == 3
    assert meta["threshold_dropped"] == 1
    assert meta["cap_dropped"] == 1
    assert meta["filtered_count"] == 2
    assert meta["final_source_count"] == 1


def test_empty_input_safe():
    out, meta = apply_relevance_threshold([], min_score=0.30, max_results=5)
    assert out == []
    assert meta["candidate_count"] == 0
    assert meta["final_source_count"] == 0


def test_chunk_without_score_treated_as_zero():
    chunks = [
        _chunk("has score", 0.5),
        {"chunk_id": "x", "document_id": "1", "content": "no score", "score": None},
    ]
    out, meta = apply_relevance_threshold(chunks, min_score=0.30, max_results=5)
    assert len(out) == 1
    assert meta["threshold_dropped"] == 1


def test_max_results_none_disables_cap():
    chunks = [_chunk(f"chunk{i}", 0.5 - i * 0.01) for i in range(20)]
    out, meta = apply_relevance_threshold(chunks, min_score=0.0, max_results=None)
    assert len(out) == 20
    assert meta["cap_dropped"] == 0

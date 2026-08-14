"""
Phase 34A.1 — Exact-match boosting tests.

Covers:
- Exact error-code boost raises chunk score above a higher-vector-score
  non-matching chunk.
- Exact technical-term boost.
- Pure numeric codes use word-boundary matching (no substring false-positives).
- No identifiers → no boost (no-op).
"""

import pytest

from app.rag.exact_match import (
    apply_exact_match_boost,
    compute_exact_match_score,
)
from app.rag.query_analysis import analyze_query


def _chunk(content, title="", score=0.5, source_type="native_text"):
    return {
        "chunk_id": "c_" + str(abs(hash(content)) % 100000),
        "document_id": "1",
        "chunk_index": 0,
        "content": content,
        "title": title,
        "score": score,
        "source_type": source_type,
    }


# ---------------------------------------------------------------------------
# Score function tests
# ---------------------------------------------------------------------------


def test_exact_match_score_no_analysis():
    chunk = _chunk("Anything containing 902 in the OCR text")
    result = compute_exact_match_score(chunk, None)
    assert result["score"] == 0.0
    assert result["matched_codes"] == []


def test_exact_match_score_empty_analysis():
    from app.rag.query_analysis import QueryAnalysis
    chunk = _chunk("Anything containing 902 in the OCR text")
    a = QueryAnalysis(query="hello")
    result = compute_exact_match_score(chunk, a)
    assert result["score"] == 0.0


def test_exact_match_hits_error_code():
    chunk = _chunk("Error 902 - message delivery failed")
    a = analyze_query("How do I troubleshoot error 902?")
    result = compute_exact_match_score(chunk, a)
    assert "902" in result["matched_codes"]
    assert result["score"] > 0


def test_exact_match_hits_ora_code():
    chunk = _chunk("ORA-12541: TNS:no listener")
    a = analyze_query("ORA-12541 please help")
    result = compute_exact_match_score(chunk, a)
    assert "ORA-12541" in result["matched_codes"]


def test_exact_match_hits_http_code():
    chunk = _chunk("server returned HTTP 500")
    a = analyze_query("HTTP 500 server error")
    result = compute_exact_match_score(chunk, a)
    assert "HTTP 500" in result["matched_codes"]


def test_word_boundary_protection_for_numeric_codes():
    """Numeric codes must NOT match inside larger numbers."""
    chunk = _chunk("order id 19021 was processed")
    a = analyze_query("How do I troubleshoot error 902?")
    result = compute_exact_match_score(chunk, a)
    # "902" must not match "19021"
    assert "902" not in result["matched_codes"]
    assert result["score"] == 0.0


def test_exact_match_technical_term():
    chunk = _chunk("The system rejected the message with code rejected-forbidden-country")
    a = analyze_query("Why did rejected-forbidden-country trigger 902?")
    result = compute_exact_match_score(chunk, a)
    assert "rejected-forbidden-country" in result["matched_terms"]
    assert result["score"] > 0


def test_underscore_term_matches_hyphenated_chunk():
    chunk = _chunk("error in message-delivery-failed path")
    a = analyze_query("error in message_delivery_failed path")
    result = compute_exact_match_score(chunk, a)
    assert "message_delivery_failed" in result["matched_terms"]


# ---------------------------------------------------------------------------
# Boost application tests
# ---------------------------------------------------------------------------


def test_boost_no_op_when_no_identifiers():
    chunks = [_chunk("unrelated content"), _chunk("more content")]
    out = apply_exact_match_boost(chunks, None)
    assert len(out) == 2
    # No metadata injected when no identifiers are present.
    assert "_exact_match_count" not in out[0]


def test_boost_reranks_exact_match_above_higher_vector_score():
    """A chunk with exact "902" should rank above a chunk with higher
    vector score but no "902"."""
    matching = _chunk("Error 902 - rejected-forbidden-country", score=0.65)
    nonmatching = _chunk("totally unrelated article about something else", score=0.76)
    a = analyze_query("How do I troubleshoot error 902?")
    out = apply_exact_match_boost([nonmatching, matching], a)
    assert out[0]["content"].startswith("Error 902")
    assert out[0]["_exact_match_count"] >= 1


def test_boost_attaches_diagnostics():
    chunk = _chunk("Error 902 - rejected-forbidden-country")
    a = analyze_query("Why did rejected-forbidden-country trigger 902?")
    out = apply_exact_match_boost([chunk], a)
    assert "_exact_match_count" in out[0]
    assert "_exact_match_boost" in out[0]
    assert "_matched_codes" in out[0]
    assert "_matched_terms" in out[0]


def test_return_only_scored_drops_non_matching_chunks():
    matching = _chunk("Error 902", score=0.5)
    noise = _chunk("unrelated content", score=0.9)
    a = analyze_query("How do I troubleshoot error 902?")
    out = apply_exact_match_boost([noise, matching], a, return_only_scored=True)
    assert len(out) == 1
    assert "902" in out[0]["content"]


def test_boost_is_multiplicative():
    chunk = _chunk("Error 902", score=0.50)
    a = analyze_query("How do I troubleshoot error 902?")
    out = apply_exact_match_boost([chunk], a)
    # With default boost 0.30, score should be 0.50 * 1.30 = 0.65
    assert abs(out[0]["score"] - 0.65) < 1e-6


def test_boost_capped_per_chunk():
    """Multiple accidental hits do not run away."""
    chunk = _chunk("902 902 902 902 902", score=0.5)
    a = analyze_query("error 902")  # only one code in analysis
    out = apply_exact_match_boost([chunk], a)
    # Boost is per-query-analysis-identifier, not per-occurrence in the
    # chunk, so it should still be 0.5 * 1.30 = 0.65.
    assert abs(out[0]["score"] - 0.65) < 1e-6


def test_boost_does_not_mutate_input():
    chunk = _chunk("Error 902", score=0.5)
    a = analyze_query("How do I troubleshoot error 902?")
    _ = apply_exact_match_boost([chunk], a)
    assert chunk["score"] == 0.5  # input unchanged

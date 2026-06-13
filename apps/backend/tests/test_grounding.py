"""
Tests for Answer Grounding (Phase 10)

Tests the grounding checks that prevent hallucinations and enforce citations:
- No chunks = no LLM call
- Low score = no answer
- Answer without citation rejected
- Valid chunks = answer with citations
- Debug info shows threshold decision
"""

import pytest
from app.rag.grounding import (
    check_retrieval_guardrail,
    check_minimum_relevance,
    check_citations,
    check_answer_grounding,
    apply_grounding_checks,
    get_debug_info,
    NO_CHUNKS_MESSAGE,
    LOW_RELEVANCE_MESSAGE,
    NO_CITATIONS_MESSAGE,
)


class TestRetrievalGuardrail:
    """Test retrieval guardrail - no chunks means no LLM call."""

    def test_no_chunks_returns_block(self):
        """If no chunks retrieved, should block and return fallback."""
        chunks = []
        should_block, message, meta = check_retrieval_guardrail(chunks)
        
        assert should_block is True
        assert message == NO_CHUNKS_MESSAGE
        assert meta["blocked_reason"] == "no_chunks_retrieved"
        assert meta["chunk_count"] == 0

    def test_with_chunks_allows_continuation(self):
        """If chunks are retrieved, should not block."""
        chunks = [{"content": "test", "score": 0.9}]
        should_block, message, meta = check_retrieval_guardrail(chunks)
        
        assert should_block is False
        assert message is None
        assert meta["chunk_count"] == 1


class TestMinimumRelevance:
    """Test minimum relevance threshold - low scores mean no answer."""

    def test_below_threshold_blocks(self):
        """If top chunk score is below threshold, should block."""
        chunks = [{"content": "test", "score": 0.2}]
        threshold = 0.3
        
        should_block, message, meta = check_minimum_relevance(chunks, threshold)
        
        assert should_block is True
        assert message == LOW_RELEVANCE_MESSAGE
        assert meta["threshold_checked"] is True
        assert meta["threshold_used"] == threshold
        assert meta["top_score"] == 0.2

    def test_above_threshold_allows_continuation(self):
        """If top chunk score is above threshold, should not block."""
        chunks = [{"content": "test", "score": 0.8}]
        threshold = 0.3
        
        should_block, message, meta = check_minimum_relevance(chunks, threshold)
        
        assert should_block is False
        assert message is None
        assert meta["top_score"] == 0.8

    def test_at_threshold_allows_continuation(self):
        """If top chunk score equals threshold, should not block (inclusive)."""
        chunks = [{"content": "test", "score": 0.3}]
        threshold = 0.3
        
        should_block, message, meta = check_minimum_relevance(chunks, threshold)
        
        assert should_block is False
        assert meta["top_score"] == 0.3

    def test_no_score_allows_continuation(self):
        """If no score available, should allow (score might be optional)."""
        chunks = [{"content": "test"}]
        
        should_block, message, meta = check_minimum_relevance(chunks, 0.3)
        
        assert should_block is False
        assert meta["threshold_checked"] is False

    def test_empty_chunks_no_check(self):
        """Empty chunks should not check threshold."""
        chunks = []
        
        should_block, message, meta = check_minimum_relevance(chunks, 0.3)
        
        assert should_block is False
        assert meta["threshold_checked"] is False


class TestCitations:
    """Test citation enforcement - answers must include citations."""

    def test_answer_with_citations_has_citations(self):
        """Answer with [1], [2] style citations should be recognized."""
        answer = "According to [1], the capital of France is Paris. [2] confirms this."
        
        has_citations, meta = check_citations(answer)
        
        assert has_citations is True
        assert meta["citation_count"] == 2
        assert 1 in meta["citations_found"]
        assert 2 in meta["citations_found"]

    def test_answer_without_citations_no_citations(self):
        """Answer without citations should be detected."""
        answer = "The capital of France is Paris."
        
        has_citations, meta = check_citations(answer)
        
        assert has_citations is False
        assert meta["citation_count"] == 0

    def test_answer_with_duplicate_citations(self):
        """Duplicate citations should be counted once."""
        answer = "[1] says Paris. [1] is correct about Paris."
        
        has_citations, meta = check_citations(answer)
        
        assert has_citations is True
        assert meta["citation_count"] == 1

    def test_answer_with_multiple_citations(self):
        """Answer with multiple citations in various formats."""
        answer = "According to [1, 2, 3], this is true. Also see [4] and [5]."
        
        has_citations, meta = check_citations(answer)
        
        assert has_citations is True
        assert meta["citation_count"] == 5


class TestAnswerGrounding:
    """Test answer grounding check - verifies answer is based on context."""

    def test_answer_with_citations_is_grounded(self):
        """Answer with citations is considered grounded."""
        chunks = [{"content": "test", "score": 0.9}]
        answer = "According to [1], the answer is yes."
        
        is_grounded, message, meta = check_answer_grounding(answer, chunks, require_citations=True)
        
        assert is_grounded is False  # Not blocked
        assert message is None

    def test_answer_without_citations_is_blocked(self):
        """Answer without citations should be blocked."""
        chunks = [{"content": "test", "score": 0.9}]
        answer = "The answer is yes."
        
        is_grounded, message, meta = check_answer_grounding(answer, chunks, require_citations=True)
        
        assert is_grounded is True
        assert message == NO_CITATIONS_MESSAGE

    def test_no_citations_not_required(self):
        """When citations not required, answer without citations is OK."""
        chunks = [{"content": "test", "score": 0.9}]
        answer = "The answer is yes."
        
        is_grounded, message, meta = check_answer_grounding(answer, chunks, require_citations=False)
        
        assert is_grounded is False


class TestApplyGroundingChecks:
    """Test combined grounding checks."""

    def test_no_chunks_blocked(self):
        """Empty chunks should be blocked at retrieval guardrail."""
        chunks = []
        
        should_block, message, meta = apply_grounding_checks(
            chunks=chunks,
            answer=None,
            threshold=0.3,
            require_citations=False,
        )
        
        assert should_block is True
        assert message == NO_CHUNKS_MESSAGE

    def test_low_score_blocked(self):
        """Low score chunks should be blocked at minimum relevance."""
        chunks = [{"content": "test", "score": 0.1}]
        
        should_block, message, meta = apply_grounding_checks(
            chunks=chunks,
            answer=None,
            threshold=0.3,
            require_citations=False,
        )
        
        assert should_block is True
        assert message == LOW_RELEVANCE_MESSAGE

    def test_valid_chunks_no_answer_yet(self):
        """Valid chunks but no answer should proceed (only retrieval checks)."""
        chunks = [{"content": "test", "score": 0.9}]
        
        should_block, message, meta = apply_grounding_checks(
            chunks=chunks,
            answer=None,
            threshold=0.3,
            require_citations=False,
        )
        
        assert should_block is False

    def test_valid_chunks_with_valid_answer(self):
        """Valid chunks with answer containing citations should proceed."""
        chunks = [{"content": "test", "score": 0.9}]
        answer = "According to [1], this is correct."
        
        should_block, message, meta = apply_grounding_checks(
            chunks=chunks,
            answer=answer,
            threshold=0.3,
            require_citations=True,
        )
        
        assert should_block is False

    def test_valid_chunks_with_no_citation_answer_blocked(self):
        """Valid chunks but answer without citations should be blocked."""
        chunks = [{"content": "test", "score": 0.9}]
        answer = "This is the answer."
        
        should_block, message, meta = apply_grounding_checks(
            chunks=chunks,
            answer=answer,
            threshold=0.3,
            require_citations=True,
        )
        
        assert should_block is True
        assert message == NO_CITATIONS_MESSAGE


class TestGetDebugInfo:
    """Test debug info generation."""

    def test_debug_info_with_chunks(self):
        """Debug info should include chunk details."""
        chunks = [
            {"content": "First chunk content", "score": 0.9, "source_file_name": "doc1.txt"},
            {"content": "Second chunk content", "score": 0.7, "source_file_name": "doc2.txt"},
        ]
        threshold = 0.3
        grounding_result = (False, None, {"chunk_count": 2})
        
        debug = get_debug_info(chunks, threshold, grounding_result)
        
        assert debug["retrieval"]["chunk_count"] == 2
        assert debug["retrieval"]["top_score"] == 0.9
        assert "doc1.txt" in debug["retrieval"]["source_files"]
        assert debug["threshold"]["configured"] == threshold
        assert debug["threshold"]["top_score_met"] is True
        assert debug["grounding"]["blocked"] is False

    def test_debug_info_threshold_decision(self):
        """Debug info should show threshold decision."""
        chunks = [{"content": "test", "score": 0.2}]
        threshold = 0.5
        grounding_result = (True, LOW_RELEVANCE_MESSAGE, {"blocked_reason": "low_score"})
        
        debug = get_debug_info(chunks, threshold, grounding_result)
        
        assert debug["threshold"]["top_score_met"] is False
        assert debug["grounding"]["blocked"] is True

    def test_debug_info_empty_chunks(self):
        """Debug info should handle empty chunks."""
        chunks = []
        threshold = 0.3
        grounding_result = (True, NO_CHUNKS_MESSAGE, {"blocked_reason": "no_chunks"})
        
        debug = get_debug_info(chunks, threshold, grounding_result)
        
        assert debug["retrieval"]["chunk_count"] == 0
        assert debug["retrieval"]["top_score"] is None


class TestDefaultMessages:
    """Test default fallback messages are properly defined."""

    def test_no_chunks_message(self):
        """NO_CHUNKS_MESSAGE should be descriptive."""
        assert "don't have enough information" in NO_CHUNKS_MESSAGE.lower() or "could not find enough information" in NO_CHUNKS_MESSAGE.lower()
        assert "sources" in NO_CHUNKS_MESSAGE.lower()

    def test_low_relevance_message(self):
        """LOW_RELEVANCE_MESSAGE should indicate relevance issue."""
        assert "don't have enough" in LOW_RELEVANCE_MESSAGE.lower() or "could not find enough" in LOW_RELEVANCE_MESSAGE.lower()
        assert "relevant" in LOW_RELEVANCE_MESSAGE.lower()

    def test_no_citations_message(self):
        """NO_CITATIONS_MESSAGE should indicate citation issue."""
        assert "don't have enough information" in NO_CITATIONS_MESSAGE.lower() or "could not find enough information" in NO_CITATIONS_MESSAGE.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_none_score_handled(self):
        """None score should be handled gracefully."""
        chunks = [{"content": "test", "score": None}]
        
        should_block, message, meta = check_minimum_relevance(chunks, 0.3)
        
        assert should_block is False  # Should allow when score is None

    def test_negative_score_handled(self):
        """Negative score should be handled."""
        chunks = [{"content": "test", "score": -0.1}]
        
        should_block, message, meta = check_minimum_relevance(chunks, 0.3)
        
        assert should_block is True

    def test_very_high_score_allows(self):
        """Very high score should definitely allow."""
        chunks = [{"content": "test", "score": 1.0}]
        
        should_block, message, meta = check_minimum_relevance(chunks, 0.3)
        
        assert should_block is False

    def test_custom_threshold(self):
        """Custom threshold should override default."""
        chunks = [{"content": "test", "score": 0.5}]
        
        should_block_default, _, _ = check_minimum_relevance(chunks, 0.3)
        should_block_custom, _, _ = check_minimum_relevance(chunks, 0.7)
        
        assert should_block_default is False
        assert should_block_custom is True
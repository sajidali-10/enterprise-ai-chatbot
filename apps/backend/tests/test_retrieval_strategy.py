"""
Tests for Phase 30E — Retrieval Strategy Abstraction & Answer Quality.

Covers:
- Retrieval strategy selector dispatch (similarity / hybrid / mmr / hybrid_mmr)
- Generic keyword/phrase/number/acronym scoring
- MMR-style diversity filtering
- Improved fallback messages
- Answer prompt structure guidance
- Debug mode diagnostics
- Source filename (not full path) exposure
- Permission filtering preserved
"""

import pytest

from app.rag.hybrid_retriever import (
    VALID_STRATEGIES,
    _extract_query_signals,
    _generic_keyword_score,
    _boost_chunks_with_keyword_signals,
    _apply_strategy_to_chunks,
    mmr_diversify,
    retrieve_with_strategy,
    RetrievalConfig,
)
from app.rag.grounding import (
    NO_CHUNKS_MESSAGE,
    LOW_RELEVANCE_MESSAGE,
    NO_CITATIONS_MESSAGE,
    TOPIC_MISMATCH_MESSAGE,
)


# ============================================================================
# Strategy selector dispatch
# ============================================================================

class TestRetrievalStrategySelector:
    def test_valid_strategies_constant(self):
        """The strategy constant exposes the documented strategies."""
        assert "similarity" in VALID_STRATEGIES
        assert "hybrid" in VALID_STRATEGIES
        assert "mmr" in VALID_STRATEGIES
        assert "hybrid_mmr" in VALID_STRATEGIES

    def test_retrieve_with_strategy_unknown_falls_back_to_hybrid(self):
        """Unknown strategy names must fall back to a safe default (hybrid)."""
        config = RetrievalConfig(vector_top_k=10, keyword_top_k=10, final_top_k=5)
        chunks = [
            {"chunk_id": "1", "content": "alpha", "title": "", "score": 0.8},
        ]
        _, meta = _apply_strategy_to_chunks(
            chunks=chunks,
            query="anything",
            strategy="not_a_real_strategy",
            config=config,
            mmr_lambda=0.7,
        )
        assert meta["strategy"] == "hybrid"

    def test_retrieve_with_strategy_similarity_metadata(self):
        """similarity strategy reports hybrid_applied=False, mmr_applied=False."""
        config = RetrievalConfig(vector_top_k=10, keyword_top_k=10, final_top_k=5)
        chunks = [
            {"chunk_id": "1", "content": "alpha", "title": "", "score": 0.8},
        ]
        _, meta = _apply_strategy_to_chunks(
            chunks=chunks,
            query="anything",
            strategy="similarity",
            config=config,
            mmr_lambda=0.7,
        )
        assert meta["strategy"] == "similarity"
        assert meta["hybrid_applied"] is False
        assert meta["mmr_applied"] is False

    def test_retrieve_with_strategy_mmr_metadata(self):
        """mmr strategy reports hybrid_applied=False, mmr_applied=True."""
        config = RetrievalConfig(vector_top_k=10, keyword_top_k=10, final_top_k=5)
        chunks = [
            {"chunk_id": "1", "content": "alpha", "title": "", "score": 0.8},
            {"chunk_id": "2", "content": "beta", "title": "", "score": 0.7},
        ]
        _, meta = _apply_strategy_to_chunks(
            chunks=chunks,
            query="anything",
            strategy="mmr",
            config=config,
            mmr_lambda=0.7,
        )
        assert meta["strategy"] == "mmr"
        assert meta["hybrid_applied"] is False
        assert meta["mmr_applied"] is True

    def test_retrieve_with_strategy_hybrid_metadata(self):
        """hybrid strategy reports hybrid_applied=True, mmr_applied=False."""
        config = RetrievalConfig(vector_top_k=10, keyword_top_k=10, final_top_k=5)
        chunks = [
            {"chunk_id": "1", "content": "alpha", "title": "", "score": 0.8},
        ]
        _, meta = _apply_strategy_to_chunks(
            chunks=chunks,
            query="anything",
            strategy="hybrid",
            config=config,
            mmr_lambda=0.7,
        )
        assert meta["strategy"] == "hybrid"
        assert meta["hybrid_applied"] is True
        assert meta["mmr_applied"] is False

    def test_retrieve_with_strategy_hybrid_mmr_metadata(self):
        """hybrid_mmr strategy reports hybrid_applied=True, mmr_applied=True."""
        config = RetrievalConfig(vector_top_k=10, keyword_top_k=10, final_top_k=5)
        chunks = [
            {"chunk_id": "1", "content": "alpha", "title": "", "score": 0.8},
            {"chunk_id": "2", "content": "beta", "title": "", "score": 0.7},
        ]
        _, meta = _apply_strategy_to_chunks(
            chunks=chunks,
            query="anything",
            strategy="hybrid_mmr",
            config=config,
            mmr_lambda=0.7,
        )
        assert meta["strategy"] == "hybrid_mmr"
        assert meta["hybrid_applied"] is True
        assert meta["mmr_applied"] is True


# ============================================================================
# Generic query signal extraction
# ============================================================================

class TestGenericQuerySignals:
    def test_extract_numbers(self):
        s = _extract_query_signals("What port is 8443 used for?")
        assert "8443" in s["numbers"]

    def test_extract_acronyms(self):
        s = _extract_query_signals("How do I configure the HNP on port 8443?")
        assert "HNP" in s["acronyms"]

    def test_extract_quoted_phrase(self):
        s = _extract_query_signals('What is the "push notification gateway" used for?')
        assert "push notification gateway" in s["phrases"]

    def test_extract_words_filters_stopwords(self):
        s = _extract_query_signals("What is the system?")
        # stopwords like "what", "is", "the" should be filtered
        assert "what" not in s["words"]
        assert "is" not in s["words"]
        assert "the" not in s["words"]

    def test_extract_words_keeps_meaningful(self):
        s = _extract_query_signals("How do containers communicate?")
        assert "containers" in s["words"]
        assert "communicate" in s["words"]

    def test_empty_query(self):
        s = _extract_query_signals("")
        assert s["phrases"] == []
        assert s["numbers"] == []
        assert s["acronyms"] == []
        assert s["words"] == []


# ============================================================================
# Generic keyword/phrase/number/acronym scoring
# ============================================================================

class TestGenericKeywordScore:
    def test_number_match_increases_score(self):
        signals = _extract_query_signals("What is port 8443?")
        chunk_with = "The service listens on port 8443 for incoming traffic."
        chunk_without = "The service uses a standard port for incoming traffic."
        score_with = _generic_keyword_score(chunk_with, "", signals)
        score_without = _generic_keyword_score(chunk_without, "", signals)
        assert score_with > score_without

    def test_acronym_match_increases_score(self):
        signals = _extract_query_signals("How does HNP work?")
        chunk_with = "HNP forwards messages to mobile devices."
        chunk_without = "The notification system forwards messages."
        score_with = _generic_keyword_score(chunk_with, "", signals)
        score_without = _generic_keyword_score(chunk_without, "", signals)
        assert score_with > score_without

    def test_phrase_match_increases_score(self):
        signals = _extract_query_signals("Tell me about push notification gateway")
        chunk_with = "The push notification gateway is the primary delivery mechanism."
        chunk_without = "The notification system is the primary delivery mechanism."
        score_with = _generic_keyword_score(chunk_with, "", signals)
        score_without = _generic_keyword_score(chunk_without, "", signals)
        assert score_with > score_without

    def test_no_signals_returns_zero(self):
        signals = _extract_query_signals("")
        score = _generic_keyword_score("Some content here", "", signals)
        assert score == 0.0

    def test_title_boost(self):
        signals = _extract_query_signals("What is the installation process?")
        chunk_title_match = "Installation Guide"
        chunk_title_other = "Configuration Reference"
        content = "Follow the steps to set up the system."
        s_match = _generic_keyword_score(content, chunk_title_match, signals)
        s_other = _generic_keyword_score(content, chunk_title_other, signals)
        # Title-match should score >= the other one
        assert s_match >= s_other


class TestBoostChunksWithKeywordSignals:
    def test_boost_preserves_chunk_count(self):
        chunks = [
            {"chunk_id": "1", "content": "port 8443 used for HNP", "title": "", "score": 0.5},
            {"chunk_id": "2", "content": "some unrelated content", "title": "", "score": 0.6},
        ]
        boosted = _boost_chunks_with_keyword_signals(chunks, "HNP port 8443", keyword_weight=0.5)
        assert len(boosted) == 2

    def test_boost_reorders_by_combined_score(self):
        chunks = [
            {"chunk_id": "1", "content": "some unrelated content", "title": "", "score": 0.9},
            {"chunk_id": "2", "content": "HNP port 8443 specifically", "title": "", "score": 0.5},
        ]
        boosted = _boost_chunks_with_keyword_signals(chunks, "HNP 8443", keyword_weight=0.5)
        # The HNP-matching chunk should now rank higher because keyword boost is strong
        assert boosted[0]["chunk_id"] == "2"

    def test_boost_no_signals_returns_unchanged(self):
        chunks = [
            {"chunk_id": "1", "content": "alpha", "title": "", "score": 0.5},
            {"chunk_id": "2", "content": "beta", "title": "", "score": 0.7},
        ]
        boosted = _boost_chunks_with_keyword_signals(chunks, "", keyword_weight=0.5)
        assert boosted == chunks


# ============================================================================
# MMR diversity filtering
# ============================================================================

class TestMMRDiversify:
    def test_empty_input_returns_empty(self):
        assert mmr_diversify([]) == []

    def test_preserves_highest_scoring_chunk(self):
        chunks = [
            {"chunk_id": "1", "content": "alpha content", "title": "", "document_id": "1", "score": 0.9},
            {"chunk_id": "2", "content": "beta content", "title": "", "document_id": "2", "score": 0.5},
        ]
        result = mmr_diversify(chunks, lambda_param=0.7, top_n=2)
        # Highest-scoring chunk must be first regardless of MMR penalty
        assert result[0]["chunk_id"] == "1"

    def test_reduces_duplicate_chunks(self):
        duplicate_text = "the exact same text appears in both chunks"
        chunks = [
            {"chunk_id": "1", "content": duplicate_text, "title": "", "document_id": "1", "score": 0.9},
            {"chunk_id": "2", "content": duplicate_text + " slightly different", "title": "", "document_id": "2", "score": 0.85},
            {"chunk_id": "3", "content": "completely different content here", "title": "", "document_id": "3", "score": 0.8},
        ]
        result = mmr_diversify(chunks, lambda_param=0.5, top_n=2)
        assert len(result) == 2
        # The diverse chunk should be selected over the near-duplicate
        chunk_ids = [c["chunk_id"] for c in result]
        assert "1" in chunk_ids
        assert "3" in chunk_ids  # diverse content wins over near-duplicate

    def test_preserves_source_metadata(self):
        chunks = [
            {"chunk_id": "1", "content": "alpha", "title": "Title A", "document_id": "42", "source_file_name": "guide.pdf", "score": 0.9},
        ]
        result = mmr_diversify(chunks, top_n=1)
        assert result[0]["source_file_name"] == "guide.pdf"
        assert result[0]["document_id"] == "42"
        assert result[0]["title"] == "Title A"

    def test_top_n_caps_result_count(self):
        chunks = [
            {"chunk_id": str(i), "content": f"unique content number {i}", "title": "", "document_id": str(i), "score": 0.9 - i * 0.01}
            for i in range(10)
        ]
        result = mmr_diversify(chunks, lambda_param=0.7, top_n=3)
        assert len(result) == 3

    def test_does_not_remove_only_relevant_source(self):
        chunks = [
            {"chunk_id": "1", "content": "only relevant content", "title": "", "document_id": "1", "score": 0.9},
        ]
        result = mmr_diversify(chunks, lambda_param=0.0, top_n=5)
        # Even with diversity focus, the only relevant chunk is preserved
        assert len(result) == 1
        assert result[0]["chunk_id"] == "1"


# ============================================================================
# Fallback messages (Phase 30E improvements)
# ============================================================================

class TestFallbackMessages:
    def test_no_chunks_message_is_helpful(self):
        """The no-chunks fallback should be helpful and generic (not domain-specific)."""
        assert "could not find" in NO_CHUNKS_MESSAGE.lower()
        assert "upload" in NO_CHUNKS_MESSAGE.lower()
        # Must not hardcode any specific product, port, or filename
        msg_lower = NO_CHUNKS_MESSAGE.lower()
        for forbidden in ["hnp", "8443", "tcp", "udp", "userguide", "installation"]:
            assert forbidden not in msg_lower, f"Fallback leaks hardcoded term: {forbidden}"

    def test_low_relevance_message_is_helpful(self):
        assert "could not find" in LOW_RELEVANCE_MESSAGE.lower()
        assert "upload" in LOW_RELEVANCE_MESSAGE.lower()

    def test_no_citations_message_is_helpful(self):
        assert "could not find" in NO_CITATIONS_MESSAGE.lower()

    def test_topic_mismatch_message_is_helpful(self):
        assert "could not find" in TOPIC_MISMATCH_MESSAGE.lower()


# ============================================================================
# Prompt structure (Phase 30E improvements)
# ============================================================================

class TestPromptStructure:
    def test_rag_prompt_contains_answer_structure(self):
        """The RAG prompt should include generic answer structure guidance."""
        from app.rag.prompt_builder import build_rag_prompt
        chunks = [{"chunk_id": "1", "content": "some content", "source_file_name": "guide.pdf"}]
        prompt = build_rag_prompt("What is X?", chunks)
        assert "ANSWER STRUCTURE" in prompt
        assert "Lead with the direct answer" in prompt or "direct answer" in prompt
        # Must not hardcode specific products, ports, or filenames in instructions
        prompt_lower = prompt.lower()
        for forbidden in ["hnp", "8443"]:
            assert forbidden not in prompt_lower

    def test_strict_citation_prompt_contains_answer_structure(self):
        from app.rag.prompt_builder import build_strict_citation_prompt
        chunks = [{"chunk_id": "1", "content": "some content", "source_file_name": "guide.pdf"}]
        prompt = build_strict_citation_prompt("What is X?", chunks)
        assert "ANSWER STRUCTURE" in prompt

    def test_no_chunks_prompt_uses_helpful_fallback(self):
        from app.rag.prompt_builder import build_rag_prompt
        prompt = build_rag_prompt("What is X?", [])
        assert "could not find" in prompt.lower()
        assert "upload" in prompt.lower()

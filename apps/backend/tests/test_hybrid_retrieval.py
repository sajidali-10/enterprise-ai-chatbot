"""
Tests for Hybrid Retrieval

Tests hybrid result merging and reranking functionality.
"""

import pytest
from app.rag.hybrid_retriever import (
    _normalize_scores,
    _fuse_scores,
    ScoredChunk,
    RetrievalConfig,
)
from app.rag.reranker import (
    MockReranker,
    NoOpReranker,
    RerankResult,
    get_reranker,
)


class TestNormalizeScores:
    """Tests for score normalization."""

    def test_normalize_empty_list(self):
        assert _normalize_scores([]) == []

    def test_normalize_single_value(self):
        assert _normalize_scores([0.5]) == [0.5]

    def test_normalize_two_values(self):
        result = _normalize_scores([0.0, 1.0])
        assert result == pytest.approx([0.0, 1.0])

    def test_normalize_middle_values(self):
        result = _normalize_scores([0.2, 0.5, 0.8])
        # 0.2 -> 0.0, 0.5 -> 0.5, 0.8 -> 1.0
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_normalize_same_values(self):
        result = _normalize_scores([0.5, 0.5, 0.5])
        # All same should become 0.5
        assert result == pytest.approx([0.5, 0.5, 0.5])


class TestFuseScores:
    """Tests for score fusion."""

    def test_fuse_empty_results(self):
        result = _fuse_scores([], [])
        assert result == []

    def test_fuse_vector_only(self):
        vector_results = [
            {"chunk_id": "v1", "document_id": "1", "chunk_index": 0, "content": "test1", "source_file_name": "a.txt", "title": "A", "score": 0.9},
            {"chunk_id": "v2", "document_id": "1", "chunk_index": 1, "content": "test2", "source_file_name": "b.txt", "title": "B", "score": 0.7},
        ]
        result = _fuse_scores(vector_results, [])
        assert len(result) == 2
        assert result[0].chunk_id == "v1"
        assert result[0].is_from_vector is True
        assert result[0].is_from_keyword is False
        assert result[1].chunk_id == "v2"

    def test_fuse_keyword_only(self):
        keyword_results = [
            {"chunk_id": "k1", "document_id": "2", "chunk_index": 0, "content": "test1", "source_file_name": "c.txt", "title": "C", "score": 0.8},
        ]
        result = _fuse_scores([], keyword_results)
        assert len(result) == 1
        assert result[0].chunk_id == "k1"
        assert result[0].is_from_vector is False
        assert result[0].is_from_keyword is True

    def test_fuse_combined_different_chunks(self):
        vector_results = [
            {"chunk_id": "v1", "document_id": "1", "chunk_index": 0, "content": "vec content", "source_file_name": "a.txt", "title": "A", "score": 0.9},
        ]
        keyword_results = [
            {"chunk_id": "k1", "document_id": "2", "chunk_index": 0, "content": "kw content", "source_file_name": "b.txt", "title": "B", "score": 0.8},
        ]
        result = _fuse_scores(vector_results, keyword_results)
        assert len(result) == 2
        # v1 should be first (higher normalized vector score)
        assert result[0].chunk_id == "v1"

    def test_fuse_combined_same_chunk(self):
        """Test that same chunk from both sources gets combined score."""
        vector_results = [
            {"chunk_id": "same1", "document_id": "1", "chunk_index": 0, "content": "shared content", "source_file_name": "a.txt", "title": "A", "score": 0.9},
        ]
        keyword_results = [
            {"chunk_id": "same1", "document_id": "1", "chunk_index": 0, "content": "shared content", "source_file_name": "a.txt", "title": "A", "score": 0.8},
        ]
        result = _fuse_scores(vector_results, keyword_results)
        assert len(result) == 1
        assert result[0].chunk_id == "same1"
        assert result[0].is_from_vector is True
        assert result[0].is_from_keyword is True
        assert result[0].vector_score is not None
        assert result[0].keyword_score is not None
        # Combined score should be higher than either alone
        assert result[0].fused_score > max(result[0].vector_score, result[0].keyword_score) * 0.7

    def test_fuse_respects_weights(self):
        vector_results = [
            {"chunk_id": "v1", "document_id": "1", "chunk_index": 0, "content": "test1", "source_file_name": "a.txt", "title": "A", "score": 1.0},
            {"chunk_id": "v2", "document_id": "1", "chunk_index": 1, "content": "test2", "source_file_name": "b.txt", "title": "B", "score": 0.5},
        ]
        result = _fuse_scores(vector_results, [], vector_weight=0.8, keyword_weight=0.2)
        assert len(result) == 2
        # v1 has normalized score 1.0, fused = 1.0 * 0.8 = 0.8
        assert result[0].fused_score == pytest.approx(0.8)


class TestMockReranker:
    """Tests for mock reranker."""

    def test_rerank_returns_same_number_of_chunks(self):
        chunks = [
            {"chunk_id": "c1", "document_id": "1", "chunk_index": 0, "content": "test content", "source_file_name": "a.txt", "title": "A", "score": 0.5},
            {"chunk_id": "c2", "document_id": "1", "chunk_index": 1, "content": "test content two", "source_file_name": "b.txt", "title": "B", "score": 0.6},
        ]
        reranker = MockReranker()
        results = reranker.rerank("test content", chunks)
        assert len(results) == 2

    def test_rerank_respects_top_n(self):
        chunks = [
            {"chunk_id": f"c{i}", "document_id": "1", "chunk_index": i, "content": f"content {i}", "source_file_name": f"f{i}.txt", "title": f"T{i}", "score": 0.5}
            for i in range(10)
        ]
        reranker = MockReranker()
        results = reranker.rerank("content", chunks, top_n=5)
        assert len(results) == 5

    def test_rerank_sorts_by_rerank_score(self):
        chunks = [
            {"chunk_id": "c1", "document_id": "1", "chunk_index": 0, "content": "python programming language", "source_file_name": "a.txt", "title": "Programming Guide", "score": 0.3},
            {"chunk_id": "c2", "document_id": "1", "chunk_index": 1, "content": "python is a great language", "source_file_name": "b.txt", "title": "Python Intro", "score": 0.3},
        ]
        reranker = MockReranker()
        results = reranker.rerank("python", chunks)
        # c2 should rank higher because query term appears in title (title_boost=0.1)
        # c1 has no title boost since "python" is not in "Programming Guide"
        assert results[0].chunk_id == "c2"
        assert results[0].rerank_score >= results[1].rerank_score

    def test_rerank_boosts_title_match(self):
        chunks = [
            {"chunk_id": "c1", "document_id": "1", "chunk_index": 0, "content": "some generic content about testing", "source_file_name": "a.txt", "title": "Testing Guide", "score": 0.5},
            {"chunk_id": "c2", "document_id": "1", "chunk_index": 1, "content": "testing is important", "source_file_name": "b.txt", "title": "Quality", "score": 0.5},
        ]
        reranker = MockReranker()
        results = reranker.rerank("testing", chunks)
        # c1 has "testing" in both content AND title, should rank higher
        assert results[0].chunk_id == "c1"


class TestNoOpReranker:
    """Tests for no-op reranker."""

    def test_noop_preserves_order(self):
        chunks = [
            {"chunk_id": "c1", "document_id": "1", "chunk_index": 0, "content": "first", "source_file_name": "a.txt", "title": "A", "score": 0.9},
            {"chunk_id": "c2", "document_id": "1", "chunk_index": 1, "content": "second", "source_file_name": "b.txt", "title": "B", "score": 0.8},
        ]
        reranker = NoOpReranker()
        results = reranker.rerank("any query", chunks)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"

    def test_noop_preserves_scores(self):
        chunks = [
            {"chunk_id": "c1", "document_id": "1", "chunk_index": 0, "content": "test", "source_file_name": "a.txt", "title": "A", "score": 0.7},
        ]
        reranker = NoOpReranker()
        results = reranker.rerank("query", chunks)
        assert results[0].rerank_score == 0.7
        assert results[0].original_score == 0.7


class TestGetReranker:
    """Tests for reranker factory."""

    def test_get_reranker_none_returns_noop(self):
        reranker = get_reranker(None)
        assert isinstance(reranker, NoOpReranker)

    def test_get_reranker_passthrough_returns_noop(self):
        reranker = get_reranker("passthrough")
        assert isinstance(reranker, NoOpReranker)

    def test_get_reranker_noop_returns_noop(self):
        reranker = get_reranker("noop")
        assert isinstance(reranker, NoOpReranker)

    def test_get_reranker_mock_returns_mock(self):
        reranker = get_reranker("mock")
        assert isinstance(reranker, MockReranker)

    def test_get_reranker_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker type"):
            get_reranker("unknown_reranker")


class TestRetrievalConfig:
    """Tests for retrieval config."""

    def test_default_config(self):
        config = RetrievalConfig()
        assert config.vector_top_k == 10
        assert config.keyword_top_k == 10
        assert config.final_top_k == 5
        assert config.min_score == 0.0
        assert config.reranker_type == "noop"

    def test_custom_config(self):
        config = RetrievalConfig(
            vector_top_k=20,
            keyword_top_k=15,
            final_top_k=8,
            min_score=0.2,
            reranker_type="mock",
            vector_weight=0.6,
            keyword_weight=0.4,
        )
        assert config.vector_top_k == 20
        assert config.keyword_top_k == 15
        assert config.final_top_k == 8
        assert config.min_score == 0.2
        assert config.reranker_type == "mock"
        assert config.vector_weight == 0.6
        assert config.keyword_weight == 0.4
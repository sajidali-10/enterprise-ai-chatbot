"""
Tests for citation formatting, grouping, and excerpt selection.
"""

import pytest
from app.rag.citations import (
    get_confidence_label,
    clean_excerpt,
    extract_keywords,
    score_excerpt_relevance,
    select_best_excerpts_for_source,
    format_citations,
    group_citations_by_source,
)


class TestGetConfidenceLabel:
    """Tests for confidence label mapping."""
    
    def test_high_confidence(self):
        assert get_confidence_label(0.85) == "High"
        assert get_confidence_label(0.80) == "High"
        assert get_confidence_label(1.0) == "High"
    
    def test_medium_confidence(self):
        assert get_confidence_label(0.79) == "Medium"
        assert get_confidence_label(0.60) == "Medium"
        assert get_confidence_label(0.65) == "Medium"
    
    def test_low_confidence(self):
        assert get_confidence_label(0.59) == "Low"
        assert get_confidence_label(0.30) == "Low"
        assert get_confidence_label(0.0) == "Low"
    
    def test_none_score(self):
        assert get_confidence_label(None) == "Low"


class TestCleanExcerpt:
    """Tests for excerpt cleaning and truncation."""
    
    def test_short_content_unchanged(self):
        content = "This is a short excerpt."
        assert clean_excerpt(content) == content
    
    def test_sentence_boundary_truncation(self):
        # Should truncate at sentence boundary if reasonable
        content = "First sentence. Second sentence with more content here. Third sentence that is longer still. Fourth and final sentence here."
        result = clean_excerpt(content, max_length=80)
        assert result.endswith(".")
        assert "..." not in result or len(result) <= 85
    
    def test_whitespace_cleanup(self):
        content = "  Multiple   spaces   and\t\ttabs.  "
        result = clean_excerpt(content)
        assert "  " not in result
        assert result == "Multiple spaces and tabs."
    
    def test_noisy_prefix_removal(self):
        content = "1. This is a numbered item."
        result = clean_excerpt(content)
        assert not result.startswith("1.")
        assert "This is a numbered item." in result
    
    def test_bullet_prefix_removal(self):
        content = "- This is a bullet point."
        result = clean_excerpt(content)
        assert not result.startswith("-")
    
    def test_empty_content(self):
        assert clean_excerpt("") == ""
        assert clean_excerpt(None) == ""


class TestExtractKeywords:
    """Tests for keyword extraction."""
    
    def test_basic_extraction(self):
        text = "Docker containers and images are important components."
        keywords = extract_keywords(text)
        assert "docker" in keywords
        assert "containers" in keywords
        assert "images" in keywords
        assert "components" in keywords
    
    def test_stopwords_filtered(self):
        text = "the and for are but not you all can"
        keywords = extract_keywords(text)
        # All stopwords should be filtered
        assert len(keywords) == 0
    
    def test_short_words_filtered(self):
        text = "Docker is a tool. It is good."
        keywords = extract_keywords(text)
        # "is", "a", "it" should be filtered (< 3 chars or stopwords)
        assert "is" not in keywords
        assert "a" not in keywords
        assert "docker" in keywords
        assert "tool" in keywords
        assert "good" in keywords


class TestScoreExcerptRelevance:
    """Tests for excerpt relevance scoring."""
    
    def test_question_keyword_overlap(self):
        excerpt = "Docker images are templates for containers."
        question = "What are Docker images?"
        answer = "Docker images are templates."
        
        score = score_excerpt_relevance(excerpt, 0.7, question, answer)
        # Should have positive score with overlap
        assert score > 0.35  # At least 50% of 0.7 base
    
    def test_answer_keyword_overlap(self):
        excerpt = "Docker containers run applications."
        question = "What is Docker?"
        answer = "Containers run applications."
        
        score = score_excerpt_relevance(excerpt, 0.5, question, answer)
        # Answer overlap should boost score
        assert score > 0.5 * 0.5  # Base score with some answer overlap
    
    def test_generic_intro_penalty(self):
        """Generic intro content like 'What is Docker?' should be penalized."""
        excerpt = "What is Docker? Docker is a platform designed to help developers."
        question = "What are Docker image components?"
        answer = "Docker images contain base images, application code, dependencies, and metadata."
        
        score = score_excerpt_relevance(excerpt, 0.6, question, answer)
        # Generic content should be significantly penalized
        assert score < 0.6 * 0.6  # Should be penalized by 0.5 factor
    
    def test_why_question_penalty(self):
        """Generic 'why' questions should be penalized."""
        excerpt = "Why do we need Docker? Consistency across environments."
        question = "What are the components of a Docker image?"
        answer = "Docker images consist of base images and layers."
        
        score = score_excerpt_relevance(excerpt, 0.6, question, answer)
        # Should be penalized for generic "Why do we need" content
        assert score < 0.6 * 0.6
    
    def test_component_specific_boost(self):
        """Excerpts with component-specific keywords should score higher."""
        generic_excerpt = "What is Docker? Docker is a platform designed to help developers."
        specific_excerpt = "Docker images consist of a base image like Alpine or Ubuntu, application code, and metadata."
        
        question = "What are the components of a Docker image?"
        answer = "Docker images have base images, application code, dependencies, and metadata."
        
        generic_score = score_excerpt_relevance(generic_excerpt, 0.7, question, answer)
        specific_score = score_excerpt_relevance(specific_excerpt, 0.7, question, answer)
        
        # Specific content should score significantly higher than generic
        assert specific_score > generic_score
    
    def test_empty_excerpt(self):
        assert score_excerpt_relevance("", 0.5, "question", "answer") == 0.0


class TestSelectBestExcerptsForSource:
    """Tests for answer-aware excerpt selection."""
    
    def test_selects_most_relevant(self):
        citations = [
            {
                "index": 1,
                "source_file_name": "doc.pdf",
                "content_snippet": "Docker images are templates for containers.",
                "relevance_score": 0.9,
            },
            {
                "index": 2,
                "source_file_name": "doc.pdf",
                "content_snippet": "What is Docker? Introduction.",
                "relevance_score": 0.6,
            },
            {
                "index": 3,
                "source_file_name": "doc.pdf",
                "content_snippet": "Docker containers run applications.",
                "relevance_score": 0.8,
            },
        ]
        
        question = "What are Docker components?"
        answer = "Docker images and containers are key components."
        
        selected = select_best_excerpts_for_source(citations, question, answer, max_excerpts=2)
        
        # Should select 2 excerpts
        assert len(selected) == 2
        # First should be the one about images (answer relevance)
        assert "images" in selected[0]["content_snippet"].lower()
    
    def test_deduplicates_similar_excerpts(self):
        citations = [
            {
                "index": 1,
                "source_file_name": "doc.pdf",
                "content_snippet": "Docker images are templates.",
                "relevance_score": 0.9,
            },
            {
                "index": 2,
                "source_file_name": "doc.pdf",
                "content_snippet": "Docker images are templates for containers.",
                "relevance_score": 0.85,
            },
        ]
        
        selected = select_best_excerpts_for_source(citations, "question", "answer")
        
        # Should deduplicate - only one selected
        assert len(selected) == 1
    
    def test_respects_max_excerpts(self):
        citations = [
            {"index": i, "source_file_name": "doc.pdf", "content_snippet": f"Content {i}", "relevance_score": 0.5 + i * 0.1}
            for i in range(10)
        ]
        
        selected = select_best_excerpts_for_source(citations, "question", "answer", max_excerpts=3)
        
        assert len(selected) <= 3


class TestFormatCitations:
    """Tests for citation formatting."""
    
    def test_basic_formatting(self):
        chunks = [
            {
                "content": "This is the content of chunk one.",
                "source_file_name": "doc1.pdf",
                "score": 0.85,
                "id": "chunk-1",
            },
            {
                "content": "This is chunk two content here.",
                "source_file_name": "doc2.pdf",
                "score": 0.75,
                "id": "chunk-2",
            },
        ]
        
        citations = format_citations(chunks)
        
        assert len(citations) == 2
        assert citations[0]["index"] == 1
        assert citations[0]["source_file_name"] == "doc1.pdf"
        assert citations[0]["relevance_score"] == 0.85
        assert citations[0]["chunk_id"] == "chunk-1"
    
    def test_excerpt_cleaning(self):
        chunks = [
            {
                "content": "1. First item with lots of content that should be truncated to a reasonable length for display purposes.",
                "source_file_name": "doc.pdf",
                "score": 0.8,
                "id": "chunk-1",
            },
        ]
        
        citations = format_citations(chunks)
        
        # Should have cleaned excerpt
        assert not citations[0]["content_snippet"].startswith("1.")


class TestGroupCitationsBySource:
    """Tests for grouped source generation."""
    
    def test_groups_by_source_file_name(self):
        citations = [
            {"index": 1, "source_file_name": "doc1.pdf", "content_snippet": "Content 1", "relevance_score": 0.8},
            {"index": 2, "source_file_name": "doc2.pdf", "content_snippet": "Content 2", "relevance_score": 0.7},
            {"index": 3, "source_file_name": "doc1.pdf", "content_snippet": "Content 3", "relevance_score": 0.9},
        ]
        
        grouped = group_citations_by_source(citations)
        
        assert len(grouped) == 2
        # Should be sorted by highest score descending
        assert grouped[0]["source_file_name"] == "doc1.pdf"
        assert grouped[0]["highest_score"] == 0.9
        assert grouped[0]["sections_used"] == 2
    
    def test_answer_aware_selection(self):
        citations = [
            {"index": 1, "source_file_name": "doc.pdf", "content_snippet": "What is Docker? Introduction.", "relevance_score": 0.6},
            {"index": 2, "source_file_name": "doc.pdf", "content_snippet": "Docker images are templates for containers.", "relevance_score": 0.9},
        ]
        
        question = "What are Docker components?"
        answer = "Docker images are key components."
        
        grouped = group_citations_by_source(citations, question, answer, max_excerpts=1)
        
        # Should select the relevant excerpt about images, not the intro
        assert len(grouped[0]["excerpts"]) == 1
        assert "images" in grouped[0]["excerpts"][0].lower()
    
    def test_confidence_labels(self):
        citations = [
            {"index": 1, "source_file_name": "high.pdf", "content_snippet": "High", "relevance_score": 0.9},
            {"index": 2, "source_file_name": "med.pdf", "content_snippet": "Med", "relevance_score": 0.7},
            {"index": 3, "source_file_name": "low.pdf", "content_snippet": "Low", "relevance_score": 0.4},
        ]
        
        grouped = group_citations_by_source(citations)
        
        assert grouped[0]["confidence"] == "High"
        assert grouped[1]["confidence"] == "Medium"
        assert grouped[2]["confidence"] == "Low"
    
    def test_knowledge_base_hides_debug_details(self):
        """When debug_mode=False, indices should be empty and show_debug_details False."""
        citations = [
            {"index": 1, "source_file_name": "doc.pdf", "content_snippet": "Content 1", "relevance_score": 0.8},
            {"index": 2, "source_file_name": "doc.pdf", "content_snippet": "Content 2", "relevance_score": 0.7},
        ]
        
        grouped = group_citations_by_source(citations, debug_mode=False)
        
        assert len(grouped) == 1
        # highest_score is still present for sorting (not hidden)
        assert grouped[0]["highest_score"] == 0.8
        # indices should be empty in non-debug mode
        assert grouped[0]["indices"] == []
        # show_debug_details should be False
        assert grouped[0]["show_debug_details"] is False
    
    def test_debug_mode_shows_debug_details(self):
        """When debug_mode=True, show_debug_details is True and indices are populated."""
        citations = [
            {"index": 1, "source_file_name": "doc.pdf", "content_snippet": "Content 1", "relevance_score": 0.8},
            {"index": 2, "source_file_name": "doc.pdf", "content_snippet": "Content 2", "relevance_score": 0.7},
        ]
        
        grouped = group_citations_by_source(citations, debug_mode=True)
        
        assert len(grouped) == 1
        # highest_score should be present
        assert grouped[0]["highest_score"] == 0.8
        # indices should be populated
        assert grouped[0]["indices"] == [1, 2]
        # show_debug_details should be True
        assert grouped[0]["show_debug_details"] is True
    
    def test_sections_used_shows_total_but_excerpts_limited_to_3(self):
        """sections_used should reflect all citations, but excerpts limited to 3."""
        citations = [
            {"index": i, "source_file_name": "doc.pdf", "content_snippet": f"Content {i}", "relevance_score": 0.5 + i * 0.05}
            for i in range(1, 6)  # 5 citations
        ]
        
        grouped = group_citations_by_source(citations)
        
        assert len(grouped) == 1
        # sections_used should show total (5)
        assert grouped[0]["sections_used"] == 5
        # But excerpts should be at most 3
        assert len(grouped[0]["excerpts"]) <= 3

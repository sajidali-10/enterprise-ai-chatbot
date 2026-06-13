import pytest
from app.rag.prompt_builder import build_rag_prompt
from app.rag.citations import format_citations, group_citations_by_source, get_confidence_label

def test_build_rag_prompt_with_chunks():
    chunks = [
        {"source_file_name": "doc1.txt", "content": "The capital of France is Paris.", "score": 0.9},
        {"source_file_name": "doc2.txt", "content": "Paris is also known as the City of Light.", "score": 0.7},
    ]
    prompt = build_rag_prompt("What is the capital of France?", chunks)
    assert "Paris" in prompt
    assert "[1]" in prompt or "Source: doc1.txt" in prompt

def test_build_rag_prompt_no_chunks():
    prompt = build_rag_prompt("What is the capital of France?", [])
    assert "could not find enough" in prompt and "information" in prompt

def test_format_citations():
    chunks = [
        {"source_file_name": "doc1.txt", "content": "The capital of France is Paris.", "score": 0.9},
        {"source_file_name": "doc2.txt", "content": "Paris is also known as the City of Light.", "score": 0.7},
    ]
    citations = format_citations(chunks)
    assert len(citations) == 2
    assert citations[0]["source_file_name"] == "doc1.txt"
    assert citations[0]["index"] == 1
    assert "index" in citations[0]
    assert "content_snippet" in citations[0]

def test_format_citations_empty():
    citations = format_citations([])
    assert len(citations) == 0


def test_get_confidence_label():
    """Test confidence label conversion from scores."""
    assert get_confidence_label(0.90) == "High"
    assert get_confidence_label(0.80) == "High"
    assert get_confidence_label(0.79) == "Medium"
    assert get_confidence_label(0.60) == "Medium"
    assert get_confidence_label(0.59) == "Low"
    assert get_confidence_label(0.0) == "Low"
    assert get_confidence_label(None) == "Low"


def test_group_citations_by_source_single_document():
    """Test grouping multiple citations from the same document."""
    citations = [
        {"index": 1, "source_file_name": "Docker Guide.pdf", "content_snippet": "Docker is a platform...", "relevance_score": 0.98},
        {"index": 2, "source_file_name": "Docker Guide.pdf", "content_snippet": "Containers are...", "relevance_score": 0.85},
        {"index": 3, "source_file_name": "Docker Guide.pdf", "content_snippet": "Images are...", "relevance_score": 0.70},
    ]
    grouped = group_citations_by_source(citations)
    
    assert len(grouped) == 1
    assert grouped[0]["source_file_name"] == "Docker Guide.pdf"
    assert grouped[0]["sections_used"] == 3
    assert grouped[0]["highest_score"] == 0.98
    assert grouped[0]["confidence"] == "High"
    assert len(grouped[0]["excerpts"]) == 3  # default max is 3
    # indices empty when not in debug mode (default)
    assert grouped[0]["indices"] == []
    assert grouped[0]["show_debug_details"] is False


def test_group_citations_by_source_multiple_documents():
    """Test grouping citations from different documents."""
    citations = [
        {"index": 1, "source_file_name": "Docker Guide.pdf", "content_snippet": "Docker is...", "relevance_score": 0.98},
        {"index": 2, "source_file_name": "Kubernetes.txt", "content_snippet": "Kubernetes is...", "relevance_score": 0.75},
        {"index": 3, "source_file_name": "Docker Guide.pdf", "content_snippet": "Containers...", "relevance_score": 0.65},
    ]
    grouped = group_citations_by_source(citations)
    
    assert len(grouped) == 2
    # Should be sorted by highest score descending
    assert grouped[0]["source_file_name"] == "Docker Guide.pdf"
    assert grouped[0]["highest_score"] == 0.98
    assert grouped[0]["show_debug_details"] is False
    assert grouped[1]["source_file_name"] == "Kubernetes.txt"


def test_group_citations_by_source_max_excerpts():
    """Test that max_excerpts limit is respected."""
    citations = [
        {"index": i, "source_file_name": "Doc.pdf", "content_snippet": f"Content {i}", "relevance_score": 1.0 - (i * 0.1)}
        for i in range(1, 6)  # 5 citations from same doc
    ]
    # Limit to 2 excerpts
    grouped = group_citations_by_source(citations, max_excerpts=2)
    
    assert len(grouped[0]["excerpts"]) == 2
    assert grouped[0]["sections_used"] == 5  # All 5 sections still counted


def test_group_citations_by_source_empty():
    """Test grouping with empty citations list."""
    grouped = group_citations_by_source([])
    assert len(grouped) == 0


def test_group_citations_confidence_levels():
    """Test confidence labeling for different score levels."""
    citations_high = [{"index": 1, "source_file_name": "A.pdf", "content_snippet": "Test", "relevance_score": 0.90}]
    citations_medium = [{"index": 1, "source_file_name": "B.pdf", "content_snippet": "Test", "relevance_score": 0.65}]
    citations_low = [{"index": 1, "source_file_name": "C.pdf", "content_snippet": "Test", "relevance_score": 0.40}]
    
    grouped_high = group_citations_by_source(citations_high)
    grouped_medium = group_citations_by_source(citations_medium)
    grouped_low = group_citations_by_source(citations_low)
    
    assert grouped_high[0]["confidence"] == "High"
    assert grouped_medium[0]["confidence"] == "Medium"
    assert grouped_low[0]["confidence"] == "Low"
import pytest
from app.rag.prompt_builder import build_rag_prompt
from app.rag.citations import format_citations

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
"""
Tests for RAG LLM Integration (Phase 8)

Verifies that:
1. Retrieved context is properly included in LLM prompts
2. Citations are returned from RAG responses
3. No-source fallback works when retrieval fails
4. Mock provider simulates real LLM behavior for RAG
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.rag.prompt_builder import build_rag_prompt
from app.rag.citations import format_citations
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.mock_provider import MockProvider
from app.rag.answer_generator import generate_answer_with_rag
from app.main import app


# ==============================================================================
# Prompt Builder Tests
# ==============================================================================

class TestRagPromptBuilder:
    """Tests for RAG prompt construction."""
    
    def test_prompt_includes_context_chunks(self):
        """Retrieved context chunks are included in the prompt."""
        chunks = [
            {"source_file_name": "faq.txt", "content": "Our support hours are 9 AM to 5 PM EST.", "score": 0.95},
            {"source_file_name": "policy.txt", "content": "Response time is within 24 hours.", "score": 0.85},
        ]
        prompt = build_rag_prompt("What are your support hours?", chunks)
        
        assert "faq.txt" in prompt
        assert "policy.txt" in prompt
        assert "9 AM to 5 PM EST" in prompt
        assert "24 hours" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
    
    def test_prompt_includes_user_question(self):
        """User question is included in the prompt."""
        chunks = [
            {"source_file_name": "doc.txt", "content": "Sample content.", "score": 0.9},
        ]
        prompt = build_rag_prompt("How do I reset my password?", chunks)
        
        assert "How do I reset my password?" in prompt
    
    def test_prompt_has_citation_guidelines(self):
        """Prompt instructs the model to cite sources."""
        chunks = [
            {"source_file_name": "doc.txt", "content": "Sample content.", "score": 0.9},
        ]
        prompt = build_rag_prompt("Test question?", chunks)
        
        assert "Cite" in prompt or "[" in prompt
        assert "ANSWER:" in prompt
    
    def test_no_source_fallback_prompt(self):
        """When no chunks are retrieved, prompt says insufficient information."""
        prompt = build_rag_prompt("How do I reset my password?", [])
        
        assert "could not find enough information" in prompt.lower() or "don't know" in prompt.lower()
        assert "do not make up" in prompt.lower()


# ==============================================================================
# Citations Tests
# ==============================================================================

class TestCitations:
    """Tests for citation formatting."""
    
    def test_citations_contain_index(self):
        """Citations include numeric index for referencing."""
        chunks = [
            {"source_file_name": "doc1.txt", "content": "Content 1.", "score": 0.9},
            {"source_file_name": "doc2.txt", "content": "Content 2.", "score": 0.8},
        ]
        citations = format_citations(chunks)
        
        assert len(citations) == 2
        assert citations[0]["index"] == 1
        assert citations[1]["index"] == 2
    
    def test_citations_contain_source(self):
        """Citations include source file name."""
        chunks = [
            {"source_file_name": "support_faq.txt", "content": "FAQ content.", "score": 0.9},
        ]
        citations = format_citations(chunks)
        
        assert citations[0]["source_file_name"] == "support_faq.txt"
    
    def test_citations_contain_content_snippet(self):
        """Citations include content snippet."""
        chunks = [
            {"source_file_name": "doc.txt", "content": "This is the full content of the document.", "score": 0.9},
        ]
        citations = format_citations(chunks)
        
        assert "content_snippet" in citations[0]
        assert len(citations[0]["content_snippet"]) <= 203  # 200 + "..."
    
    def test_citations_include_score(self):
        """Citations include relevance score when available."""
        chunks = [
            {"source_file_name": "doc.txt", "content": "Content.", "score": 0.87},
        ]
        citations = format_citations(chunks)
        
        assert citations[0]["relevance_score"] == 0.87
    
    def test_empty_chunks_returns_empty_citations(self):
        """Empty chunk list returns empty citations list."""
        citations = format_citations([])
        assert citations == []


# ==============================================================================
# Mock Provider RAG Tests
# ==============================================================================

class TestMockProviderRag:
    """Tests for mock provider RAG behavior."""
    
    def test_mock_provider_detects_rag_prompt(self):
        """Mock provider recognizes RAG prompts with INFORMATION and USER QUESTION."""
        provider = MockProvider()
        rag_prompt = """INFORMATION:
[1] Source: test_doc.txt
This is test content about AI and machine learning.

USER QUESTION: What topics are covered?
"""
        request = ChatRequest(message=rag_prompt)
        response = provider.chat(request)
        
        assert isinstance(response, ChatResponse)
        assert response.role == MessageRole.assistant
    
    def test_mock_provider_includes_citations_in_response(self):
        """Mock provider response references the retrieved sources."""
        provider = MockProvider()
        rag_prompt = """INFORMATION:
[1] Source: faq.txt
Our support hours are 9 AM to 5 PM EST.

USER QUESTION: What are support hours?
"""
        request = ChatRequest(message=rag_prompt)
        response = provider.chat(request)
        
        # Mock should mention something about the retrieved info
        assert "faq" in response.message.lower() or "support hours" in response.message.lower() or "9 am" in response.message.lower()
    
    def test_mock_provider_handles_non_rag_prompt(self):
        """Mock provider handles regular prompts normally."""
        provider = MockProvider()
        request = ChatRequest(message="Hello!")
        response = provider.chat(request)
        
        assert isinstance(response, ChatResponse)
        assert "Hello" in response.message


# ==============================================================================
# RAG Pipeline Integration Tests
# ==============================================================================

class TestRagPipeline:
    """Integration tests for the full RAG pipeline with mock provider."""
    
    def test_generate_answer_with_rag_returns_citations(self):
        """RAG pipeline returns citations when chunks are found."""
        # Mock the retriever to return known chunks
        with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
            mock_retrieve.return_value = (
                [
                    {"source_file_name": "test.txt", "content": "Test content.", "score": 0.9},
                ],
                {}
            )
            
            answer, citations, metadata = generate_answer_with_rag("Test question?")
            
            assert isinstance(answer, str)
            assert len(answer) > 0
            assert isinstance(citations, list)
            assert len(citations) > 0
            assert citations[0]["source_file_name"] == "test.txt"
    
    def test_generate_answer_with_rag_handles_empty_chunks(self):
        """RAG pipeline handles empty retrieval gracefully."""
        with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
            mock_retrieve.return_value = ([], {})
            
            answer, citations, metadata = generate_answer_with_rag("Test question?")
            
            assert "could not find" in answer.lower() or "not enough information" in answer.lower()
            assert citations == []
    
    def test_chat_endpoint_rag_mode_returns_citations(self):
        """Chat API returns citations in RAG mode."""
        client = TestClient(app)
        
        with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
            mock_retrieve.return_value = (
                [
                    {"source_file_name": "policy.pdf", "content": "Our return policy allows 30 days.", "score": 0.95},
                ],
                {}
            )
            
            response = client.post("/api/chat", json={
                "message": "What is the return policy?",
                "mode": "rag"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["role"] == "assistant"
            assert "citations" in data
            assert data["citations"] is not None
            assert len(data["citations"]) > 0


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestErrorHandling:
    """Tests for error handling in RAG pipeline."""
    
    def test_provider_error_returns_error_message(self):
        """Provider errors are returned as part of the response, not silently hidden."""
        with patch('app.rag.answer_generator.get_llm_provider') as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = ChatResponse(
                message="[OpenRouter Error] API key invalid",
                role=MessageRole.assistant
            )
            mock_get_provider.return_value = mock_provider
            
            with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
                mock_retrieve.return_value = (
                    [{"source_file_name": "doc.txt", "content": "Content.", "score": 0.9}],
                    {}
                )
                
                answer, citations, metadata = generate_answer_with_rag("Test?")
                
                # Error message should be in the answer
                assert "[OpenRouter Error]" in answer or "Error" in answer
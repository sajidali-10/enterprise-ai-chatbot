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
            
            assert "don't have" in answer.lower() or "could not find" in answer.lower() or "not enough information" in answer.lower()
            assert citations == []
    
    def test_chat_endpoint_rag_mode_returns_citations(self):
        """Chat API returns citations in RAG mode."""
        client = TestClient(app, headers={"X-Dev-User": "admin_user"})
        
        with patch('app.api.chat.generate_answer_with_rag_audit') as mock_generate:
            mock_generate.return_value = (
                "Our return policy allows 30 days.",
                [{"index": 1, "source_file_name": "policy.pdf", "content_snippet": "Our return policy allows 30 days.", "relevance_score": 0.95}],
                {"retrieval_info": "test"}
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
    
    def test_provider_error_blocked_by_grounding_guardrail(self):
        """Provider errors are blocked by Phase 10 grounding guardrail (no citations).
        
        Phase 10 grounding blocks answers that lack citations. This is correct production
        behavior - internal provider errors should NOT be leaked to users. The grounding
        guardrail returns a safe fallback message instead.
        """
        with patch('app.rag.answer_generator.get_llm_provider') as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = ChatResponse(
                message="[OpenRouter Error] API key invalid",
                role=MessageRole.assistant
            )
            mock_get_provider.return_value = mock_provider
            
            with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
                mock_retrieve.return_value = (
                    [{"source_file_name": "doc.txt", "content": "This is a test document about testing.", "score": 0.9}],
                    {}
                )
                
                answer, citations, metadata = generate_answer_with_rag("What is testing?")
                
                # Phase 10 grounding blocks answers without citations
                # Internal provider errors are NOT leaked to users
                assert citations == []
                assert metadata.get("blocked") is True
                assert metadata.get("block_reason") == "answer_lacks_citations"
                # Safe fallback message is returned instead of exposing internal error
                assert "don't have enough information" in answer.lower() or "could not find enough information" in answer.lower()


# ==============================================================================
# Citation Repair Flow Tests (Phase 10.6)
# ==============================================================================

class TestCitationRepairFlow:
    """Tests for Phase 10.6 citation repair flow.
    
    Verifies that:
    1. Initial LLM call without citations triggers retry when retrieval is strong
    2. Backend citation attachment runs when retry fails to produce citations
    3. Fallback still happens when retrieval is weak (below threshold)
    4. Citation metadata is properly tracked through the repair flow
    """
    
    def test_strong_retrieval_retry_and_attachment_produces_citations(self):
        """Backend citation attachment runs when LLM retry still lacks citations.
        
        Given:
        - Strong retrieval (top_score 0.97)
        - LLM first response WITHOUT citations
        - LLM retry response WITHOUT citations
        
        Expected:
        - backend_citations_attached = true
        - final answer includes citations like [1], [2]
        - blocked = false
        - citation_count > 0
        """
        with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
            with patch('app.rag.answer_generator.get_llm_provider') as mock_get_provider:
                # Strong retrieval with high score
                mock_retrieve.return_value = (
                    [
                        {"id": "chunk1", "source_file_name": "docker.txt", 
                         "content": "Docker images are immutable packages that include the application code, dependencies, and OS libraries.", "score": 0.97},
                        {"id": "chunk2", "source_file_name": "docker.txt", 
                         "content": "Docker containers are runtime instances created from Docker images.", "score": 0.95},
                    ],
                    {}
                )
                
                # Mock LLM: first call returns answer without citations,
                # second call (retry) also returns answer without citations
                mock_provider = MagicMock()
                mock_provider.set_temperature = MagicMock()
                
                # Track number of calls
                call_count = [0]
                def mock_chat(request):
                    call_count[0] += 1
                    # Both calls return answer without citations
                    return MagicMock(message="Docker images are immutable packages. Docker containers are runtime instances.")
                
                mock_provider.chat.side_effect = mock_chat
                mock_get_provider.return_value = mock_provider
                
                answer, citations, metadata = generate_answer_with_rag(
                    "what are the components of docker?",
                    min_relevance_score=0.3
                )
                
                # Citation repair flow should have been triggered
                assert metadata.get("blocked") is False, f"Expected blocked=False but got blocked={metadata.get('blocked')}"
                
                # Answer should now have citations from backend attachment
                # Citations may be [1, 2] format (multiple citations on same sentence)
                import re
                citation_pattern = r'\[\d+(?:,\s*\d+)*\]'
                assert re.search(citation_pattern, answer), f"Expected citations in answer but got: {answer}"
                
                # Citation repair metadata should be present
                citation_repair = metadata.get("grounding", {}).get("citation_repair", {})
                assert citation_repair.get("retry_attempted") is True, "Retry should have been attempted"
                assert citation_repair.get("attachment_attempted") is True, "Backend attachment should have been attempted"
                assert citation_repair.get("final_has_citations") is True, "Final answer should have citations"
                assert citation_repair.get("final_citation_count", 0) > 0, "Citation count should be > 0"
    
    def test_weak_retrieval_falls_back_regardless_of_citations(self):
        """Fallback happens when retrieval is weak even if LLM produces something.
        
        Given:
        - Weak retrieval (top_score 0.2, below threshold)
        - LLM response without citations
        
        Expected:
        - Fallback message returned
        - blocked = true
        - No unsupported answer returned
        """
        with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
            with patch('app.rag.answer_generator.get_llm_provider') as mock_get_provider:
                # Weak retrieval with low score (below typical threshold)
                mock_retrieve.return_value = (
                    [
                        {"id": "chunk1", "source_file_name": "doc.txt", 
                         "content": "Some content about docker.", "score": 0.2},
                    ],
                    {}
                )
                
                mock_provider = MagicMock()
                mock_provider.chat.return_value = MagicMock(
                    message="Docker has several components including images and containers."
                )
                mock_get_provider.return_value = mock_provider
                
                answer, citations, metadata = generate_answer_with_rag(
                    "what are the components of docker?",
                    min_relevance_score=0.5  # 0.5 threshold, 0.2 is below
                )
                
                # Should block because retrieval is weak
                assert metadata.get("blocked") is True
                assert "could not find enough information" in answer.lower() or \
                       "relevant information" in answer.lower() or \
                       "low relevance" in answer.lower()
    
    def test_no_retry_when_retrieval_is_weak(self):
        """No retry attempted when retrieval score is below threshold.
        
        Given:
        - Weak retrieval (top_score 0.25, below 0.3 threshold)
        - LLM response without citations
        
        Expected:
        - No retry attempted (because top_score < min_relevance_score)
        - blocked = true
        """
        with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
            mock_retrieve.return_value = (
                [
                    {"id": "chunk1", "source_file_name": "docker.txt", 
                     "content": "Docker is a containerization platform.", "score": 0.25},
                ],
                {}
            )
            
            with patch('app.rag.answer_generator.get_llm_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_provider.chat.return_value = MagicMock(
                    message="Docker components include images and containers."
                )
                mock_get_provider.return_value = mock_provider
                
                answer, citations, metadata = generate_answer_with_rag(
                    "what are the components of docker?",
                    min_relevance_score=0.3
                )
                
                # When retrieval is weak, grounding blocks BEFORE citation repair flow runs
                # So no citation_repair metadata is initialized (not False, just absent)
                citation_repair = metadata.get("grounding", {}).get("citation_repair")
                assert citation_repair is None, \
                    f"Citation repair should not run when retrieval is weak, got: {citation_repair}"
                assert metadata.get("blocked") is True

    def test_initial_response_with_citations_skips_retry(self):
        """When initial LLM response already has citations, skip retry.
        
        Given:
        - Strong retrieval
        - LLM response WITH citations
        
        Expected:
        - No retry attempted
        - blocked = false
        - answer with citations returned
        """
        with patch('app.rag.answer_generator.retrieve_chunks_with_settings') as mock_retrieve:
            with patch('app.rag.answer_generator.get_llm_provider') as mock_get_provider:
                mock_retrieve.return_value = (
                    [
                        {"id": "chunk1", "source_file_name": "docker.txt", 
                         "content": "Docker images are immutable packages.", "score": 0.97},
                    ],
                    {}
                )
                
                mock_provider = MagicMock()
                mock_provider.set_temperature = MagicMock()
                call_count = [0]
                def mock_chat(request):
                    call_count[0] += 1
                    # LLM already includes citation
                    return MagicMock(message="Docker images are immutable packages that include everything needed [1].")
                
                mock_provider.chat.side_effect = mock_chat
                mock_get_provider.return_value = mock_provider
                
                answer, citations, metadata = generate_answer_with_rag(
                    "what are the components of docker?",
                    min_relevance_score=0.3
                )
                
                # Should only be called once (no retry needed)
                assert call_count[0] == 1, f"Expected 1 LLM call but got {call_count[0]}"
                
                citation_repair = metadata.get("grounding", {}).get("citation_repair", {})
                assert citation_repair.get("retry_attempted") is False
                assert citation_repair.get("final_has_citations") is True
                assert metadata.get("blocked") is False
"""
Tests for chat mode behavior.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestChatModes:
    """Tests for different chat modes: general_chat, knowledge_base, debug."""
    
    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)
    
    def test_general_chat_no_sources(self, client):
        """General Chat mode should not return citations or sources."""
        with patch('app.api.chat.generate_answer_without_rag') as mock_generate:
            mock_generate.return_value = "This is a general answer."
            
            response = client.post("/api/chat", json={
                "message": "What is Python?",
                "mode": "general_chat"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert data.get("citations") is None
            assert data.get("grouped_sources") is None
            assert data.get("debug_info") is None
    
    def test_general_chat_legacy_normal_mode(self, client):
        """Legacy 'normal' mode should map to general_chat."""
        with patch('app.api.chat.generate_answer_without_rag') as mock_generate:
            mock_generate.return_value = "General answer."
            
            response = client.post("/api/chat", json={
                "message": "Hello?",
                "mode": "normal"  # Legacy mode
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data.get("citations") is None
    
    def test_knowledge_base_has_sources(self, client):
        """Knowledge Base mode should return citations and grouped sources."""
        with patch('app.api.chat.generate_answer_with_rag') as mock_generate:
            mock_generate.return_value = (
                "This is a RAG answer.",
                [{"index": 1, "source_file_name": "doc.pdf", "content_snippet": "Snippet", "relevance_score": 0.8}],
                {}
            )
            
            response = client.post("/api/chat", json={
                "message": "What is in the document?",
                "mode": "knowledge_base"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert data.get("citations") is not None
            assert data.get("grouped_sources") is not None
    
    def test_knowledge_base_legacy_rag_mode(self, client):
        """Legacy 'rag' mode should map to knowledge_base."""
        with patch('app.api.chat.generate_answer_with_rag') as mock_generate:
            mock_generate.return_value = (
                "RAG answer.",
                [{"index": 1, "source_file_name": "doc.pdf", "content_snippet": "Snippet", "relevance_score": 0.8}],
                {}
            )
            
            response = client.post("/api/chat", json={
                "message": "Question?",
                "mode": "rag"  # Legacy mode
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data.get("citations") is not None
    
    def test_knowledge_base_no_debug_info_default(self, client):
        """Knowledge Base mode without debug flag should not return debug info."""
        with patch('app.api.chat.generate_answer_with_rag') as mock_generate:
            mock_generate.return_value = (
                "RAG answer.",
                [{"index": 1, "source_file_name": "doc.pdf", "content_snippet": "Snippet", "relevance_score": 0.8}],
                {"retrieval_info": "some data"}
            )
            
            response = client.post("/api/chat", json={
                "message": "Question?",
                "mode": "knowledge_base"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data.get("debug_info") is None
    
    def test_debug_mode_requires_admin(self, client):
        """Debug mode should fall back to knowledge_base for non-admin users."""
        with patch('app.api.chat.HAS_SECURITY', True):
            with patch('app.api.chat.authenticate_request') as mock_auth:
                mock_auth_context = MagicMock()
                mock_auth_context.is_authenticated = True
                mock_auth_context.is_admin.return_value = False
                mock_auth.return_value = mock_auth_context
                
                with patch('app.api.chat.generate_answer_with_rag_audit') as mock_generate:
                    mock_generate.return_value = (
                        "RAG answer (not debug).",
                        [{"index": 1, "source_file_name": "doc.pdf", "content_snippet": "Snippet", "relevance_score": 0.8}],
                        {"retrieval_info": "data"}
                    )
                    
                    response = client.post("/api/chat", json={
                        "message": "Question?",
                        "mode": "debug"
                    }, headers={"X-Dev-User": "regular_user"})
                    
                    assert response.status_code == 200
                    # Should get answer but debug_info should be None (fallback to knowledge_base)
                    data = response.json()
                    assert data.get("debug_info") is None
    
    def test_debug_mode_returns_debug_info_for_admin(self, client):
        """Debug mode should return debug info for admin users."""
        with patch('app.api.chat.HAS_SECURITY', True):
            with patch('app.api.chat.authenticate_request') as mock_auth:
                mock_auth_context = MagicMock()
                mock_auth_context.is_authenticated = True
                mock_auth_context.is_admin.return_value = True
                mock_auth.return_value = mock_auth_context
                
                with patch('app.api.chat.generate_answer_with_rag_audit') as mock_generate:
                    mock_generate.return_value = (
                        "RAG answer with debug.",
                        [{"index": 1, "source_file_name": "doc.pdf", "content_snippet": "Snippet", "relevance_score": 0.8}],
                        {"retrieval_method": "hybrid", "chunks_retrieved": 5}
                    )
                    
                    response = client.post("/api/chat", json={
                        "message": "Question?",
                        "mode": "debug"
                    }, headers={"X-Dev-User": "admin_user"})
                    
                    print(f"DEBUG Response: {response.status_code} - {response.text}")
                    assert response.status_code == 200
                    data = response.json()
                    # Should have debug_info for admin in debug mode
                    assert data.get("debug_info") is not None


class TestChatModeEnum:
    """Tests for ChatMode enum values."""
    
    def test_enum_values(self):
        from app.api.chat import ChatMode
        
        assert ChatMode.GENERAL_CHAT == "general_chat"
        assert ChatMode.KNOWLEDGE_BASE == "knowledge_base"
        assert ChatMode.DEBUG == "debug"
    
    def test_enum_comparison(self):
        from app.api.chat import ChatMode
        
        assert ChatMode.GENERAL_CHAT == "general_chat"
        assert ChatMode.KNOWLEDGE_BASE == "knowledge_base"
        assert ChatMode.DEBUG == "debug"


class TestKnowledgeBaseModeDisplay:
    """Tests for Knowledge Base mode display behavior."""
    
    def test_knowledge_base_hides_debug_details(self, client):
        """Knowledge Base mode should hide debug details (indices) but keep score internal."""
        with patch('app.api.chat.generate_answer_with_rag') as mock_generate:
            mock_generate.return_value = (
                "Docker images contain base images, application code, and dependencies.",
                [{"index": 1, "source_file_name": "docker.pdf", "content_snippet": "Base image content", "relevance_score": 0.85}],
                {"retrieval_info": "test"}
            )
            
            response = client.post("/api/chat", json={
                "message": "What are Docker image components?",
                "mode": "knowledge_base"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "grouped_sources" in data
            assert data["grouped_sources"] is not None
            
            for source in data["grouped_sources"]:
                # Score is always included for sorting, but show_debug_details flag controls display
                assert source.get("highest_score") is not None
                # Indices should be empty in knowledge_base mode
                assert source.get("indices") == []
                # show_debug_details should be False
                assert source.get("show_debug_details") is False
    
    def test_debug_mode_shows_raw_scores_and_indices(self, client):
        """Debug mode should show raw scores and indices for admin users."""
        with patch('app.api.chat.HAS_SECURITY', True):
            with patch('app.api.chat.authenticate_request') as mock_auth:
                mock_auth_context = MagicMock()
                mock_auth_context.is_authenticated = True
                mock_auth_context.is_admin.return_value = True
                mock_auth.return_value = mock_auth_context
                
                with patch('app.api.chat.generate_answer_with_rag_audit') as mock_generate:
                    mock_generate.return_value = (
                        "Docker images contain base images, application code, and dependencies.",
                        [
                            {"index": 1, "source_file_name": "docker.pdf", "content_snippet": "Base image content", "relevance_score": 0.85},
                            {"index": 2, "source_file_name": "docker.pdf", "content_snippet": "Layers content", "relevance_score": 0.75},
                        ],
                        {"retrieval_info": "test"}
                    )
                    
                    response = client.post("/api/chat", json={
                        "message": "What are Docker image components?",
                        "mode": "debug"
                    }, headers={"X-Dev-User": "admin_user"})
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "grouped_sources" in data
                    
                    for source in data["grouped_sources"]:
                        # Raw score should be present in debug mode
                        assert source.get("highest_score") is not None
                        assert isinstance(source["highest_score"], float)
                        # Indices should be populated
                        assert len(source.get("indices", [])) > 0
                        # show_debug_details should be True
                        assert source.get("show_debug_details") is True


class TestExcerptRelevanceSelection:
    """Tests for excerpt relevance selection in different scenarios."""
    
    def test_docker_components_prefers_specific_chunks(self, client):
        """Docker component questions should prefer image/base image chunks over general intro."""
        # Import and test the excerpt selection logic directly since mocking bypasses it
        from app.rag.citations import select_best_excerpts_for_source
        
        # Simulate citations with a mix of generic intro and specific component content
        citations = [
            {
                "index": 1,
                "source_file_name": "docker.pdf",
                "content_snippet": "What is Docker? Docker is a platform designed to help developers.",
                "relevance_score": 0.75
            },
            {
                "index": 2,
                "source_file_name": "docker.pdf",
                "content_snippet": "Why do we need Docker? Consistency Across Environments...",
                "relevance_score": 0.72
            },
            {
                "index": 3,
                "source_file_name": "docker.pdf",
                "content_snippet": "Docker image layers include base images like Alpine or Ubuntu.",
                "relevance_score": 0.80
            },
            {
                "index": 4,
                "source_file_name": "docker.pdf",
                "content_snippet": "Image metadata includes environment variables and exposed ports.",
                "relevance_score": 0.78
            },
        ]
        
        question = "Components of a Docker Image?"
        answer = "Docker images consist of base images, application code, dependencies, and metadata."
        
        # Test that answer-aware selection works
        selected = select_best_excerpts_for_source(citations, question, answer, max_excerpts=3)
        
        assert len(selected) == 3  # Should select top 3 after dedup
        
        # Get selected content
        selected_content = " ".join([c["content_snippet"].lower() for c in selected])
        
        # Should contain specific content about images/components
        assert "image" in selected_content or "base" in selected_content or "alpine" in selected_content
        
        # The specific content should be ranked higher than generic intro
        # First selected should be one of the specific ones (indices 3 or 4)
        first_content = selected[0]["content_snippet"].lower()
        assert "alpine" in first_content or "metadata" in first_content or "environment" in first_content

    def test_sections_used_shows_total_but_excerpts_limited(self, client):
        """sections_used should show total chunks, but excerpts should be limited to best 3."""
        with patch('app.api.chat.generate_answer_with_rag') as mock_generate:
            # More than 3 citations from same source
            mock_generate.return_value = (
                "Answer text.",
                [
                    {"index": i, "source_file_name": "doc.pdf", "content_snippet": f"Content {i}", "relevance_score": 0.5 + i * 0.05}
                    for i in range(1, 6)  # 5 citations
                ],
                {}
            )
            
            response = client.post("/api/chat", json={
                "message": "Test question?",
                "mode": "knowledge_base"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            for source in data.get("grouped_sources", []):
                # sections_used should show total (5)
                assert source["sections_used"] == 5
                # But excerpts should be at most 3
                assert len(source["excerpts"]) <= 3

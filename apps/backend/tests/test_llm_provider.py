"""
Tests for LLM Provider Abstraction (Phase 7)

Tests provider selection, fallback behavior, and provider interface.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ==============================================================================
# Provider Selection Tests
# ==============================================================================

def test_llm_provider_mock_selected_by_default():
    """When LLM_PROVIDER is not set, mock provider is used."""
    with patch.dict(os.environ, {}, clear=True):
        from app.services.llm import get_llm_provider
        from app.services.llm.mock_provider import MockProvider
        
        # Force reimport to pick up env
        import importlib
        import app.services.llm
        importlib.reload(app.services.llm)
        
        provider = get_llm_provider()
        assert isinstance(provider, MockProvider)


def test_llm_provider_openai_selected_when_configured():
    """When LLM_PROVIDER=openai, OpenAI provider is used."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
        import importlib
        import app.services.llm
        importlib.reload(app.services.llm)
        
        from app.services.llm.openai_compatible_provider import OpenAICompatibleProvider
        provider = app.services.llm.get_llm_provider()
        assert isinstance(provider, OpenAICompatibleProvider)


def test_llm_provider_mock_explicitly_set():
    """When LLM_PROVIDER=mock, mock provider is used."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "mock"}):
        import importlib
        import app.services.llm
        importlib.reload(app.services.llm)
        
        from app.services.llm.mock_provider import MockProvider
        provider = app.services.llm.get_llm_provider()
        assert isinstance(provider, MockProvider)


def test_llm_provider_case_insensitive():
    """Provider selection is case insensitive."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "OPENAI"}):
        import importlib
        import app.services.llm
        importlib.reload(app.services.llm)
        
        from app.services.llm.openai_compatible_provider import OpenAICompatibleProvider
        provider = app.services.llm.get_llm_provider()
        assert isinstance(provider, OpenAICompatibleProvider)


def test_llm_provider_openrouter_selected():
    """When LLM_PROVIDER=openrouter, OpenRouter provider is used."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}):
        import importlib
        import app.services.llm
        importlib.reload(app.services.llm)
        
        from app.services.llm.openrouter_provider import OpenRouterProvider
        provider = app.services.llm.get_llm_provider()
        assert isinstance(provider, OpenRouterProvider)


def test_llm_provider_ollama_selected():
    """When LLM_PROVIDER=ollama, Ollama provider is used."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
        import importlib
        import app.services.llm
        importlib.reload(app.services.llm)
        
        from app.services.llm.ollama_provider import OllamaProvider
        provider = app.services.llm.get_llm_provider()
        assert isinstance(provider, OllamaProvider)


def test_llm_provider_litellm_selected():
    """When LLM_PROVIDER=litellm, LiteLLM provider is used."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "litellm"}):
        import importlib
        import app.services.llm
        importlib.reload(app.services.llm)
        
        from app.services.llm.litellm_provider import LiteLLMProvider
        provider = app.services.llm.get_llm_provider()
        assert isinstance(provider, LiteLLMProvider)


def test_llm_provider_fallback_to_mock_for_unknown():
    """Unknown provider names fall back to mock."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_provider"}):
        import importlib
        import app.services.llm
        importlib.reload(app.services.llm)
        
        from app.services.llm.mock_provider import MockProvider
        provider = app.services.llm.get_llm_provider()
        assert isinstance(provider, MockProvider)


# ==============================================================================
# Provider Interface Tests
# ==============================================================================

def test_mock_provider_returns_chat_response():
    """Mock provider returns ChatResponse with assistant role."""
    from app.services.llm.mock_provider import MockProvider
    from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
    
    provider = MockProvider()
    request = ChatRequest(message="Hello")
    response = provider.chat(request)
    
    assert isinstance(response, ChatResponse)
    assert response.role == MessageRole.assistant
    assert len(response.message) > 0


def test_mock_provider_handles_rag_prompt():
    """Mock provider recognizes RAG prompts with INFORMATION and USER QUESTION."""
    from app.services.llm.mock_provider import MockProvider
    from app.schemas.chat import ChatRequest, ChatResponse
    
    provider = MockProvider()
    rag_prompt = """INFORMATION:
[1] Source: test_doc.txt
This is test content about AI and machine learning.

USER QUESTION: What topics are covered?
"""
    request = ChatRequest(message=rag_prompt)
    response = provider.chat(request)
    
    assert isinstance(response, ChatResponse)
    # Mock RAG response should contain the question
    assert "machine learning" in response.message.lower() or "test" in response.message.lower()


def test_mock_provider_handles_question():
    """Mock provider responds to questions."""
    from app.services.llm.mock_provider import MockProvider
    from app.schemas.chat import ChatRequest
    
    provider = MockProvider()
    request = ChatRequest(message="What is 2+2?")
    response = provider.chat(request)
    
    assert len(response.message) > 0


def test_mock_provider_handles_greeting():
    """Mock provider has special greeting response."""
    from app.services.llm.mock_provider import MockProvider
    from app.schemas.chat import ChatRequest
    
    provider = MockProvider()
    request = ChatRequest(message="hello")
    response = provider.chat(request)
    
    assert "Hello" in response.message


# ==============================================================================
# Chat API Integration Tests
# ==============================================================================

def test_chat_endpoint_works_with_mock_provider():
    """Chat endpoint works when mock provider is configured."""
    # Mock provider is default in tests
    client = TestClient(app, headers={"X-Dev-User": "admin_user"})
    response = client.post("/api/chat", json={"message": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert len(data["message"]) > 0


def test_chat_endpoint_rejects_empty_message():
    """Chat endpoint validates empty message."""
    client = TestClient(app, headers={"X-Dev-User": "admin_user"})
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_endpoint_rag_mode():
    """Chat endpoint works in RAG mode."""
    client = TestClient(app, headers={"X-Dev-User": "admin_user"})
    response = client.post("/api/chat", json={
        "message": "What topics are covered?",
        "mode": "rag"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"


# Import app for tests
from app.main import app
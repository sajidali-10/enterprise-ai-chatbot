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


# ==============================================================================
# LiteLLM Gateway Provider Tests (Phase 15)
# ==============================================================================

from unittest.mock import patch, MagicMock

from app.services.llm.litellm_provider import LiteLLMProvider
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole


def test_litellm_provider_builds_correct_openai_request():
    """
    LiteLLMProvider calls the LiteLLM Gateway OpenAI-compatible endpoint
    with the correct URL, headers, and payload structure.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = LiteLLMProvider(
            model="openrouter-gpt-oss",
            base_url="http://litellm:4000",
            master_key="test-master-key",
            timeout=60.0,
        )
        request = ChatRequest(message="Hello")
        response = provider.chat(request)

        # Verify POST was called
        assert mock_client.post.called
        call_kwargs = mock_client.post.call_args

        # Verify URL is OpenAI-compatible
        url = call_kwargs[0][0]
        assert url == "http://litellm:4000/v1/chat/completions"

        # Verify headers include Bearer auth
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer test-master-key"
        assert headers["Content-Type"] == "application/json"
        # Note: str(call_kwargs) will include the key in the headers dict — this is
        # expected (the key IS sent in the request). The real secrets-leak check is
        # in test_litellm_provider_does_not_log_secrets which captures actual log output.

        # Verify payload
        payload = call_kwargs[1]["json"]
        assert payload["model"] == "openrouter-gpt-oss"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

        # Verify response
        assert isinstance(response, ChatResponse)
        assert response.message == "Test response"
        assert response.role == MessageRole.assistant


def test_litellm_provider_handles_missing_master_key():
    """
    When LITELLM_MASTER_KEY is not set, LiteLLMProvider returns a
    clear fallback message — not a stack trace.
    """
    with patch.dict(os.environ, {"LITELLM_MASTER_KEY": ""}, clear=False):
        with patch.dict(os.environ, {"LITELLM_BASE_URL": "http://litellm:4000"}, clear=False):
            with patch.dict(os.environ, {"LITELLM_MODEL": "openrouter-gpt-oss"}, clear=False):
                import importlib
                import app.services.llm.litellm_provider
                importlib.reload(app.services.llm.litellm_provider)

                provider = app.services.llm.litellm_provider.LiteLLMProvider()
                request = ChatRequest(message="Hello")
                response = provider.chat(request)

                assert "[LiteLLM Gateway]" in response.message
                assert "master key not configured" in response.message
                # Should NOT raise an exception
                assert isinstance(response, ChatResponse)


def test_litellm_provider_handles_connection_error():
    """
    When the LiteLLM Gateway is unreachable, LiteLLMProvider returns a
    graceful fallback — not a stack trace.
    """
    import httpx

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = mock_client

        provider = LiteLLMProvider(
            model="openrouter-gpt-oss",
            base_url="http://litellm:4000",
            master_key="test-master-key",
        )
        request = ChatRequest(message="Hello")
        response = provider.chat(request)

        assert "[LiteLLM Gateway]" in response.message
        assert "unreachable" in response.message
        assert isinstance(response, ChatResponse)


def test_litellm_provider_handles_http_404():
    """
    When the model is not found (404), LiteLLMProvider returns a
    clear fallback message — not a stack trace.
    """
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"error": {"message": "Model not found"}}

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_response.raise_for_status = MagicMock()  # no-op on 404 test
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = LiteLLMProvider(
            model="nonexistent-model",
            base_url="http://litellm:4000",
            master_key="test-master-key",
        )
        request = ChatRequest(message="Hello")
        response = provider.chat(request)

        assert "[LiteLLM Gateway]" in response.message
        assert "not found" in response.message.lower()
        assert isinstance(response, ChatResponse)


def test_litellm_provider_handles_http_401():
    """
    When auth fails (401/403), LiteLLMProvider returns a clear fallback
    message — not a stack trace or secret exposure.
    """
    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = LiteLLMProvider(
            model="openrouter-gpt-oss",
            base_url="http://litellm:4000",
            master_key="wrong-key",
        )
        request = ChatRequest(message="Hello")
        response = provider.chat(request)

        assert "[LiteLLM Gateway]" in response.message
        assert "authentication failed" in response.message.lower()
        # Verify master key is NOT in the response message
        assert "wrong-key" not in response.message
        assert isinstance(response, ChatResponse)


def test_litellm_provider_handles_rate_limit_429():
    """
    When rate limited (429), LiteLLMProvider returns a clear fallback
    message — not a stack trace.
    """
    mock_response = MagicMock()
    mock_response.status_code = 429

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = LiteLLMProvider(
            model="openrouter-gpt-oss",
            base_url="http://litellm:4000",
            master_key="test-master-key",
        )
        request = ChatRequest(message="Hello")
        response = provider.chat(request)

        assert "[LiteLLM Gateway]" in response.message
        assert "rate limit" in response.message.lower()
        assert isinstance(response, ChatResponse)


def test_litellm_provider_handles_timeout():
    """
    When request times out, LiteLLMProvider returns a clear fallback
    message — not a stack trace.
    """
    import httpx

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
        mock_client_cls.return_value = mock_client

        provider = LiteLLMProvider(
            model="openrouter-gpt-oss",
            base_url="http://litellm:4000",
            master_key="test-master-key",
            timeout=10.0,
        )
        request = ChatRequest(message="Hello")
        response = provider.chat(request)

        assert "[LiteLLM Gateway]" in response.message
        assert "timed out" in response.message.lower()
        assert isinstance(response, ChatResponse)


def test_litellm_provider_does_not_log_secrets():
    """
    LiteLLMProvider must never log the master key or API key.
    We verify by checking that error messages don't contain secrets.
    """
    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        secret_key = "super-secret-master-key-12345"
        provider = LiteLLMProvider(
            model="openrouter-gpt-oss",
            base_url="http://litellm:4000",
            master_key=secret_key,
        )
        request = ChatRequest(message="Hello")

        # Capture any logged output
        import logging
        import io

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("app.services.llm.litellm_provider")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        try:
            response = provider.chat(request)
        finally:
            logger.removeHandler(handler)

        log_output = log_capture.getvalue()
        response_output = response.message

        # Verify secrets are NOT in logs or response
        assert secret_key not in log_output, f"Master key leaked to logs: {log_output}"
        assert secret_key not in response_output, f"Master key leaked in response: {response_output}"


def test_litellm_provider_repr_includes_model_only():
    """
    LiteLLMProvider __repr__ includes model and base_url but never secrets.
    """
    provider = LiteLLMProvider(
        model="openrouter-gpt-oss",
        base_url="http://litellm:4000",
        master_key="super-secret",
    )
    repr_str = repr(provider)
    assert "openrouter-gpt-oss" in repr_str
    assert "litellm" in repr_str
    assert "super-secret" not in repr_str


# ==============================================================================
# Provider Info Endpoint Tests (Phase 15)
# ==============================================================================


def test_provider_info_endpoint_returns_no_secrets():
    """
    /api/health/provider must never expose API keys, master keys,
    or full URLs with credentials.
    """
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "litellm",
        "LITELLM_MODEL": "openrouter-gpt-oss",
        "LITELLM_BASE_URL": "http://litellm:4000",
        "LITELLM_MASTER_KEY": "super-secret-master-key",
        "LITELLM_ENABLED": "true",
    }, clear=False):
        client = TestClient(app)
        response = client.get("/api/health/provider")
        assert response.status_code == 200
        data = response.json()

        # Must include expected fields
        assert "provider" in data
        assert "model" in data
        assert "base_url_host" in data
        assert "gateway_mode" in data
        assert "litellm_enabled" in data

        # Secrets must NOT be present
        response_str = str(data)
        assert "super-secret" not in response_str
        assert "master-key" not in response_str.lower()

        # base_url_host must be just a hostname, not a full URL
        assert "://" not in data["base_url_host"]
        assert "@" not in data["base_url_host"]  # no auth info


def test_provider_info_endpoint_openrouter_mode():
    """
    /api/health/provider returns correct fields for direct OpenRouter mode.
    """
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_MODEL": "google/gemini-2.0-flash-exp",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
    }, clear=False):
        client = TestClient(app)
        response = client.get("/api/health/provider")
        assert response.status_code == 200
        data = response.json()

        assert data["provider"] == "openrouter"
        assert data["model"] == "google/gemini-2.0-flash-exp"
        assert data["base_url_host"] == "openrouter.ai"
        assert data["gateway_mode"] is False


def test_provider_info_endpoint_litellm_gateway_mode():
    """
    /api/health/provider returns correct fields for LiteLLM Gateway mode.
    """
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "litellm",
        "LITELLM_MODEL": "openrouter-gpt-oss",
        "LITELLM_BASE_URL": "http://litellm:4000",
        "LITELLM_ENABLED": "true",
    }, clear=False):
        client = TestClient(app)
        response = client.get("/api/health/provider")
        assert response.status_code == 200
        data = response.json()

        assert data["provider"] == "litellm"
        assert data["model"] == "openrouter-gpt-oss"
        assert data["base_url_host"] == "litellm"
        assert data["gateway_mode"] is True
        assert data["litellm_enabled"] is True


# Import app for tests
from app.main import app
"""
LLM Service

Provider abstraction layer for LLM access.
Supports multiple providers with configuration-driven selection.

Usage:
    from app.services.llm import get_llm_provider
    
    provider = get_llm_provider()  # Selects based on LLM_PROVIDER env var
    response = provider.chat(request)

Provider Selection (via LLM_PROVIDER env var):
- "mock" (default): MockProvider - returns fake responses, no API key needed
- "openai": OpenAICompatibleProvider - OpenAI API
- "openrouter": OpenRouterProvider - OpenRouter unified API
- "ollama": OllamaProvider - local Ollama inference
- "litellm": LiteLLMProvider - unified interface to 100+ LLMs

Unknown values fall back to "mock".
"""

import os
from app.services.llm.base import LlmProvider
from app.services.llm.mock_provider import MockProvider
from app.services.llm.openai_compatible_provider import OpenAICompatibleProvider


# Lazy imports to avoid ImportError for optional providers
def _get_openrouter_provider() -> LlmProvider:
    from app.services.llm.openrouter_provider import OpenRouterProvider
    return OpenRouterProvider()


def _get_ollama_provider() -> LlmProvider:
    from app.services.llm.ollama_provider import OllamaProvider
    return OllamaProvider()


def _get_litellm_provider() -> LlmProvider:
    from app.services.llm.litellm_provider import LiteLLMProvider
    return LiteLLMProvider()


# Provider registry
_PROVIDERS = {
    "mock": lambda: MockProvider(),
    "openai": lambda: OpenAICompatibleProvider(),
    "openrouter": _get_openrouter_provider,
    "ollama": _get_ollama_provider,
    "litellm": _get_litellm_provider,
}


def get_llm_provider() -> LlmProvider:
    """
    Get the configured LLM provider based on LLM_PROVIDER environment variable.
    
    Returns:
        LlmProvider instance for the configured provider.
        
    Raises:
        No exceptions - falls back to MockProvider for any error.
    
    Example:
        # Use default (mock)
        provider = get_llm_provider()
        
        # Use OpenAI
        os.environ["LLM_PROVIDER"] = "openai"
        provider = get_llm_provider()
    """
    provider_name = os.getenv("LLM_PROVIDER", "mock").lower().strip()
    
    # Get provider constructor, default to mock
    provider_fn = _PROVIDERS.get(provider_name, _PROVIDERS["mock"])
    
    try:
        return provider_fn()
    except Exception as e:
        # Any provider initialization error falls back to mock
        import logging
        logging.warning(f"Failed to initialize LLM provider '{provider_name}': {e}. Using mock.")
        return MockProvider()


__all__ = [
    "LlmProvider",
    "MockProvider",
    "OpenAICompatibleProvider",
    "get_llm_provider",
]
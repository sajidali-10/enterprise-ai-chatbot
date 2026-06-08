"""
LiteLLM Provider

Unified interface to 100+ LLMs through LiteLLM library.
Supports OpenAI, Anthropic, Azure, OpenRouter, Ollama, and more.

Configuration:
- LLM_PROVIDER=litellm
- LITELLM_MODEL=e.g., "gpt-4o-mini", "claude-3-haiku", "openrouter/anthropic/claude-3-haiku"
- LITELLM_API_KEY=your-api-key (if not using Ollama local)
- LITELLM_BASE_URL=override-endpoint (optional)
- LITELLM_API_BASE=alternate-endpoint (optional, for proxies)
"""

import os
from typing import Optional

from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.base import LlmProvider


class LiteLLMProvider(LlmProvider):
    """
    LiteLLM-based provider for unified LLM access.
    
    Supports any model that LiteLLM supports:
    - OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.
    - Anthropic: claude-3-5-sonnet, claude-3-opus, claude-3-haiku
    - Azure: azure/gpt-4o
    - OpenRouter: openrouter/anthropic/claude-3-haiku
    - Ollama: ollama/llama3.1
    - And 100+ more providers
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    ):
        self.model = model or os.getenv("LITELLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "")
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL", "")
        self.api_base = api_base or os.getenv("LITELLM_API_BASE", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_kwargs = kwargs
        
        # Check if litellm is available
        self._litellm_available = self._check_litellm()
    
    def _check_litellm(self) -> bool:
        """Check if litellm package is installed."""
        try:
            import litellm
            return True
        except ImportError:
            return False
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request through LiteLLM.
        
        Falls back to mock response if:
        - LiteLLM not installed
        - API key not configured (and not using local Ollama)
        - Request fails
        """
        # Fallback if litellm not installed
        if not self._litellm_available:
            return self._mock_fallback("LiteLLM not installed. Install with: pip install litellm")
        
        # Fallback if no API key and not using local provider
        if not self.api_key and not self._is_local_provider():
            return self._mock_fallback(
                f"LITELLM_API_KEY not set and model '{self.model}' is not a local provider."
            )
        
        try:
            import litellm
            
            # Build litellm arguments
            litellm_args = {
                "model": self.model,
                "messages": [{"role": "user", "content": request.message}],
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            
            # Add optional params if set
            if self.api_key:
                litellm_args["api_key"] = self.api_key
            if self.base_url:
                litellm_args["base_url"] = self.base_url
            elif self.api_base:
                litellm_args["api_base"] = self.api_base
            
            # Add any extra kwargs (temperature, max_tokens, etc.)
            litellm_args.update(self.extra_kwargs)
            
            # Make the call
            response = litellm.completion(**litellm_args)
            
            # Extract content from response
            content = response.choices[0].message.content
            return ChatResponse(message=content, role=MessageRole.assistant)
            
        except Exception as e:
            return self._mock_fallback(f"LiteLLM error: {str(e)}")
    
    def _is_local_provider(self) -> bool:
        """Check if model is a local provider (doesn't need API key)."""
        local_prefixes = ["ollama/", "ollama/", "local/", "localhost:"]
        return any(self.model.lower().startswith(p) for p in local_prefixes)
    
    def _mock_fallback(self, error_message: str) -> ChatResponse:
        """Return a mock response when LiteLLM cannot be used."""
        return ChatResponse(
            message=f"[LiteLLM Fallback] {error_message}. Using mock response instead.",
            role=MessageRole.assistant,
        )
    
    def __repr__(self) -> str:
        return f"LiteLLMProvider(model={self.model!r})"
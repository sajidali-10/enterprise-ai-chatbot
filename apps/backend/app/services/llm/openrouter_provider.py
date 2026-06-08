"""
OpenRouter Provider

Direct OpenRouter API integration for accessing 100+ LLMs through a unified API.

Configuration:
- LLM_PROVIDER=openrouter
- OPENROUTER_API_KEY=your-api-key (required)
- OPENROUTER_MODEL=model-name (default: google/gemini-2.0-flash-exp)
- OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 (default)
- OPENROUTER_SITE_URL=your-site-url (optional, for ranking)
- OPENROUTER_SITE_NAME=your-site-name (optional)
"""

import os
from typing import Optional

import httpx

from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.base import LlmProvider


class OpenRouterProvider(LlmProvider):
    """
    OpenRouter API provider.
    
    OpenRouter provides unified access to 100+ LLMs from various providers
    including OpenAI, Anthropic, Google, Meta, Mistral, and more.
    
    See available models: https://openrouter.ai/models
    """
    
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "google/gemini-2.0-flash-exp"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or os.getenv("OPENROUTER_MODEL", self.DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_kwargs = kwargs
        
        # Determine if configured
        self._configured = bool(self.api_key)
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to OpenRouter API.
        
        Falls back to mock response if:
        - OPENROUTER_API_KEY not set
        - Request fails
        """
        if not self._configured:
            return self._mock_fallback(
                "OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys"
            )
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.message}],
        }
        
        # Add any extra params (temperature, max_tokens, etc.)
        payload.update(self.extra_kwargs)
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return ChatResponse(message=content, role=MessageRole.assistant)
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return self._mock_fallback("OpenRouter API key is invalid or expired.")
            elif e.response.status_code == 402:
                return self._mock_fallback("OpenRouter credit limit exceeded.")
            elif e.response.status_code == 429:
                return self._mock_fallback("OpenRouter rate limit exceeded. Try again later.")
            else:
                return self._mock_fallback(f"OpenRouter HTTP error {e.response.status_code}: {str(e)}")
        except Exception as e:
            return self._mock_fallback(f"OpenRouter error: {str(e)}")
    
    def _mock_fallback(self, error_message: str) -> ChatResponse:
        """Return a mock response when OpenRouter cannot be used."""
        return ChatResponse(
            message=f"[OpenRouter Fallback] {error_message}. Using mock response instead.",
            role=MessageRole.assistant,
        )
    
    def __repr__(self) -> str:
        return f"OpenRouterProvider(model={self.model!r})"
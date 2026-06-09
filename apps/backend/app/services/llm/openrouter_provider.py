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

Recommended low-cost models:
- google/gemini-2.0-flash-exp (fastest, cheapest)
- anthropic/claude-3-haiku (good quality)
- meta-llama/llama-3-8b-instruct (open source)
- mistralai/mistral-7b-instruct (balanced)
"""

import logging
import os
from typing import Optional

import httpx

from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.base import LlmProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(LlmProvider):
    """
    OpenRouter API provider.
    
    OpenRouter provides unified access to 100+ LLMs from various providers
    including OpenAI, Anthropic, Google, Meta, Mistral, and more.
    
    See available models: https://openrouter.ai/models
    
    Safe fallback behavior:
    - If OPENROUTER_API_KEY is not set, returns clear error message (not silent)
    - If API call fails, returns error message in dev mode
    - Never silently hides errors in production
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
        self.provider_name = "openrouter"
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to OpenRouter API.
        
        Safe fallback behavior:
        - If OPENROUTER_API_KEY not set: returns clear error message
        - On API errors (401, 402, 429, etc.): returns error with details
        - On network errors: logs error and returns message
        """
        if not self._configured:
            error_msg = (
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY environment variable. "
                "Get your key at: https://openrouter.ai/keys"
            )
            logger.warning(f"OpenRouter: {error_msg}")
            return ChatResponse(
                message=f"[OpenRouter Not Configured] {error_msg}",
                role=MessageRole.assistant,
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
                
                if response.status_code == 401:
                    error_msg = "OpenRouter API key is invalid or expired."
                    logger.error(f"OpenRouter: {error_msg}")
                    return ChatResponse(
                        message=f"[OpenRouter Error] {error_msg} Please check your OPENROUTER_API_KEY.",
                        role=MessageRole.assistant,
                    )
                elif response.status_code == 402:
                    error_msg = "OpenRouter credit limit exceeded."
                    logger.error(f"OpenRouter: {error_msg}")
                    return ChatResponse(
                        message=f"[OpenRouter Error] {error_msg} Please add credits at: https://openrouter.ai/credits",
                        role=MessageRole.assistant,
                    )
                elif response.status_code == 429:
                    error_msg = "OpenRouter rate limit exceeded."
                    logger.warning(f"OpenRouter: {error_msg}")
                    return ChatResponse(
                        message=f"[OpenRouter Error] {error_msg} Please wait and try again.",
                        role=MessageRole.assistant,
                    )
                
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return ChatResponse(message=content, role=MessageRole.assistant)
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {str(e)}"
            logger.error(f"OpenRouter: {error_msg}")
            return ChatResponse(
                message=f"[OpenRouter Error] {error_msg}",
                role=MessageRole.assistant,
            )
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"OpenRouter: {error_msg}")
            return ChatResponse(
                message=f"[OpenRouter Error] {error_msg}",
                role=MessageRole.assistant,
            )
    
    def __repr__(self) -> str:
        return f"OpenRouterProvider(model={self.model!r})"
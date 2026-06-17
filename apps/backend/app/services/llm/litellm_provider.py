"""
LiteLLM Gateway Provider (Phase 15)

Calls a self-hosted LiteLLM Gateway via its OpenAI-compatible HTTP API.
This provider does NOT use the litellm Python library — it uses httpx directly.

Configuration (environment variables):
- LLM_PROVIDER=litellm
- LITELLM_BASE_URL: Gateway base URL (e.g., http://litellm:4000)
- LITELLM_MODEL: Model name as defined in LiteLLM config (e.g., openrouter-gpt-oss)
- LITELLM_MASTER_KEY: Gateway master key for authentication
- LITELLM_TIMEOUT: Optional request timeout in seconds (default: 120)

The gateway's OpenAI-compatible endpoint is:
  {LITELLM_BASE_URL}/v1/chat/completions

Available models (defined in infra/litellm/config.yaml):
- openrouter-gpt-oss    → google/gemini-2.0-flash-exp via OpenRouter
- openrouter-claude     → anthropic/claude-3-haiku via OpenRouter
- openrouter-llama      → meta-llama/llama-3-8b-instruct via OpenRouter
- openrouter-free       → mistralai/mistral-7b-instruct via OpenRouter
- local-ollama          → llama3.2 via local Ollama

To switch back to direct OpenRouter: set LLM_PROVIDER=openrouter (no gateway needed).
"""

import logging
import os
from typing import Optional

import httpx

from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.base import LlmProvider

logger = logging.getLogger(__name__)


class LiteLLMProvider(LlmProvider):
    """
    LiteLLM Gateway provider — calls the self-hosted LiteLLM proxy
    using the OpenAI-compatible /v1/chat/completions endpoint.

    Authentication: Bearer token using LITELLM_MASTER_KEY.
    No API keys are stored here — keys are managed by the LiteLLM Gateway
    via its config.yaml and environment variables.
    """

    DEFAULT_TIMEOUT = 120.0  # seconds

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        master_key: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        self.model = model or os.getenv("LITELLM_MODEL", "openrouter-gpt-oss")
        base = base_url or os.getenv("LITELLM_BASE_URL", "http://litellm:4000")
        self.base_url = base.rstrip("/")
        self.master_key = master_key or os.getenv("LITELLM_MASTER_KEY", "")
        self.timeout = timeout or float(os.getenv("LITELLM_TIMEOUT", str(self.DEFAULT_TIMEOUT)))
        self.extra_kwargs = kwargs

        self.provider_name = "litellm"

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to the LiteLLM Gateway.

        Error handling:
        - No master key configured → mock fallback with clear message
        - Gateway unreachable (connection error) → mock fallback
        - HTTP 401/403 → mock fallback (auth issue)
        - HTTP 404 → mock fallback (model not found)
        - HTTP 429 → mock fallback (rate limited)
        - HTTP 500+ → mock fallback (server error)
        """
        if not self.master_key:
            msg = (
                "LiteLLM Gateway master key not configured. "
                "Set LITELLM_MASTER_KEY environment variable."
            )
            logger.warning(f"LiteLLM Gateway: {msg}")
            return self._fallback_response(msg)

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.master_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.message}],
        }

        # Pass through any extra kwargs (temperature, max_tokens, etc.)
        # but exclude known httpx/response fields
        excluded = {"timeout", "model", "messages"}
        for key, value in self.extra_kwargs.items():
            if key not in excluded:
                payload[key] = value

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)

                if response.status_code == 401 or response.status_code == 403:
                    msg = (
                        "LiteLLM Gateway authentication failed. "
                        "Check LITELLM_MASTER_KEY."
                    )
                    logger.warning(f"LiteLLM Gateway: {msg}")
                    return self._fallback_response(msg)

                elif response.status_code == 404:
                    msg = (
                        f"LiteLLM Gateway: model '{self.model}' not found. "
                        "Check LITELLM_MODEL and that the model is defined in config.yaml."
                    )
                    logger.warning(f"LiteLLM Gateway: {msg}")
                    return self._fallback_response(msg)

                elif response.status_code == 429:
                    msg = "LiteLLM Gateway rate limit exceeded. Please wait and try again."
                    logger.warning(f"LiteLLM Gateway: {msg}")
                    return self._fallback_response(msg)

                elif response.status_code >= 500:
                    msg = f"LiteLLM Gateway server error ({response.status_code}). Please try again later."
                    logger.error(f"LiteLLM Gateway: {msg}")
                    return self._fallback_response(msg)

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return ChatResponse(message=content, role=MessageRole.assistant)

        except httpx.ConnectError:
            msg = (
                "LiteLLM Gateway is unreachable. "
                "Ensure the litellm service is running (docker compose up -d litellm)."
            )
            logger.warning(f"LiteLLM Gateway: {msg}")
            return self._fallback_response(msg)

        except httpx.TimeoutException:
            msg = f"LiteLLM Gateway request timed out after {self.timeout}s."
            logger.warning(f"LiteLLM Gateway: {msg}")
            return self._fallback_response(msg)

        except httpx.HTTPStatusError as e:
            msg = f"LiteLLM Gateway HTTP error {e.response.status_code}: {str(e)}"
            logger.error(f"LiteLLM Gateway: {msg}")
            return self._fallback_response(msg)

        except Exception as e:
            msg = f"LiteLLM Gateway request failed: {str(e)}"
            logger.error(f"LiteLLM Gateway: {msg}")
            return self._fallback_response(msg)

    def _fallback_response(self, error_message: str) -> ChatResponse:
        """Return a gracefully-handled fallback response (no stack traces)."""
        return ChatResponse(
            message=f"[LiteLLM Gateway] {error_message}",
            role=MessageRole.assistant,
        )

    def __repr__(self) -> str:
        return f"LiteLLMProvider(model={self.model!r}, base_url={self.base_url!r})"
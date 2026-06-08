"""
Ollama Provider

Direct Ollama API integration for local LLM inference.

Configuration:
- LLM_PROVIDER=ollama
- OLLAMA_HOST=http://localhost:11434 (default, or set OLLAMA_BASE_URL)
- OLLAMA_MODEL=model-name (default: llama3.2)
- OLLAMA_KEEP_ALIVE=5m (default, how long to keep model in memory)

Note: Ollama must be running locally. Install from https://ollama.ai/
"""

import os
from typing import Optional

import httpx

from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.base import LlmProvider


class OllamaProvider(LlmProvider):
    """
    Ollama local LLM provider.
    
    Ollama runs LLMs locally, providing:
    - Privacy (no data leaves your machine)
    - No API costs
    - Fast inference with local GPU
    
    Common models:
    - llama3.2, llama3.2:1b, llama3.2:3b
    - mistral, mixtral
    - codellama:7b, codellama:13b
    - qwen2.5:7b, qwen2.5:14b
    - deepseek-r1:7b, deepseek-r1:14b
    - gemma2:2b, gemma2:9b
    """
    
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.2"
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        keep_alive: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        self.host = host or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", self.DEFAULT_HOST)
        self.model = model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        self.timeout = timeout
        self.extra_kwargs = kwargs
        
        self.base_url = self.host.rstrip("/")
        
        # Check if Ollama is reachable
        self._available = self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama server is running."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to Ollama API.
        
        Falls back to mock response if:
        - Ollama server is not running
        - Model not installed
        - Request fails
        """
        if not self._available:
            return self._mock_fallback(
                f"Ollama server not available at {self.base_url}. "
                f"Is Ollama running? Install from https://ollama.ai/"
            )
        
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.message}],
            "stream": False,
        }
        
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        
        # Add any extra params (temperature, top_p, etc.)
        payload.update(self.extra_kwargs)
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["message"]["content"]
                return ChatResponse(message=content, role=MessageRole.assistant)
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return self._mock_fallback(
                    f"Model '{self.model}' not found. "
                    f"Install with: ollama pull {self.model}"
                )
            else:
                return self._mock_fallback(f"Ollama HTTP error {e.response.status_code}: {str(e)}")
        except Exception as e:
            return self._mock_fallback(f"Ollama error: {str(e)}")
    
    def _mock_fallback(self, error_message: str) -> ChatResponse:
        """Return a mock response when Ollama cannot be used."""
        return ChatResponse(
            message=f"[Ollama Fallback] {error_message}. Using mock response instead.",
            role=MessageRole.assistant,
        )
    
    def __repr__(self) -> str:
        return f"OllamaProvider(model={self.model!r}, host={self.host!r})"
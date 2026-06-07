import os
import httpx
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.base import LlmProvider

class OpenAICompatibleProvider(LlmProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        if not self.api_key:
            return ChatResponse(
                message="Error: OPENAI_API_KEY is not set. Using mock fallback.",
                role=MessageRole.assistant,
            )
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.message}],
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return ChatResponse(message=content, role=MessageRole.assistant)
        except Exception as e:
            return ChatResponse(
                message=f"Error calling OpenAI-compatible API: {str(e)}",
                role=MessageRole.assistant,
            )